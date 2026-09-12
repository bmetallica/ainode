# Umbau — moderne LLMs im Cluster, und Profile

Ziel: Qwen3.8 und vergleichbare Modelle laufen im Cluster genauso zuverlässig
wie solo, alles Nötige bringt AINode selbst mit, und die Bedienung geht über
das UI. Dazu eine Profilfunktion, mit der sich mehrere Modelle gemeinsam
konfigurieren, laden und beim Start automatisch bereitstellen lassen.

Stand vor dem Umbau: `main` @ a4a0a67, 1132 Tests.

---

## 1. Befund

### 1.1 Das Katalog-Rezept greift nur solo

`catalog_recipe()` wird an **einer** Stelle angewendet — `handle_model_load`
(`models/api_routes.py:691`). `handle_sharding_launch` kennt es nicht.

Für Qwen3.8 heißt das konkret: solo bekommt es `engine_image`
`vllm/vllm-openai:v0.27.1`, `--reasoning-parser qwen3`, `--tool-call-parser
qwen3_coder`, `--kv-cache-dtype auto` und das MTP-Speculative-Config. Verteilt
bekommt es **nichts davon** und scheitert an demselben argparse-Fehler, der
gerade solo behoben wurde.

Das ist der wichtigste Einzelbefund: derselbe Modellklick führt je nach Anzahl
ausgewählter Knoten zu zwei verschiedenen Startkommandos.

### 1.2 `proven_tp` wird nirgends durchgesetzt

Der Katalog führt pro Modell ein `proven_tp` (bei Qwen3.8: 1). Die UI nutzt es
als Untergrenze bei der Empfehlung, aber nichts hindert daran, ein Modell mit
`proven_tp=1` über vier Knoten zu spannen. Bei einem Modell mit
Speculative-Decoding-Konfiguration ist das kein Geschmacksfrage — die
Draft-TP-Größe hängt daran.

### 1.3 Kein Weg, mehrere Modelle gemeinsam zu beschreiben

Der Endausbau des Betreibers ist „drei Modelle, dauerhaft, auf drei Knoten".
Heute heißt das: drei Mal klicken, in der richtigen Reihenfolge, mit jeweils
korrekt gesetztem Speicheranteil — und nach einem Neustart kommen nur die
Solo-Instanzen über `instances.json` zurück, verteilte gar nicht.

Es fehlt eine Beschreibung des **Sollzustands**, die sich anwenden und beim
Start wiederherstellen lässt.

---

## 2. Entscheidungen

Getroffen, nicht erfragt — mit Begründung, damit sie revidierbar sind.

| Frage | Entscheidung | Warum |
|---|---|---|
| Was heißt „Profil anwenden"? | **Konvergieren**: was nicht im Profil steht, wird gestoppt; was fehlt, gestartet; was passt, bleibt. | „Nur hinzufügen" ließe alte Instanzen Speicher halten und macht den Zustand nach zwei Anwendungen unvorhersagbar. Ein Profil beschreibt einen Sollzustand, keine Aktionsliste. |
| Reihenfolge beim Anwenden | **Seriell**, in Profilreihenfolge | Zwei gleichzeitige vLLM-Starts auf einem Unified-Memory-Knoten konkurrieren um Speicher; einer wird OOM-gekillt. Steht so schon im bestehenden Replay-Code. |
| Verhältnis zu `instances.json` | Ein gesetztes **Default-Profil ersetzt** den Manifest-Replay. Ohne Default bleibt alles wie bisher. | Zwei Quellen für „was soll laufen" widersprechen sich sonst. Ohne Profil ändert sich nichts. |
| Knotenbindung im Profil | `node_ids`, mit Namensfallback | Eine ID ist eindeutig; verschwindet sie, meldet das Anwenden das klar, statt still woanders zu starten. |
| Fehler beim Anwenden | Weitermachen, am Ende sammeln | Ein Modell, das nicht lädt, darf die anderen zwei nicht verhindern. |
| Speicherort | `~/.ainode/profiles.json` | Neben `config.json` und `instances.json`, gleiche Backup-Einheit. |

---

## 3. Umbau

### Teil A — Rezeptparität (Voraussetzung für alles Weitere)

`catalog_recipe` wird aus `handle_model_load` in eine gemeinsame Funktion
gezogen, die beide Launch-Routen anwenden. Danach ist ein Modellstart
unabhängig von der Knotenzahl identisch konfiguriert.

Zusätzlich: `proven_tp` wird beim Planen berücksichtigt. Ein Modell mit
`proven_tp=1` auf mehreren Knoten läuft **nicht** TP=N, sondern bekommt eine
Aufteilung, die zu ihm passt (Pipeline), oder eine klare Ablehnung.

*Aufwand: ~0,5 T. Risiko: gering, rein additiv für den verteilten Pfad.*

### Teil B — Profilmodell und API

Neues Modul `ainode/profiles/`:

```
Profile           name, description, entries[], created_at, updated_at
ProfileEntry      model, node_ids[], strategy, gpu_memory_utilization,
                  max_model_len, max_num_seqs, kv_cache_dtype, quantization,
                  engine_image, extra_vllm_args[], served_model_name[],
                  trust_remote_code, embedding (bool)
ProfileStore      laden/speichern/CRUD, default_profile
```

`ProfileEntry` deckt genau die Felder ab, die `parse_load_overrides` annimmt,
plus Platzierung — ein Eintrag ist damit ein serialisierter Launch.

Ein Eintrag mit `embedding: true` wird über den Embedding-Manager geladen statt
als vLLM-Instanz, damit ein Profil den kompletten Endausbau beschreiben kann.

Routen:

```
GET    /api/profiles                 Liste + welches Default ist
POST   /api/profiles                 anlegen
GET    /api/profiles/{name}
PUT    /api/profiles/{name}          ersetzen
DELETE /api/profiles/{name}
POST   /api/profiles/{name}/apply    konvergieren (seriell, Bericht je Eintrag)
POST   /api/profiles/{name}/default  als Default setzen ("" löscht ihn)
POST   /api/profiles/capture         laufenden Zustand als Profil sichern
```

`capture` ist der Weg, wie ein Betreiber realistisch zu einem Profil kommt:
erst von Hand richtig einstellen, dann festhalten — nicht ein Formular mit
zwölf Feldern dreimal ausfüllen.

*Aufwand: ~1 T.*

### Teil C — Anwenden und Start

`apply_profile(app, profile)`:
1. Ist-Zustand ermitteln (InstanceManager + Embedding-Manager).
2. Vergleichen: was passt, was fehlt, was ist zu viel.
3. Überzähliges stoppen.
4. Fehlendes seriell starten, jeweils über die bestehenden Launch-Pfade —
   kein zweiter Startweg, der auseinanderdriftet.
5. Bericht je Eintrag zurückgeben.

Beim Start ersetzt ein gesetztes Default-Profil den Manifest-Replay
(`replay_instances_on_startup`). Ohne Default bleibt der bestehende Weg.

*Aufwand: ~1 T.*

### Teil D — UI

Neuer Reiter **Profiles** neben Cluster/Chat/Server/Models/Training/Config:

- Liste der Profile, Default markiert
- Pro Profil: Einträge als Tabelle (Modell, Knoten, Aufteilung, Speicher),
  Knöpfe *Anwenden*, *Als Default*, *Bearbeiten*, *Löschen*
- **Aktuellen Zustand sichern** als primärer Weg zum ersten Profil
- Beim Anwenden: Fortschritt je Eintrag, Fehler im Klartext

*Aufwand: ~1 T.*

### Teil E — Katalog und Doku

- Qwen3.8-27B im Katalog auf `proven_tp` prüfen und als clustertauglich
  kennzeichnen, wo belegt.
- `docs/mesh/ANLEITUNG-3-NODE-MESH.md` um einen Abschnitt „Endausbau als
  Profil" erweitern.
- Updateanleitung.

*Aufwand: ~0,5 T.*

---

## 4. Was dieser Umbau nicht macht

- **Keine automatische Speicherplanung.** Welches Modell wie viel bekommt,
  entscheidet weiterhin der Betreiber. Eine Automatik bräuchte verlässliche
  Größenangaben pro Quantisierung, die der Katalog nur für kuratierte Modelle
  hat.
- **Keine Modellumverteilung bei Knotenausfall.** Ein Profil beschreibt einen
  Sollzustand, stellt ihn aber nicht laufend wieder her. Der bestehende
  DEGRADED-Pfad mit Relaunch-Knopf bleibt der Weg dafür.
- **Kein Profil-Import/Export über Knoten hinweg.** `profiles.json` lässt sich
  kopieren; ein Sync-Mechanismus wäre ein eigenes Stück Arbeit.

---

## 5. Reihenfolge

A → B → C → D → E. A ist Voraussetzung, weil ein Profil sonst je nach
Knotenzahl unterschiedlich startet. D erst nach C, damit das UI gegen eine
fertige API gebaut wird.
