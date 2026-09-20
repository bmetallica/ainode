# update.md — Befunde und Plan

Aufgenommen am 2026-09-20 auf `main` @ c80304d (2147 Tests). Anlass: zwei Nodes
sind beim Laden von MiniMax-M3 abgestürzt, das UI friert beim Laden kurz ein,
die Instanzkarten sind überladen, geladene Modelle fehlen unter „Modelle", und
die Modelle- und Server-Seiten brauchen lange.

Teil 1 sind die Ursachen, jede mit Fundstelle im Code und einer ehrlichen
Angabe, ob sie bewiesen oder vermutet ist. Teil 2 ist der Plan, nach Risiko
geordnet. Teil 3 sind die Fragen, die ich ohne eine Messung auf der Hardware
nicht beantworten kann.

---

## Teil 1 — Befunde

### B1 · Es gibt keinen Schutz des Host-Speichers  (Ursache des Absturzes)

**Bewiesen.** Die einzige Zulassungsprüfung vor einem Start steht in
`ainode/models/api_routes.py::append_solo_instance` und hat drei Lücken, die
alle auf denselben Absturz hinauslaufen:

1. Sie greift **nur bei gestapelten Loads** (`if not is_primary and not
   replaced_primary`). Das erste Modell auf einem Knoten wird ungeprüft
   gestartet.
2. Sie prüft **nur die Summe der `gpu_memory_utilization`-Anteile** gegen eine
   feste Obergrenze von 0.90. Das ist ein Verhältnis, kein Speicher — und auf
   GB10 ist „GPU-Speicher" derselbe physische Speicher wie der des Betriebs\
   systems. Ein Anteil von 0.85 lässt rechnerisch 15 % übrig, aber wenn davon
   der Kernel, die Page-Cache und AINode selbst leben müssen, ist das zu wenig.
3. Der **verteilte Pfad prüft gar nichts.** `ainode/engine/sharding_routes.py`
   ruft `backend.start_distributed()` ohne jede Speicherprüfung auf — und genau
   dieser Pfad hat die beiden Nodes mitgenommen.

Dazu kommt: es gibt **keine laufende Überwachung**. Ist der Start einmal durch,
schaut niemand mehr hin. Wenn vLLM nach dem Laden der Gewichte den KV-Cache
anlegt und dabei über das Ende des Speichers hinausläuft, gibt es nichts, das
eingreift, bevor das System steht.

### B2 · Der Ladepfad blockiert die Event-Loop  (UI weg, Node verschwindet)

**Bewiesen.** Beide Startpfade rufen synchrone, minutenlange Arbeit direkt im
aiohttp-Handler auf:

| Pfad | Stelle | Was dort synchron passiert |
|---|---|---|
| solo | `models/api_routes.py::handle_model_load` → `append_solo_instance` | `existing.backend.stop()` (docker stop), `_fetch_weights_from_a_peer()` (**ssh + rsync, Gigabytes**), `backend.start()` |
| verteilt | `engine/sharding_routes.py::handle_sharding_launch` | `backend.start_distributed()` — SSH zu allen Peers, Image-Abgleich, Gewichte spiegeln, Ray formen |

`append_solo_instance` ist eine gewöhnliche `def`, kein `async def`, und wird
ohne `run_in_executor` aufgerufen. Solange sie läuft, läuft die Event-Loop
dieses Prozesses nicht — und in derselben Loop hängt
`ainode/discovery/broadcast.py::BroadcastSender`, das per
`asyncio.create_task(self._broadcast_loop())` die UDP-Ankündigung sendet.

Damit erklärt sich die beobachtete Reihenfolge vollständig:

* der Knoten, der lädt, **hört auf zu senden** → der Head stuft ihn als stale
  und dann offline ein → er verschwindet aus der Clusteransicht;
* sein eigenes UI und seine API antworten in dieser Zeit nicht;
* er taucht wieder auf, sobald die Loop weiterläuft — „als ob er sich frisch
  verbindet", weil die Ankündigung neu eintrifft.

Dass **zuerst das Head-UI wegging**, hat vermutlich dieselbe Ursache über einen
Umweg: bis #119 hat `_selectNodeIds` im Formular die Head-Node **immer**
mitaktiviert. Eine Auswahl „nur Node 3" war in Wahrheit „Head + Node 3", also
zwei Knoten, also der **verteilte** Pfad — und der läuft auf dem **Head**.
Damit blockiert erst der Head (start_distributed), dann Node 3 (Ray-Worker),
und Node 3 kommt zuletzt zurück. Das deckt sich exakt mit der Beschreibung.
Die Auswahl ist seit #119 korrigiert; das Blockieren nicht.

> **Wichtig für B1:** ein Speicherwächter, der als asyncio-Task läuft, wird von
> genau diesem Problem mit lahmgelegt — und zwar in dem Moment, in dem er
> gebraucht wird. Der Wächter muss in einem **eigenen Thread** laufen.

### B3 · Die Instanzkarten sind überladen

**Gestaltung, kein Fehler.** `renderInstances()` in `app.js` packt heute alles
auf die Karte: Modellname, Achsen-Badge, Knotenliste, Ladezeiten, Fehlertext,
Kompilier-Cache-Knopf, Degraded-Hinweis, Fortschrittsbalken, Assistenten-Antwort,
Relaunch, Kernel-Cache, Unload. Das ist gewachsen, weil jede neue Information
irgendwo hin musste, und ist inzwischen der unübersichtlichste Teil des UI.

### B4 · Geladene Modelle fehlen unter „Modelle"

**Bewiesen, zwei getrennte Ursachen.**

1. **Einmalig geladen, nie aktualisiert.** In `app.js`:
   `if (!this.state.catalog) { … fetch('/api/models') … }` und dasselbe Muster
   für `state.downloadedModels`. Beide werden genau einmal pro Seitenaufruf
   geholt. Alles, was danach heruntergeladen oder geladen wird, erscheint bis
   zum nächsten Neuladen der Seite nicht.
2. **Die Seite zeigt nur die Platte des Heads.** `/api/models/downloaded`
   scannt `models_dir` **dieses** Knotens. Ein Modell, das auf Node 3 liegt und
   dort läuft, taucht in der Modelle-Seite des Heads nie auf — auch nicht, wenn
   die Instanzliste es zeigt. Die Instanzliste kommt aus den Ankündigungen, die
   Modelle-Seite aus einem lokalen Verzeichnis-Scan. Zwei Quellen, zwei
   Wahrheiten.

### B5 · `/api/models` läuft bei jedem Aufruf über alle Modellverzeichnisse

**Bewiesen.** `registry.py::_dir_size_gb`:

```python
total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
```

Das wird aus `list_available()` **pro Katalogeintrag, der auf Platte liegt**
aufgerufen, und aus `list_downloaded()` sogar **zweimal pro Modell**. Bei euren
Modellbeständen sind das hunderte Gigabyte und zehntausende `stat()`-Aufrufe
pro Seitenaufruf — und `app.js` ruft `/api/models` an sechs Stellen auf, unter
anderem in `populateLaunchModels()`.

Das erklärt die langsame Modelle-Seite und macht nebenbei die Page-Cache kaputt,
die der nächste Modellstart gebrauchen könnte.

### B6 · Die Server-Seite probt seriell mit 2-Sekunden-Timeouts

**Bewiesen.** `api/server_routes.py::handle_server_status` ruft
`_probe_loaded_models` für die primäre Instanz und dann in einer Schleife für
jede gestapelte Instanz — nacheinander, jeweils mit `total=2`. Vier Instanzen,
von denen zwei nicht antworten, sind vier Sekunden Wartezeit, bevor die Seite
etwas zeigt. Danach holt das Frontend `/api/server/endpoints` als zweiten,
getrennten Aufruf.

### B7 · Die Ladezeit-Zeile ist nicht zu sehen

**Vermutlich kein Fehler, sondern ein fehlendes Update.** Die Zeile kommt mit
#120 und liegt seit heute auf `main`. Ein Blick, der das entscheidet:

```bash
curl -sS localhost:3000/api/status | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('load_timeline' in d, d.get('load_timeline'))"
```

`False` heißt: das Update ist auf dem Head noch nicht ausgerollt. `True` mit
leerer Liste heißt: ausgerollt, aber seit dem Neustart wurde nichts geladen —
die Liste füllt sich erst beim nächsten Start. Kommt dann immer noch nichts auf
der Karte an, ist es ein echter Fehler und gehört in Schritt S4 mit aufgeräumt.

---

## Teil 2 — Plan

Reihenfolge nach Risiko: was die Maschine umbringt zuerst, was nur nervt
zuletzt. Jeder Schritt ist ein eigener PR.

### S1 · Host-Speicherwächter  ⟵ zuerst

**Ziel:** kein vLLM-Container darf den Host mitnehmen.

* Neuer `ainode/safety/memory_guard.py`, der in einem **eigenen Thread** läuft
  (siehe B2 — ein asyncio-Task wäre genau dann taub, wenn er gebraucht wird)
  und alle 2 Sekunden `MemAvailable` aus `/proc/meminfo` liest. Direkt aus
  `/proc`, nicht über psutil: das ist der Wert des **Hosts**, auch aus dem
  Container heraus, und er ist nicht von einer cgroup-Grenze verfälscht.
* Zwei Schwellen, beide im UI einstellbar:
  * **Warnung** (Vorgabe 8 GB): keine neuen Loads mehr zulassen, Banner im UI.
  * **Kritisch** (Vorgabe 4 GB): die **zuletzt gestartete** Instanz sofort
    abschießen — `docker kill`, nicht `stop`, weil ein sauberes Herunterfahren
    zehn Sekunden dauert und die Maschine die nicht mehr hat.
* Der Abschuss setzt `load_error` auf der Instanz auf einen Satz, der sagt was
  passiert ist und warum: *„Vom Speicherwächter beendet: es waren nur noch
  N MB Host-Speicher frei (Grenze: 4096 MB). Diese Instanz war die zuletzt
  gestartete."* Der Fehlerassistent aus #118 kann den dann erklären.
* Vorgabewerte als Preset: `dgx-spark` (8/4 GB) und `generic`. Konfiguration in
  `NodeConfig`, damit sie in `config.json` steht und pro Knoten gilt.
* Läuft auf **jedem** Knoten, nicht nur auf dem Head — abgestürzt sind die
  Nodes.

**Prüfung auf der Hardware:** Schwelle testweise auf einen Wert knapp unter dem
aktuell freien Speicher setzen, irgendein Modell laden, zusehen wie der Wächter
zuschlägt und die Meldung auf der Karte erscheint — ohne dass der Host steht.

**Risiko:** ein zu eifriger Wächter beendet ein gesundes Modell. Gegenmittel:
die kritische Schwelle muss zweimal in Folge unterschritten werden, und der
Wächter fasst nie eine Instanz an, die älter als der laufende Start ist, solange
eine jüngere existiert.

### S2 · Zulassungsprüfung vor dem Start — der Planer als Torwächter

**Ziel:** der Absturz aus B1 soll gar nicht erst anfangen.

* Der Planer aus #119 rechnet bereits aus, was eine Belegung kostet. Beide
  Startpfade fragen ihn **vor** dem Start und lehnen ab, wenn nach der geplanten
  Belegung weniger als die Warnschwelle Host-Speicher übrig bliebe.
* Die Ablehnung nennt Zahlen und einen Ausweg (weniger `max-model-len`,
  niedrigeres `gpu-memory-utilization`, ein Knoten mehr) — nicht nur „passt
  nicht".
* Ein bewusstes Übersteuern bleibt möglich (`"force": true` im Body, Häkchen im
  Formular), weil der Planer konservativ rechnet und der Betreiber es besser
  wissen darf. Ohne Häkchen gilt der Plan.
* Damit fällt die alte 0.90-Anteilsregel aus `append_solo_instance` weg: sie
  wird durch eine Rechnung in Gigabyte ersetzt, die auch für den ersten Load
  und für den verteilten Pfad gilt.

### S3 · Den Ladepfad aus der Event-Loop nehmen

**Ziel:** kein Knoten verschwindet mehr aus dem Cluster, weil er lädt.

* `append_solo_instance` und `backend.start_distributed()` wandern in
  `run_in_executor`. Beide sind bereits synchrone Funktionen — es ist ein
  `await loop.run_in_executor(None, …)` an zwei Stellen, kein Umbau.
* Die Antwort des Endpunkts kommt dann sofort („launching"), was sie ohnehin
  schon tut — nur eben, ohne vorher minutenlang zu blockieren.
* Danach prüfen, ob noch weitere blockierende Aufrufe in Handlern stehen; die
  Kandidaten sind `engine.stop()` im Unload-Pfad und `_fetch_weights_from_a_peer`.
* Gegenprobe: ein Test, der den Ladepfad mit einer künstlich langsamen
  `start()`-Attrappe aufruft und dabei nachweist, dass die Loop weiterläuft.

**Prüfung auf der Hardware:** während eines Ladevorgangs auf Node 3 im Browser
die Clusteransicht offen lassen. Erfolg: kein Knoten verschwindet, das Head-UI
bleibt bedienbar, der Fortschritt der Karte läuft durchgehend weiter.

### S4 · Instanzkarte aufräumen, Details in einen Dialog

**Ziel:** die Karte zeigt vier Dinge, alles andere ist einen Klick entfernt.

Auf der Karte bleibt:

* Modellname
* Status als **ein** Wort mit Farbe: `LÄDT` (gelb, mit Fortschritt), `BEREIT`
  (grün), `FEHLER` (rot), `GESTOPPT` (grau)
* Knopf **Details**
* Knopf **Entladen**

In den Details-Dialog wandert: die Ladezeit-Aufschlüsselung, der vollständige
Fehlertext, *EXPLAIN THIS ERROR* mit der Antwort des Assistenten, Kernel-Cache
leeren, Relaunch, die Startparameter (aus `/api/instances/launch-config`), die
Knoten und die Achse, Port und Instanz-ID.

Dabei gleich B7 prüfen: wenn die Ladezeit im Dialog erscheint, war es nur das
fehlende Update; wenn nicht, liegt der Fehler in der Kette
Record → Ankündigung → `/api/nodes` → Karte, und die ist im Dialog leichter zu
sehen als auf der überladenen Karte.

### S5 · Modelle-Seite: Clustersicht, Aktualisierung, Größen-Cache

Drei Dinge, die zusammengehören:

1. **Größen zwischenspeichern** (behebt B5). `_dir_size_gb` bekommt einen Cache
   mit Schlüssel `(Pfad, mtime des Verzeichnisses)`; außerdem zählt nur, was
   zählt — Gewichtsdateien, wofür `planner/facts.py::weight_bytes_on_disk`
   schon existiert. Nach Download und Löschen wird der Eintrag verworfen.
2. **Aktualisieren statt einmal holen** (behebt B4.1). `state.catalog` und
   `state.downloadedModels` bekommen ein TTL und werden nach jedem Download,
   Löschen und Modellstart neu geholt.
3. **Die Platten aller Knoten** (behebt B4.2). Ein neuer Endpunkt sammelt, was
   wo liegt — pro Modell die Liste der Knoten, die es haben. Die Karte zeigt
   das als kleine Knotenpunkte, und „auf diesen Knoten spiegeln" wird damit zu
   einer sichtbaren Aktion statt zu einem Nebeneffekt des Ladens.

### S6 · Server-Seite parallelisieren

`handle_server_status` sammelt die Proben mit `asyncio.gather` statt
nacheinander und senkt das Timeout auf 1 s; `/api/server/endpoints` ist eine
Konstante und gehört in dieselbe Antwort statt in einen zweiten Aufruf. Aus
„vier Sekunden nacheinander" wird „eine Sekunde insgesamt".

### S7 · UI-Durchgang

Zum Schluss, wenn die harten Sachen sitzen: alle Ansichten einmal auf dieselben
drei Fragen abklopfen —

* holt sie Daten, die sie nicht braucht (wie B5)?
* holt sie Daten einmal und nie wieder (wie B4.1)?
* zeigt sie den Zustand **dieses** Knotens, wo der des Clusters gemeint ist
  (wie B4.2)?

Das sind die drei Muster, die in jedem der obigen Befunde stecken; ich erwarte,
sie noch an weiteren Stellen zu finden.

---

## Teil 3 — Was ich nicht weiß

1. **Warum genau das Head-UI zuerst wegging.** Die Erklärung über die erzwungene
   Head-Auswahl (B2) passt auf die beschriebene Reihenfolge, ist aber nicht
   bewiesen. Nach dem Ausrollen von #119 und S3 muss der Effekt verschwunden
   sein; tut er das nicht, fehlt eine Ursache.
2. **Was vor der ersten Engine-Zeile passiert.** Container starten und torch
   importieren ist bis heute ungemessen — genau die Lücke, die #120 sichtbar
   macht. Erst danach lässt sich sagen, ob ein warm gehaltener Engine-Container
   etwas bringen würde.
3. **Ob 4 GB die richtige kritische Schwelle sind.** Der Wert ist geschätzt. Er
   ist einstellbar, und nach dem ersten echten Eingriff des Wächters wissen wir,
   ob er zu früh oder zu spät kommt.
