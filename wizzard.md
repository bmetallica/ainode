# Profil-Wizard — Plan und Roadmap

Stand: 2026-09-29, `main` nach #237. **Nur Plan — noch nichts davon gebaut.**

Dein Wunsch, in eigenen Worten zusammengefasst:

> Ein Wizard, in dem man fertige Profile plant: Name → Modelle → Nodes →
> dann die Parameter (Cache usw.) passend zu den zugeordneten Nodes, live
> gekoppelt (erhöhe ich den Cache von A auf Node 1, bekommt B dort weniger).
> Embedding- und Image-Modelle gehören dazu. In der Node-Konfiguration eine
> maximale VRAM-Grenze. Mit dem Wizard erstellte Profile lassen sich löschen
> und vollständig im Wizard wieder bearbeiten.

Unten: was es schon gibt, was daraus folgt, wie ich es bauen würde, was ich
ergänzen würde, die Entscheidungen, die ich von dir brauche, und die Roadmap
in PR-großen Schritten.

---

## 1. Was schon da ist — und was fehlt

| Baustein | Stand | Für den Wizard |
|---|---|---|
| **Profile** (`profiles/store.py`, `apply.py`) | `ProfileEntry` ist ein serialisierter Launch: Modell, `kind` (llm/embedding/image), `node_ids`, Achse, gmu, `max_model_len`, KV-Dtype, Flags, Image-Knöpfe. `apply` **konvergiert** den Node, auf dem es läuft: stoppt dort, was nicht im Profil steht, startet seriell, wartet je Modell. Standardprofil beim Boot. | Das Ergebnis des Wizards **ist** ein Profil. Anwenden, Default, Boot-Restore gibt es schon — der Wizard muss nur gute Einträge erzeugen. |
| **Planer** (`planner/compute.py`) | Plant **ein** Modell gegen freie Node-Budgets: Split, Cache, Fenster, Parallelität, Prognose; nutzt Messungen (#219, #228, #229). | Rechenkern pro Modell wiederverwendbar. **Fehlt:** mehrere Modelle, die sich **einen** Node teilen. |
| **Image-Planer** (`plan_for_image`) | Gewichte + Spitze abhängig von `max_image_size`. | Fester Block im Node-Budget. |
| **Embeddings** (`embeddings/manager.py`) | In-Process im AINode-Container (sentence-transformers), kein vLLM, keine gmu. | Fester Block (Gewichte + Aufschlag) auf dem Node, auf dem er läuft. **Fehlt:** Größenschätzung im Planer. |
| **Speichergrenzen** | Memory-Guard (Warn/Kritisch, GiB), Utilization-Cap aus *aktuell freiem* Speicher, Planer-Headroom. | **Fehlt:** eine vom Operator gesetzte **Obergrenze pro Node**. |
| **UI** | Profilkarten (anlegen per „Capture“, anwenden, Default, löschen), Launch-Seitenleiste mit Live-Plan (#214–#216). | **Fehlt:** der Wizard selbst. |

Der Kern der Aufgabe ist also nicht die Oberfläche, sondern ein **Planer für
einen ganzen Node-Haushalt** — der Rest setzt auf Vorhandenem auf.

---

## 2. Das Speichermodell, auf dem alles steht

Warum „Cache von A hoch → B weniger“ überhaupt so ist, und woran der Wizard
rechnen muss:

**Budget pro Node** (was Profile dort belegen dürfen):

```
budget(n) = min(VRAM-Grenze(n), Gesamt(n))   ← neue Einstellung, §5
            − System-Reserve (4 GB)
            − Guard-Linie + Abstand          ← wie heute im Planer
```

Bewusst **nicht** „aktuell frei“: ein Profil beschreibt den Node, wie er nach
dem Anwenden aussieht. Dafür muss das Anwenden auf **jedem** beteiligten Node
stoppen, was nicht im Profil steht — heute tut `apply` das nur auf dem Node,
auf dem es läuft (E1, Phase 7).
Was heute läuft, ist für die Planung irrelevant (für die Vorschau nicht, §4.5).

**Was jede Instanz auf jedem ihrer Nodes belegt:**

| Art | Belegung auf Node n | Stellschraube |
|---|---|---|
| LLM (vLLM) | `gmu × Gesamt(n)` — vLLM füllt das mit Gewichten/Rang + Engine + KV-Cache | Cache-Größe ⇄ Kontext × Sitzungen ⇄ gmu |
| LLM über mehrere Nodes | dieselbe gmu auf **allen** seinen Nodes (vLLM gibt allen Rängen denselben Anteil) → der engste Node begrenzt | wie oben, aber an mehrere Node-Budgets gekoppelt |
| Image (diffusers) | Gewichte + Spitze eines Laufs (wächst mit `max_image_size`) | Auflösung |
| Embedding | Gewichte × ~1,3 (Aktivierungen, Batch) | keine (evtl. Batchgröße) |

**Die Bedingung:** Für jeden Node n gilt `Σ Belegung aller Instanzen auf n ≤ budget(n)`.

Daraus folgen die Kopplungen, die du beschrieben hast — und eine, die man
leicht übersieht: Ein Modell, das über Node 1 **und** Node 2 läuft, koppelt die
beiden Nodes. Mehr Cache für A (nur Node 1) kann dadurch C (Node 1+2) kleiner
machen, und weil C auf Node 2 dann weniger belegt, wird dort Platz für B frei.
Der Wizard muss das zeigen, sonst wirkt es wie Zufall.

**Reihenfolge beim Start:** vLLM prüft beim Start, ob `gmu × Gesamt` gerade
frei ist. Solange die Summe ins Budget passt und seriell gestartet wird (macht
`apply` schon), ist die Reihenfolge egal. Ich würde trotzdem **große zuerst**
starten — ein Image- oder Embedding-Modell, das zuerst lädt, fragmentiert nichts,
aber der Page-Cache eines gerade gelesenen 150-GB-Checkpoints drückt „frei“
kurzzeitig.

---

## 3. Der Verteiler — was „live gekoppelt“ technisch heißt

Pro LLM-Instanz gibt es eine Größe, die sich verteilen lässt: den KV-Cache
(pro Rang). Alles andere (Gewichte, Engine, Image-Spitze, Embeddings) ist fest.

Jedes LLM hat im Wizard einen von drei Modi:

1. **Festgelegt über Nutzung** — „Kontext 131072 × 2 Sitzungen“ → daraus der
   nötige Cache (mit gemessenen Kosten pro Token, wo vorhanden).
2. **Festgelegt über Größe** — Schieber „Cache: 24 GB“ → daraus Tokens, und
   Kontext × Sitzungen passend abgeleitet (wie heute im Launch-Formular).
3. **Automatisch** — bekommt einen Anteil vom Rest, gewichtet nach einer
   Priorität (Standard: gleich).

**Algorithmus** (klein, deterministisch, erklärbar):

1. Pro Node: Budget minus alle festen Blöcke minus alle festgelegten Caches
   (Modi 1+2). Wird ein Node dabei negativ → **Konflikt** (rot), mit Angabe,
   welche Modelle auf dem Node zusammen zu viel wollen.
2. Den Rest per *progressive filling* auf die automatischen Modelle verteilen:
   alle wachsen gleichmäßig (nach Priorität), ein Modell stoppt, sobald
   irgendeiner seiner Nodes voll ist; die anderen wachsen weiter. Das ist das
   Standardverfahren für geteilte Kapazität über mehrere Ressourcen
   (max-min fair) und ergibt genau das Verhalten „A hoch → B runter“.
3. Pro Instanz daraus: gmu (vom engsten Node), Tokens, Kontext, Sitzungen,
   `--max-num-seqs`, Prognose-Balken — mit derselben Arithmetik wie
   `plan_for` (Messwerte zuerst, dann MoE-Kalibrierung, dann Schätzung).
4. Grenzen: jede gmu ≤ Rezept-`recommended_gmu`; ein Cache unter einer
   Anfrage (Fenster < 4096) ist ein Fehler, keine Lösung.

**Wo es läuft:** auf dem Server (`planner/household.py`, reine Funktion + ein
Endpunkt `POST /api/planner/household` mit dem ganzen Entwurf). Der Server hat
die Modell-Fakten, Messungen und Node-Daten; der Browser schickt den Entwurf
(debounced, 300 ms, wie heute) und zeichnet das Ergebnis. Die Zeichenfunktionen
gehören in `lib.js` (#236) und werden mit Node getestet.

**Was der Operator beim Schieben sieht:** pro Node ein gestapelter Balken
(je Modell ein Segment, Farbe pro Modell, fest/Cache getrennt schraffiert), die
Summe gegen die Grenze, und an jedem betroffenen Modell der neue Wert mit
Delta („B: 180k → 131k Tokens“). Ein festgelegtes Modell bewegt sich nie von
allein; stößt der Operator an, bekommt er „Node 1 ist voll — B ist fest auf
48 GB, lösen?“.

---

## 4. Der Ablauf im Wizard

Vollbild-Dialog mit Schrittleiste; jeder Schritt jederzeit zurück; der Entwurf
wird im Browser zwischengespeichert (ein versehentlich geschlossener Tab
verliert nichts).

### 4.1 Start
- **Neues Profil** · **Profil bearbeiten** (Liste der Profile) · *Duplizieren*.
- Bestehende, per „Capture“ entstandene Profile lassen sich ebenfalls im Wizard
  öffnen: ihre Einträge werden übernommen, alle Caches als „festgelegt über
  Nutzung“ mit den gespeicherten Werten.

### 4.2 Name
- Name (bestehende Regel: 64 Zeichen, `[A-Za-z0-9 _.-]`), Beschreibung
  optional, Häkchen „als Standard beim Boot“.

### 4.3 Modelle
- Auswahl aus heruntergeladenen Modellen, gruppiert **LLM · Embedding · Image**,
  mit Größe auf Platte und Badge *gemessen/geschätzt*.
- Optional auch nicht heruntergeladene (Katalog/Suche) → Hinweis, dass beim
  Anwenden erst geladen und gespiegelt wird (mit Größe und grober Dauer).
- Dasselbe Modell **zweimal** (auf verschiedenen Nodes) ist erlaubt — das sind
  Replikate (O3 aus `upgrade-fixes.md`, siehe §6).

### 4.4 Zuordnung Modelle → Nodes
- Spalten = Nodes (Name, Gesamt, **Grenze**, online/offline), Karten = Modelle;
  ziehen oder per Häkchen zuordnen.
- LLM über mehrere Nodes: Achse Tensor/Pipeline (Regeln wie heute: TP nur 1/2/4/8
  und Köpfe teilbar, PP höchstens Layer-Zahl). Image und Embedding: genau ein Node.
- **Sofortprüfung nur auf Gewichte:** passt die Summe der festen Blöcke plus
  Mindestcache je Modell nicht auf einen Node, wird die Spalte rot, bevor es
  zum Feintuning geht. Vorschlag „Automatisch zuordnen“ (größtes zuerst auf den
  Node mit dem meisten Platz, Split nur wenn nötig) als Startpunkt.

### 4.5 Parameter (der Kern)
- Oben: je Node der gestapelte Balken (§3), live.
- Darunter je Modell eine Karte:
  - **LLM:** Modus (Nutzung / Größe / Automatisch + Priorität), Kontext,
    Sitzungen, Cache-Schieber, KV-Dtype, Ergebnis (Tokens, gmu, `max-num-seqs`),
    Badges *gemessen/kalibriert/geschätzt*, Warnungen (Vision + fp8, nvfp4_ds_mla,
    Drafter bei >1 Sitzung, …). Erweitert: Rezeptflags, eigene Flags (geprüft
    gegen das Image, F2).
  - **Image:** `max_image_size`, Schritte, Standardgröße → Spitze.
  - **Embedding:** nur Anzeige der Belegung.
- Konflikte oben gesammelt, jeder mit Sprung zur Karte.

### 4.6 Übersicht und Speichern
- Pro Node: was läuft, mit welcher gmu, in welcher Startreihenfolge.
- **Vorschau gegen den Ist-Zustand:** „wird gestoppt: X auf Node 2 · bleibt: Y ·
  wird neu gestartet (andere Parameter): Z · wird geladen: W (86 GB)“ — weil
  `apply` konvergiert, ist das keine Kosmetik, sondern die Warnung vor
  Ausfallzeit.
- Knöpfe: **Speichern** · **Speichern und anwenden** · (Standard setzen).

### 4.7 Bearbeiten und Löschen
- Bearbeiten öffnet denselben Wizard mit allen Entscheidungen (Modi,
  Prioritäten, Achsen, festgelegte Werte) — deshalb speichert der Wizard neben
  den Einträgen seinen **Entwurf** (§7).
- Löschen: Bestätigung; ist es das Standardprofil, wird der Standard entfernt;
  ist es gerade angewendet, Frage „Modelle weiterlaufen lassen oder stoppen?“.

---

## 5. VRAM-Grenze pro Node

- Neues Feld `memory_limit_gb` in `NodeConfig` (0 = keine Grenze), einstellbar
  in **Config → Nodes** für jeden Node vom Head aus (Übertragung an den Node mit
  dem Cluster-Key, wie beim Passwort).
- Der Node sendet sie in seiner Ankündigung mit → der Head plant mit den Grenzen
  aller Nodes, auch im normalen Launch-Formular.
- Wirkt überall, wo heute ein Budget entsteht: `node_budgets` (Planer +
  Wizard), `cap_utilization` (gmu-Obergrenze = Grenze/Gesamt), Admission
  (Launch über der Grenze wird abgelehnt, „Launch anyway“ wie bisher).
- Der Memory-Guard bleibt, was er ist (Schutz des Hosts); die Grenze ist eine
  **Planungs- und Zulassungsgrenze**. Anzeige in der Cluster-Ansicht als
  Markierung im Node-Balken.

Warum das über den Wizard hinaus nützt: der Head hatte ~27 GB weniger nutzbar
als die Peers (R3) — mit einer Grenze plant man das einmal ein, statt es bei
jedem Start neu zu erleben.

---

## 6. Was ich ergänzen würde

1. **Replikate statt Split** (O3): Qwen3-Coder-Next einmal pro Node statt TP=2
   ist schneller und ausfallsicherer. Im Wizard ist das einfach „dasselbe
   Modell zweimal, je ein Node“. Dazu braucht der **Router** Lastverteilung
   (heute nimmt er den ersten Treffer) — least-outstanding-requests über die
   Replikate. Ich würde O3 in diesen Plan ziehen statt separat zu bauen.
2. **„Automatisch planen“** als Startpunkt in 4.4/4.5 (Zuordnung + Caches aus
   Prioritäten), damit ein Profil in drei Klicks steht und nur noch
   nachgeschärft wird.
3. **Speichern ohne Anwenden prüft trotzdem:** Modelle auf Platte? Auf den
   richtigen Nodes (Spiegelung)? Flags gegen das Image (F2)? Parser (F1)?
4. **Anwenden mit Fortschritt** in einem Panel (wie Update/Download), pro
   Eintrag Status; bei Fehler: „restliche Einträge trotzdem starten / abbrechen
   / vorheriges Profil wiederherstellen“.
5. **Nach dem Anwenden messen** und im Profil vermerken, was tatsächlich
   belegt wurde (#228/#229) — beim nächsten Öffnen zeigt der Wizard „geplant
   78 GB, gemessen 81 GB“.
6. **Client-Configs** (opencode) direkt aus dem Profil erzeugen — die Limits
   stehen im Plan schon fest (Fenster, Sitzungen, Cache-Anteil, B4).

---

## 7. Datenmodell

Profil bleibt kompatibel; neu ist ein optionaler Block:

```json
{
  "name": "Endausbau",
  "entries": [ … ProfileEntry wie bisher, daraus wird angewendet … ],
  "wizard": {
    "version": 1,
    "models": [
      {"id": "m1", "model": "Qwen/Qwen3-Coder-Next", "kind": "llm",
       "node_ids": ["n1"], "strategy": "solo",
       "mode": "usage", "max_model_len": 131072, "sessions": 2,
       "cache_gb": null, "priority": 1, "kv_cache_dtype": "fp8",
       "extra_vllm_args": []},
      {"id": "m2", "model": "BAAI/bge-m3", "kind": "embedding", "node_ids": ["n2"]}
    ],
    "planned": {"n1": {"budget_gb": 110, "used_gb": 104.2}, "…": {}}
  }
}
```

`entries` ist immer die Wahrheit für `apply` und Boot; `wizard` ist nur das,
was man zum Wiederbearbeiten braucht. Ein Profil ohne `wizard`-Block
(Capture) öffnet der Wizard mit abgeleiteten Werten (§4.1).

---

## 8. Entscheidungen, die ich von dir brauche

| # | Frage | Mein Vorschlag |
|---|---|---|
| E1 | Soll ein Wizard-Profil beim Anwenden **alles andere** auf seinen Nodes stoppen? Heute konvergiert `apply` nur den Node, auf dem es ausgeführt wird (den Head): ein Solo-Modell auf Spark2, das nicht im Profil steht, läuft weiter — und belegt dort genau den Speicher, den der Wizard für etwas anderes eingeplant hat. | Ja: jeder Node, den das Profil benutzt, wird konvergiert (Anwenden fächert an die Peers aus); Nodes ohne Eintrag bleiben unangetastet. Die Vorschau (4.6) zeigt vorher, was wo gestoppt wird. |
| E2 | VRAM-Grenze: nur Planung + Zulassung, oder soll der Guard darüber hinaus eingreifen? | Nur Planung + Zulassung. |
| E3 | Automatische Verteilung: gleiche Anteile, nach Priorität, oder proportional zur Modellgröße? | Nach Priorität, Standard gleich. |
| E4 | O3 (Replikate + Lastverteilung im Router) in diesen Plan aufnehmen? | Ja (Phase 6). |
| E5 | Dürfen im Wizard auch noch **nicht heruntergeladene** Modelle gewählt werden (Download beim Anwenden)? | Ja, mit deutlichem Hinweis auf Größe/Dauer. |

---

## 9. Roadmap

Jede Phase ist ein oder zwei PRs mit eigenem Hardware-Test; nach jeder Phase
ist etwas Nutzbares da.

| Phase | Inhalt | Ergebnis für dich | Test auf den Sparks |
|---|---|---|---|
| **0** | VRAM-Grenze pro Node (§5): Feld, Config-UI für alle Nodes vom Head, Ankündigung, Planer/Cap/Admission | Grenze setzen, normale Launches halten sich daran | Grenze Spark1 = 100 GB → Seitenleiste plant nur noch in 100 GB; Launch darüber wird abgelehnt |
| **1** | Haushalts-Planer (§3) als reine Funktion + `POST /api/planner/household`; Größenschätzung für Embeddings; Tests für Kopplung (A↑ → B↓, Mehrnode-Kopplung, Konflikte) | per API prüfbar | Entwurf mit zwei Modellen auf Spark1 an die API → Summe ≤ Budget, A↑ senkt B |
| **2** | Profil-Datenmodell `wizard` (§7), Validierung beim Speichern, Löschen mit Default/aktiv-Behandlung | Profile können Entwürfe tragen | per API anlegen, lesen, löschen |
| **3** | UI Schritte 4.1–4.4 (Start, Name, Modelle, Zuordnung inkl. Automatisch zuordnen) | Wizard bis zur Zuordnung | Profil mit LLM + Embedding + Image auf 3 Nodes zuordnen |
| **4** | UI Schritt 4.5 (Parameter, Live-Balken, Modi, Konflikte) + 4.6 (Übersicht, Vorschau, Speichern/Anwenden) | **Der Wizard ist benutzbar** | dein Beispiel: A↑ auf Node 1 → B sinkt live; speichern + anwenden → alles läuft, Messung ≈ Plan |
| **5** | Bearbeiten (voller Wizard), Capture-Profile öffnen, Duplizieren | Profile pflegen | bestehendes Profil öffnen, Cache ändern, anwenden → nur das geänderte Modell startet neu |
| **6** | Replikate + Lastverteilung im Router (O3) | ein Modell auf zwei Nodes, Anfragen verteilt | zwei parallele opencode-Sitzungen landen auf beiden Nodes |
| **7** | Anwenden konvergiert jeden beteiligten Node (E1), mit Fortschrittspanel, Fehlerwegen, Messung ins Profil, opencode-Config aus Profil | Komfort | Fehler in einem Eintrag → Rest startet, Wiederherstellung möglich |

Aufwand grob: Phase 0 S–M, 1 M, 2 S, 3 M, 4 L, 5 M, 6 M, 7 M.

Ich würde mit Phase 0 und 1 anfangen — beide sind auch ohne Wizard nützlich
und alles Weitere steht auf ihnen.
