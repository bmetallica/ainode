# Upgrade- und Fix-Liste

Stand: 2026-09-29, `main` bei `24bab34` (#218). Nur eine Liste — nichts davon
ist umgesetzt.

Jeder Punkt sagt, **wo** er sitzt, **warum** er zählt, **was** ich tun würde
und wie groß das ist (S = Stunden, M = ein Tag, L = mehrere Tage). Und jeder
sagt, ob er **belegt** ist (im Code nachgelesen oder auf dem Cluster gemessen)
oder nur **vermutet** — die Vermutungen stehen dabei, damit niemand sie für
Befunde hält.

---

## Reihenfolge, die ich vorschlagen würde

1. **K1–K3** (Sicherheit) — ein offenes Admin-API mit Docker-Socket ist das
   größte Einzelrisiko im Projekt, und K2 ist eine Aussperrfalle.
2. **B1–B3** — verteilte Instanzen überleben keinen Neustart, und der Import
   kann die ganze API minutenlang einfrieren.
3. **P1–P3** — die Planer-Zahlen, die du täglich siehst.
4. Der Rest nach Lust und Bedarf.

---

## K — Kritisch

### K1 · Admin-API ist im LAN offen, und sie hat den Docker-Socket
**Belegt.** `config.host = "0.0.0.0"` (`core/config.py:64`), Auth ist per
Default aus (`auth/middleware.py:29`), und der Container mountet
`/var/run/docker.sock` (`service/systemd.py:77`) sowie `~/.ssh` lesend.

Damit kann jedes Gerät im Netz ohne Anmeldung u. a.
`POST /api/update/run` (git pull + Rebuild + Neustart),
`POST /api/models/delete-repo`, `POST /api/cluster/unload`,
`PUT /api/secrets/{key}` und `POST /api/sharding/launch` aufrufen. Wer den
Docker-Socket steuert, ist faktisch root auf dem Host — und über die
SSH-Schlüssel auf allen drei Nodes.

**Vorschlag:** (a) `/v1/*` (OpenAI-Proxy) und `/api/*` (Verwaltung) trennen;
Verwaltung standardmäßig nur von `127.0.0.1` und dem Cluster-Subnetz annehmen.
(b) Auth für `/api/*` einschaltbar machen, *ohne* dass das UI kaputtgeht (→ K2).
(c) Im README klar sagen, dass Port 3000 nicht ins Internet gehört.
**Aufwand:** M.

### K2 · API-Keys einschalten sperrt das eigene Dashboard aus
**Belegt.** Die Middleware verlangt `Authorization: Bearer …` für alles außer
`/`, `/onboarding`, `/api/health`, `/static/`, `/api/onboarding/`
(`auth/middleware.py:23`). `app.js` sendet diesen Header **nirgends** — das
einzige Vorkommen ist ein Beispieltext (`app.js:6947`), `fetchJSON` (`:532`)
setzt ihn nicht.

Folge: `POST /api/auth/enable` → jede folgende UI-Anfrage bekommt 401,
einschließlich `POST /api/auth/disable`. Rückweg nur per Shell
(`auth.json` editieren).

**Vorschlag:** Session-Cookie für das Dashboard (Login-Seite oder
einmaliger Token beim ersten Aufruf), Bearer-Keys nur für `/v1/*` und
externe Clients. Mindestens: Warnung im UI vor dem Einschalten und ein
Test, der das Aussperren nachstellt. **Aufwand:** M.

### K3 · CSRF auf die Verwaltungsrouten
**Vermutet, zu prüfen.** CORS lässt nur `localhost`/`127.0.0.1` als Origin zu
(`api/server.py:~905`) — aber CORS verhindert nur das *Lesen* der Antwort,
nicht das *Absenden*. aiohttps `request.json()` prüft den Content-Type nicht;
ein Formular von einer beliebigen Webseite mit `enctype="text/plain"` und
JSON-förmigem Body würde vermutlich als gültige Anfrage durchgehen. Der Browser
des Operators steht im selben LAN.

**Prüfen:** `curl -X POST -H 'Content-Type: text/plain' -d '{"model":"x"}'
localhost:3000/api/models/unload` — wird das angenommen, ist es real.
**Vorschlag:** für alle ändernden Routen `Content-Type: application/json`
erzwingen (erzwingt einen Preflight) oder einen CSRF-Token. **Aufwand:** S.

---

## B — Bugs

### B1 · Verteilte Instanzen überleben keinen AINode-Neustart
**Belegt.** Das Manifest speichert nur Solo-Instanzen
(`models/api_routes.py:122`: „distributed (head) instances are out of scope
for auto-replay"), und die Übernahme laufender Container aus #203 kennt nur
`ainode_image-*` und `ainode-vllm-node-solo-*` (`models/api_routes.py:845`) —
**nicht** `vllm_node`, den Container des eugr-Launchers.

Folge: Smaug auf zwei Nodes läuft weiter, aber nach `systemctl restart ainode`
steht es nicht mehr unter INSTANCES, der Router kennt es nicht, und der
Load-Tab/Planer halten die Nodes für frei. Genau der Fehler von #203, nur für
deinen Hauptanwendungsfall.

**Vorschlag:** verteilte Starts ins Manifest (Nodes, Achse, Rang-Zahl, Flags);
Übernahme erkennt `vllm_node` + Port und fragt `/v1/models`; Replay nur, wenn
alle Peers erreichbar sind. **Aufwand:** M.

### B2 · Import aus `/model-import` kann die API minutenlang einfrieren
**Belegt im Code, Auswirkung vermutet.** `handle_take_dropbox` ruft
`shutil.move` **direkt im async-Handler** auf (`models/import_routes.py:499`).
Liegen `/model-import` und `~/.ainode/models` auf demselben Dateisystem, ist das
ein `rename` — sofort. Liegen sie auf verschiedenen (eigene Platte, USB,
anderer Mount), kopiert `shutil.move` 129 GB **im Event-Loop**: UI, Proxy,
Discovery und Speicherwächter-Telemetrie stehen, bis die Kopie fertig ist.

Ebenso, kleiner: `shutil.rmtree` beim Löschen eines Repos
(`models/api_routes.py:1611`, `:2139`) und beim Compile-Cache (`:275`).

**Vorschlag:** alle in `run_in_executor`; der Import als Job mit Fortschritt
(wie der Download), damit ein geschlossener Tab ihn nicht abbricht.
**Aufwand:** S.

### B3 · Der verteilte Start schreibt nie `config.max_model_len`
**Belegt.** `sharding_routes.py:~609` setzt `config.model`, Peers und
Parallelgrößen, aber nicht `max_model_len`/`kv_cache_dtype`/
`gpu_memory_utilization`. #218 hat den **Load-Tab** repariert (liest jetzt die
Instanz), aber jeder andere Leser der geteilten `NodeConfig` sieht weiter die
Werte eines früheren Solo-Loads — u. a. der Boot-Pfad nach Neustart, der Status-
Broadcast und alles, was `app["config"]` fragt.

**Vorschlag:** entweder konsequent aus der Instanz lesen (und `app["config"]`
nur noch als Node-Default behandeln) oder beim verteilten Start dieselben
Felder persistieren wie der Solo-Pfad (`_persist_primary_overrides`).
**Aufwand:** S–M.

### B4 · Die opencode-Config prüft das Fenster, aber nicht den Cache
**Belegt.** `clients/opencode.py:65` (`_launched_context`) liest nur
`--max-model-len`. Ob `Kontext × gleichzeitige Anfragen` in den KV-Cache passt,
prüft niemand. `scripts/opencode-limits.py` macht es bereits (mit vLLMs eigener
Token-Zahl) — der Generator nicht.

**Vorschlag:** `kv_tokens` aus der Messung (#219 speichert sie jetzt) oder aus
dem Plan nehmen und `limit.context` auf `kv_tokens / max_num_seqs` kappen.
**Aufwand:** S.

### B5 · Belegungs-Balken zeigt zu wenig, wenn man die Länge fährt
**Belegt.** Seit #214 lässt das UI `concurrency` weg, wenn der Operator zuletzt
das Kontextfeld geändert hat (`app.js:~2232`). `handle_plan` setzt dann
`concurrency=1` (`planner/api_routes.py:362`), und `_forecast` rechnet den
„benutzt"-Anteil für **eine** Anfrage. Der Balken unterschätzt also genau dann,
wenn man das Fenster größer zieht.

**Vorschlag:** `concurrency` für die Prognose immer mitschicken, nur für die
*Ableitung* weglassen (zwei getrennte Parameter). **Aufwand:** S.

### B6 · „Zuletzt bearbeitetes Feld" wird nie zurückgesetzt
**Belegt.** `state.launchLastEdited` (`app.js:1754`) bleibt nach einem
Modellwechsel stehen. Beim nächsten Modell wird deshalb weiter eine der beiden
Größen nicht an den Planer geschickt, obwohl der Operator sie für dieses Modell
nie angefasst hat.

**Vorschlag:** beim Modell- und Node-Wechsel auf `null`. **Aufwand:** S.

### B7 · Aus FOLLOWUPS.md noch offen
- **Phantom-Zeile** nach fehlgeschlagener Speicherprüfung in `/api/server/status`
  („launching", kein Container dahinter).
- **Veraltete `ClusterNode`-Einträge** auf lang laufenden Nodes; `_routing_table`
  akzeptiert Status `online` (Discovery-Gesundheit) statt Engine-Lebendigkeit.

Beide brauchen laut FOLLOWUPS erst eine Reproduktion. **Aufwand:** je S–M.

---

## P — Planer-Genauigkeit

### P1 · `TP_REPLICATION = 1.05` ist für MoE unter Expert-Parallelismus zu hoch
**Gemessen.** Smaug-Flash, TP=2: geschätzt 87,6 GB/Node, vLLM meldet
`Model loading took 67.7 GiB` (= 72,7 GB). #219 ersetzt die Schätzung **nach**
dem ersten erfolgreichen Start — davor lehnt #218 bei einem unvermessenen MoE
womöglich Starts ab, die passen würden.

**Vorschlag:** nicht die Konstante raten, sondern Messpunkte sammeln
(#219 macht das jetzt) und ab 3–4 MoE-Messungen einen eigenen Faktor für
„MoE + EP" ableiten; bis dahin im Plan anzeigen, dass die Gewichte *geschätzt*
sind und die Ablehnung auf einer Schätzung beruht. **Aufwand:** S (Anzeige),
M (Faktor aus Daten).

### P2 · Indexer-Caches werden nicht mitgerechnet
**Belegt, Größe unbekannt.** DeepSeek-V4 (`index_head_dim`, `index_n_heads`,
`index_topk`) und Qwen3.8-Flash-Next (`indexer_budget`, `indexer_head_dim`,
`indexer_kv_heads`, `indexer_compress_ratio`) haben einen Sparse-Attention-
Indexer mit eigenem Cache. `grep indexer ainode/planner/` → 0 Treffer. Die
Token-Zahlen sind für diese Modelle optimistisch.

**Vorschlag:** gegen vLLMs `GPU KV cache size` kalibrieren (#219 liest die
jetzt) statt die Formel zu raten. **Aufwand:** M.

### P3 · Rekurrenter Zustand hybrider Modelle fehlt
**Belegt, Größe unbekannt.** Qwen3-Next/Qwen3.5-MoE/Qwen3.8 halten pro
*Sequenz* einen festen Zustand der linearen Attention. Der Planer rechnet nur
die Full-Attention-Layer pro Token; der Zustand × `max_num_seqs` fehlt. Bei
vielen parallelen Sessions wächst der Fehler.
**Vorschlag:** Zustandsgröße aus `config.json` (`linear_*`-Felder) ableiten und
pro Sequenz addieren; gegen Messung prüfen. **Aufwand:** M.

### P4 · Engine-Overhead aus Messungen kalibrieren
**Gemessen, widersprüchlich.** Smaug bei 0,66: 67,7 GiB Gewichte + 13,72 GiB
Cache = 81,4 GiB — mehr als 0,66 × Gesamtspeicher. Entweder zählt vLLM den
Pool anders als wir annehmen, oder „Model loading took" enthält Speicher, der
danach freigegeben wird. `ENGINE_OVERHEAD_GB = 2.5` ist nie gemessen worden.
**Vorschlag:** `Available KV cache memory` + Gewichte + Pool aus mehreren
Starts nebeneinanderlegen, dann entscheiden. **Aufwand:** S (Auswertung).

### P5 · ds_mla-Skalenblock (8 Byte/Layer/Token) fehlt
**Belegt.** 576 statt gemessener 584 Byte — 1,4 % zu wenig, im Overhead
verborgen. Nur relevant, wenn P4 den Overhead enger macht. **Aufwand:** S.

### P6 · Messung nur auf dem Head
**Belegt.** `measure/recorder.py` misst `MemAvailable` vor/nach auf dem
eigenen Node. Bei einem verteilten Start wird der Peer nie gemessen — und wenn
der Peer enger ist als der Head, ist er der, der zählt.
**Vorschlag:** Peers melden ihren Abfall per Broadcast/MQTT; die Messung nimmt
das Maximum. **Aufwand:** M.

### P7 · Messungen gehören zu einem Engine-Build
**Vermutet.** Das Base-Image kommt aus eugrs rollendem
`prebuilt-vllm-current` (`scripts/build-base-image.sh:139`). Eine Messung von
letzter Woche kann von einem anderen vLLM stammen. **Vorschlag:** vLLM-Version
in jede Messung schreiben und ältere als „anderer Build" kennzeichnen.
**Aufwand:** S.

---

## R — Robustheit und Betrieb

### R1 · Keine Log-Rotation
**Belegt.** `distributed.log` wird nur angehängt (`engine/backends/eugr.py:1459`,
`:1657`, `:1695`); bei dir waren es schon 35 MB. `read_log_tail` liest zwar nur
die letzten 2 MB, aber die Datei wächst unbegrenzt, und `grep` über alle Logs
(wie in den Anleitungen) wird langsam.
**Vorschlag:** pro Start eine neue Datei oder Größenrotation (z. B. 50 MB × 5).
**Aufwand:** S.

### R2 · Blockierender `docker ps` beim Start
**Belegt, klein.** `replay_instances_on_startup` ruft `subprocess.run` direkt
im async-Code (`models/api_routes.py:~992`). Beim Start unkritisch, aber in
Executor verschieben kostet nichts. **Aufwand:** S.

### R3 · Grundlast auf dem Head ungeklärt
**Gemessen, Ursache offen.** Der Head hatte ohne geladenes Modell ~27 GB
weniger nutzbar als die Peers — das kostet bei Smaug fünf parallele Sessions.
Kandidaten: zwei Registry-Container, Page-Cache, AINode selbst, Überbleibsel.
**Vorschlag:** einmal `docker stats`, `free -g`, `smem` auf dem Head; ggf.
Registries auf einen Peer verlegen. **Aufwand:** S (Diagnose).

### R4 · Gemischte Versionen während eines Rolling-Updates
**Vermutet.** Seit #212 meldet ein Node `gpu_memory_gb` dezimal (128), ein noch
nicht aktualisierter Peer binär (119). Während `update-cluster.sh` läuft, plant
der Head mit beiden Einheiten gleichzeitig.
**Vorschlag:** im Broadcast Rohwerte (Bytes/MiB) senden, Umrechnung nur beim
Leser. **Aufwand:** S.

### R5 · Engine-Image nicht reproduzierbar gepinnt
**Belegt.** Siehe P7. Ein Rebuild kann still eine andere vLLM-Version bringen,
und Rezeptflags (`--tool-call-parser deepseek_v4` usw.) können damit verschwinden.
**Vorschlag:** Wheel-Version im Image-Tag und in `image.env` festhalten;
Rezepte gegen das Image prüfen (→ F2). **Aufwand:** S–M.

---

## O — Optimierungen

### O1 · Planer liest bei jedem Tastendruck die Modelldateien
**Belegt.** Jede Feldänderung plant neu (350 ms Debounce), und `local_facts` →
`weight_bytes_on_disk` (`planner/facts.py:124`) macht ein `rglob` + `stat`
über alle Shards. `_dir_size_gb` hat einen Cache, das hier nicht.
**Vorschlag:** denselben mtime-Cache verwenden. **Aufwand:** S.

### O2 · Gewichte einmal statt dreimal (NFS über die Fabric)
**Aus dem MiaAI-Rezept, von dir zurückgestellt.** Head exportiert
`~/.ainode/models` per NFSv4 über CX7, Peers mounten lesend. Spart pro
Modell 2× Plattenplatz und die Spiegelzeit (Smaug: 2 × 167 GB).
**Aufwand:** L — braucht eine Entscheidung (Default oder Option).

### O3 · Modell für zwei Sessions: zwei Kopien statt TP=2, wo es passt
**Belegt als Rechnung.** Für Modelle, die auf einen Node passen
(Qwen3-Coder-Next NVFP4, KAT-Coder NVFP4), ist „eine Kopie pro Node, TP=1"
schneller und ausfallsicherer. Der Planer schlägt das nie vor.
**Vorschlag:** Plan-Option „Replikate" neben Tensor/Pipeline, Router verteilt
Anfragen. **Aufwand:** M–L.

---

## W — Wartbarkeit und Tests

### W1 · 55 Testdateien prüfen Quelltext-Strings
**Belegt.** Viele Tests lesen `app.js` oder `.py` als Text und suchen Zeichen-
ketten (`"line + measured + warn" in app_js`). Das ist schnell geschrieben und
bricht bei jeder Umformulierung — so geschehen bei #216. Gleichzeitig hat
die ganze Suite einen zerschnittenen `Recorder` durchgelassen (#219).
**Vorschlag:** reine JS-Funktionen (`renderOccupancy`, `reflectPlanIntoFields`,
`resumeCheckNote`, `limits`) in ein kleines Modul und mit `node --test`
prüfen; Struktur-Tests (Klasse hat ihre Methoden) statt String-Tests.
**Aufwand:** M.

### W2 · `app.js` ist ein 9.400-Zeilen-Objekt
**Belegt.** Launch-Formular, Server-Ansicht, Downloads, Import, Training,
Update und Topologie in einer Datei, ohne Build-Schritt. Jeder Merge der
letzten Tage hatte hier seine Konflikte.
**Vorschlag:** in ES-Module nach Ansicht aufteilen (kein Bundler nötig,
`<script type="module">`). **Aufwand:** L, am besten schrittweise.

### W3 · Zwei Engine-Backends laufen auseinander
**Belegt.** `NvidiaBackend` (1.630 Zeilen) und `EugrBackend` (1.750 Zeilen);
FOLLOWUPS listet Flags, die nur einer kann; der eugr-Pfad ignoriert das
Katalog-`engine_image`. Auf deinem Cluster läuft praktisch nur eugr.
**Vorschlag:** entscheiden, ob `nvidia` noch gebraucht wird; wenn nein,
entfernen oder einfrieren. **Aufwand:** M.

### W4 · Kein pytest auf dem Host, nur in CI
**Belegt.** Lokal fehlt pytest (ich habe mit einer venv gearbeitet).
`pip install -e ".[dev]"` steht im CLAUDE.md, ist aber nicht eingerichtet.
**Aufwand:** S.

---

## F — Ergänzungen

### F1 · Reasoning-Parser erkennen statt raten
Smaug, DeepSeek und Qwen3.8 denken standardmäßig, keins der Rezepte hat einen
`--reasoning-parser` — weil ein falscher Name den Start in Sekunde drei
killt. **Vorschlag:** wie #205 die Env-Registry, die gültigen Parser-Namen aus
dem Image fragen (`vllm.reasoning`-Registry) und dann Familie → Parser
automatisch setzen. Dann stimmt auch `reasoning: true` in opencode wieder.
**Aufwand:** S–M.

### F2 · Rezept-Flags vor dem Start gegen das Image prüfen
`vllm serve --help` liefert im `vllm-node`-Image nichts (siehe Verlauf), aber
der Parser lässt sich per Python bauen (`EngineArgs.add_cli_args`). Damit
ließe sich jede Rezept- und Advanced-Zeile vor dem Start prüfen — die ganze
Klasse `--quantization modelopt_fp4` / „unrecognized arguments" würde vor dem
Laden der Gewichte abgelehnt statt danach. **Aufwand:** M.

### F3 · Client-Configs neu erzeugen, wenn sich der Start ändert
Heute ist die opencode-Config ein Schnappschuss; nach einem Neuladen mit
anderem Fenster ist sie still falsch (genau dein 608.512-Fall). **Vorschlag:**
Hinweis im UI „Config veraltet seit Start X" bzw. ein stabiler Endpunkt, den
opencode direkt abruft. **Aufwand:** S.

### F4 · Modellkatalog-Einträge verifizieren
`smaug-flash` und `qwen3.8-flash-next-nvfp4` stehen auf `verified=False`.
Nach dem ersten guten Start: Messwerte aus #219 übernehmen, Beschreibung mit
echten Zahlen, `verified=True`. **Aufwand:** S je Modell.

---

## H — Hardware-Experimente, die Fragen beantworten würden

| # | Experiment | beantwortet |
|---|---|---|
| H1 | Qwen3.8-Flash-Next (24 Heads) mit TP=3 über alle drei Nodes | ob `TENSOR_SIZES = (1,2,4,8)` zu streng ist |
| H2 | Smaug mit `fp8_ds_mla` statt `fp8` | ob der gepackte MLA-Cache hier wirklich mehr Token gibt |
| H3 | `--reasoning-parser deepseek_v4` auf Smaug | F1, und ob opencode mit `reasoning: true` stabil läuft |
| H4 | zwei, drei weitere MoE-Starts mit #219 | P1 und P4 auf Daten statt auf einem Messpunkt |
| H5 | Head-Speicher analysieren (`docker stats`, `smem`) | R3 |
