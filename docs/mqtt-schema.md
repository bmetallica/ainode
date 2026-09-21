# MQTT — vollständiges Schema

Was AINode veröffentlicht, auf welchem Topic, mit welchen Feldern, und was
jedes davon bedeutet. Stand: 0.6.0.

Gedacht als Nachschlagewerk beim Bau von Dashboards und Alarmen. Wo ein Feld
fehlen kann, steht das dabei — AINode lässt eine Messung, die es nicht lesen
kann, **weg**, statt eine Null zu senden. Eine Null bedeutet hier also immer
„gemessen und null", nie „nicht verfügbar". Das ist Absicht: ein Dashboard,
das auf einer immer-null-Metrik aufbaut, sieht gesund aus, während der Knoten
voll ist.

---

## 1. Aufbau der Topics

```
<prefix>/<node_id>/status              online / offline  (retained, Last Will)
<prefix>/<node_id>/system              CPU, Speicher, Platte, Netz, Temperaturen
<prefix>/<node_id>/gpu                 Auslastung, Speicher, Temperatur
<prefix>/<node_id>/fabric              RoCE-Links: Durchsatz und Fehler
<prefix>/<node_id>/models              was geladen ist, und wie es benutzt wird
<prefix>/<node_id>/engine/<modell>     was die Engine selbst meldet
<prefix>/<node_id>/safety              der Host-Speicherwächter
<prefix>/<node_id>/transfers           laufende Downloads und Spiegelungen
<prefix>/cluster                       die Flottensicht (nur Head)
<prefix>/<node_id>/logs/ainode         das Log des Orchestrators
<prefix>/<node_id>/logs/vllm/<modell>  das Log jeder Engine-Instanz
<prefix>/<node_id>/events/launch       ein Ladevorgang ist fertig (Ereignis)
```

Manche Topics erscheinen nur, wenn es etwas zu sagen gibt: `gpu` fehlt ohne
NVIDIA-Treiber, `fabric` ohne RDMA-Karte, `safety` ohne Speicherwächter,
`transfers` wenn gerade nichts überträgt, und `engine/<modell>` erst, sobald
die Instanz antwortet. Ein leeres Topic im Takt wäre Rauschen.

`<prefix>` ist frei wählbar (Settings → Monitoring, Vorgabe `ainode`).
`<node_id>` ist die Knoten-ID aus `config.json`, nicht der Hostname.

Zwei Dinge fallen aus dem Muster, beide mit Grund:

* **`<prefix>/cluster` trägt keine Knoten-ID.** Die Flottensicht ist keine
  Eigenschaft des Knotens, der sie zufällig veröffentlicht. Sie kommt
  ausschließlich vom **Head** — ein Member würde sein eigenes Teilbild unter
  dasselbe Topic schreiben, und wer zuletzt sendet, gewinnt. Ein Sender pro
  Tatsache.
* **`logs/vllm/<modell>`** hat eine zusätzliche Ebene, weil es pro Knoten
  mehrere Instanzen geben kann. Die Modell-ID wird topic-sicher gemacht:
  alles außer `A-Za-z0-9._-` wird zu `_`. Aus
  `demon-zombie/MiniMax-M2.7-AWQ-4bit` wird
  `demon-zombie_MiniMax-M2.7-AWQ-4bit`.

Alle Nutzlasten sind **ein JSON-Objekt pro Nachricht**, UTF-8.

### Abonnieren

```bash
mosquitto_sub -h <broker> -t 'ainode/#' -v            # alles
mosquitto_sub -h <broker> -t 'ainode/+/status' -v     # wer lebt
mosquitto_sub -h <broker> -t 'ainode/+/gpu' -v        # GPU aller Knoten
mosquitto_sub -h <broker> -t 'ainode/+/engine/#' -v   # alle Engines
mosquitto_sub -h <broker> -t 'ainode/+/logs/#' -v     # alle Logs
mosquitto_sub -h <broker> -t 'ainode/+/events/#' -v   # nur Ereignisse
mosquitto_sub -h <broker> -t 'ainode/cluster' -v      # nur die Flottensicht
```

### Takt, QoS, Retain

| Einstellung | Vorgabe | Bedeutung |
|---|---|---|
| Intervall | 30 s | 1–3600. Eine Sekunde ist zum Zuschauen bei einem Vorgang, nicht zum Dauerbetrieb. |
| QoS | 0 | 0, 1 oder 2, wie vom Broker unterstützt. |
| Retain | aus | Praktisch nach einem Broker-Neustart. **Vorsicht:** eine behaltene Nachricht eines verschwundenen Knotens sieht aus wie ein lebender Knoten. |

Alle Knoten veröffentlichen **ihre eigenen** Werte. Nur auf dem Head
konfiguriert, bekommst du Telemetrie nur vom Head — *Apply to all nodes*
kopiert die Einstellungen samt Passwort über das Clusternetz auf alle Knoten.

---

## 2. Felder, die jede Nachricht trägt

Jede Nutzlast (außer den Logs, die dasselbe in kürzerer Form tragen) beginnt
mit der Knotenidentität, damit eine einzelne Nachricht für sich allein
auswertbar ist:

| Feld | Typ | Bedeutung |
|---|---|---|
| `node_id` | string | Knoten-ID aus `config.json`. |
| `node_name` | string | Anzeigename. Kann leer sein. |
| `version` | string | AINode-Version des sendenden Knotens. |
| `timestamp` | float | Unix-Zeit in Sekunden, auf Millisekunden gerundet. |

---

## 3. `<prefix>/<node_id>/status` — lebt dieser Knoten?

```json
{ "node_id": "spark-1", "node_name": "SPARK1",
  "status": "online", "timestamp": 1789935612.233 }
```

Zwei Nachrichten, beide **retained**:

* beim Verbinden `online`;
* `offline`, sobald die Verbindung abreißt — und zwar veröffentlicht vom
  **Broker**, nicht vom Knoten. Das ist ein *Last Will*: der Broker hält die
  Nachricht seit dem Verbindungsaufbau bereit und sendet sie, wenn die
  Verbindung wegfällt, egal wie — abgeschossener Prozess, gezogenes Kabel,
  eingefrorener Knoten. Auf dieser Seite muss dafür nichts mehr laufen, und
  genau darin liegt der Sinn.

Ein geplantes Herunterfahren sendet sein `offline` selbst, bevor es die
Verbindung schließt. Beides heißt „offline"; nur eines davon ist ein Grund,
nachts aufzustehen — wer das unterscheiden will, achtet darauf, ob kurz
danach wieder ein `online` kommt.

> **Das ist die richtige Grundlage für „Knoten weg"**, nicht das Ausbleiben
> von `system`-Nachrichten. Mit `retain` sieht die letzte Messung eines vor
> einer Stunde gestorbenen Knotens nämlich genauso frisch aus wie die eines
> lebenden.

---

## 4. `<prefix>/<node_id>/system`

Gesundheit des Knotens. Gemessen mit psutil; alles, was diese Plattform nicht
liefert, fehlt.

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "cpu":    { "cores": 20, "percent": 3.4,
              "load_1m": 0.81, "load_5m": 0.70, "load_15m": 0.74 },
  "memory": { "total_mb": 124928.0, "used_mb": 64110.2,
              "available_mb": 58121.4, "percent": 51.3,
              "swap_used_mb": 0.0, "swap_total_mb": 8189.0 },
  "disk":   { "/": { "total_gb": 1863.0, "free_gb": 402.1, "percent": 78.4 } },
  "network":{ "enp1s0f0": { "bytes_sent": 50295393784, "bytes_recv": 36227108899,
                            "link_mbit": 400000, "tx_mbit_s": 12.4, "rx_mbit_s": 3.1 } },
  "temperature_c": { "cpu": 46.0 },
  "uptime_seconds": 2288124.2
}
```

| Feld | Bedeutung |
|---|---|
| `cpu.cores` | logische Kerne |
| `cpu.percent` | Auslastung seit der letzten Messung, nicht seit dem Start |
| `cpu.load_1m/5m/15m` | Lastdurchschnitt; fehlt auf Plattformen ohne `getloadavg` |
| `memory.*_mb` | Megabyte. `available_mb` ist `MemAvailable`, enthält also zurückgewinnbaren Page-Cache |
| `memory.swap_*` | fehlt, wenn kein Swap konfiguriert ist |
| `disk` | ein Eintrag pro Pfad: `/` plus das Modellverzeichnis. Zwei Pfade auf demselben Dateisystem erscheinen **einmal** |
| `network` | ein Eintrag pro Interface. `bytes_*` sind Zähler seit Systemstart, `*_mbit_s` sind Raten zwischen zwei Messungen |
| `network.*.link_mbit` | ausgehandelte Verbindungsgeschwindigkeit; fehlt, wenn der Treiber keine meldet |
| `temperature_c` | ein Eintrag pro Sensor, Bezeichnung normalisiert; fehlt komplett, wenn keine Sensoren lesbar sind |
| `uptime_seconds` | Systemlaufzeit, nicht AINode-Laufzeit |

> **Auf GB10 wichtig:** `memory` ist hier **derselbe** Speicher, den die GPU
> benutzt — es gibt kein getrenntes VRAM. Ein Alarm auf `memory.percent` ist
> auf dieser Hardware gleichzeitig ein GPU-Speicher-Alarm. Siehe auch den
> Speicherwächter in Abschnitt 8.

---

## 5. `<prefix>/<node_id>/fabric` — die RoCE-Links

Das Topic, das es gibt, weil `system.network` für RDMA **null** meldet: RDMA
schreibt direkt aus einer HCA in den Speicher der anderen, der Kernel-Stack
sieht kein einziges Paket, und `/proc/net/dev` zählt nur, was der Stack
behandelt hat. Auf einem Knoten, der nichts als RDMA macht, ist das nichts.

Ein Eintrag pro Port, Schlüssel ist `<hca>:<port>`:

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "ports": {
    "rocep1s0:1": {
      "hca": "rocep1s0", "port": "1",
      "state": "ACTIVE", "phys_state": "LinkUp",
      "rate": "400 Gb/sec (4X NDR)", "link_gbit": 400.0,
      "tx_bytes": 8130947072, "rx_bytes": 7714402304,
      "tx_packets": 19203441, "rx_packets": 18855012,
      "tx_mbit_s": 18422.1, "rx_mbit_s": 17801.4,
      "tx_percent": 4.6, "rx_percent": 4.5,
      "errors": { "link_downed": 0, "port_rcv_errors": 0, "symbol_error": 0 },
      "errors_total": 0, "errors_new": 0
    }
  }
}
```

| Feld | Bedeutung |
|---|---|
| `tx_bytes` / `rx_bytes` | **Bereits in Bytes.** Der Kernel zählt in `port_xmit_data` 32-Bit-**Wörter**; AINode multipliziert einmal mit 4, damit es flussabwärts niemand vergessen kann. Wer selbst nachrechnet, muss daran denken |
| `tx_mbit_s` / `rx_mbit_s` | Rate zwischen zwei Messungen. Fehlt bei der **ersten** Messung — eine Rate braucht zwei Werte, und eine aus einem einzigen erfundene wäre eine Schätzung im Gewand einer Messung |
| `tx_percent` / `rx_percent` | Anteil am Link, **pro Richtung**. Nicht addiert: ein Link kann in eine Richtung gesättigt und in die andere leer sein, und eine Summe versteckt genau das |
| `state`, `phys_state` | `ACTIVE`/`LinkUp` im Normalfall. Alles andere heißt: das Kabel ansehen |
| `errors` | die einzelnen Fehlerzähler, soweit die Firmware sie führt |
| `errors_total` | ihre Summe — **eine** Zahl zum Alarmieren, damit kein Dashboard wissen muss, welche sechs Zähler es auf welcher Firmware gibt |
| `errors_new` | der Zuwachs seit der letzten Messung. **Das ist der Wert, auf den du alarmierst:** ein Zähler, der seit dem Aufbau der Maschine auf 3 steht, ist kein Fehler, der gerade passiert |

> **Auf einem switchlosen Ring sind die Fehlerzähler wichtiger als der
> Durchsatz.** Jeder Link ist ein einzelnes Kabel zu einem Nachbarn, es gibt
> keine Redundanz, hinter der sich ein sterbendes Kabel verstecken könnte. Es
> zeigt sich hier, lange bevor NCCL mitten in einem Lauf aufgibt.

---

## 6. `<prefix>/<node_id>/safety` — der Speicherwächter

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "memory_guard_enabled": true,
  "host_available_mb": 58121, "host_total_mb": 124928,
  "warn_mb": 8192, "critical_mb": 4096,
  "blocking_launches": false, "below_critical": false,
  "stops_total": 0
}
```

| Feld | Bedeutung |
|---|---|
| `host_available_mb` | `MemAvailable` des **Hosts**, direkt aus `/proc/meminfo` |
| `warn_mb` / `critical_mb` | die **durchgesetzten** Grenzen, nicht die eingestellten. Die Reserve wird gegen die Größe der Maschine gedeckelt (15 % des Gesamtspeichers), und ein Alarm auf einer Zahl, die der Wächter gar nicht benutzt, geht zum falschen Zeitpunkt los |
| `blocking_launches` | unter der Warngrenze: neue Modelle werden abgelehnt |
| `below_critical` | unter der kritischen Grenze: beim **zweiten** Mal in Folge wird die zuletzt gestartete Engine abgeschossen |
| `stops_total` | wie viele Engines dieser Prozess bisher beenden musste |
| `last_stop` | die letzte davon: `at`, `model`, `reason`, `available_mb`, `critical_mb`. Fehlt, wenn es keine gab |
| `host_memory_readable` | nur vorhanden und dann `false`, wenn `/proc/meminfo` nicht lesbar war. Der Wächter greift dann **nicht** ein — er ist ein Netz, kein Tor, das bei Unkenntnis schließt |

**Dieses Topic wird außer der Reihe gesendet, wenn der Wächter zuschlägt.**
Er misst alle zwei Sekunden, die Telemetrie sendet alle dreißig — das eine
Ereignis, auf das es ankommt, wäre sonst eine halbe Minute zu spät, oder gar
nicht, wenn der Knoten dazwischen untergeht.

---

## 7. `<prefix>/<node_id>/gpu`

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "utilization_percent": 87,
  "memory_used_mb": 64110,
  "memory_total_mb": 124928,
  "temperature_c": 61
}
```

Das Topic **fehlt ganz**, wenn NVML nichts liefert (kein NVIDIA-Treiber, keine
GPU) — statt eine Nachricht mit einem `error`-Feld zu senden.

> **Auf GB10:** NVML meldet für den Speicher `used = 0`, genau wie
> `nvidia-smi` dort `[N/A]` anzeigt. AINode erkennt das und liefert
> ersatzweise die Host-Werte aus psutil — der Unified-Speicher **ist** der
> GPU-Speicher. `memory_used_mb` und `memory_total_mb` sind hier also
> dieselben Zahlen wie unter `system.memory`, in Megabyte.

---

## 8. `<prefix>/<node_id>/models`

Was dieser Knoten bedient und wie er benutzt wird.

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "loaded": [
    { "model": "demon-zombie/MiniMax-M2.7-AWQ-4bit",
      "api_port": 8000, "status": "serving", "nodes": 2,
      "gpu_memory_utilization": 0.87, "max_model_len": 65536,
      "load_phase": "ready" }
  ],
  "embeddings": ["nomic-ai/nomic-embed-text-v1.5"],
  "requests_total": 1842,
  "errors_total": 3,
  "latency_ms": { "p50": 1180.0, "p95": 4320.5, "p99": 9800.2 },
  "uptime_seconds": 84021.4,
  "per_model": {
    "demon-zombie/MiniMax-M2.7-AWQ-4bit": {
      "requests": 1842, "errors": 3,
      "avg_latency_ms": 2140.7,
      "tokens_generated": 412093,
      "avg_tokens_per_second": 98.5
    }
  }
}
```

### `loaded[]` — je vLLM-Instanz auf diesem Knoten

| Feld | Bedeutung |
|---|---|
| `model` | HF-Repo-ID, wie geladen |
| `api_port` | OpenAI-Port dieser Instanz. Die primäre hat 8000, gestapelte 8001, 8002 … |
| `status` | `starting`, `serving`, `failed` |
| `nodes` | über wie viele Knoten diese Instanz läuft (1 = solo) |
| `gpu_memory_utilization` | der Wert, mit dem sie gestartet wurde; fehlt, wenn keiner gesetzt war |
| `max_model_len` | Kontextfenster; fehlt, wenn nicht gesetzt |
| `load_phase` | `starting`, `distributing`, `loading_weights`, `distributed_init`, `profiling`, `ready`, `failed`; fehlt, wenn der Backend keine meldet |
| `kind` | `image` bei einem Bildmodell. **Fehlt** bei einem LLM — der Normalfall bleibt unbeschriftet, damit ein Dashboard von vor der Bildgenerierung unverändert liest |

`embeddings` sind In-Process-Modelle, keine vLLM-Instanzen — sie haben keinen
eigenen Port und erscheinen deshalb nicht in `loaded`.

### `per_model` — gemessen, nicht geschätzt

| Feld | Bedeutung |
|---|---|
| `requests` / `errors` | Zähler seit dem Start dieses AINode-Prozesses |
| `avg_latency_ms` | fehlt, solange keine Anfrage gezählt wurde |
| `tokens_generated` | fehlt, wenn die Route keine Tokenzahl meldet |
| `avg_tokens_per_second` | Tokens geteilt durch die Zeit, die **tatsächlich mit Generieren** verbracht wurde — nicht durch die Laufzeit. Ein Modell, das in einer Stunde zehn Anfragen bedient hat, ist nicht langsam, sondern unbeschäftigt; durch die Laufzeit geteilt behauptet die Zahl das Gegenteil |

`latency_ms` sind die Perzentile über alle Anfragen dieses Knotens. Fehlt,
solange nichts gemessen wurde. Ein Mittelwert würde den Schwanz verstecken,
und der Schwanz ist das, was ein Nutzer merkt.

`uptime_seconds` ist hier die Laufzeit des AINode-Prozesses, im Unterschied zu
`system.uptime_seconds` (Systemlaufzeit).

---

## 9. `<prefix>/<node_id>/engine/<modell>` — was die Engine selbst meldet

Alles in `models` ist **am Proxy** gemessen: was durch AINode ging. Das
beschreibt den Verkehr, nicht die Engine. Die Fragen, die man in einer
geschäftigen Stunde wirklich hat, beantwortet vLLM selbst — auf seinem eigenen
Prometheus-Endpunkt, je Instanz. AINode liest den aus und destilliert ihn.

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "instance": "demon-zombie_MiniMax-M2.7-AWQ-4bit",
  "kv_cache_percent": 81.37,
  "requests_running": 7, "requests_waiting": 3, "requests_swapped": 0,
  "requests_total_in_flight": 10,
  "preemptions_total": 42,
  "prompt_tokens_total": 1000000, "generation_tokens_total": 412093,
  "time_to_first_token_s": 0.3, "time_per_output_token_s": 0.0102,
  "e2e_latency_s": 2.1, "queue_time_s": 0.04,
  "spec_accepted_tokens_total": 300, "spec_draft_tokens_total": 400,
  "spec_acceptance_rate": 0.75
}
```

| Feld | Warum es zählt |
|---|---|
| `kv_cache_percent` | Wie voll der KV-Cache ist. Die Zahl, die entscheidet, wie viele Leute das Modell gleichzeitig benutzen können — und die bei GLM hier vollgelaufen ist |
| `preemptions_total` | Anfragen, die aus dem Cache verdrängt und später neu gerechnet wurden. **Ein steigender Zähler ist die Frühwarnung, bevor „das Modell ist plötzlich langsam"** |
| `requests_running` / `_waiting` | Warteschlangentiefe — ob `--max-num-seqs` auch nur ungefähr passt |
| `requests_swapped` | Anfragen, die auf den Host ausgelagert wurden. Fehlt bei Builds, die nicht auslagern |
| `requests_total_in_flight` | die Summe, als eine Zahl für „gerade über der Kapazität" |
| `time_to_first_token_s` | Prefill-Kosten, getrennt vom Decode. Eine prefill- und eine decodegebundene Last sehen in Tokens pro Sekunde gleich aus und brauchen entgegengesetzte Maßnahmen |
| `time_per_output_token_s` | der Kehrwert der Decode-Geschwindigkeit, wie die Engine sie misst |
| `e2e_latency_s`, `queue_time_s` | Gesamtdauer und Wartezeit, jeweils als Mittelwert |
| `spec_acceptance_rate` | ob sich ein spekulativer Drafter lohnt. Unter etwa 0,5 kostet er mehr, als er spart |

### Wenn die Instanz Bilder macht

Dasselbe Topic, dieselbe Form, eigene Namen — ein Dashboard soll nicht wissen
müssen, welche Engine hinter einem Modell steckt, um zu sehen wie beschäftigt
es ist:

| Feld | Bedeutung |
|---|---|
| `images_generated_total` | erzeugte Bilder seit dem Start der Instanz |
| `image_seconds_total` | dafür aufgewendete Sekunden |
| `seconds_per_image` | der Mittelwert daraus; fehlt, solange nichts erzeugt wurde |
| `steps_per_second` | Entrauschungsschritte pro Sekunde — die eigentliche Geschwindigkeit, unabhängig davon wie viele Schritte eingestellt sind |
| `requests_running` | läuft gerade eine Generierung? Immer 0 oder 1: die Engine serialisiert, weil zwei gleichzeitige Läufe auf dieser Hardware die Spitze verdoppeln statt die Wartezeit zu halbieren |

Ein Bildmodell hat **keinen** KV-Cache, also fehlen `kv_cache_percent`,
`preemptions_total` und die Token-Zähler. Das ist kein Ausfall, sondern die
Abwesenheit von etwas, das es dort nicht gibt.

Die Zeitfelder sind **Mittelwerte** (`sum/count` des Histogramms), keine
Perzentile: ein vollständiger Bucket-Satz sind Dutzende Zeilen pro Histogramm
und ohne ein Prometheus dahinter unbenutzbar.

Feldnamen wandern zwischen vLLM-Versionen; AINode kennt je Feld mehrere
Schreibweisen und lässt weg, was diese Version nicht meldet. Eine Instanz, die
noch lädt oder gestorben ist, hat keinen `/metrics`-Endpunkt und bekommt
deshalb kein Topic — beides steht ohnehin in `models` und im Log.

---

## 10. `<prefix>/<node_id>/transfers` — was gerade kopiert wird

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "downloads": [
    { "model": "sparkarena/Minimax-M3-v0-NVFP4-REAP50",
      "status": "downloading", "percent": 41.5,
      "downloaded_bytes": 53500000000, "total_bytes": 128900000000 }
  ],
  "mirror": { "running": true, "model": "org/big", "done": 1, "total": 3 }
}
```

Nur vorhanden, solange etwas läuft. `downloads` listet Jobs im Status
`downloading`, `starting` oder `queued`; `mirror` erscheint während eines
Spiegellaufs zu den Peers. Ein 200-GB-Checkpoint braucht Stunden, und ein bei
40 % über Nacht stehengebliebener Transfer ist genau das, wofür es Telemetrie
gibt.

---

## 11. `<prefix>/cluster` — nur vom Head

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "nodes_total": 3,
  "nodes_online": 3,
  "vram_total_gb": 366.0,
  "versions_agree": true,
  "nodes": [
    { "node_id": "spark-1", "node_name": "SPARK1", "status": "serving",
      "model": "demon-zombie/MiniMax-M2.7-AWQ-4bit", "gpu_memory_gb": 122.0,
      "gpu_memory_used_mb": 64110, "gpu_memory_total_mb": 124928,
      "gpu_memory_used_percent": 51.3, "gpu_utilization_percent": 87.0,
      "version": "0.6.0",
      "instances": [ { "model": "…", "api_port": 8000, "status": "serving" } ] }
  ]
}
```

| Feld | Bedeutung |
|---|---|
| `nodes_total` | alle bekannten Knoten, auch offline gegangene |
| `nodes_online` | mit Status `online`, `serving` oder `member-ready` |
| `vram_total_gb` | Summe über **alle** Knoten, auch offline — Kapazität der Flotte, nicht des Moments |
| `nodes[].status` | Gesundheit aus Sicht des Heads: `online`, `stale`, `offline`. Als online gezählt werden `online`, `serving` und `member-ready` — die beiden letzten kommen von Knoten, deren Build ihren Dienstzustand statt ihrer Gesundheit meldet |
| `nodes[].model` | das primäre Modell des Knotens; `""` wenn keines |
| `nodes[].gpu_memory_*` | fehlen, wenn der Knoten keine Gesamtgröße meldet |
| `nodes[].instances[]` | gestapelte Instanzen; fehlt, wenn der Knoten nur sein primäres Modell hat |
| `nodes[].version` | der Build dieses Knotens; fehlt bei einem Peer, der zu alt ist, um ihn mitzuschicken |
| `versions_agree` | ob die Flotte sich über ihren Build einig ist. **`false` ist ein Alarm:** eugrs Launcher vergleicht die Engine-Images der Knoten und bricht einen verteilten Start ab, wenn sie abweichen — Minuten nach dem Start. Ein Peer, der gar keine Version meldet, zählt **nicht** als Uneinigkeit; er sagt nur nichts, und daraus einen Konflikt zu erfinden hieße, wegen eines längst erledigten Updates zu alarmieren |
| `versions` | nur wenn sie sich uneinig sind: die gefundenen Versionen |

Diese Werte stammen aus den UDP-Ankündigungen der Knoten, nicht aus einer
Abfrage — ein Knoten, der gerade nicht sendet, trägt hier seinen letzten
Stand. `status` sagt, wie alt der ist.

---

## 12. `<prefix>/<node_id>/events/launch` — ein Ladevorgang ist fertig

Das einzige Topic, das ein **Ereignis** trägt statt eines Zustands. Nicht
retained: eine behaltene Nachricht würde bei jeder Wiederverbindung eines
Dashboards erneut ausgeliefert und einen Ladevorgang von letzter Woche als
Neuigkeit melden.

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "model": "Qwen/Qwen3.8",
  "outcome": "ready",
  "seconds": 292.0,
  "timeline": [
    { "phase": "starting", "seconds": 38.0 },
    { "phase": "loading_weights", "seconds": 71.0 },
    { "phase": "profiling", "seconds": 183.0 }
  ]
}
```

| Feld | Bedeutung |
|---|---|
| `outcome` | `ready` oder `failed` |
| `seconds` | Gesamtdauer. Eigenes Feld, damit ein Dashboard für die eine Zahl, nach der alle fragen, keine Liste summieren muss |
| `timeline` | wohin die Zeit ging, Phase für Phase — `starting` ist Container starten und torch importieren, `profiling` ist Kernel kompilieren und den KV-Cache bemessen |
| `error` | nur bei `failed`: der Grund, auf 1000 Zeichen gekürzt. Ein Ereignis ist kein Log; den Rest trägt das Log-Topic |

Das Ereignis wird beim nächsten Sendetakt veröffentlicht, nicht in der
Sekunde des Abschlusses. Bei einem Ladevorgang von Minuten ist das ein
Rundungsfehler — und der Preis dafür, dass es **jeden** Start erfasst: über
die API, über ein angewandtes Profil und über das Wiederherstellen nach einem
Neustart. Der Speicherwächter ist der umgekehrte Fall; dort *ist* die
Zeitnähe die Nachricht, deshalb sendet er außer der Reihe.

---

## 13. Logs

Standardmäßig **aus**. Einschalten in Settings → Monitoring → *forward log
lines*. Dort stehen auch das Zeilenlimit pro Nachricht und der Mindest-Level
für die AINode-Zeilen.

### `<prefix>/<node_id>/logs/ainode`

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "source": "ainode", "timestamp": 1789935612.233,
  "count": 3,
  "lines": [
    "2026-09-20T18:14:02 INFO ainode.engine.backends.eugr: Starting solo vLLM via the launcher",
    "2026-09-20T18:14:31 WARNING ainode.safety.memory_guard: host memory at 3980 MB, below the 4096 MB line (1/2)",
    "2026-09-20T18:14:33 ERROR ainode.safety.memory_guard: Stopped by the host memory guard …"
  ]
}
```

### `<prefix>/<node_id>/logs/vllm/<modell>`

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "source": "vllm", "instance": "demon-zombie_MiniMax-M2.7-AWQ-4bit",
  "timestamp": 1789935612.233,
  "count": 12, "dropped": 84, "skipped_bytes": 131072,
  "lines": ["(EngineCore pid=189) INFO 09-20 18:14:55 [core.py:123] Initializing a V1 LLM engine …"]
}
```

| Feld | Bedeutung |
|---|---|
| `source` | `ainode` oder `vllm` |
| `instance` | nur bei `vllm`: die topic-sichere Modell-ID, identisch mit der letzten Topic-Ebene |
| `lines` | die Zeilen, **älteste zuerst** |
| `count` | `len(lines)` — bequem für Dashboards, die nicht zählen wollen |
| `dropped` | wie viele Zeilen wegen des Limits **nicht** mitgeschickt wurden; fehlt, wenn keine |
| `skipped_bytes` | wie viele Bytes übersprungen wurden, weil die Datei weiter gewachsen war als eine Nachricht tragen darf; fehlt, wenn keine |

### Was nicht gesendet wird, und warum

MQTT ist ein schlechter Transport für einen Feuerwehrschlauch, und ein
vLLM-Log ist einer: es zeichnet mehrmals pro Sekunde eine Fortschrittsanzeige
neu und schreibt eine Zeile pro Gewichts-Shard. Deshalb:

* Es geht **nie ein ganzes Log** raus, sondern nur, was seit der letzten
  Veröffentlichung dazugekommen ist.
* Fortschrittsanzeigen und Shard-Zähler werden verworfen (`…%|`, `it/s]`,
  `s/it]`, `Loading safetensors checkpoint shards`).
* Pro Nachricht höchstens das eingestellte Zeilenlimit (Vorgabe 100) **und**
  höchstens 64 KB. Was nicht passt, wird gezählt und gemeldet, nicht
  verschwiegen.
* Eine stille Instanz sendet **gar nichts** — kein leeres Rauschen im Broker.
* Nach einem Neustart beginnt AINode am Dateiende. Die erste Nachricht ist
  also nicht der komplette letzte Ladevorgang.
* Eine angefangene letzte Zeile wartet auf ihr Newline, statt über zwei
  Nachrichten zerrissen zu werden.
* Ein Relaunch schreibt eine neue Datei; dem wird gefolgt. Genauso einem
  `truncate`.

Jede Instanz schreibt in **ihre eigene** Logdatei (`vllm.log` für die
primäre, `vllm-<port>.log` für gestapelte), sonst gäbe es „pro Instanz" gar
nicht.

---

## 14. Rezepte

### Ein Knoten ist voll, bevor es kracht

```
ainode/+/safety  →  blocking_launches == true     der Wächter lehnt Starts ab
                    below_critical == true        er ist im Begriff einzugreifen
                    stops_total steigt            er hat eine Engine beendet
```

Besser als ein eigener Schwellwert auf `system.memory`, weil hier die Grenzen
stehen, die der Wächter **wirklich** durchsetzt — du müsstest sonst deine
Alarmschwelle jedes Mal nachziehen, wenn du die Reserve verstellst. Und
`stops_total` kommt außer der Reihe, also in Sekunden statt beim nächsten
Takt.

### Ein Kabel im Ring wird schlecht

```
ainode/+/fabric  →  ports.*.errors_new > 0        ein Kabel ansehen
                    ports.*.state != "ACTIVE"     der Link ist unten
```

`errors_new`, nicht `errors_total`: ein Zähler, der seit dem Aufbau der
Maschine auf 3 steht, ist kein Fehler, der gerade passiert.

### Ein Modell lädt nicht durch

### Ein Modell lädt nicht durch

```
ainode/+/events/launch  →  outcome == "failed"
ainode/+/models         →  loaded[].load_phase bleibt > 10 min ungleich "ready"
```

Das Ereignis bringt den Grund gleich mit (`error`) und die Aufschlüsselung,
wohin die Zeit ging (`timeline`). Der vollständige Text steht auf
`ainode/<node>/logs/vllm/<modell>`.

### Ein Modell wird langsam, und du willst wissen warum

```
ainode/+/engine/#  →  kv_cache_percent > 90        der Cache ist die Grenze
                      preemptions_total steigt     Anfragen werden verdrängt
                      requests_waiting > 0         die Warteschlange staut
                      time_to_first_token_s steigt  es ist der Prefill, nicht das Decode
```

Diese vier unterscheiden die Fälle, die in „Tokens pro Sekunde" identisch
aussehen und entgegengesetzte Maßnahmen brauchen: mehr Cache (weniger
Kontext, `--kv-cache-dtype fp8`, ein Knoten mehr) gegen weniger Parallelität
(`--max-num-seqs`).

### Ein Knoten ist weg

```
ainode/+/status  →  status == "offline"
```

Die richtige Quelle: das kommt vom **Broker** per Last Will, nicht vom toten
Knoten. Ohne Timeout in deinem Dashboard, und ohne die Retain-Falle — die
letzte Messung eines vor einer Stunde gestorbenen Knotens sähe sonst genauso
frisch aus wie die eines lebenden. `ainode/cluster` mit
`nodes_online < nodes_total` ist die Gegenprobe aus Sicht des Heads.

### Die Flotte ist uneinig über ihren Build

```
ainode/cluster   →  versions_agree == false
```

Ein verteilter Start bricht daran ab, Minuten nachdem er angefangen hat. Hier
siehst du es vorher.

### Durchsatz je Modell

```
ainode/+/models  →  per_model.<modell>.avg_tokens_per_second
```

Nicht mit `nodes` multiplizieren: der Wert ist bereits der der Instanz, egal
über wie viele Knoten sie verteilt ist.

---

## 15. Wo das im Code steht

| Thema | Datei |
|---|---|
| Topics, Verbindung, Sendeschleife, Last Will, Ereignisse | `ainode/telemetry/mqtt.py` |
| Nutzlasten `system`/`gpu`/`models`/`cluster`/`safety`/`fabric`/`transfers` | `ainode/telemetry/payloads.py` |
| Engine-Metriken: Prometheus lesen und destillieren | `ainode/telemetry/engine_metrics.py` |
| Logs: Puffer, Tail, Nutzlasten | `ainode/telemetry/logs.py` |
| Launch-Ereignisse | `ainode/telemetry/events.py` |
| Einstellungen und API | `ainode/telemetry/api_routes.py` |
| Systemmessung | `ainode/metrics/system.py` |
| RoCE-Zähler | `ainode/metrics/fabric.py` |
| GPU- und Anfragemessung | `ainode/metrics/collector.py` |
| Der Speicherwächter selbst | `ainode/safety/memory_guard.py` |

Tests, die das Schema festhalten: `tests/test_telemetry.py`,
`tests/test_mqtt_logs.py`, `tests/test_fabric_metrics.py`,
`tests/test_engine_metrics.py`, `tests/test_mqtt_availability.py`,
`tests/test_mqtt_events_and_extras.py` — und `tests/test_mqtt_schema_doc.py`,
das **dieses Dokument** gegen die Payload-Bauer prüft. Ändert sich ein Feld,
fällt dort ein Test, und dann gehört diese Datei mit angepasst.
