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
<prefix>/<node_id>/system
<prefix>/<node_id>/gpu
<prefix>/<node_id>/models
<prefix>/cluster
<prefix>/<node_id>/logs/ainode
<prefix>/<node_id>/logs/vllm/<modell>
```

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
mosquitto_sub -h <broker> -t 'ainode/#' -v          # alles
mosquitto_sub -h <broker> -t 'ainode/+/gpu' -v      # GPU aller Knoten
mosquitto_sub -h <broker> -t 'ainode/+/logs/#' -v   # alle Logs
mosquitto_sub -h <broker> -t 'ainode/cluster' -v    # nur die Flottensicht
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

## 3. `<prefix>/<node_id>/system`

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

## 4. `<prefix>/<node_id>/gpu`

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

## 5. `<prefix>/<node_id>/models`

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

`embeddings` sind In-Process-Modelle, keine vLLM-Instanzen — sie haben keinen
eigenen Port und erscheinen deshalb nicht in `loaded`.

### `per_model` — gemessen, nicht geschätzt

| Feld | Bedeutung |
|---|---|
| `requests` / `errors` | Zähler seit dem Start dieses AINode-Prozesses |
| `avg_latency_ms` | fehlt, solange keine Anfrage gezählt wurde |
| `tokens_generated` | fehlt, wenn die Route keine Tokenzahl meldet |
| `avg_tokens_per_second` | Tokens geteilt durch die Zeit, die **tatsächlich mit Generieren** verbracht wurde — nicht durch die Laufzeit. Ein Modell, das in einer Stunde zehn Anfragen bedient hat, ist nicht langsam, sondern unbeschäftigt; durch die Laufzeit geteilt behauptet die Zahl das Gegenteil |

`uptime_seconds` ist hier die Laufzeit des AINode-Prozesses, im Unterschied zu
`system.uptime_seconds` (Systemlaufzeit).

---

## 6. `<prefix>/cluster` — nur vom Head

```json
{
  "node_id": "spark-1", "node_name": "SPARK1",
  "version": "0.6.0", "timestamp": 1789935612.233,
  "nodes_total": 3,
  "nodes_online": 3,
  "vram_total_gb": 366.0,
  "nodes": [
    { "node_id": "spark-1", "node_name": "SPARK1", "status": "serving",
      "model": "demon-zombie/MiniMax-M2.7-AWQ-4bit", "gpu_memory_gb": 122.0,
      "gpu_memory_used_mb": 64110, "gpu_memory_total_mb": 124928,
      "gpu_memory_used_percent": 51.3, "gpu_utilization_percent": 87.0,
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

Diese Werte stammen aus den UDP-Ankündigungen der Knoten, nicht aus einer
Abfrage — ein Knoten, der gerade nicht sendet, trägt hier seinen letzten
Stand. `status` sagt, wie alt der ist.

---

## 7. Logs

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

## 8. Rezepte

### Ein Knoten ist voll, bevor es kracht

Auf GB10 teilen sich GPU und Betriebssystem einen Speicher, also reicht ein
Alarm:

```
ainode/+/system  →  memory.available_mb < 8192    Warnung
                    memory.available_mb < 4096    der Wächter greift jetzt ein
```

Dieselben Grenzen setzt der Host-Speicherwächter durch (Settings → Memory
Guard). Ein Alarm darauf sagt dir dasselbe wie das Log — nur früher.

### Ein Modell lädt nicht durch

```
ainode/+/models  →  loaded[].load_phase bleibt > 10 min ungleich "ready"
                    loaded[].status == "failed"
```

Der zugehörige Grund steht auf `ainode/<node>/logs/vllm/<modell>`.

### Ein Knoten ist weg

```
ainode/cluster   →  nodes_online < nodes_total
```

Zuverlässiger als das Ausbleiben von `ainode/<node>/system`, weil ein
verschwundener Knoten ja gerade nichts mehr sendet — und mit `retain` sähe
seine letzte Nachricht auf ewig frisch aus.

### Durchsatz je Modell

```
ainode/+/models  →  per_model.<modell>.avg_tokens_per_second
```

Nicht mit `nodes` multiplizieren: der Wert ist bereits der der Instanz, egal
über wie viele Knoten sie verteilt ist.

---

## 9. Wo das im Code steht

| Thema | Datei |
|---|---|
| Topics, Verbindung, Sendeschleife | `ainode/telemetry/mqtt.py` |
| Nutzlasten `system`/`gpu`/`models`/`cluster` | `ainode/telemetry/payloads.py` |
| Logs: Puffer, Tail, Nutzlasten | `ainode/telemetry/logs.py` |
| Einstellungen und API | `ainode/telemetry/api_routes.py` |
| Systemmessung | `ainode/metrics/system.py` |
| GPU- und Anfragemessung | `ainode/metrics/collector.py` |

Tests, die das Schema festhalten: `tests/test_telemetry.py` und
`tests/test_mqtt_logs.py`. Ändert sich ein Feld, ändert sich dort ein Test —
und dann gehört diese Datei mit angepasst.
