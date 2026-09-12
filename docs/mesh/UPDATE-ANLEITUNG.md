# Update auf 0.6.0 — Profile, Rezepte im Cluster, Gemma 4

Für den bestehenden Cluster `ai-vkv` (Spark1 Head `192.168.1.2`, Spark2
`192.168.1.3`, Spark3 `192.168.1.4`). Rechne mit 10–20 Minuten, davon das
meiste Warten auf den Orchestrator-Build.

**Das Engine-Image muss nicht neu gebaut werden.** Geändert hat sich nur der
Orchestrator (`ainode:dev`) — `vllm-node:latest` und die gepinnte NCCL bleiben,
wie sie sind. Das spart die 15–25 Minuten aus Schritt 6.

---

## Was dieses Update ändert

| Änderung | Warum es dich betrifft |
|---|---|
| Katalog-Rezepte gelten auch für **verteilte** Starts | Qwen3.8 über drei Knoten bekam bisher weder sein Engine-Image noch `--reasoning-parser`/`--tool-call-parser`/Speculative-Config und starb am argparse-Fehler (Exit-Code 2). Derselbe Klick führte je nach Knotenzahl zu zwei verschiedenen Kommandos. |
| Ein Advanced-Feld wirft das Rezept nicht mehr weg | `--max-num-seqs` für Qwen3.8 zu setzen hat vorher Reasoning-Parser, Tool-Parser und MTP-Config mitgenommen. Jetzt gewinnt dein Wert **pro Flag**, der Rest des Rezepts bleibt. |
| `proven_tp` begrenzt den Tensor-Split | Ein Modell, das nur bei TP=1 verifiziert ist, wird über mehrere Knoten als **Pipeline** gefahren statt seine Attention-Heads in einer nie getesteten Breite zu zerlegen. |
| Der Standard-Backend (eugr) beachtet alle Launch-Felder | `served_model_name`, `kv_cache_dtype` und `trust_remote_code` wurden im UI angenommen, gespeichert — und auf dem Weg zu vLLM stillschweigend verworfen. Wer in OpenWebUI einen kurzen Modellnamen wollte, bekam die volle Repo-ID. |
| Tool-Calling automatisch | Open WebUI schickt bei jedem Chat mit Werkzeug `tool_choice: "auto"`. vLLM lehnt das ab, solange die Engine nicht mit `--enable-auto-tool-choice` **und** einem passenden `--tool-call-parser` gestartet wurde. Das hatten genau drei kuratierte Modelle; alles von HF Geladene nicht. AINode leitet den Parser jetzt aus der Modellfamilie ab. |
| Parser für Qwen3.8 und Nemotron 3.5 korrigiert | Beide standen auf `qwen3_coder`; die Rezepte, aus denen die Einträge stammen, benutzen für genau diese Modelle `qwen3_xml`. `qwen3_coder` gehört zur Qwen3-Coder-/3.5-Generation. |
| **Profile** | Mehrere Modelle als einen Zustand beschreiben, anwenden und beim Start wiederherstellen — inklusive verteilter Instanzen, die `instances.json` nie abbilden konnte. Siehe Schritt 12 der [Cluster-Anleitung](ANLEITUNG-3-NODE-MESH.md). |
| Gemma 4 26B-A4B (NVFP4) im Katalog | Mit den Flags aus eugrs erprobtem Rezept (MIT, im Code als Herkunft vermerkt). **Anmerkung:** Das Modell heißt 26B-A4B, nicht 31B — eine 31B-Variante gibt es in der Referenzimplementierung nicht. |

---

## Schritt 1 — Laufenden Zustand sichern (auf Spark1)

Falls beim Update etwas schiefgeht, willst du wissen, was lief:

```bash
ssh Spark1
curl -s localhost:3000/api/cluster/resources | jq '.instances, .distributed_instances' \
  > ~/ainode-vor-update.json
cp ~/.ainode/config.json ~/.ainode/config.json.bak
```

`~/.ainode/instances.json` und `config.json` werden vom Update **nicht**
angefasst; der neue Stand liest beide unverändert weiter.

---

## Schritt 2 — Neuen Stand holen und bauen (auf Spark1)

```bash
ssh Spark1
cd ~/ainode          # das Verzeichnis aus Schritt 6 Weg B
git pull
scripts/build-ainode-image.sh
```

**Erfolg:** am Ende steht `ainode:dev` mit frischem Zeitstempel:

```bash
docker images ainode:dev
```

Baut das Skript nicht, weil der Build-Kontext fehlt: das Skript **muss** aus dem
Repo-Verzeichnis heraus laufen (`cd ~/ainode`), nicht aus `$HOME`.

---

## Schritt 3 — Image auf die anderen beiden Knoten

```bash
docker save ainode:dev | ssh Spark2 docker load
docker save ainode:dev | ssh Spark3 docker load
```

**Erfolg:** beide melden `Loaded image: ainode:dev`.

Das dauert je Knoten ein bis zwei Minuten. Läuft es über 10G statt über einen
Direktlink — das ist hier richtig so: `docker load` spricht die
Management-Adresse an, und der Weg wird einmal beim Update benutzt, nicht im
Betrieb.

---

## Schritt 4 — Dienst neu starten, Member zuerst

Die Member zuerst, damit der Head beim Hochkommen einen vollständigen Cluster
sieht:

```bash
ssh Spark2 'sudo systemctl restart ainode'
ssh Spark3 'sudo systemctl restart ainode'
ssh Spark1 'sudo systemctl restart ainode'
```

**Erfolg:**

```bash
curl -s localhost:3000/api/status | jq '{version, node_id, engine_ready}'
```

zeigt `"version": "0.6.0"`. Zeigt es weiter die alte Nummer, läuft noch der
alte Container:

```bash
docker ps --filter name=ainode --format '{{.Image}} {{.Status}}'
sudo systemctl restart ainode
```

Und alle drei Knoten wieder da:

```bash
curl -s localhost:3000/api/cluster/resources | jq '.nodes[] | {node_id, status}'
```

---

## Schritt 5 — Prüfen, dass das Rezept jetzt auch verteilt greift

Das ist der eigentliche Grund für dieses Update. Qwen3.8 über zwei Knoten
starten und das erzeugte Kommando ansehen:

```bash
curl -s -X POST localhost:3000/api/sharding/launch \
  -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3.8-27B-NVFP4","node_ids":["<head-id>","<peer-id>"]}' | jq

docker exec ainode cat /opt/spark-vllm-docker/examples/ainode-distributed.sh
```

**Erfolgskriterium:** im Skript stehen `--reasoning-parser qwen3`,
`--tool-call-parser qwen3_coder` und `--speculative_config`, und die Instanz
läuft in `--pipeline-parallel-size 2` statt in TP=2 (Qwen3.8 ist bei TP=1
verifiziert). Vorher enthielt dieses Skript keines dieser Flags.

Fehlt etwas, ist es kein Rätselraten — der Grund steht im Log:

```bash
docker exec ainode tail -50 /root/.ainode/logs/distributed.log
```

---

## Schritt 6 — Profil anlegen (optional, aber der Sinn der Sache)

Sobald deine drei Modelle so laufen, wie du sie willst:

1. Weboberfläche → Reiter **Profiles**
2. Name `Endausbau` eintragen → **Save current state**
3. **Set default**

Ab dann stellt ein `systemctl restart ainode` diesen Satz selbst wieder her.
Details, inklusive der Speicheranteile für drei gleichzeitige Modelle, in
Schritt 12 der [Cluster-Anleitung](ANLEITUNG-3-NODE-MESH.md).

---

## Wenn du zurück willst

Der alte Orchestrator liegt noch als Image vor, solange du ihn nicht gelöscht
hast:

```bash
docker images | grep ainode
```

Zurückrollen heißt: alten Tag in `AINODE_IMAGE` eintragen und den Dienst neu
starten. Ein gesetztes Standardprofil stört dabei nicht — ein älterer Stand
kennt `profiles.json` schlicht nicht und macht weiter, was er vorher gemacht
hat (`instances.json`).

---

## Bekannte Stolpersteine

| Symptom | Ursache | Behebung |
|---|---|---|
| `/api/status` zeigt nach dem Neustart die alte Version | Container wurde nicht ersetzt | `docker ps --filter name=ainode`, dann `sudo systemctl restart ainode` |
| Verteilter Start endet mit Exit-Code 2 | Rezept fehlt weiterhin → auf dem Knoten läuft noch der alte Orchestrator | Schritt 3 für diesen Knoten wiederholen |
| „engine image not present on peer" | Das Modell verlangt ein Engine-Image, das ein Member nicht hat | passiert ab diesem Stand nicht mehr von selbst — AINode legt es dort ab. Bleibt die Meldung: `docker images` auf dem Member prüfen |
| Open WebUI: „server does not allow automatic tool calls" | Modell ohne Tool-Parser gestartet, oder Familie nicht erkannt | Launch-Panel → *Advanced* → **Tool calling** auf den passenden Parser stellen und neu starten. Welcher es ist, steht in der Auswahlliste neben dem Namen |
| Werkzeuge werden aufgerufen, aber die Antwort ist Unsinn | falscher Parser für dieses Modell | derselbe Weg — bei Qwen 3.6+ `qwen3_xml`, bei Qwen3-Coder/3.5 `qwen3_coder` |
| Profil anwenden stoppt ein Modell, das du behalten wolltest | „Anwenden" konvergiert — was nicht im Profil steht, wird gestoppt | Profil mit **Save current state** neu aufnehmen, wenn alles läuft |
