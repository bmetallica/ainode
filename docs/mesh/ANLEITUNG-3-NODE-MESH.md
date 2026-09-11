# AINode auf 3 DGX Sparks im Ring — Schritt für Schritt

Für genau eine Konfiguration: **drei DGX Sparks, ringförmig ohne Switch
verkabelt**, zusätzlich alle drei am gemeinsamen 10G-Ethernet (`enP7s7`).
Der Cluster wurde mit **NVIDIA Sync** aufgesetzt.

Jeder Schritt ist **pro Knoten einzeln** ausgeschrieben. Wo etwas auf allen
drei Knoten identisch ist, steht es einmal mit dem Hinweis „auf allen drei".

Namen in dieser Anleitung: `spark1`, `spark2`, `spark3`. `spark1` ist der Head
(Web-UI + API). Ersetze sie durch deine echten Hostnamen.

---

## Bevor du anfängst: was NVIDIA Sync bereits gemacht hat

NVIDIA Sync koppelt Sparks paarweise über den 200G-QSFP-Link und vergibt dabei
selbst IP-Adressen auf den ConnectX-7-Interfaces. **Für einen Dreier-Ring
reicht das nicht**, und die vergebenen Adressen kollidieren in der Regel mit
dem Schema, das der Ring braucht: dort muss jeder der vier CX7-Ports pro Knoten
in einem *eigenen* /24 liegen, und die beiden Enden eines Kabels müssen sich
dasselbe /24 teilen.

Sieh dir zuerst an, was aktuell gesetzt ist — **auf allen drei**:

```bash
ip -br -4 addr show | grep -E 'enp1s0f|enP2p1s0f|enP7s7'
ls /etc/netplan/
```

Wenn dort bereits CX7-Adressen stehen, die nicht dem Schema in Schritt 2
entsprechen, ersetzt du sie dort. Die von NVIDIA Sync eingerichtete
SSH-Verbindung und der Hostname-Eintrag bleiben nützlich — die nimmst du mit.

> **Ehrlicher Hinweis:** Ich kann nicht prüfen, was deine Sync-Version konkret
> geschrieben hat. Vergleiche die Ausgabe oben mit Schritt 2 und passe nur an,
> was abweicht. Lösche keine Netplan-Datei, ohne vorher hineingesehen zu haben.

---

## Schritt 1 — Verkabelung prüfen

Der Ring braucht eine bestimmte Steckung: **Port 0 des einen Sparks an Port 1
des nächsten** (nicht Port 0 an Port 0, wie beim Zweier-Setup).

```
spark1 Port 0  ──────  Port 1 spark2
spark2 Port 0  ──────  Port 1 spark3
spark3 Port 0  ──────  Port 1 spark1
```

Zusätzlich: alle drei mit dem **RJ-45-10G-Port** (`enP7s7`) am selben Switch /
im selben flachen LAN. Das ist im Ring nicht optional — es ist der einzige Weg,
auf dem jeder Knoten jeden erreicht.

Prüfen — **auf allen drei**:

```bash
ibdev2netdev | grep 'Up)' | wc -l
```

**Erfolg: `4`.** Vier aktive CX7-Links bedeuten „beide QSFP-Ports verkabelt".
Steht dort `2`, ist nur ein Kabel gesteckt oder eine Seite ist down — dann
stimmt die Verkabelung nicht und alles Weitere greift nicht.

Fehlt `ibdev2netdev`, geht es auch ohne:

```bash
for d in /sys/class/infiniband/*/ports/1/state; do cat "$d"; done
```
Erfolg: vier Zeilen, alle beginnend mit `4:` (ACTIVE).

---

## Schritt 2 — Netplan, pro Knoten einzeln

Jeder der vier CX7-Ports bekommt ein **eigenes /24**. Die beiden Enden eines
Kabels teilen sich ein /24. MTU 9000 auf allen CX7-Interfaces.

> **Wichtig:** `enP7s7` (10G) wird hier **nicht** angefasst — das bleibt bei
> DHCP oder deiner bestehenden statischen Konfiguration.

### spark1

`/etc/netplan/40-cx7.yaml`:

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:            # Port 0 → spark2 Port 1
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.177.11/24]
    enP2p1s0f0np0:          # Port 0, zweiter Twin
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.178.11/24]
    enp1s0f1np1:            # Port 1 → spark3 Port 0
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.187.11/24]
    enP2p1s0f1np1:          # Port 1, zweiter Twin
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.188.11/24]
```

### spark2

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:            # Port 0 → spark3 Port 1
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.197.12/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.198.12/24]
    enp1s0f1np1:            # Port 1 → spark1 Port 0
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.177.12/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.178.12/24]
```

### spark3

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:            # Port 0 → spark1 Port 1
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.187.13/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.188.13/24]
    enp1s0f1np1:            # Port 1 → spark2 Port 0
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.197.13/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.198.13/24]
```

### Anwenden — auf allen drei

```bash
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

**Nie zwei Interfaces ins selbe Subnetz legen.** Das verwirrt die
Autoerkennung und zerlegt das Routing — sowohl bei AINode als auch bei NCCL.

### Verkabelung gegen die Adressen prüfen

Vom **spark1** aus:

```bash
ping -c2 192.168.177.12    # → spark2, über Port 0
ping -c2 192.168.187.13    # → spark3, über Port 1
```
Vom **spark2** aus:
```bash
ping -c2 192.168.197.13    # → spark3, über Port 0
```

**Erfolg: alle drei antworten.** Wenn nicht, sind zwei Kabel vertauscht —
korrigiere die Steckung, nicht die Adressen.

---

## Schritt 3 — 10G-Ethernet prüfen

**Auf allen drei:**

```bash
ip -br -4 addr show enP7s7
```

**Erfolg:** Status `UP` und eine Adresse aus deinem LAN, z. B.
`10.0.0.11/24`. Alle drei müssen im **selben** Subnetz liegen und sich
gegenseitig pingen können. Notiere die drei Adressen — du brauchst sie gleich.

In dieser Anleitung: `spark1 = 10.0.0.11`, `spark2 = 10.0.0.12`, `spark3 = 10.0.0.13`.

---

## Schritt 4 — Passwortloses SSH vom Head zu den anderen

Nur vom Head (`spark1`) zu den beiden anderen — **über die 10G-Adressen**, nicht
über die CX7-Adressen.

Auf **spark1**:

```bash
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id -o StrictHostKeyChecking=accept-new 10.0.0.12
ssh-copy-id -o StrictHostKeyChecking=accept-new 10.0.0.13
```

Prüfen:

```bash
ssh -o BatchMode=yes 10.0.0.12 hostname
ssh -o BatchMode=yes 10.0.0.13 hostname
```

**Erfolg:** beide geben ihren Hostnamen aus, ohne nach einem Passwort zu fragen.

Falls NVIDIA Sync bereits Schlüssel verteilt hat, funktioniert das ggf. sofort —
dann überspringe `ssh-copy-id`.

---

## Schritt 5 — Gemeinsames Modellverzeichnis

**Auf allen drei:**

```bash
sudo mkdir -p /mnt/shared-models
```

Der Installer bricht ab, wenn das Verzeichnis fehlt. Ein leeres Verzeichnis
genügt zum Start.

**Empfohlen:** exportiere es per NFS vom `spark1` und mounte es auf den anderen
beiden — **über die 10G-Adresse**, nicht über eine CX7-Adresse. Der Ring ist
nicht voll vermascht gedacht für NFS, und AINode verteilt Modellgewichte
ohnehin selbst über die Direktlinks (siehe Schritt 9).

---

## Schritt 6 — Image bereitstellen (einmalig)

Dieser Fork veröffentlicht sein **eigenes** Image. Vor der ersten Installation
muss eines existieren. Zwei Wege — Details in
[`BOOTSTRAP.md`](BOOTSTRAP.md):

**A) Über die CI** (braucht einen self-hosted Runner mit den Labels
`[self-hosted, dgx-spark, aarch64]`):

```bash
gh workflow run publish-image.yml -f push=true
```

Danach **das GHCR-Package auf public stellen**: GitHub → Packages → `ainode` →
Package settings → Change visibility. Der Installer löst Tags anonym auf.

**B) Lokal auf einem Spark bauen** (kein Registry nötig):

```bash
scripts/build-base-image.sh
docker build -f scripts/Dockerfile.ainode -t ainode:dev .
```

Dann das Image auf die anderen beiden bringen:

```bash
docker save ainode:dev | ssh 10.0.0.12 docker load
docker save ainode:dev | ssh 10.0.0.13 docker load
```

---

## Schritt 7 — Installation, pro Knoten einzeln

`INSTALLER` steht für:
`https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh`

### spark1 — der Head

```bash
curl -fsSL $INSTALLER | bash
```

**Ohne `--job`.** Der Head wird als `solo` installiert und erst beim Start eines
verteilten Modells zum Head — das macht die Weboberfläche in Schritt 9. So
startet der Knoten auch dann sauber, wenn einer der anderen aus ist.

Bei einem lokal gebauten Image (Weg B):
```bash
AINODE_IMAGE=ainode:dev bash -c "$(curl -fsSL $INSTALLER)"
```

### spark2 — Member

```bash
curl -fsSL $INSTALLER | bash -s -- --job worker
```

### spark3 — Member

```bash
curl -fsSL $INSTALLER | bash -s -- --job worker
```

`--job worker` setzt `distributed_mode: "member"`: kein eigenes Modell, kein
eigener Engine-Start. Der Knoten meldet sich per UDP-Discovery und wartet
darauf, dass der Head ihm einen Rang zuweist.

---

## Schritt 8 — Koordinations-Interface setzen, auf allen drei

Der Installer schreibt `cluster_interface: "enP2p1s0f1np1"` — einen CX7-Port.
Im Ring erreicht dieser Port nur **einen** Nachbarn, taugt also nicht als
Cluster-Adresse. AINode erkennt den Mesh automatisch und weicht auf `enP7s7`
aus; wir tragen es trotzdem explizit ein, damit es auch dann stimmt, wenn
einmal nur zwei Links aktiv sind (z. B. ein Kabel gezogen).

**Auf allen drei:**

```bash
sudo systemctl stop ainode

python3 - <<'EOF'
import json, pathlib
p = pathlib.Path.home() / ".ainode" / "config.json"
cfg = json.loads(p.read_text())
cfg["coord_interface"] = "enP7s7"     # Ray, Discovery, SSH
cfg["rdma_hcas"] = []                 # leer = automatisch alle vier RoCE-Geräte
cfg["cluster_id"] = "spark-mesh"      # auf allen drei identisch!
p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
EOF

sudo systemctl start ainode
```

`cluster_id` muss auf allen drei **gleich** sein — nur Knoten mit derselben ID
sehen einander.

---

## Schritt 9 — Prüfen, dass der Mesh erkannt wird

**Auf allen drei:**

```bash
docker exec ainode ainode doctor
```

**Erfolg — genau diese Zeilen, ohne Warnung:**

```
Fabric                 mesh
Active CX7 links       4
Coordination iface     enP7s7
Coordination IP        10.0.0.11             ← die 10G-Adresse, keine 192.168.x
NCCL_IB_HCA            roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1
NCCL_IB_MERGE_NICS     0
NCCL_IB_SUBNET_AWARE_ROUTING  1
NCCL_NET_PLUGIN        none
```

| Ausgabe | Bedeutung |
|---|---|
| `Fabric direct` | nur 2 Links aktiv → zurück zu Schritt 1, ein Kabel fehlt |
| `Fabric unknown` | eine andere Zahl als 2 oder 4 → Verkabelung prüfen |
| `Coordination IP <none>` | `enP7s7` hat keine Adresse → Schritt 3 |
| eine `192.168.x`-Adresse als Coordination IP | `coord_interface` wurde nicht gesetzt → Schritt 8 |

Dann vom **spark1** prüfen, dass alle drei einander sehen:

```bash
curl -s localhost:3000/api/cluster/resources \
  | jq '.nodes[] | {hostname, fabric_ip, ib_ips}'
```

**Erfolg:** drei Einträge; jeder mit seiner **10G-Adresse** als `fabric_ip` und
**vier** `192.168.1xx.x`-Adressen in `ib_ips`. Die `fabric_ip` darf in `ib_ips`
nicht vorkommen — das sind zwei getrennte Wege: Koordination über 10G,
Massentransfer über die Direktlinks.

---

## Schritt 10 — Modell über alle drei Knoten starten

Im Browser auf `http://<spark1>:3000`.

1. **MODELS** → gewünschtes Modell herunterladen und warten, bis es fertig ist.
2. Im Launch-Bereich: Modell auswählen.
3. **Sharding**: `Pipeline` wählen.
4. **Nodes**: alle drei Punkte aktivieren.
5. **LAUNCH**.

**Warum Pipeline und nicht Tensor:** Tensor-Parallel teilt die Attention-Heads
auf die Ränge auf, und Head-Zahlen sind Zweierpotenzen — **TP=3 unterstützt kein
gängiges Modell**. Wählst du trotzdem `Tensor` mit drei Knoten, lehnt AINode das
ab, bevor irgendetwas startet, und nennt die Alternativen.

Lässt du die Auswahl auf Automatik (bzw. rufst die API ohne `strategy` auf),
ergibt sich bei drei Knoten ohnehin **PP=3**.

Dasselbe per API:

```bash
curl -s -X POST http://<spark1>:3000/api/sharding/launch \
  -H 'Content-Type: application/json' \
  -d '{"model":"<repo/modell>","node_ids":["<id1>","<id2>","<id3>"],"strategy":"pipeline"}' | jq
```

Die `node_ids` stehen in der Ausgabe aus Schritt 9.

**Erfolg:**
```json
{"status":"launching","strategy":"pipeline","pipeline_parallel_size":3,
 "tensor_parallel_size":1,"parallel_plan":{"label":"PP=3"}}
```

Dann in den Logs mitlesen:

```bash
docker exec ainode tail -f /root/.ainode/logs/distributed.log
```

**Erfolg:** der Serve erreicht READY **und erzeugt Tokens**. Ein `200` auf
`/v1/models` ist kein Beweis — auf GB10 kann die Engine READY melden und beim
ersten echten Prompt sterben. Teste mit einer langen Eingabe:

```bash
curl -s http://<spark1>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<repo/modell>","messages":[{"role":"user","content":"Erkläre in 200 Wörtern, wie Pipeline-Parallelität funktioniert."}]}' | jq -r '.choices[0].message.content'
```

### Was du bei 3 Knoten wählen solltest

| Situation | Strategie | Warum |
|---|---|---|
| Modell passt nicht auf zwei Knoten | **Pipeline (PP=3)** | teilt die Gewichte über alle drei; der einzige Weg für große Modelle |
| Modell passt auf einen Knoten, du willst Durchsatz | **Data (DP=3)** | drei volle Kopien, mehr parallele Anfragen — **auf dieser Hardware ungetestet** |
| Modell passt auf zwei Knoten | **Tensor (TP=2)**, nur zwei Knoten auswählen | der bewährte Pfad; der dritte Knoten bleibt frei |

---

## Schritt 11 — Wenn ein Knoten ausfällt

Fällt ein Member aus, läuft die Instanz auf dem Head weiter, hat aber die Ränge
verloren, die Ray dort platziert hatte — **sie kann nicht mehr bedienen**.

Im UI wird sie nach etwa 30 Sekunden bernsteinfarben als `DEGRADED` markiert,
mit dem Namen des verlorenen Knotens und einem **RELAUNCH**-Button.

RELAUNCH startet sie auf den verbliebenen Knoten neu und **plant dabei neu**:
aus PP=3 wird TP=2, nicht das unmögliche TP=3. Passt das Modell nicht mehr auf
zwei Knoten, wird abgelehnt — mit Angabe, welcher Knoten wie viel Platz hat.

AINode startet das **nicht von selbst** neu: ein Modell, das über drei Knoten
liegt, passt meist nicht auf zwei, und ein automatischer Versuch würde einen
sichtbaren Ausfall gegen ein OOM tauschen.

Prüfen:

```bash
curl -s http://<spark1>:3000/api/cluster/resources \
  | jq '.distributed_instances[] | {model, degraded, missing_peer_ips, surviving_node_ids}'
```

Der Head selbst startet auch dann sauber, wenn ein Member aus ist — er kommt
mit Web-UI hoch, ohne Engine, und schreibt den Grund ins Journal:

```bash
journalctl -u ainode -n 50
```

---

## Fehlerbehebung

| Symptom | Ursache | Behebung |
|---|---|---|
| `doctor` sagt `Fabric direct` | nur 2 CX7-Links aktiv | Schritt 1 — zweites Kabel / Port down |
| `doctor` sagt `Coordination IP <none>` | `enP7s7` ohne Adresse | Schritt 3 |
| Nur ein Knoten in `/api/cluster/resources` | unterschiedliche `cluster_id` oder `discovery_port` | Schritt 8; Port ist 5679 |
| Launch: „No fabric IP known for node(s)" | der Member kündigt keine Adresse an | auf dem Member `ainode doctor`, dann Schritt 8 |
| Launch: „Tensor-parallel across 3 nodes is not supported" | `Tensor` bei drei Knoten gewählt | Pipeline wählen — so ist es gedacht |
| Engine startet, stirbt beim ersten Prompt | GB10/sm120-FlashInfer unter CUDA-Graph-Capture | bekannt; `--enforce-eager` wird automatisch gesetzt. Tritt es trotzdem auf: Logs mitschicken |
| Launcher-Fehler „ibdev2netdev not found" | unvollständige `.env` | sollte nicht mehr vorkommen — AINode bricht vorher mit klarer Meldung ab. Wenn doch: Meldung mitschicken |
| Modelltransfer läuft über 10G statt Direktlink | kein gemeinsames Subnetz gefunden | Schritt 9, Feld `ib_ips` prüfen |

Logs:

```bash
docker exec ainode tail -100 /root/.ainode/logs/distributed.log   # verteilter Launch
docker exec ainode tail -100 /root/.ainode/logs/vllm.log          # Solo-Engine
journalctl -u ainode -n 100                                       # Dienst selbst
```

---

## Was in dieser Konfiguration läuft und was nicht

**Läuft:**
- Koordination (Ray, Discovery, SSH) über 10G — der einzige Weg, auf dem jeder
  Knoten jeden erreicht
- NCCL über alle vier RoCE-Geräte, Routing im Ring macht NCCL selbst
- Modellgewichte über die Direktlinks, nicht über 10G
- Pipeline-Parallel über alle drei Knoten
- Tensor-Parallel über beliebige **zwei** der drei (jedes Paar ist direkt
  verkabelt)
- Ausfall eines Members: sichtbar, mit Ein-Klick-Neustart auf dem Rest

**Läuft nicht / ungetestet:**
- **Tensor-Parallel über drei Knoten** — kein gängiges Modell unterstützt TP=3.
  Keine Einschränkung von AINode, sondern der Modelle.
- **Data-Parallel** ist implementiert, aber auf dieser Hardware **nicht
  verifiziert**. Die Automatik wählt es deshalb nie. Wenn du es erfolgreich
  fährst, trag es in `ainode/engine/AGENTS.md` ein.
- **Ausfall des Heads** — dann ist die Weboberfläche weg. Es gibt keine
  automatische Übernahme durch einen anderen Knoten.
- **Automatischer Neustart nach Knotenausfall** — bewusst nicht, siehe
  Schritt 11.
