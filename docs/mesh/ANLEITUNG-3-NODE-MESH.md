# AINode auf dem Cluster `ai-vkv` — Schritt für Schritt

Für genau diesen Cluster: **drei DGX Sparks, ringförmig ohne Switch**, alle drei
zusätzlich am flachen 10G-LAN. Aufgesetzt mit **NVIDIA Sync**.

Alle Befehle enthalten die echten Adressen — nichts zu ersetzen. Jeder Schritt
ist pro Knoten einzeln ausgeschrieben; wo etwas überall gleich ist, steht es
einmal mit dem Hinweis „auf allen drei".

**Head ist `Spark1`** (Web-UI + API).

## Deine Topologie

| | Spark1 | Spark2 | Spark3 |
|---|---|---|---|
| **LAN** `enP7s7` | `192.168.1.2` | `192.168.1.3` | `192.168.1.4` |
| Port 0 `enp1s0f0np0` | `10.100.36.1` | `10.100.34.2` | `10.100.32.1` |
| Port 0 `enP2p1s0f0np0` | `10.100.37.1` | `10.100.35.2` | `10.100.33.1` |
| Port 1 `enp1s0f1np1` | `10.100.32.2` | `10.100.36.2` | `10.100.34.1` |
| Port 1 `enP2p1s0f1np1` | `10.100.33.2` | `10.100.37.2` | `10.100.35.1` |

Daraus ergeben sich sechs Punkt-zu-Punkt-Netze, je eins pro Kabel-Twin:

| Netz | Verbindet |
|---|---|
| `10.100.36.0/24`, `10.100.37.0/24` | Spark1 **Port 0** ↔ Spark2 **Port 1** |
| `10.100.34.0/24`, `10.100.35.0/24` | Spark2 **Port 0** ↔ Spark3 **Port 1** |
| `10.100.32.0/24`, `10.100.33.0/24` | Spark3 **Port 0** ↔ Spark1 **Port 1** |

```
        Spark1
       /      \
 32/33          36/37
     /            \
 Spark3 --34/35-- Spark2
```

> **NVIDIA Sync hat das korrekt aufgesetzt — du musst am Netz nichts ändern.**
> Ich hatte in einer früheren Fassung dieser Anleitung angenommen, Sync würde
> für einen Dreier-Ring ein kollidierendes Schema schreiben. Für diesen Cluster
> stimmt das nicht: die Steckung folgt dem Muster „Port 0 an Port 1 des
> Nachbarn", jedes Interface liegt in einem eigenen Subnetz, und beide Enden
> eines Kabels teilen sich eins. Genau das braucht der Ring.
>
> Bei drei Knoten ist ein Ring zugleich **voll vermascht**: jedes Paar hat ein
> direktes Kabel. Deshalb funktioniert auch Tensor-Parallel über **je zwei**
> beliebige der drei Knoten.

### Eine Sache zum Nachsehen

Die Adresse für Spark3 stand in deiner Ausgabe als `192.168.4` — vermutlich ein
Tippfehler für `192.168.1.4`. Die Anleitung geht davon aus. Prüf es kurz:

```bash
ssh Spark3 ip -br -4 addr show enP7s7
```

Wäre Spark3 tatsächlich in einem anderen Subnetz als Spark1 und Spark2, würde
die Koordination nicht funktionieren — alle drei müssen sich über `enP7s7`
gegenseitig erreichen.

---

## Schritt 1 — Verkabelung und Adressen bestätigen

**Auf allen drei** — vier aktive CX7-Links:

```bash
ibdev2netdev | grep 'Up)' | wc -l
```

**Erfolg: `4`.** Genau darauf stützt sich die Mesh-Erkennung von AINode.

Fehlt `ibdev2netdev`:

```bash
for f in /sys/class/infiniband/*/ports/1/state; do cat "$f"; done
```
Erfolg: vier Zeilen, alle beginnend mit `4:` (ACTIVE).

Dann vom **Spark1** aus die drei Direktlinks prüfen:

```bash
ping -c2 10.100.36.2      # -> Spark2 über Port 0
ping -c2 10.100.32.1      # -> Spark3 über Port 1
ssh Spark2 ping -c2 10.100.34.1   # Spark2 -> Spark3 über Port 0
```

**Erfolg: alle drei antworten.** Damit ist der Ring geschlossen.

---

## Schritt 2 — Netzkonfiguration

**Nichts zu tun.** NVIDIA Sync hat die Interfaces korrekt vergeben — dieser
Schritt existiert nur, damit die Nummerierung nicht springt.

Die Regeln, nach denen es korrekt ist, stehen hier trotzdem: an ihnen erkennst
du eine kaputte Konfiguration, ohne die Adressen nachschlagen zu müssen.

1. **Steckung:** Port 0 eines Sparks an Port 1 des nächsten, im Kreis.
2. Jedes der vier CX7-Interfaces pro Knoten liegt in einem **eigenen** Subnetz.
3. Die beiden Enden **eines Kabels** teilen sich ein Subnetz.
4. MTU 9000 auf allen vier.
5. `enP7s7` bleibt unangetastet.

Regel 2 ist die, die am ehesten kippt, und AINode wie NCCL stolpern beide
darüber. Kontrolle — **auf allen drei**:

```bash
ip -br -4 addr show | grep -E 'enp1s0f|enP2p1s0f'
```

**Erfolg:** vier Zeilen, vier verschiedene `10.100.3x`-Netze, MTU 9000.
Taucht ein Netz zweimal auf, ist das der Fehler.

Musst du die Konfiguration je neu erzeugen, ist `nvidia-sync` der Weg — nicht
eine handgeschriebene Netplan-Datei. Die Werte stehen in der Tabelle oben.

---

## Schritt 3 — 10G-Ethernet prüfen

**Auf allen drei:**

```bash
ip -br -4 addr show enP7s7
```

**Erfolg:** `UP` und die erwartete Adresse — Spark1 `192.168.1.2`,
Spark2 `192.168.1.3`, Spark3 `192.168.1.4`. Alle drei im selben `/24`.

Gegenseitige Erreichbarkeit vom **Spark1**:

```bash
ping -c2 192.168.1.3 && ping -c2 192.168.1.4
```

Dieses Netz trägt Ray, die UDP-Discovery und SSH. Im Ring ist es der einzige
Weg, auf dem jeder Knoten jeden erreicht — über die `10.100.3x`-Links erreicht
jeder Knoten nur seine zwei direkten Nachbarn, nicht sich selbst als Gruppe.

---

## Schritt 4 — Passwortloses SSH vom Head zu den anderen

NVIDIA Sync hat dir die Aliase `Spark1` / `Spark2` / `Spark3` eingerichtet. Die
sind für deine eigenen Befehle bequem — **AINode benutzt sie nicht.** Der Head
verbindet sich mit `<ssh_user>@192.168.1.3`, also über die IP. Ein Alias in
`~/.ssh/config` mit eigenem Key oder Benutzernamen greift dabei unter Umständen
nicht.

Deshalb genau so testen, wie AINode es tut — auf **Spark1**:

```bash
ssh -o BatchMode=yes "$USER@192.168.1.3" hostname
ssh -o BatchMode=yes "$USER@192.168.1.4" hostname
```

**Erfolg:** beide geben ihren Hostnamen aus, ohne Passwortabfrage. Schlägt es
fehl, obwohl `ssh Spark2` funktioniert, liegt es am Alias — dann nachlegen:

```bash
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id -o StrictHostKeyChecking=accept-new "$USER@192.168.1.3"
ssh-copy-id -o StrictHostKeyChecking=accept-new "$USER@192.168.1.4"
```

Der Container mountet `~/.ssh` des Host-Benutzers read-only, der Key muss also
dem Benutzer gehören, unter dem du installierst.

---

## Schritt 5 — Gemeinsames Modellverzeichnis

**Auf allen drei:**

```bash
sudo mkdir -p /mnt/shared-models
```

Der Installer bricht ab, wenn das Verzeichnis fehlt. Ein leeres Verzeichnis
genügt zum Start.

**Empfohlen:** exportiere es per NFS von `Spark1` (`192.168.1.2`) und mounte es
auf den anderen beiden — **über die 10G-Adresse**, nicht über eine
`10.100.3x`-Adresse. Der Ring ist
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
docker save ainode:dev | ssh Spark2 docker load
docker save ainode:dev | ssh Spark3 docker load
```

---

## Schritt 7 — Installation, pro Knoten einzeln

`INSTALLER` steht für:
`https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh`

### Spark1 — der Head (`192.168.1.2`)

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

### Spark2 — Member (`192.168.1.3`)

```bash
curl -fsSL $INSTALLER | bash -s -- --job worker
```

### Spark3 — Member (`192.168.1.4`)

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
cfg["cluster_id"] = "ai-vkv"          # auf allen drei identisch!
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
Coordination IP        192.168.1.2           ← die 10G-Adresse, keine 10.100.3x
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
| eine `10.100.3x`-Adresse als Coordination IP | `coord_interface` wurde nicht gesetzt → Schritt 8 |

Dann vom **spark1** prüfen, dass alle drei einander sehen:

```bash
curl -s localhost:3000/api/cluster/resources \
  | jq '.nodes[] | {hostname, fabric_ip, ib_ips}'
```

**Erfolg:** drei Einträge — `fabric_ip` ist `192.168.1.2` / `.3` / `.4`, und
`ib_ips` enthält je **vier** `10.100.3x`-Adressen. Die `fabric_ip` darf in `ib_ips`
nicht vorkommen — das sind zwei getrennte Wege: Koordination über 10G,
Massentransfer über die Direktlinks.

---

## Schritt 10 — Modell über alle drei Knoten starten

Im Browser auf `http://192.168.1.2:3000`.

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
curl -s -X POST http://192.168.1.2:3000/api/sharding/launch \
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
curl -s http://192.168.1.2:8000/v1/chat/completions \
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
curl -s http://192.168.1.2:3000/api/cluster/resources \
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

---
