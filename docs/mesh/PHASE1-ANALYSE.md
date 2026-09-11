# Phase 1 — Analyse: AINode auf 3× DGX Spark im Ring/Mesh

Stand: 2026-09-11 · Baseline gemessen: `pytest tests/` → **723 passed, 1 xfailed**
ainode @ `0cfde90` (v0.5.6) · spark-vllm-docker @ `346dc04`, gepinnt im Base-Image auf `c026c92`

---

## 1. Wie ainode heute das Cluster-Interface ermittelt und nutzt

**Es ermittelt es gar nicht — es steht in der Config.** `NodeConfig.cluster_interface`
(`ainode/core/config.py:123`, Default `"eno1"`) ist ein **einzelner String**.
`scripts/install.sh:217` schreibt hart `"cluster_interface": "enP2p1s0f1np1"`.
Ein grep über `onboarding/`, `cli/`, `api/` findet **keine** Autodetection.

Dieser eine String bedient gleichzeitig vier verschiedene Rollen:

| Rolle | Ort |
|---|---|
| NCCL/Gloo/UCX-Socket-Interface | `eugr.py:387-392`, `nvidia.py:530-556` |
| `fabric_ip` im Discovery-Announcement | `api/server.py:237` → `hca_discovery.detect_fabric_ip()` |
| SSH-Ziel + Ray-Adresse für Peers | `sharding_routes.py:174-186`, `nvidia.py:1224-1300` |
| Subnetz-Filter für die HCA-Auswahl | `eugr.py:410 _cluster_subnet()` → `_detect_ib_hca()` |

**Env-Weitergabe an Worker/Container** — zwei sehr verschiedene Wege:

* **eugr-Backend** (Default, `engine_backend: "eugr"`): `_write_eugr_env()` (`eugr.py:593`)
  schreibt `/opt/spark-vllm-docker/.env` mit `CLUSTER_NODES`, `ETH_IF`, `IB_IF`,
  `MASTER_PORT`, `SSH_USER` und `CONTAINER_*`-Variablen. launch-cluster.sh macht aus
  jedem `CONTAINER_X` ein `-e X` (`launch-cluster.sh:422-437`). Zusätzlich der per-Node-Shim
  `scripts/nccl-env-init.sh`, per NFS (`/mnt/shared-models`) verteilt und via
  `--entrypoint` in jeden `vllm_node` gehängt, weil eine cluster-weite `.env` bei
  heterogener HCA-Benennung falsch wäre.
* **nvidia-Backend** (opt-in): `_build_nccl_env()` (`nvidia.py:513`) baut das Dict,
  `_build_ray_docker_cmd()` macht `-e`-Flags daraus, `_ssh_launch_worker()` (`nvidia.py:1224`)
  schickt das per SSH **an die fabric_ip** des Peers.

Gesetzt werden heute: `VLLM_HOST_IP`, `MASTER_ADDR/PORT`, `NCCL_SOCKET_IFNAME`,
`GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `UCX_NET_DEVICES`, `OMPI_MCA_btl_tcp_if_include`,
`NCCL_IB_GID_INDEX=3`, `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA`, und (nur nvidia)
`NCCL_IB_SUBNET_AWARE_ROUTING=1`.
**Nicht vorhanden im gesamten Repo:** `NCCL_NET_PLUGIN`, `NCCL_IB_MERGE_NICS`.

## 2. UDP-Discovery

`discovery/broadcast.py`: `BroadcastSender` schickt alle 5 s ein JSON-`NodeAnnouncement`
an `<broadcast>`, `BroadcastListener` bindet und pflegt eine Registry mit
Health-Reaper (online <15 s, stale <30 s). Peers landen in
`ClusterState._nodes` (`discovery/cluster.py`) als `ClusterNode`.

**Port: Code-Default ist 5678** (`config.py:31`, `broadcast.py:113/181`), nicht 5679.
5679 steht nur in `scripts/install.sh:221` und in CLAUDE.md/README.

**Trennung NCCL-Knoten vs. Transfer-Knoten: nein.** Es gibt zwei Adress-Begriffe —
`DiscoveredNode.peer_ip` (UDP-Quell-IP, also Mgmt-LAN) und `NodeAnnouncement.fabric_ip`
(`broadcast.py:62`) — aber sie werden **genau andersherum** benutzt als der Mesh es braucht:
der Kommentar dort ("BUG D") verwirft die Mgmt-IP absichtlich und benutzt die
RoCE-Fabric-IP für SSH, Ray-Bootstrap **und** Modelltransfer (`nvidia.py:1302
_ensure_peer_has_model`). Eine zweite Liste für Datei-/Image-Verteilung existiert nicht.

## 3. Wo die Parallelitätsstrategie festgelegt wird

**TP ist im Launch-Pfad fest verdrahtet.**

* `engine/sharding.py` kennt `ShardingStrategy.{TENSOR,PIPELINE}_PARALLEL` und
  `ShardingConfig.pipeline_parallel_size` — aber nur als **Planer/Preview**
  (`/api/sharding/plan`). Der Launch benutzt ihn nicht.
* `sharding_routes.py:110` liest `strategy` und kommentiert es selbst weg:
  *"We accept but don't gate on strategy here"*. Gesetzt wird stur
  `tensor_parallel_size = 1 + len(chosen_peers)` (Zeilen 240, 260).
* `eugr.py:547 _tp_size()` = `1 + len(peer_ips)`; das generierte Launch-Script
  hat `--pipeline-parallel-size 1` hart drin (`eugr.py:666`).
* `nvidia.py:1420 _tp_size()` dito; `_build_vllm_serve_args` kennt nur `--tensor-parallel-size`.
* `InstanceRecord` (`discovery/instance.py:27`) hat nur `tensor_parallel_size`.
* UI: `templates/index.html:148-151` hat Pills *Tensor* / *Pipeline*; `app.js:1124-1150`
  schickt `strategy: 'tensor'|'pipeline'`, während die API-Doku `tensor_parallel`/
  `pipeline_parallel` erwartet — und das Feld ohnehin ignoriert wird.
  `recommendLaunch` (`app.js:1019`) setzt die Pill zwangsweise auf *tensor* und
  probiert nur TP ∈ {1,2,4,8}.
  → **Der Pipeline-Pill ist heute reines Placebo.**
* Data-Parallel: existiert nirgends im Inferenzpfad (nur DDP im Training,
  `training/engine.py:830`).

## 4. Was spark-vllm-docker konkret macht

**`autodiscover.sh detect_interfaces()`** — `ibdev2netdev | awk '/Up\)/'` liefert
(RoCE-Dev, Netdev)-Paare. Sanity-Checks: jedes `enp*`-ohne-großes-P muss eine IP haben;
keine zwei Interfaces im selben Subnetz. Dann **Heuristik über die Anzahl**:
* **2 aktiv → non-mesh:** `IB_IF` = die zwei erkannten RoCE-Twins, `ETH_IF` = das
  aktive `enp*`-Interface ohne großes P.
* **4 aktiv → mesh:** `IB_IF="rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"`
  (hart kodiert), `ETH_IF=enP7s7`, sonst `wlP9s9` mit Warnung, sonst harter Fehler.
  Zusätzlich werden exportiert: `CONTAINER_NCCL_NET_PLUGIN=none`,
  `CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1`, `CONTAINER_NCCL_IB_MERGE_NICS=0`.
* alles andere → Fehler.

**`detect_copy_hosts()`** — non-mesh: `COPY = CLUSTER`. Mesh: scannt die Subnetze von
`enp1s0f0np0` und `enp1s0f1np1` (nc auf :22, dann `ssh … nvidia-smi | grep "NVIDIA GB10"`)
und dedupliziert über die `ETH_IF`-IP des Remote-Hosts, damit ein Host mit zwei
IB-Adressen nicht doppelt gezählt wird. `save_config()` schreibt **`CLUSTER_NODES`
(Koordination, 10G) und `COPY_HOSTS` (Direktlinks) als zwei getrennte Zeilen**;
`hf-download.sh` und `build-and-copy.sh` benutzen `COPY_HOSTS`.

**`launch-cluster.sh get_env_flags()` (Zeile 1316)** — pro Knoten:
`VLLM_HOST_IP`/`RAY_NODE_IP_ADDRESS`/`RAY_OVERRIDE_NODE_IP_ADDRESS` = die
**CLUSTER_NODES-IP (10G)**; `MN_IF_NAME`/`UCX_NET_DEVICES`/`NCCL_SOCKET_IFNAME`/
`OMPI_MCA_btl_tcp_if_include`/`GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME` = **ETH_IF (10G)**;
`NCCL_IB_HCA=IB_IF` = **alle vier RoCE-Devices**, `NCCL_IB_DISABLE=0`.
Genau die Trennung, die wir wollen: **Koordination über Ethernet, Datenpfad über RoCE,
Routing macht NCCL selbst.**

**`docs/NETWORKING.md`** bestätigt Verkabelung (Port 0 ↔ Port 1 des Nachbarn, je eigenes
/24 pro Twin) und sagt explizit: *"tensor-parallel … works best with a power of 2 …
A 3-node mesh is mainly useful for pipeline parallelism or data parallelism."*
Der 10G-Pfad ist dort nicht optional: *"For 3-node mesh we have to use 10G interface
for OOB communication!"*

**Parallelität:** `parse_parallelism_from_text` (`launch-cluster.sh:1186`) kennt
`-tp/-pp/-dp` und validiert `TP*PP*DP ≤ Knotenzahl`. Default ist **`NO_RAY_MODE=true`**
(Zeile 43) — also vLLMs natives `--nnodes/--node-rank/--headless`; Ray nur mit `--ray`.

---

## 5. Deine Annahmen — was sich bestätigt, was nicht

### Bestätigt
* **"Geht von genau einem aktiven NIC pro Cluster-Subnetz aus"** — ja, und schlimmer als
  gedacht: derselbe String ist Socket-Interface, SSH-Ziel, Announce-Adresse und HCA-Filter.
* **"Unterstützt nur Tensor-Parallel"** — im Launch-Pfad ja. PP existiert nur im
  Planer und als toter UI-Pill.
* **Lizenzen** — spark-vllm-docker MIT, ainode Apache-2.0, beide LICENSE-Dateien geprüft.

### Nicht bestätigt / abweichend
1. **Discovery-Port ist 5679** — nein, Code-Default ist **5678**. 5679 kommt nur aus
   install.sh und der Doku. Beides funktioniert, aber Default-Config ≠ Doku.
2. **"Gibt es schon eine Trennung NCCL-Knoten vs. Transfer-Knoten?"** — nein. Die
   vorhandene Trennung (Mgmt-IP vs. Fabric-IP) läuft **umgekehrt**: die Mgmt-IP wird
   absichtlich verworfen, RoCE trägt SSH + Ray + Modelltransfer. Für den Mesh ist das
   die falsche Richtung.
3. **Das eugr-Backend ist weiter vom Mesh entfernt als es aussieht.**
   `_detect_ib_hca()` filtert HCAs auf das Subnetz von `cluster_interface`. Setzt man
   `cluster_interface=enP7s7`, liegt **keine** RoCE-HCA in diesem Subnetz → `IB_IF=`
   leer → launch-cluster.sh fällt in seine eigene Autodiscovery → braucht
   `ibdev2netdev`, das im ainode-Image **nicht installiert ist**
   (`scripts/Dockerfile.ainode:46-66` bringt nur iproute2 & Co.) → harter Abbruch.
   Das ist kein Restrisiko, sondern ein vorhersagbarer Fehlschlag.
4. **Das nvidia-Backend ist umgekehrt näher dran als gedacht.**
   `hca_discovery.build_nccl_ib_hca_whitelist()` filtert **nur** nach GID-Index 3, nicht
   nach Subnetz — liefert im Mesh also von selbst alle vier HCAs. Und
   `NCCL_IB_SUBNET_AWARE_ROUTING=1` setzt es bereits. Es fehlen nur `NCCL_NET_PLUGIN=none`,
   `NCCL_IB_MERGE_NICS=0` und die Entkopplung Socket-IF ↔ SSH/Ray-Adresse.
5. **Die 10G-Rolle ist im Code schon halb da:** `hca_discovery.probe_path_mtu()` hat
   `mgmt_iface="enP7s7"` als Default-Parameter — aber nirgends konfigurierbar und
   nirgends sonst benutzt.
6. **Mesh-Support in eugr ist im gepinnten Base-Image bereits enthalten**
   (`EUGR_COMMIT=c026c92`; dessen `autodiscover.sh` hat `MESH_MODE`). Wir umgehen ihn
   aktiv: weil ainode `ETH_IF` **und** `IB_IF` in die `.env` schreibt, kehrt
   `detect_interfaces()` sofort zurück (Zeile 57), `MESH_MODE` bleibt `"false"`, die drei
   Mesh-NCCL-Exports laufen nie — und `detect_copy_hosts()` wird außerhalb von `--setup`
   ohnehin nicht aufgerufen.
7. **launch-cluster.sh benutzt per Default kein Ray.** ainode erzwingt Ray, indem es
   `--distributed-executor-backend ray` ins generierte Script schreibt. Für PP/DP ist der
   no-ray-Pfad (`--nnodes/--node-rank/--headless`) der von der Referenz vorgesehene.

---

## 6. Umsetzungsvorschlag mit Aufwand

Randbedingung 2-/4-Knoten-TP: **jede** neue Verzweigung hängt an der Mesh-Erkennung.
Bei 2 aktiven CX7-Interfaces liefert sie `False` und der heutige Code-Pfad läuft
unverändert. Regression-Gate ist die gemessene Baseline (723 Tests).

### A) Netzwerk-Abstraktion — ~1–1,5 Tage, 3 Commits
* **A1** Neues `ainode/cluster/topology.py`: `detect_cx7_links()` (über sysfs, ohne
  `ibdev2netdev` — Bausteine existieren in `hca_discovery.list_local_hcas()` und
  `eugr._detect_ib_hca()`), `is_mesh()` (4 aktiv = Mesh, 2 = non-Mesh, sonst unbekannt),
  `coordination_interface()` (Mesh → enP7s7 / wlP9s9, sonst `cluster_interface`).
  Reines Lesen, mit fake-sysfs voll unit-testbar. Herkunftshinweis auf
  `autodiscover.sh detect_interfaces()` (MIT) im Docstring. **~0,5 T**
* **A2** Config: `cluster_interface` bleibt (Kompatibilität), neu `coord_interface`
  und `rdma_hcas`, beide optional/leer = autodetect. Dabei die 5678/5679-Inkonsistenz
  aufräumen. **~0,2 T**
* **A3** Beide Backends auf `topology` umstellen. nvidia: im Mesh-Fall
  `NCCL_NET_PLUGIN=none`, `NCCL_IB_MERGE_NICS=0`, `NCCL_IB_HCA`= alle vier, Socket-IFs
  auf `coord_interface`. eugr: `ETH_IF=coord_interface`, `IB_IF`= alle HCAs (im Mesh
  ohne Subnetzfilter) plus die drei `CONTAINER_*`-Mesh-Variablen — und sicherstellen,
  dass `ETH_IF`/`IB_IF` **nie leer** geschrieben werden. **~0,5–0,8 T**

### B) Discovery: zweite Adressliste — ~0,5–1 Tag, 2 Commits
* **B1** `NodeAnnouncement` um `ib_ips: List[str]` und `coord_ip: str` erweitern (neben
  dem bestehenden `fabric_ip`), durch `ClusterNode` durchreichen. `from_json` wirft
  unbekannte Keys weg → ältere Peers bleiben kompatibel. **~0,3 T**
* **B2** Helfer `transfer_address_for(node)`: beste **lokal erreichbare** IB-Adresse,
  sonst `fabric_ip`, sonst `peer_ip`. Benutzt von `_ensure_peer_has_model` und dem
  Image-/NFS-Pfad. Koordination (SSH für Ray-Start, Discovery) geht auf `coord_ip`.
  **Wichtig:** im 3er-Mesh erreicht Head A den Knoten C **nicht** direkt über IB — der
  Helfer muss das per lokalem Subnetzvergleich erkennen und dann auf `coord_ip`
  zurückfallen. eugr löst das per SSH-Scan; lokal geht es ohne Scan. **~0,4 T**

### C) Pipeline-/Data-Parallel — ~3–4 Tage, in 6 Schritten
* **C1** Datenmodell: `pipeline_parallel_size` / `data_parallel_size` in
  `InstanceRecord` + `NodeAnnouncement`, `world_size = tp*pp*dp`. Defaults 1 → Wire-Format
  bleibt abwärtskompatibel. **~0,5 T**
* **C2** `ParallelPlan` + Validierung: `tp*pp*dp == Knotenzahl`, TP nur 1/2/4/8, bei
  3 Knoten TP=3 mit klarer Meldung ablehnen. Portierung von
  `parse_parallelism_from_text` (MIT, mit Hinweis). **~0,5 T**
* **C3** nvidia-Backend: `_build_vllm_serve_args` um pp/dp, `_tp_size()` → `_parallel_plan()`. **~0,5 T**
* **C4** eugr-Backend: Launch-Script mit `-tp/-pp/-dp`; Entscheidung Ray vs. no-ray.
  **~0,7 T + Klärungsrisiko** (siehe offene Punkte).
* **C5** `sharding_routes.handle_sharding_launch`: `strategy` tatsächlich auswerten
  (heute No-op), Plan bauen, ungültige Kombination ablehnen statt still TP zu nehmen. **~0,5 T**
* **C6** UI: Pill-Werte auf `tensor|pipeline|data` (Backend akzeptiert beide
  Schreibweisen), `recommendLaunch` um PP/DP, Badge `PP=3` statt `TP=3`,
  Fit-Hinweis pro Strategie. **~0,8 T**

### Offene Punkte
* Ob die gepinnte vLLM-Version `--data-parallel-size` über mehrere Knoten sinnvoll
  unterstützt, ist **unverifiziert** — hier gibt es kein GPU/Image zum Prüfen. Falls
  nein: C mit PP ausliefern, DP in Zustandsmodell und UI vorbereiten und im Launch
  sauber ablehnen.
* `ibdev2netdev` fehlt im ainode-Image. Unsere sysfs-Erkennung umgeht das, aber jeder
  Pfad, der in launch-cluster.shs eigene Autodiscovery fällt, bleibt kaputt. Vorschlag:
  in A3 hart sicherstellen, dass es nie dazu kommt — **und** `infiniband-diags` ins
  Image legen (2 Zeilen), damit der Fallback ehrlich ist.

### Vorab-Check auf echter Hardware (bestätigt/widerlegt meine Lesart, ändert nichts)
Auf **jedem** der drei Knoten:
```bash
ibdev2netdev | grep 'Up)' | wc -l          # Erfolg: 4  (=> Mesh-Heuristik greift)
ip -br -4 addr show enP7s7                 # Erfolg: UP + eine IP im gemeinsamen LAN
ls /sys/class/infiniband/                  # Erfolg: 4 Devices (rocep*/roceP* oder mlx5_*)
python3 -c "import json;print(json.load(open('/root/.ainode/config.json'))['cluster_interface'])"
                                           # erwarte: enP2p1s0f1np1  (= heute falsch für Mesh)
```
Wenn Zeile 1 nicht 4 ergibt, stimmt meine Mesh-Annahme nicht und A1 braucht eine
andere Heuristik.
