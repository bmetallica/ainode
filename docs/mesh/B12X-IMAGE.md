# Das B12X-Engine-Image (experimentell)

Manche Modelle laufen nur auf einem eigenen Serving-Stack. Im Katalog sind das
derzeit **Qwen3.8-Flash-Next FP4-Local** (`nvidia/Qwen3.8-Flash-Next-FP4-Local`,
siehe unten) und **GLM 5.3 Flash** (`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`):
seine halbe Kommandozeile — `--attention-backend B12X`, `--moe-backend b12x`,
`--linear-backend b12x`, `--load-format b12x` — existiert in keinem anderen
Image. Gegen das Standard-Image gestartet, endet es in argparse mit Exit-Code 2.

> **Experimentell.** Das ist upstreams eigene Einordnung, und wir haben darauf
> noch kein Modell serviert. Der Katalogeintrag ist `verified=False`.

---

## Image holen

```bash
ssh Spark1
cd ~/ainode
scripts/build-b12x-image.sh --nodes Spark2,Spark3
```

Das **kompiliert nicht**, es zieht ein fertiges Image von upstream — Minuten,
keine halbe Stunde. Ohne `--nodes` bleibt es lokal; mit `--nodes` geht es per
`docker save | ssh docker load` auf die Peers.

Hast du den [Image-Cache](REGISTRY-CACHE.md) eingerichtet, ist der schnellere
Weg, es auf jedem Knoten einzeln zu ziehen — dann holt der Cache die Layer
einmal aus dem Netz:

```bash
for h in Spark1 Spark2 Spark3; do ssh $h 'cd ~/ainode && scripts/build-b12x-image.sh' & done; wait
```

**Erfolg:** `docker images | grep b12x` zeigt auf allen drei Knoten dieselbe
Image-ID. Der Launcher vergleicht IDs über die Knoten hinweg, nicht Tags — zwei
unterschiedliche Builds mit gleichem Namen lehnt er ab.

---

## Qwen3.8-Flash-Next FP4-Local

NVIDIAs per Distillation quantisierte Fassung des Flash-Next: 102 GB statt
133 GB, laut Modellkarte GPQA Diamond 91,5 % und ToolEvalBench 88,7 % (BF16:
92,2 / 89,4). Ihre NVFP4-Skalen liegen in einem neuen, komprimierten Format,
das nur das B12X-Image liest — und zwar erst das von November. Das Image dafür
also einmal neu holen (der Standard-Commit im Skript ist 68afa9f, 2026-11-21):

```bash
ssh Spark1
cd ~/ainode
scripts/build-b12x-image.sh --nodes Spark2,Spark3
```

Der Katalogeintrag ist eugrs `recipes/qwen3.8-flash-next-nvfp4-cluster.yaml`:
TP=2 auf zwei Knoten, fp8-KV-Cache (NVIDIAs eigenes Rezept für diesen
Checkpoint, Bild-Eingaben eingeschlossen), MTP mit vier Entwurfs-Tokens,
B12X-Lader und -Kernel, FlashInfer-Autotuning aus, `--max-num-seqs 16`.

**Clients unverändert lassen:** Im Wizard unter *More settings → API name(s)*
den alten Namen eintragen, `nvidia/Qwen3.8-Flash-Next-NVFP4`. Das Modell
antwortet dann unter beiden Namen, und die opencode-Konfigurationen der Clients
bleiben, wie sie sind.

**Mehr Cache statt Tempo:** die ~51 GB n-Gramm-Embedding-Tabellen (PLE) liegen
im Rezept im GPU-Speicher. Auf die NVMe ausgelagert, steht der Platz dem
KV-Cache zur Verfügung — mehr Sitzungen bei langem Kontext, auf Kosten eines
Plattenzugriffs pro Token. In den Extra-Args des Modells:

```
--additional-config {"ple_table_memory":"disk"}
```

und unter *Engine environment* `B12X_DISK_BACKEND=io_uring`. Beides ist die
Konfiguration der Modellkarte für einen einzelnen Spark.

## Modell starten

Nichts weiter einzustellen: der Katalogeintrag nennt das Image selbst
(`engine_image_eugr`), und AINode übergibt es dem Launcher.

1. **MODELS** → GLM 5.3 Flash herunterladen
2. Im Launch-Panel das Modell wählen
3. **Genau zwei Knoten** aktivieren — das Rezept ist `cluster_only` mit TP=2,
   auf einem Knoten läuft es nicht, auf dreien gäbe es kein gültiges TP
4. **LAUNCH**

Die 16 Umgebungsvariablen (`CUTE_DSL_ARCH`, `B12X_POLICY_MODE`, die
`INSTANTTENSOR_*`-Gruppe) kommen aus dem Katalog mit. Eigene ergänzt du unter
*Advanced → Engine environment*, eine `NAME=wert`-Zeile pro Variable.

---

## Bekannte Hürden

**`RuntimeError: the initial b12x loader requires GPU host page tables`**

Der schnelle B12X-Lader verlangt eine Plattformfähigkeit, die der
Engine-Container nicht in jeder Umgebung hat. Der restliche B12X-Stack hängt
nicht daran — Attention-, MoE- und Linear-Backend funktionieren weiter. Im
Launch-Panel unter *Advanced → Extra vLLM args*:

```
--load-format auto
```

Dein Wert gewinnt über den des Rezepts, alles andere bleibt.

**`Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/models/…'`**

AINode serviert ein heruntergeladenes Modell aus seinem Verzeichnis statt über
die Repo-ID — das spart bei jedem Start einen Neudownload. Der B12X-Lader
erwartet an dieser Stelle eine Repo-ID und kommt mit dem Pfad nicht zurecht.
Tritt zusammen mit dem obigen Fehler auf; verschwindet mit `--load-format auto`
ebenfalls.

---

## Für ein anderes B12X-Modell

Ohne Katalogeintrag geht es auch, es ist nur Handarbeit — im Launch-Panel unter
*Advanced*:

| Feld | Wert |
|---|---|
| Engine image | `vllm-node-b12x` |
| Extra vLLM args | die Flags aus dem Rezept |
| Engine environment | die `env:`-Einträge, eine Zeile je Variable |

Ein selbst eingetragenes Image wird immer befolgt — im Gegensatz zu einem aus
dem Katalog, den der eugr-Pfad nur beachtet, wenn er für ihn gedacht ist.

---

## Warum zwei eugr-Commits

Das B12X-Image wird aus einem **neueren** eugr-Stand geholt als das
Basis-Image: `--exp-b12x` gibt es erst seit einem späteren Commit, der
gepinnte Stand des Basis-Images kennt den Schalter nicht und antwortet mit
seinem Usage-Text. Deshalb hat das Skript einen eigenen Pin
(`EUGR_B12X_COMMIT`, per Umgebungsvariable überschreibbar) und einen eigenen
Checkout unter `scripts/_eugr-b12x` — der des Basis-Images wird gepatcht und
verträgt keinen Commit-Wechsel.

Das ist unkritisch: Der **Launcher** kommt aus unserem eigenen Image (dort auf
`EUGR_COMMIT` gepinnt) und muss sich mit dem Engine-Image nur über die `.env`
einig sein, nicht darüber, welche Kernel einkompiliert wurden.

---

## Warum das ein eigenes Skript ist

`scripts/build-base-image.sh` baut aus Quellen und hängt an upstreams
vorgebauten vLLM-Wheels. `--exp-b12x` ist damit unvereinbar — upstreams eigene
Begründung: „B12X vLLM wheels are not published". Die Patches jenes Skripts
(NCCL-Pin, uv-Override) zielen auf den Quellbau und gehen hier ins Leere. Beide
Skripte benutzen aber denselben eugr-Commit: zwei Engine-Images aus
verschiedenen Launcher-Generationen wären sich über die `.env` nicht einig.
