# Das B12X-Engine-Image (experimentell)

Manche Modelle laufen nur auf einem eigenen Serving-Stack. Im Katalog ist das
derzeit **GLM 5.3 Flash** (`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`):
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

## Warum das ein eigenes Skript ist

`scripts/build-base-image.sh` baut aus Quellen und hängt an upstreams
vorgebauten vLLM-Wheels. `--exp-b12x` ist damit unvereinbar — upstreams eigene
Begründung: „B12X vLLM wheels are not published". Die Patches jenes Skripts
(NCCL-Pin, uv-Override) zielen auf den Quellbau und gehen hier ins Leere. Beide
Skripte benutzen aber denselben eugr-Commit: zwei Engine-Images aus
verschiedenen Launcher-Generationen wären sich über die `.env` nicht einig.
