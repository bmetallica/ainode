# images.md — Bildgenerierung in AINode

Ziel: `Qwen/Qwen-Image-2.1` (oder ein vergleichbares Diffusionsmodell) läuft
auf Node 3 neben qwen3.8 und dem Embedding-Modell, gestartet und überwacht wie
jedes andere Modell — dieselbe Karte, derselbe Speicherwächter, dasselbe
Profil, dieselbe Telemetrie. Kein Nebencontainer, den AINode nicht kennt.

Aufgenommen am 2026-09-21 auf `main` @ 96041b0 (2371 Tests).

---

## 1. Warum es heute nicht geht

Nachgeprüft, nicht vermutet:

```
Qwen/Qwen-Image-2.1     pipeline_tag = text-to-image   library_name = diffusers
                        47,4 GB                        diffusers:QwenImage21Pipeline
Verzeichnisse:          model_index.json, transformer/, text_encoder/, vae/,
                        scheduler/, processor/
```

Zwei getrennte Gründe, und nur einer davon ist ein Filter:

1. **Die Suche blendet es bewusst aus.**
   `ModelManager.SERVABLE_PIPELINE_TAGS` in `ainode/models/registry.py` ist
   `("text-generation", "image-text-to-text")`.
   Was die Suche zeigt, soll ladbar sein; `text-to-image` gehört heute nicht
   dazu. Den Filter allein zu öffnen wäre das Schlechteste von allem: man
   fände das Modell, lüde 47 GB, und der Start scheiterte mit einer Meldung,
   die niemandem sagt warum.

2. **vLLM kann es nicht laden**, und daran ändert kein Flag etwas. Das ist
   kein Checkpoint mit einer Architektur, die vLLM kennt, sondern ein
   Pipelineverzeichnis: ein MMDiT-Transformer, ein Text-Encoder, ein VAE und
   ein Scheduler, zusammengehalten von `model_index.json`. vLLM bedient
   Text- und Vision-*Verstehen*; Bild-*Erzeugung* ist eine andere Rechnung
   mit anderem Speicherverlauf und ohne KV-Cache.

Vor allem anderen gilt es das zu bestätigen — ein Befehl, der die Frage
endgültig klärt:

```bash
docker exec vllm_node python -c "
from vllm.model_executor.models.registry import ModelRegistry
print([a for a in ModelRegistry.get_supported_archs() if 'Image' in a or 'Diff' in a])"
```

Eine leere Liste heißt: dieser Plan gilt.

---

## 2. Die Architektur in einem Satz

AINode bekommt eine **zweite Engine-Art**. Nicht einen Sonderweg, sondern
einen weiteren Wert für `engine_backend`, der dieselbe `EngineBackend`-
Schnittstelle erfüllt wie `eugr` und `nvidia` — `start`, `stop`, `wait_ready`,
`is_running`, `health_check`, `logs`, `log_path`, `load_phase`, `kill`.

Alles, was AINode heute um Instanzen herum kann, hängt an dieser Schnittstelle
und am `InstanceManager`. Wer sie erfüllt, bekommt umsonst: die Instanzkarte
mit Statuswort und Details, den Speicherwächter, der ihn notfalls abschießt,
das Zulassungstor, die Ladephasen mit Zeitmessung, den Fehlerassistenten, die
Platzierung, die Profile, die Log-Weiterleitung und die Telemetrie.

Das ist der eigentliche Grund, es *in* AINode zu bauen statt daneben: nicht
der Komfort eines Knopfes, sondern dass ein 47-GB-Prozess auf einem Knoten mit
Unified Memory unter derselben Aufsicht steht wie alles andere. Ein
ComfyUI-Container neben AINode teilt sich denselben physischen Speicher und
ist für den Wächter unsichtbar — er würde die falsche Instanz abschießen.

---

## 3. Die Bausteine

### S1 · Modalität im Katalog, in der Suche und im Record

Heute ist „Modell" gleichbedeutend mit „LLM". Das muss sichtbar aufgebrochen
werden, bevor irgendetwas geladen wird.

* `ModelInfo` bekommt `modality: str = "text"` (`text` | `image`). Der
  HF-Sweep setzt es aus dem `pipeline_tag`.
* `ModelManager.SERVABLE_PIPELINE_TAGS` wird um `text-to-image` erweitert — aber die
  Suchergebnisse tragen die Modalität, und die UI zeigt sie als Abzeichen. Was
  gefunden wird, ist damit weiterhin ladbar, nur eben von einer anderen
  Engine.
* `InstanceRecord` bekommt `kind` (heute implizit `llm`), und
  `ainode/profiles/store.py` kennt bereits `KIND_LLM` und `KIND_EMBEDDING` —
  dort kommt `KIND_IMAGE` dazu. Ein Profil, das Node 3 wiederherstellt, muss
  wissen, welcher Art jede Zeile ist.

**Abnahme:** die HF-Suche findet `Qwen/Qwen-Image-2.1` und kennzeichnet es als
Bildmodell; ein Klick auf *Load* bietet die Bild-Engine an, nicht vLLM.

### S2 · Das Engine-Image

Ein neues Image `ainode-diffusers`, **FROM dem bestehenden eugr-Basisimage**.

Das ist die wichtigste Einzelentscheidung des Plans. Der schwierige Teil auf
dieser Hardware sind nicht die Bibliotheken, sondern die Räder: torch mit
CUDA für aarch64 und Blackwell. Dieser Stack ist im Basisimage vorhanden und
auf dem Cluster bewährt. Darauf kommen nur noch

```
pip install diffusers accelerate safetensors
```

Ein fertiges Image von außen zu holen hieße, denselben torch-Stack noch einmal
zu riskieren — und genau daran ist auf diesem Cluster schon einmal ein Tag
draufgegangen (`Failed to find C compiler`, DeepGEMM, die
InstantTensor-Puffergrenze).

**Abnahme:** `docker run --rm ainode-diffusers python -c "import torch, diffusers;
print(torch.cuda.is_available(), diffusers.__version__)"` sagt `True`.

### S3 · Der Server

Ein kleines Skript im Image, gestartet wie eugrs Launch-Skript: geschrieben,
in den Container kopiert, ausgeführt. Es lädt die Pipeline einmal und bedient:

```
POST /v1/images/generations     OpenAI-kompatibel: prompt, size, n,
                                response_format=b64_json, dazu steps,
                                guidance_scale, seed, negative_prompt
GET  /health                    lädt noch / bereit / Fehler
GET  /metrics                   Prometheus: Bilder, Sekunden je Bild,
                                Schritte je Sekunde, laufende Anfragen
```

**Selbst geschrieben, nicht fremdbezogen.** Zwei Gründe: wir bestimmen den
Vertrag, an dem später Proxy, UI und Telemetrie hängen; und ich kann für
keinen Fremdserver belegen, dass er auf GB10 mit aarch64 läuft — das müsste
ohnehin erst bewiesen werden, und dann kann man auch gleich das Wenige selbst
schreiben.

Der Server schreibt seine Phasen in derselben Sprache ins Log, die
`load_phase.py` schon erkennt (`loading weights`, `ready`), damit die
Ladeanzeige ohne Sonderfall funktioniert.

**Abnahme:** `curl -X POST localhost:8003/v1/images/generations -d '{"prompt":"a
red cube","size":"1024x1024"}'` liefert ein Bild in base64.

### S4 · Das Backend

`ainode/engine/backends/diffusers.py`, registriert in `get_backend` unter
`engine_backend="diffusers"`. Startet den Container mit `--network host` auf
dem zugeteilten Port, teet das Log nach `~/.ainode/logs/diffusers-<port>.log`,
erfüllt die ABC.

`kill()` ist Pflicht, nicht Kür — der Speicherwächter ruft es auf.

**Abnahme:** Laden über `/api/models/load` mit `engine_backend: "diffusers"`,
die Instanzkarte zeigt `LOADING · 40%` und dann `READY`, *Details* zeigt die
Ladezeit-Aufschlüsselung.

### S5 · Routing und der Bild-Endpunkt

* `POST /v1/images/generations` auf dem Head, geroutet wie
  `/v1/chat/completions`: `_routing_table` findet die Instanz am Modellnamen
  und leitet auf den Knoten weiter, der sie bedient. Der Code dafür existiert
  und ist modalitätsblind — er braucht nur die zweite Route.
* `/v1/models` listet Bildmodelle mit, damit ein Client sie findet. OpenAI
  tut dasselbe.

**Abnahme:** derselbe curl gegen den **Head** liefert das Bild, obwohl die
Instanz auf Node 3 läuft.

### S6 · Speicher: Wächter, Zulassung, Planer

Der heikelste Teil, und der Grund für den ganzen Plan.

* **Wächter:** funktioniert ohne Änderung, sobald die Instanz im
  `InstanceManager` steht — er schießt die zuletzt gestartete ab. Das ist
  hier sogar *besonders* richtig: der Speicherbedarf eines Diffusionslaufs
  ist keine Ebene, sondern eine Spitze. Die Gewichte liegen konstant, aber
  Aktivierungen und der VAE-Dekodierschritt wachsen mit Auflösung und
  Batchgröße, und die Spitze kommt am Ende eines Laufs. Ein Bild in 2048²,
  das ein sonst stabiles Node 3 umbringt, ist ein realistischer Fehlerfall.
* **Zulassungstor:** muss den Bedarf *anders* schätzen. Heute fragt es den
  Planer, der `config.json` liest — ein Diffusionsrepo hat keine, sondern
  `model_index.json`. Deshalb bekommt der Planer eine zweite, schlichte
  Rechnung: Gewichte von der Platte, plus ein fester Aufschlag für die
  Engine, plus ein von Auflösung und Batch abhängiger Puffer. Kein KV-Cache,
  kein `max-model-len`, keine Parallelisierungsachse.
* **Was der Planer *nicht* tun darf:** so tun, als könne er es. Für ein
  Bildmodell gibt er einen Plan ohne KV-Zahlen aus und sagt das ausdrücklich,
  statt Felder mit Nullen zu füllen.
* **Eine Auflösungsgrenze** je Instanz (`max_image_size`), die der Server
  durchsetzt. Das ist das Gegenstück zu `--max-model-len`: die eine
  Stellschraube, die verhindert, dass eine einzelne Anfrage den Knoten
  sprengt.

> **Kein CPU-Offload.** diffusers bietet `enable_model_cpu_offload()` an, und
> auf einer Maschine mit getrenntem VRAM ist das der übliche Rat. Auf GB10
> ist es sinnlos: „CPU" und „GPU" sind derselbe physische Speicher, das
> Verschieben spart kein Byte und kostet Kopien. Wer diesen Schalter hier
> vorschlägt, hat die Hardware nicht verstanden — es gehört in den Katalog\
> eintrag als ausdrückliches *nicht benutzen*.

**Abnahme:** ein Startversuch, der rechnerisch nicht passt, wird mit 507 und
einer Zahl abgelehnt; ein 2048²-Bild auf einem knappen Knoten wird vom Server
abgelehnt, nicht vom Kernel.

### S7 · UI

* **Laden:** das bestehende Formular, plus die Felder, die nur für Bilder
  gelten (Standardauflösung, Schritte, maximale Auflösung). Der Planer-Hinweis
  zeigt die Bildrechnung statt der KV-Rechnung.
* **Benutzen:** ein Panel neben dem Chat — Prompt, Größe, Schritte, Seed,
  Ergebnisgalerie. Das ist der sichtbare Teil und der kleinste.
* **Karte:** unverändert. Name, Statuswort, *Details*, *Entladen* — der
  Umbau aus #124 trägt hier ohne Zutun, was der Beleg dafür ist, dass er
  richtig war.

**Abnahme:** ein Bild lässt sich erzeugen, ohne die Shell zu öffnen.

### S8 · Telemetrie

* `models.loaded[]` bekommt `kind: "image"`.
* `engine/<modell>` funktioniert, sobald der Server `/metrics` liefert — der
  Scraper in `engine_metrics.py` ist auf vLLM-Namen eingestellt, bekommt also
  einen zweiten Satz: `images_generated_total`, `seconds_per_image`,
  `steps_per_second`, `requests_running`.
* Die Log-Weiterleitung greift ohne Änderung, weil sie an `log_path` hängt.

**Abnahme:** `mosquitto_sub -t 'ainode/+/engine/#'` zeigt während einer
Generierung bewegte Zahlen.

### S9 · Profile und Platzierung

Beide kennen heute `node_ids`, `strategy` und LLM-Startparameter. Sie
brauchen `kind` und die bildspezifischen Felder, sonst stellt ein Profil
Node 3 unvollständig wieder her. Die Platzierung aus #114 funktioniert
unverändert — „dieses Modell läuft auf Node 3" ist modalitätsblind.

---

## 4. Passt es auf Node 3?

Die Rechnung, mit den Zahlen die feststehen:

```
Gewichte Qwen-Image-2.1 (bf16)        47,4 GB
Engine, CUDA-Kontext, Aktivierungen  ~ 4–6 GB
Puffer für VAE-Dekodierung bei 1024²  ~ 2–4 GB   (wächst quadratisch mit der Kante)
                                      ─────────
                                      ~55–58 GB
```

Dazu auf Node 3: qwen3.8, das Embedding-Modell, und **8 GB Host-Reserve**, die
der Wächter freihält. Von 122 GB bleibt damit für das Bildmodell ungefähr
`122 − 8 − (qwen3.8 + nomic)`.

Die eine Zahl, die mir fehlt, ist der tatsächliche Belegungsstand. Bevor
irgendetwas gebaut wird:

```bash
curl -sS localhost:3000/api/cluster/safety/memory | python3 -m json.tool
curl -sS localhost:3000/api/nodes | python3 -m json.tool | grep -A6 '"node_id"'
```

Wenn `host_available_mb` auf Node 3 unter etwa 60 GB liegt, passt es in bf16
nicht, und die Frage lautet dann: ein Quantisat (die HF-Suche zeigt mehrere
GGUF-Varianten, die den Bedarf ungefähr halbieren, zu Lasten der Qualität),
oder qwen3.8 zieht auf einen anderen Knoten um.

**Geschwindigkeit:** ein 20-B-MMDiT über 20–50 Entrauschungsschritte bei
1024² ist bandbreitengebunden wie alles auf dieser Hardware. Ich schätze das
hier ausdrücklich **nicht** — der Versuch in Schritt 0 misst es, und dann
steht eine echte Zahl im Katalogeintrag statt einer erfundenen.

---

## 5. Reihenfolge

**Schritt 0 — der Versuch, bevor irgendetwas gebaut wird.** Eine halbe Stunde,
von Hand auf Node 3:

```bash
docker run --rm --network host -v ~/.ainode/models:/models \
  <eugr-basisimage> bash -lc "
    pip install -q diffusers accelerate &&
    python - <<'PY'
import torch, time
from diffusers import DiffusionPipeline
p = DiffusionPipeline.from_pretrained('/models/Qwen--Qwen-Image-2.1',
                                      torch_dtype=torch.bfloat16).to('cuda')
t = time.time()
img = p('a red cube on a white table', num_inference_steps=20).images[0]
print('seconds:', round(time.time()-t, 1))
img.save('/models/probe.png')
PY"
```

Das beantwortet die drei Fragen, an denen der ganze Plan hängt, und keine
davon kann ich vom Schreibtisch aus beantworten:

1. Läuft `QwenImage21Pipeline` überhaupt auf aarch64 mit Blackwell und dem
   torch dieses Images? (Der wahrscheinlichste Stolperstein: der
   Attention-Backend — flash-attn ist auf dieser Kombination schon einmal
   ausgefallen, SDPA ist der Rückfall.)
2. Wie viel Speicher belegt sie wirklich?
3. Wie lange dauert ein Bild?

**Scheitert Schritt 0, ist der Plan hinfällig** — und zwar bevor eine Zeile
Code geschrieben ist. Das ist der Sinn dieser Reihenfolge.

Danach, jeder Schritt ein PR:

| # | Schritt | Warum in dieser Reihenfolge |
|---|---|---|
| 1 | S2 Image | ohne lauffähigen Container ist alles andere Theorie |
| 2 | S3 Server | der Vertrag, an dem der Rest hängt |
| 3 | S4 Backend | erst jetzt kann AINode etwas starten |
| 4 | S1 Modalität | jetzt gibt es eine zweite Art, die sich lohnt zu benennen |
| 5 | S6 Speicher | bevor es jemand im Alltag benutzt |
| 6 | S5 Routing | Zugriff vom Head |
| 7 | S7 UI | der sichtbare Teil, zuletzt |
| 8 | S8 + S9 | Telemetrie und Profile ziehen nach |

---

## 6. Was ich nicht weiß

1. **Ob diffusers auf dieser Hardware trägt.** Der größte Einzelposten, und
   Schritt 0 entscheidet ihn. Alles danach ist gewöhnliche Arbeit.
2. **Ob 47 GB neben qwen3.8 auf Node 3 passen.** Braucht die aktuelle
   Belegung, siehe Abschnitt 4.
3. **Wie schnell es ist.** Wird gemessen, nicht geschätzt. Bei mehr als etwa
   einer Minute je Bild ist die Frage, ob es sich neben einem Coding-Modell
   auf demselben Knoten lohnt oder lieber allein läuft.
4. **Ob ein Quantisat taugt.** Die GGUF-Varianten sind Fremdkonvertierungen;
   ob diffusers sie ohne Weiteres lädt, ist offen und wäre ein eigener
   Versuch.
