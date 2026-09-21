# images.md — Bildgenerierung in AINode

Ziel: `Qwen-Image-2.1` — vorzugsweise als Quantisat — läuft auf Node 3 neben
qwen3.8 und dem Embedding-Modell, gestartet und überwacht wie jedes andere
Modell: dieselbe Karte, derselbe Speicherwächter, dasselbe Profil, dieselbe
Telemetrie. Kein Nebencontainer, den AINode nicht kennt.

**Priorität: Nebenspur.** Das steht hinter allem anderen, und der Plan ist so
geschnitten, dass es nichts blockiert und nichts Laufendes anfasst — siehe
Abschnitt 7. Der einzige Teil, der sich sofort lohnt, ist Schritt 0: eine
halbe Stunde, die abschließend klärt, ob der Rest überhaupt möglich ist.

Aufgenommen am 2026-09-21 auf `main` @ 96041b0 (2371 Tests).

---

## Stand: gebaut am 2026-09-21

| Schritt | PR | Was daraus wurde |
|---|---|---|
| S2 Engine-Image | #138 | `scripts/Dockerfile.diffusers` FROM `vllm-node`, `build-diffusers-image.sh`, Verteilung über `update-cluster.sh --images` |
| S3 Server | #138 | `ainode/engine/diffusers_server.py` — OpenAI-Bildendpunkt, `/metrics`, Auflösungsgrenze, serialisierte Läufe, kein CPU-Offload |
| S4 Backend | #138 | `ainode/engine/backends/diffusers.py`, `engine_backend="diffusers"` |
| S1 Modalität | #139 | `modality` auf `ModelInfo`, `text-to-image` in der Suche, `kind` auf Record und Profilen |
| S5 Routing | #139 | `POST /v1/images/generations` durch denselben Proxy |
| S6 Speicher | #139 | `plan_for_image`, Zulassungstor mit der Bildrechnung, `max_image_size` |
| S9 Profile | #139 | `KIND_IMAGE`, Startparameter im Profil |
| S7 UI | #140 | Bildfelder im Launch-Formular, eigene Ansicht *Images* |
| S8 Telemetrie | #140 | `kind` im `models`-Payload, Bild-Metriken auf `engine/<modell>` |

### Beim Durchsehen gefunden und behoben (#142)

Vier Fehler, alle derselben Gestalt: ein Feld, das an einem Ende eines Pfades
existiert und unterwegs verloren geht — also funktioniert das, was darauf
aufbaut, stillschweigend nie, obwohl jedes einzelne Stück richtig ist.

| | Befund |
|---|---|
| **A** | `kind` fiel aus den Instanz-Projektionen von `/api/nodes` und `/api/cluster/resources` heraus. Die *Images*-Ansicht fragt genau danach und war damit **dauerhaft leer**, bei korrektem Rest. |
| **B** | Ein selbst heruntergeladenes Bildmodell steht in keinem Katalog, also sagte nichts, welche Engine es braucht — und die Vorgabe des Knotens ist vLLM, das eine Pipeline gar nicht laden kann. Jetzt entscheidet der Checkpoint selbst: `model_index.json` statt `config.json`. |
| **C** | `record_image_speed` war toter Code; `seconds_per_image` konnte nie gefüllt werden. Ein Bildmodell hat keine Tokens, seine Geschwindigkeit ist die mittlere Latenz — eine Anfrage, ein Bild. |
| **D** | Beim Einsammeln eines Profils von den Peers wurde `kind` auf `KIND_LLM` überschrieben. Ein Bildmodell auf Node 3 — dort, wo es hingehört — kam als LLM ins Profil zurück, und das Wiederherstellen hätte vLLM auf eine Diffusers-Pipeline losgelassen. |

Dazu aus demselben Durchgang: der ältere `nvidia`-Backend schrieb weiterhin
alle Instanzen in **eine** Logdatei (dasselbe Problem, das für `eugr` schon
behoben war), die Bildeinstellungen des Formulars reisten auch bei
Textmodellen mit, und weder Suche noch Modelle-Seite kennzeichneten ein
Bildmodell als solches.

**Was das nicht ersetzt: Schritt 0.** Ob diffusers auf dieser Hardware trägt,
ist weiterhin ungemessen — der Code ist derselbe, ob der Versuch gelingt oder
nicht, aber ob er etwas erzeugt, entscheidet allein der Lauf auf Node 3.
Die beiden Katalogeinträge sind deshalb `verified=False`, und ihre
`min_memory_gb` sind konservativ geschätzt, nicht gemessen.

---

## 1. Warum es heute nicht geht

Nachgeprüft, nicht vermutet:

```
Qwen/Qwen-Image-2.1     pipeline_tag = text-to-image   library_name = diffusers
                        diffusers:QwenImage21Pipeline
Verzeichnisse und Größen:
                        text_encoder/   17,5 GB
                        transformer/    14,2 GB
                        vae/             1,4 GB
                        scheduler/, processor/, model_index.json
                        ───────────────────────
                        rund 33 GB auf der Platte
```

> **Korrektur zur ersten Fassung dieses Plans.** Dort standen 47,4 GB. Das ist
> der Wert, den der Hub als `usedStorage` meldet — er umfasst den LFS-Bestand
> des Repos über alle Revisionen und kann deutlich über dem liegen, was ein
> Checkout tatsächlich belegt. Für „passt es auf Node 3" zählt die Summe der
> Dateien, und die sind rund **33 GB**. Unser eigener Planer rechnet richtig:
> `weight_bytes_on_disk` summiert die Gewichtsdateien, nicht die Hub-Angabe.
> Die Suche zeigt weiterhin die Hub-Zahl, was hier zu Lasten der Vorsicht irrt
> — das ist die richtige Richtung, aber man muss es wissen.

Bemerkenswert daran, und für die Quantisierung in Abschnitt 5 entscheidend:
**der Text-Encoder ist der größte Brocken, nicht der Transformer.**

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

**Und es wird verteilt wie jedes andere Image.** Das fehlte in der ersten
Fassung dieses Plans und ist kein Detail: ein Image, das nur auf dem Head
liegt, ist auf Node 3 nichts wert.

AINode hat dafür zwei Wege, und beide greifen hier:

* **`scripts/update-cluster.sh`, Schritt 5b.** Verteilt heute genau ein
  Engine-Image (`vllm-node`) an alle Knoten — über die lokale Registry auf
  dem Head, wenn es eine gibt, sonst per `docker save | ssh docker load`.
  Daraus wird eine **Liste**, und `ainode-diffusers` fährt mit, sobald es
  lokal gebaut ist. Derselbe Vergleich der Image-**IDs** am Ende, aus
  demselben Grund: Tags lügen, IDs nicht.
* **Die Verteilung zur Laufzeit** (`ensure_local_image` /
  `ensure_peer_has_image` in `ainode/engine/distribute.py`), die greift, wenn
  ein Katalogrezept ein `engine_image` festnagelt. Unser Rezept tut das
  (`engine_image="ainode-diffusers:latest"`), also prüft der Ladepfad das
  Image, bevor er startet.

Der Unterschied zwischen beiden ist der Herkunftsort. Die Laufzeitverteilung
kann ein Image aus einer Registry ziehen; unseres ist lokal gebaut und liegt
in keiner. Deshalb ist `update-cluster.sh` der Weg, und die Laufzeitprüfung
ist das Netz darunter — sie muss, wenn das Image fehlt, **sagen was zu tun
ist** statt an einem `docker pull` zu scheitern, der nie klappen konnte:

> Das Engine-Image `ainode-diffusers:latest` liegt auf diesem Knoten nicht
> und ist nirgends zu ziehen — es wird lokal gebaut. Führe auf dem Head
> `scripts/update-cluster.sh --images` aus.

**Abnahme:** `docker run --rm ainode-diffusers python -c "import torch, diffusers;
print(torch.cuda.is_available(), diffusers.__version__)"` sagt `True` — und
zwar auf **allen drei Knoten**, mit identischer Image-ID.

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
                                      bf16        FP8
Gewichte auf der Platte              ~33 GB      18 GB
Engine, CUDA-Kontext, Aktivierungen  ~4–6 GB     ~4–6 GB
Puffer VAE-Dekodierung bei 1024²     ~2–4 GB     ~2–4 GB   (quadratisch mit der Kante)
                                     ────────    ────────
                                     ~39–43 GB   ~24–28 GB
```

Dazu auf Node 3: qwen3.8, das Embedding-Modell, und **8 GB Host-Reserve**, die
der Wächter freihält. Von 122 GB bleibt für das Bildmodell ungefähr
`122 − 8 − (qwen3.8 + nomic)`.

Die eine Zahl, die mir fehlt, ist der tatsächliche Belegungsstand. Bevor
irgendetwas gebaut wird:

```bash
curl -sS localhost:3000/api/cluster/safety/memory | python3 -m json.tool
curl -sS localhost:3000/api/nodes | python3 -m json.tool | grep -A6 '"node_id"'
```

Grobe Orientierung, solange die Zahl fehlt: **unter etwa 45 GB frei** ist bf16
vom Tisch und FP8 die Antwort; **unter etwa 30 GB frei** wird es auch mit FP8
eng, und dann ist die Frage nicht das Format, sondern ob qwen3.8 auf einen
anderen Knoten gehört.

Der Speicherwächter macht diese Rechnung übrigens nicht überflüssig, aber
erträglich: rechne ich mich hier um zehn Gigabyte, bekomme ich eine 507er
Absage oder im schlimmsten Fall eine beendete Instanz — nicht einen Knoten,
der neu gestartet werden muss. Das ist der Unterschied zu der Situation, in
der MiniMax-M3 zwei Knoten mitgenommen hat.

**Geschwindigkeit:** ein 20-B-MMDiT über 20–50 Entrauschungsschritte bei
1024² ist bandbreitengebunden wie alles auf dieser Hardware. Ich schätze das
hier ausdrücklich **nicht** — der Versuch in Schritt 0 misst es, und dann
steht eine echte Zahl im Katalogeintrag statt einer erfundenen.

---

## 5. Quantisierung

Ausdrücklich Teil des Plans und nicht ein späterer Gedanke: bf16 ist hier die
teuerste Variante, und auf einem Knoten, der schon ein Coding-Modell und die
Embeddings trägt, ist der Unterschied zwischen 33 GB und 18 GB der zwischen
„geht" und „geht bequem".

Was es gibt, mit echten Zahlen vom Hub:

| Repo | Größe | Format | Layout | Einschätzung |
|---|---|---|---|---|
| `Qwen/Qwen-Image-2.1` | ~33 GB | bf16 | vollständige Pipeline | Referenz. Läuft, wenn überhaupt etwas läuft |
| `Rin247/Qwen-Image-2.1-FP8` | 18,0 GB | fp8 | vollständige Pipeline | **der erste Kandidat** |
| `Rin247/Qwen-Image-2.1-INT4` | 12,5 GB | int4 | vollständige Pipeline | zweiter Kandidat, offene Frage beim Backend |
| `ModelsLab/Qwen-Image-2.1-W4A4-int4` | 4,7 GB | Nunchaku W4A4 | nur Transformer | auf dem Papier das schnellste, praktisch das riskanteste |
| `leejet/Qwen-Image-2.1-GGUF` | 2,6–7,7 GB je Datei | GGUF Q2…Q8 | nur Transformer | täuscht — siehe unten |

### Warum FP8 der erste Kandidat ist

Drei Gründe, und alle drei zählen auf dieser Hardware:

1. **Es ist eine vollständige Pipeline im diffusers-Layout.** Der Ladeweg aus
   S3 ändert sich nicht um eine Zeile — `from_pretrained` auf ein Verzeichnis,
   fertig. Alles andere in dieser Tabelle verlangt Sonderbehandlung.
2. **fp8 ist auf Blackwell nativ.** Die Tensor-Kerne rechnen es direkt; es ist
   kein Auspacken-und-in-bf16-rechnen wie bei den meisten 4-Bit-Formaten.
   Dasselbe Argument, aus dem `--kv-cache-dtype fp8` bei den LLMs hier die
   Vorgabe ist.
3. **18 GB statt 33 GB** lässt Node 3 Luft, und Luft ist auf einem
   Unified-Memory-Knoten gleichbedeutend mit Stabilität.

### Warum GGUF täuscht

Die GGUF-Dateien sehen mit 4,2 GB (Q4_K) verlockend aus, aber sie enthalten
**nur den Transformer** — und der ist mit 14,2 GB gar nicht der größte Teil.
Der Text-Encoder mit 17,5 GB bliebe unangetastet:

```
GGUF-Transformer Q4_K    4,2 GB
+ Text-Encoder bf16     17,5 GB
+ VAE                    1,4 GB
                        ────────
                       ~23 GB    — schlechter als FP8 mit 18 GB
```

Dazu kommt, dass diffusers GGUF nur über `from_single_file` mit einer
`GGUFQuantizationConfig` lädt, also einen zweiten Ladeweg bräuchte, und dass
der Rest der Pipeline trotzdem aus dem bf16-Repo kommen muss. Es gibt auch
GGUF-Text-Encoder (`pottokao/…-Text-Encoder-Heretic-GGUF`), aber dann setzt
man eine Pipeline aus zwei Fremdkonvertierungen zusammen und hat zwei
Fehlerquellen statt keiner. **Für uns nicht der erste Weg.**

### Warum Nunchaku das Risiko ist

4,7 GB und W4A4 ist beeindruckend, und es ist eine Quantisierung, die auch
rechnet statt nur zu speichern. Der Preis: Nunchaku bringt eigene CUDA-Kernel
mit, und für aarch64 mit Blackwell gibt es dafür mit ziemlicher Sicherheit
kein fertiges Rad. Das hieße selbst übersetzen — auf genau dem Stack, der uns
auf diesem Cluster schon `Failed to find C compiler` und die
DeepGEMM-Geschichte beschert hat. Ein eigener Tag Arbeit mit offenem Ausgang.

**Nicht ausgeschlossen, aber zuletzt.** Wenn FP8 läuft und zu langsam ist,
ist Nunchaku die nächste Frage; vorher nicht.

### Was in Schritt 0 mitgemessen wird

Der Versuch aus Abschnitt 7 läuft **zweimal**: einmal bf16, einmal FP8. Das
kostet fast nichts zusätzlich und beantwortet die Frage, die sonst hinterher
gestellt wird — ob das Quantisat auf dieser Hardware überhaupt lädt, wie viel
Speicher es wirklich spart, und ob die Bilder taugen. Erst wenn FP8 sowohl
lädt als auch überzeugt, wird es der Katalogeintrag; sonst bf16, und die
Quantisierung wird eine eigene Runde.

INT4 kommt in dieselbe Messreihe, sobald geklärt ist, welches
Quantisierungs-Backend das Repo voraussetzt — `torchao` wäre gutartig (reines
PyTorch, gute Chancen auf aarch64), `bitsandbytes` wäre die gleiche Wette wie
Nunchaku. Das steht in der `model_index.json` bzw. in der
`quantization_config` des Transformers und ist in zwei Minuten geklärt:

```bash
curl -sS https://huggingface.co/Rin247/Qwen-Image-2.1-INT4/raw/main/transformer/config.json \
  | python3 -m json.tool | grep -A8 quantization
```

---

## 6. Der Katalogeintrag

Am Ende steht ein kuratierter Eintrag, wie bei jedem anderen bewährten Modell
— mit dem, was die Messung ergeben hat, nicht mit dem, was plausibel klingt:

```python
"qwen-image-2.1-fp8": ModelInfo(
    id="qwen-image-2.1-fp8",
    name="Qwen-Image 2.1 (FP8)",
    hf_repo="Rin247/Qwen-Image-2.1-FP8",
    modality="image",
    size_gb=18.0,
    min_memory_gb=<aus Schritt 0>,
    description="… Sekunden je Bild bei 1024² und 20 Schritten, gemessen auf "
                "einem GB10-Knoten. Kein CPU-Offload: auf Unified Memory "
                "verschiebt es nichts und kostet Kopien.",
    engine_backend="diffusers",
    verified=<erst nach dem Lauf auf der Hardware>,
),
```

Das `verified`-Häkchen bekommt es erst, wenn es hier wirklich gelaufen ist —
dieselbe Regel wie bei den LLMs.

---

## 7. Reihenfolge und Priorität

### Priorität: eine Nebenspur

Bildgenerierung steht **hinter** allem anderen. Was das konkret heißt, damit
es nicht bei einer Absichtserklärung bleibt:

* **Nichts in diesem Plan blockiert etwas anderes.** Keiner der Schritte
  ändert den LLM-Ladepfad, den Planer für Textmodelle, das Routing von
  `/v1/chat/completions` oder das Verhalten des Wächters. Die Berührungspunkte
  sind additiv: ein zweiter `engine_backend`, eine zweite Route, ein zweites
  `kind`. Wenn dieser Plan ein Jahr liegen bleibt, fehlt nichts.
* **Node 3 bleibt ein Arbeitsknoten.** Das Bildmodell kommt dort **zuletzt**
  dazu, nicht zuerst — und zwar nicht aus Höflichkeit, sondern weil der
  Wächter die *zuletzt gestartete* Instanz abschießt. Wird es eng, stirbt
  damit automatisch das Bildmodell und nicht das Coding-Modell, an dem jemand
  gerade arbeitet. Diese Reihenfolge ist eine Eigenschaft, keine Konvention:
  sie dokumentiert sich selbst, weil der Wächter sie durchsetzt.
* **Nichts davon fasst laufende Modelle an.** Kein Schritt verlangt, qwen3.8
  oder die Embeddings neu zu starten — außer dem Update selbst, das ohnehin
  alle Dienste neu startet.
* **Schritt 0 kostet eine halbe Stunde und ist jederzeit machbar**, auch
  völlig unabhängig davon, ob der Rest je gebaut wird. Er beantwortet die
  Frage „ginge das überhaupt?" abschließend und kostet nichts als Zeit am
  Terminal und rund 33 GB Download.

Vorschlag für die tatsächliche Abfolge: Schritt 0 bei Gelegenheit, dann liegt
das Ergebnis vor. Der Rest wird gebaut, wenn nichts Dringenderes ansteht —
oder gar nicht, wenn Schritt 0 unerfreulich ausgeht.

### Schritt 0 — der Versuch, bevor irgendetwas gebaut wird

Eine halbe Stunde, von Hand auf Node 3. **Zweimal**, einmal bf16 und einmal
FP8, weil der zweite Lauf fast nichts zusätzlich kostet und die
Quantisierungsfrage gleich miterledigt:

Die Gewichte holt AINode schon heute — der Repo-Downloader lädt beliebige
Repos unabhängig vom Pipeline-Tag, das Serven ist das Einzige, was fehlt:

```bash
curl -sS -X POST localhost:3000/api/models/download-repo \
  -H 'Content-Type: application/json' \
  -d '{"hf_repo":"Rin247/Qwen-Image-2.1-FP8"}'
```

Dann der Versuch selbst:

```bash
docker run --rm --network host -v ~/.ainode/models:/models \
  <eugr-basisimage> bash -lc "
    pip install -q diffusers accelerate &&
    python - <<'PY'
import torch, time
from diffusers import DiffusionPipeline
d = '/models/Rin247--Qwen-Image-2.1-FP8'      # bzw. Qwen--Qwen-Image-2.1
p = DiffusionPipeline.from_pretrained(d, torch_dtype=torch.bfloat16).to('cuda')
t = time.time()
img = p('a red cube on a white table', num_inference_steps=20).images[0]
print('Sekunden:', round(time.time()-t, 1))
print('Spitze GB:', round(torch.cuda.max_memory_allocated()/1e9, 1))
img.save('/models/probe.png')
PY"
```

Das beantwortet die Fragen, an denen der ganze Plan hängt, und keine davon
kann ich vom Schreibtisch aus beantworten:

1. Läuft `QwenImage21Pipeline` überhaupt auf aarch64 mit Blackwell und dem
   torch dieses Images? (Der wahrscheinlichste Stolperstein: der
   Attention-Backend — flash-attn ist auf dieser Kombination schon einmal
   ausgefallen, SDPA ist der Rückfall.)
2. Lädt das **FP8-Quantisat** genauso, oder verlangt es ein
   Quantisierungs-Backend, das hier fehlt?
3. Wie viel Speicher belegen beide wirklich — inklusive der Spitze beim
   VAE-Schritt, die `max_memory_allocated()` mitnimmt?
4. Wie lange dauert ein Bild, und taugen die FP8-Bilder?

**Scheitert Schritt 0, ist der Plan hinfällig** — und zwar bevor eine Zeile
Code geschrieben ist. Das ist der Sinn dieser Reihenfolge. Scheitert nur der
FP8-Teil, wird bf16 der Katalogeintrag und die Quantisierung eine eigene
Runde.

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

## 8. Was ich nicht weiß

1. **Ob diffusers auf dieser Hardware trägt.** Der größte Einzelposten, und
   Schritt 0 entscheidet ihn. Alles danach ist gewöhnliche Arbeit.
2. **Ob es neben qwen3.8 auf Node 3 passt.** Braucht die aktuelle Belegung,
   siehe Abschnitt 4. Die Antwort entscheidet zwischen bf16 und FP8, nicht
   zwischen „geht" und „geht nicht".
3. **Ob das FP8-Repo hier lädt.** `Rin247/Qwen-Image-2.1-FP8` ist eine
   Fremdkonvertierung in vollständigem diffusers-Layout. Das Layout spricht
   dafür, dass es ohne Sonderbehandlung lädt; belegt ist es nicht. Dasselbe
   gilt für INT4, wo zusätzlich offen ist, welches Quantisierungs-Backend
   vorausgesetzt wird — `torchao` wäre gutartig, `bitsandbytes` eine Wette.
4. **Wie viel FP8 wirklich an Qualität kostet.** Bei einem Bildmodell ist das
   keine Zahl, sondern ein Blick: derselbe Prompt mit demselben Seed, beide
   Varianten nebeneinander. Gehört in Schritt 0.
5. **Wie schnell es ist.** Wird gemessen, nicht geschätzt. Bei mehr als etwa
   einer Minute je Bild ist die Frage, ob es sich neben einem Coding-Modell
   auf demselben Knoten lohnt oder lieber allein läuft.
6. **Ob Nunchaku (4,7 GB, W4A4) auf aarch64 baubar ist.** Auf dem Papier das
   attraktivste Format, praktisch eigene CUDA-Kernel ohne fertiges Rad für
   diese Architektur. Erst relevant, wenn FP8 läuft und zu langsam ist.
