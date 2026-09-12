# Image-Cache auf dem Head (Registry-Mirror)

Ein 20-GB-Engine-Image dreimal aus dem Internet zu ziehen, weil der Cluster
drei Knoten hat, ist Unsinn — die Knoten hängen mit 10G aneinander und über
Direktlinks mit 100G, die Leitung nach draußen ist das Langsamste im Haus.

AINode umgeht das schon zur Hälfte: hat der Head das Image, wird es über die
Fabric zu den Peers kopiert statt dort erneut geladen (`docker save | ssh
docker load`). Das kostet **einen** Download statt N.

Ein Registry-Cache auf dem Head ist die vollständige Lösung. Er gewinnt dort,
wo das Kopieren verliert:

| | Kopieren vom Head | Registry-Cache |
|---|---|---|
| Downloads aus dem Internet | 1 | 1 |
| Übertragung zwischen den Knoten | volles Image, unkomprimiert | nur die fehlenden Layer |
| Neue Version desselben Images | volles Image erneut | nur der geänderte Layer |
| Überlebt einen Neustart / neue Modelle | nein | ja |
| Einrichtung | keine | einmalig `daemon.json` pro Knoten |

Bei einem Image-Update — neue vLLM-Version, gleicher Unterbau — sind das
typisch 2 GB statt 20.

---

## Einrichtung (einmalig, ~10 Minuten)

### 1. Registry auf dem Head starten

Als **Pull-through-Cache** für Docker Hub. Der Cache holt sich fehlende Layer
selbst von Hub und behält sie:

```bash
ssh Spark1
sudo mkdir -p /var/lib/registry-cache
docker run -d --restart=always --name registry-cache \
  -p 5000:5000 \
  -v /var/lib/registry-cache:/var/lib/registry \
  -e REGISTRY_PROXY_REMOTEURL=https://registry-1.docker.io \
  registry:2
```

**Erfolg:**

```bash
curl -s http://192.168.1.2:5000/v2/_catalog
```
antwortet mit `{"repositories":[]}`.

Platzbedarf: rechne mit 60–100 GB, wenn mehrere Engine-Images im Umlauf sind.
Aufräumen geht mit `docker exec registry-cache registry garbage-collect
/etc/docker/registry/config.yml`.

### 2. Alle drei Knoten auf den Cache zeigen lassen

Auf **jedem** Knoten — auch auf dem Head selbst:

```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "registry-mirrors": ["http://192.168.1.2:5000"],
  "insecure-registries": ["192.168.1.2:5000"]
}
JSON
sudo systemctl restart docker
```

`insecure-registries` ist nötig, weil der Cache ohne TLS läuft. Das ist in
einem geschlossenen Cluster-Netz vertretbar; hängt der Head im offenen Netz,
gehört ein Zertifikat davor.

> **Achtung:** `systemctl restart docker` stoppt laufende Container ohne
> `--restart`-Policy. Vorher die Modelle entladen oder einen ruhigen Moment
> wählen.

**Erfolg:**

```bash
docker info | grep -A2 "Registry Mirrors"
```

### 3. Prüfen, dass der Cache greift

```bash
# auf Spark2 — zieht über den Head, nicht über die Leitung nach draußen
time docker pull vllm/vllm-openai:v0.27.1
# auf Spark1 — im Cache liegt jetzt etwas
curl -s http://192.168.1.2:5000/v2/_catalog
```

**Erfolgskriterium:** der zweite Knoten, der dasselbe Image zieht, ist um ein
Vielfaches schneller als der erste, und `_catalog` nennt `vllm/vllm-openai`.

---

## Was der Cache **nicht** abdeckt

- **GHCR** (`ghcr.io/bmetallica/ainode`). `registry-mirrors` gilt bei Docker
  nur für Docker Hub. Für GHCR bräuchte es einen zweiten Cache und explizit
  umgeschriebene Image-Namen (`192.168.1.2:5000/bmetallica/ainode:…`) — das
  ändert den Tag, und der Launcher vergleicht Tags über die Knoten hinweg.
  Für den Orchestrator lohnt es ohnehin kaum: das Image ist klein.
- **Lokal gebaute Images** (`vllm-node:latest`, `ainode:dev`). Die gibt es in
  keiner Registry. Entweder wie bisher mit `docker save | ssh docker load`
  verteilen, oder in den Cache **pushen** — dann aber als eigene Registry
  betreiben (ohne `REGISTRY_PROXY_REMOTEURL`, sonst ist sie schreibgeschützt)
  und die Modelle auf `192.168.1.2:5000/vllm-node:latest` umstellen.

Wer beides will, betreibt zwei Container: einen Pull-through-Cache auf 5000
und eine schreibbare Registry auf 5001. Das ist kein AINode-Thema mehr,
sondern normale Docker-Infrastruktur.

---

## Warum AINode das nicht selbst einrichtet

Die Änderung gehört in `/etc/docker/daemon.json` und verlangt einen Neustart
des Docker-Daemons **auf jedem Knoten**. AINode läuft in einem Container und
startet den Daemon nicht neu, unter dem es selbst läuft — und ein Werkzeug,
das ungefragt die Docker-Konfiguration des Hosts umschreibt und dabei alle
Container beendet, wäre eine unangenehme Überraschung. Die Schritte oben sind
kurz genug, um sie bewusst zu machen.
