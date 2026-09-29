"""Make `ssh <ip>` use what the operator set up for `ssh <alias>`.

Reported from a launch led by a node other than the head:

    Error: Passwordless SSH to 192.168.1.4 failed.

while `ssh admin@Spark3` worked from the same machine. The operator's
~/.ssh/config names each node by an alias — `Host Spark3`, `HostName
192.168.1.4`, `User admin`, `IdentityFile ~/.ssh/id_ed25519_nvsync_…` — and
eugr's launcher connects to the bare address. No block matches an address, so
ssh fell back to root and the default key names, and failed. The head had
worked only because the installer had copied a key there under a default name.

So before a distributed launch the container's own copy of the ssh config
(/root/.ssh/config, made by the entrypoint from the host's) gets one block per
node address, carrying the User, IdentityFile, Port and the like of the alias
that belongs to that node. An alias belongs to a node when its HostName is one
of the node's addresses, or resolves to one, or when the alias or its HostName
is the node's name. The blocks go first in the file: ssh takes the first value
it finds for each option, and the entrypoint's `Host *` block would otherwise
decide the user.

The host's file is never touched; this is the container's copy, rebuilt at
every container start.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["parse_ssh_config", "ip_blocks", "ensure_ip_aliases", "BEGIN", "END"]

BEGIN = "# >>> ainode: node addresses (generated before each distributed launch)"
END = "# <<< ainode: node addresses"

#: Options worth carrying from an alias to its address. HostName is left out
#: (the address is the host), and so is anything that names other hosts.
_CARRIED = ("user", "identityfile", "identitiesonly", "port", "certificatefile",
            "pubkeyacceptedalgorithms", "pubkeyacceptedkeytypes", "hostkeyalias",
            "userknownhostsfile", "stricthostkeychecking", "proxyjump", "proxycommand")


def parse_ssh_config(text: str) -> List[Dict[str, object]]:
    """[{patterns: [...], options: [(key, value)]}] for each Host block.
    Match blocks and anything before the first Host are ignored."""
    blocks: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    inside_ours = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == BEGIN:
            inside_ours = True
            continue
        if line == END:
            inside_ours = False
            continue
        if inside_ours or not line or line.startswith("#"):
            continue
        parts = line.replace("=", " ", 1).split(None, 1)
        key = parts[0].lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        if key == "host":
            current = {"patterns": value.split(), "options": []}
            blocks.append(current)
        elif key == "match":
            current = None
        elif current is not None:
            current["options"].append((key, value))
    return blocks


def _resolve(name: str) -> List[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(name, None)})
    except (OSError, UnicodeError):
        return []


def private_keys(directory: Path) -> List[str]:
    """Every private key in ``directory``: a file with a matching .pub, or an
    ``id_*`` file that is not one."""
    out = []
    try:
        for entry in sorted(directory.iterdir()):
            if not entry.is_file() or entry.suffix == ".pub":
                continue
            if entry.name in ("config", "known_hosts", "authorized_keys") or \
                    entry.name.startswith("known_hosts"):
                continue
            if (directory / (entry.name + ".pub")).exists() or entry.name.startswith("id_"):
                out.append(str(entry))
    except OSError:
        pass
    return out


def ip_blocks(blocks: Sequence[Dict[str, object]], nodes: Iterable[Sequence[str]],
              resolve=_resolve, keys: Sequence[str] = ()) -> str:
    """The generated section: one Host block per node whose alias is found.

    ``nodes`` are the address groups of the nodes involved — each a list of
    the node's addresses (fabric, LAN, RoCE) and its name.
    """
    aliases = []
    for block in blocks:
        patterns = [p for p in block["patterns"] if not any(c in p for c in "*?!")]
        options = dict((k, v) for k, v in block["options"])
        hostname = str(options.get("hostname") or "")
        if not patterns or "%" in hostname:
            continue
        names = {p.lower() for p in patterns}
        if hostname:
            names.add(hostname.lower())
            names.update(a.lower() for a in resolve(hostname))
        aliases.append((names, block))

    out: List[str] = []
    unmatched: List[str] = []
    for group in nodes:
        addresses = [a for a in group if a]
        wanted = {a.lower() for a in addresses}
        targets = sorted({a for a in addresses if a.replace(".", "").isdigit() or ":" in a})
        if not targets:
            continue
        match = next((block for names, block in aliases if names & wanted), None)
        carried = [(k, v) for k, v in match["options"] if k in _CARRIED] if match else []
        if not carried:
            unmatched.extend(t for t in targets if t not in unmatched)
            continue
        out.append(f"# for {' '.join(match['patterns'])}")
        out.append("Host " + " ".join(targets))
        for key, value in carried:
            out.append(f"    {_canonical(key)} {value}")
        out.append("")
    # No alias found for these — its HostName may be a name only the host
    # resolves, or there is none. Then every key there is: ssh tries each,
    # and the one the operator set up for the cluster is among them. Still
    # after any alias block, so a matched node keeps exactly its settings.
    if unmatched and keys:
        out.append("# no alias found: every key in ~/.ssh")
        out.append("Host " + " ".join(unmatched))
        for key in keys:
            out.append(f"    IdentityFile {key}")
        out.append("")
    return "\n".join(out)


def _canonical(key: str) -> str:
    return {"user": "User", "identityfile": "IdentityFile", "identitiesonly": "IdentitiesOnly",
            "port": "Port", "certificatefile": "CertificateFile",
            "pubkeyacceptedalgorithms": "PubkeyAcceptedAlgorithms",
            "pubkeyacceptedkeytypes": "PubkeyAcceptedKeyTypes",
            "hostkeyalias": "HostKeyAlias", "userknownhostsfile": "UserKnownHostsFile",
            "stricthostkeychecking": "StrictHostKeyChecking", "proxyjump": "ProxyJump",
            "proxycommand": "ProxyCommand"}.get(key, key)


def ensure_ip_aliases(nodes: Iterable[Sequence[str]],
                      path: Path = Path("/root/.ssh/config"), resolve=_resolve) -> int:
    """Rewrite the generated section of the container's ssh config for
    ``nodes``. Returns how many node blocks it wrote. Never raises."""
    try:
        text = path.read_text() if path.exists() else ""
    except OSError:
        return 0
    try:
        section = ip_blocks(parse_ssh_config(text), list(nodes), resolve=resolve,
                            keys=private_keys(path.parent))
        kept = []
        inside = False
        for line in text.splitlines():
            if line.strip() == BEGIN:
                inside = True
                continue
            if line.strip() == END:
                inside = False
                continue
            if not inside:
                kept.append(line)
        body = "\n".join(kept).lstrip("\n")
        if section:
            body = f"{BEGIN}\n{section}{END}\n\n" + body
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body if body.endswith("\n") else body + "\n")
        path.chmod(0o600)
        if section:
            logger.info("ssh: node addresses now use their aliases' settings (%d block(s)):\n%s",
                        section.count("Host "), section)
        else:
            logger.warning("ssh: no alias and no key found for the peers %s — "
                           "passwordless ssh to them will fail",
                           ", ".join(a for g in nodes for a in g if a))
        return section.count("Host ") if section else 0
    except Exception:
        logger.exception("could not write the node addresses into %s", path)
        return 0


def node_groups(app) -> List[List[str]]:
    """The address group of every other cluster member."""
    cluster = app.get("cluster_state")
    own = str(getattr(app.get("config"), "node_id", "") or "")
    out = []
    for node in (cluster.members() if cluster is not None else []):
        if node.node_id == own:
            continue
        out.append([str(getattr(node, "fabric_ip", "") or ""),
                    str(getattr(node, "peer_ip", "") or ""),
                    str(getattr(node, "node_name", "") or ""),
                    *[str(a) for a in (getattr(node, "ib_ips", None) or [])]])
    return out


def check_ssh(app, timeout: int = 8) -> List[Dict[str, object]]:
    """Map the other members' addresses to their aliases, then try a
    passwordless ``ssh <address> true`` to each — what the launcher does
    first. Blocking; run it in an executor."""
    import subprocess

    groups = node_groups(app)
    ensure_ip_aliases(groups)
    cluster = app.get("cluster_state")
    own = str(getattr(app.get("config"), "node_id", "") or "")
    results = []
    for node in (cluster.members() if cluster is not None else []):
        if node.node_id == own:
            continue
        address = str(getattr(node, "fabric_ip", "") or getattr(node, "peer_ip", "") or "")
        row: Dict[str, object] = {"node_id": node.node_id,
                                  "node_name": getattr(node, "node_name", ""),
                                  "address": address}
        if not address:
            row.update(ok=False, error="no address")
            results.append(row)
            continue
        try:
            done = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
                 address, "true"], capture_output=True, text=True, timeout=timeout + 5)
            row["ok"] = done.returncode == 0
            if not row["ok"]:
                row["error"] = (done.stderr or "").strip()[-300:]
            settings = subprocess.run(["ssh", "-G", address], capture_output=True,
                                      text=True, timeout=5).stdout
            row["user"] = next((line.split(None, 1)[1] for line in settings.splitlines()
                                if line.startswith("user ")), "")
            row["identity_files"] = [line.split(None, 1)[1] for line in settings.splitlines()
                                     if line.startswith("identityfile ")]
        except Exception as exc:
            row.update(ok=False, error=f"{exc.__class__.__name__}: {exc}")
        results.append(row)
    return results
