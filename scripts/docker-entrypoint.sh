#!/usr/bin/env bash
# AINode container entrypoint.
# Delegates to `ainode start --in-container`. The CLI reads
# ~/.ainode/config.json (mounted via -v <host>/.ainode:/root/.ainode) and
# picks solo vs head via `distributed_mode`.
set -euo pipefail

: "${AINODE_HOME:=/root/.ainode}"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

# SSH keys: the host mounts its ~/.ssh at /host-ssh (read-only). OpenSSH
# refuses to use keys not owned by the invoking user, so copy them into
# /root/.ssh owned by root with the right modes. The caller's docker run
# must use `-v $HOME/.ssh:/host-ssh:ro` (not /root/.ssh — that directory
# is created + populated here).
if [ -d /host-ssh ]; then
    rm -rf /root/.ssh
    mkdir -p /root/.ssh
    cp -rL /host-ssh/. /root/.ssh/ 2>/dev/null || true
    chown -R root:root /root/.ssh
    chmod 700 /root/.ssh
    find /root/.ssh -type f -exec chmod 600 {} +
    find /root/.ssh -type f -name '*.pub' -exec chmod 644 {} + || true

    # The container runs as root but host SSH keys belong to the host user
    # (e.g. `sem`). eugr's launcher and other tools run `ssh <host>` without
    # a username and default to $USER=root — fail. Inject a User directive
    # into ssh_config keyed on the peer IPs from config.json so
    # `ssh 10.0.0.2` means `ssh sem@10.0.0.2`.
    if command -v python3 >/dev/null && [ -f "$AINODE_HOME/config.json" ]; then
        python3 - <<'PY'
import json, os, pathlib
cfg_path = pathlib.Path(os.environ.get("AINODE_HOME", "/root/.ainode")) / "config.json"
try:
    cfg = json.loads(cfg_path.read_text())
except Exception:
    cfg = {}
# The host's ssh config can name its keys by ABSOLUTE path — nvidia-sync
# writes e.g. "IdentityFile /home/admin/.ssh/id_ed25519_nvsync_cluster_assistant".
# The keys were copied to /root/.ssh above, and /home/admin does not exist in
# this container, so ssh reported
#
#   no such identity: /home/admin/.ssh/id_ed25519_...: No such file or directory
#   admin@10.100.36.2: Permission denied (publickey,password)
#
# and every transfer to a peer failed on authentication. Point such lines at
# the copy, by file name, leaving anything that does resolve alone.
ssh_config_path = pathlib.Path("/root/.ssh/config")
if ssh_config_path.exists():
    rewritten = []
    changed = False
    for line in ssh_config_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("identityfile"):
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                named = pathlib.Path(parts[1].strip().strip('"').strip("'"))
                if not named.expanduser().exists():
                    local = pathlib.Path("/root/.ssh") / named.name
                    if local.exists():
                        indent = line[: len(line) - len(line.lstrip())]
                        line = f"{indent}IdentityFile {local}"
                        changed = True
        rewritten.append(line)
    if changed:
        ssh_config_path.write_text("\n".join(rewritten) + "\n")
        ssh_config_path.chmod(0o600)
        print("ainode: pointed ssh IdentityFile entries at /root/.ssh")

ssh_user = cfg.get("ssh_user") or ""
if ssh_user:
    # Host * rather than the peer list. The peer list is empty in config.json
    # until a distributed launch has already SUCCEEDED, so keying on it made
    # the mapping appear only after it was no longer needed — and the first
    # distributed launch on a fresh install failed with eugr's
    # "Passwordless SSH to <peer> failed", because the launcher runs plain
    # `ssh <ip>` and this container is root while the keys belong to the
    # install user.
    #
    # The wildcard is safe here: nothing else ssh's out of this container, and
    # an explicit user in a command line (distribute.py sends user@host) wins
    # over this anyway.
    ssh_config = pathlib.Path("/root/.ssh/config")
    existing = ssh_config.read_text() if ssh_config.exists() else ""
    block = "\n# Injected by AINode entrypoint for container→peer ssh\n"
    block += "Host *\n"
    block += f"    User {ssh_user}\n"
    block += "    StrictHostKeyChecking no\n"
    block += "    UserKnownHostsFile /root/.ssh/known_hosts\n"
    ssh_config.write_text(block + existing)
    ssh_config.chmod(0o600)
PY
    fi
fi

exec ainode start --in-container "$@"
