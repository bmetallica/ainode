#!/usr/bin/env python3
"""Read the last launch banner and print the opencode limits that match it.

Written because three numbers disagreed and only one of them was the engine's:

    opencode.json   "context": 608512
    Load tab        Context length 180.2K tokens
    serve command   --max-model-len 684032

The banner is the only one that is ground truth — it is the argv vLLM was given.
This reads it, applies the same arithmetic clients/opencode.py applies, and
prints the block. Standalone: it imports no ainode, so it runs against a node
whose container has not been updated, and changes nothing while it does.

Pass vLLM's own KV-cache token count as a second argument and it will also cap
the window to what the cache can back PER SESSION at the launched concurrency —
which the generator does not yet do, and which is the failure that reads as a
model that keeps stopping rather than as a memory error:

    grep -hoE "GPU KV cache size: [0-9,]+ tokens" ~/.ainode/logs/*.log | tail -1

  python3 opencode-limits.py [logdatei] [kv-token-gesamt]

Ohne Argumente nimmt es das jüngste Log unter ~/.ainode/logs/.
Die KV-Token-Zahl ist optional: mit ihr wird zusätzlich geprüft, ob das Fenster
mal der Parallelität überhaupt in den Cache passt.
"""
import glob
import json
import os
import re
import sys

MAX_OUTPUT, MARGIN_FRACTION, MIN_MARGIN = 32768, 16, 4096


def limits_for(window):
    window = max(1024, int(window))
    output = min(MAX_OUTPUT, max(256, window // 4))
    margin = max(MIN_MARGIN, window // MARGIN_FRACTION)
    context = window - output - margin
    if context < window // 4:
        output = max(256, window // 4)
        margin = max(256, window // MARGIN_FRACTION)
        context = window - output - margin
    return {"context": max(512, context), "output": output}


def _number(text) -> int:
    """An integer out of whatever a human pasted: commas, spaces, underscores."""
    digits = "".join(c for c in str(text) if c.isdigit())
    return int(digits) if digits else 0


def flag(command, name, default=0):
    m = re.search(re.escape(name) + r"[= ]+(\d+)", command)
    return int(m.group(1)) if m else default


def main():
    path = (sys.argv[1] or None) if len(sys.argv) > 1 else None
    if not path:
        logs = glob.glob(os.path.expanduser("~/.ainode/logs/*.log"))
        if not logs:
            sys.exit("kein Log unter ~/.ainode/logs/")
        path = max(logs, key=os.path.getmtime)
    # vLLM prints the figure WITH thousands separators — "GPU KV cache size:
    # 1,109,643 tokens" — and the instruction to copy it is the instruction to
    # paste commas. Refusing them was a script that could not read its own
    # documented input.
    kv_total = _number(sys.argv[2]) if len(sys.argv) > 2 else 0

    banner = None
    model = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            if "] serve command:" in line:
                banner = line.strip()
            elif "===== launch " in line:
                m = re.search(r"===== launch (\S+) at", line)
                if m:
                    model = m.group(1)
    if banner is None:
        sys.exit(f"kein 'serve command' in {path} — falsches Log?")

    window = flag(banner, "--max-model-len")
    seqs = flag(banner, "--max-num-seqs", 1)
    served = re.search(r"--served-model-name\s+(\S+)", banner)
    name = served.group(1) if served else (model or "<model>")

    print(f"Log        {path}")
    print(f"Modell     {name}")
    print(f"gestartet  --max-model-len {window:,}  --max-num-seqs {seqs}")
    if not window:
        sys.exit("kein --max-model-len im Startbefehl — die Engine nutzt das "
                 "Modellmaximum, und das muss aus der config.json kommen.")

    limit = limits_for(window)
    note = ""
    if kv_total:
        per_session = kv_total // max(1, seqs)
        print(f"Cache      {kv_total:,} Token -> {per_session:,} pro Session "
              f"bei {seqs} gleichzeitigen")
        if per_session < window:
            limit = limits_for(per_session)
            note = (f"  # gekappt: der Cache traegt nur {per_session:,} Token "
                    f"pro Session, nicht {window:,}")
    print()
    print('    "models": {')
    print(f'      "{name}": {{')
    print(f'        "name": "{name.split("/")[-1]}",')
    print('        "tool_call": true,')
    print('        "reasoning": true,')
    print('        "attachment": false,')
    print(f'        "limit": {json.dumps(limit)}{note}')
    print("      }")
    print("    }")
    if note:
        print()
        print("Der Cache ist der engere Zwang, nicht das Fenster. Entweder so "
              "eintragen, oder\nmit weniger gleichzeitigen Anfragen neu laden.")


if __name__ == "__main__":
    main()
