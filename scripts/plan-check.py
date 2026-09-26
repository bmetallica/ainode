#!/usr/bin/env python3
"""Recompute a plan by hand, in both units. Imports no ainode.

Written to check the unit fix without restarting anything — the question was
asked while a download was running, and a service restart would have taken it
down. It reads /proc/meminfo, the checkpoint's own config.json and the bytes on
disk, applies the same reserves the planner does, and prints the result twice:
once comparing decimal weights against binary memory (what the planner did),
once with both decimal (what it does now).

Standalone on purpose. A verification that imports the code it is verifying
checks nothing.

  python3 scripts/plan-check.py <model-dir> [free-node1-GiB] [free-node2-GiB ...]

"free" is MemAvailable in GiB — the "available" column of `free -g`. With no
node figures it reads the local node.

"""
import json
import sys
from pathlib import Path

SYSTEM_RESERVE_GB = 4.0
GUARD_WARN_GIB = 8.0
ENGINE_OVERHEAD_GB = 2.5
COMM_GB = 0.5
TP_REPLICATION = 1.05
LEN_GRAN = 4096
GIB = 1024 ** 3

def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024 ** 2)
    return 0.0

def mem_total_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return float(line.split()[1]) / (1024 ** 2)
    return 0.0

def headroom(total_gb):
    return round(min(8.0, max(1.0, total_gb * 0.06)), 1)

def weights_bytes(directory):
    d = Path(directory)
    snap = d if (d / "config.json").is_file() else None
    if snap is None and (d / "snapshots").is_dir():
        for sub in sorted((d / "snapshots").iterdir()):
            if (sub / "config.json").is_file():
                snap = sub
                break
    snap = snap or d
    total = 0
    for f in snap.rglob("*"):
        if f.suffix in (".safetensors", ".bin", ".pt", ".gguf"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total, snap

def facts(snap):
    c = json.loads((snap / "config.json").read_text())
    t = c.get("text_config") or c
    def g(*names):
        return next((x[n] for x in (c, t) for n in names if n in x), 0)

    layers = g("num_hidden_layers")
    interval = g("full_attention_interval")
    types = g("layer_types")
    if isinstance(types, list):
        caching = sum(1 for x in types if "full" in str(x) or x == "attention")
    elif interval:
        caching = layers // interval
    else:
        caching = layers
    lora, rope = g("kv_lora_rank"), g("qk_rope_head_dim")
    kvh, hd = g("num_key_value_heads"), g("head_dim")
    if not lora and rope and kvh == 1 and hd >= 256:
        lora = hd                                  # DeepSeek-V4-artiges MLA
    dt = 2                                       # bf16/fp16 weights
    return dict(layers=layers, caching=caching, lora=lora, rope=rope,
                kvh=kvh, hd=hd, dtype_bytes=dt, heads=g("num_attention_heads"),
                ctx=g("max_position_embeddings"), experts=g("num_experts", "n_routed_experts", "num_local_experts"))

def kv_per_token(f, kv_bytes):
    if f["lora"]:
        return int(f["caching"] * (f["lora"] + f["rope"]) * kv_bytes)
    return int(2 * f["caching"] * f["kvh"] * f["hd"] * kv_bytes)

def plan(weights_gb, frees_gb, totals_gb, f, kv_bytes, concurrency):
    out = []
    for count in (1, len(frees_gb)):
        if count < 1 or count > len(frees_gb):
            continue
        if count > 1 and f["heads"] and f["heads"] % count:
            continue
        usable = [fr - SYSTEM_RESERVE_GB - max(0.0, GUARD_WARN_GIB * GIB / 1e9 - SYSTEM_RESERVE_GB)
                  - headroom(to) for fr, to in zip(frees_gb, totals_gb)][:count]
        per = weights_gb / count * (TP_REPLICATION if count > 1 else 1.0)
        over = ENGINE_OVERHEAD_GB + (COMM_GB if count > 1 else 0.0)
        room = min(usable) - per - over
        if room <= 0:
            out.append((count, None, None, None))
            continue
        kv_gb = room * count
        toks = int(kv_gb * 1e9 / kv_per_token(f, kv_bytes))
        want = min(f["ctx"], (toks // max(1, concurrency)) // LEN_GRAN * LEN_GRAN)
        out.append((count, per, kv_gb, (toks, want)))
    return out

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    wb, snap = weights_bytes(directory)
    f = facts(snap)
    frees_gib = [float(x) for x in sys.argv[2:]] or [mem_available_gib()]
    totals_gib = [mem_total_gib()] * len(frees_gib)
    print(f"Model    {snap}")
    print(f"  Weights   {wb/1e9:.1f} GB decimal   = {wb/GIB:.1f} GiB")
    print(f"  {f['layers']} layers, {f['caching']} of them caching"
          + (f", MLA latent {f['lora']}+{f['rope']}" if f['lora'] else
             f", {f['kvh']} KV heads x {f['hd']}")
          + f", {f['heads']} attention heads, context {f['ctx']:,}")
    print()
    for label, wgb, fgb, tgb in (
            ("BEFORE  (decimal weights against binary memory)", wb/1e9,
             [x for x in frees_gib], [x for x in totals_gib]),
            ("AFTER   (both decimal)", wb/1e9,
             [x*GIB/1e9 for x in frees_gib], [x*GIB/1e9 for x in totals_gib])):
        print(label)
        for kvname, kvb in (("bf16/auto", f["dtype_bytes"]), ("fp8", 1)):
            for count, per, kv_gb, res in plan(wgb, fgb, tgb, f, kvb, 2):
                if res is None:
                    print(f"  {kvname:9} {count} node: does not fit")
                    continue
                toks, want = res
                print(f"  {kvname:9} {count} node{'s' if count>1 else ' '}: "
                      f"{per:6.1f} GB/node, KV {kv_gb:6.1f} GB = {toks:>10,} tokens"
                      f"  -> context {want:>9,} for 2 sessions")
        print()
