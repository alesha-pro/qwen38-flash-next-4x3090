#!/usr/bin/env python3
"""Fix thin-v2: restore shared_expert_gate weights from v1 (BF16).

The build script FP8-quantized shared_expert_gate.weight but the runtime
module is ReplicatedLinear with quant_config=None (BF16). The FP8 bytes
are loaded as BF16 → garbage gate values.

This script rewrites the v2 shards with shared_expert_gate restored to
BF16 from the v1 checkpoint. All other tensors are unchanged.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("v1", type=Path, help="BF16 thin checkpoint")
    parser.add_argument("v2", type=Path, help="FP8 thin-v2 checkpoint")
    parser.add_argument("out", type=Path, help="destination thin-v2-fix checkpoint")
    args = parser.parse_args()
    v1, v2, out = args.v1.resolve(), args.v2.resolve(), args.out.resolve()

    if out.exists():
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)

    # Copy non-shard files from v2
    for f in v2.iterdir():
        if f.suffix != ".safetensors" and f.name != "THIN-MANIFEST.json":
            shutil.copy2(f, out / f.name)

    fixed = 0
    for shard_name in sorted(v2.glob("model-*.safetensors")):
        print(f"[{shard_name.name}] processing ...", flush=True)

        # Read v1 BF16 shared_expert_gate weights from the same shard
        v1_path = v1 / shard_name.name
        v1_gates = {}
        if v1_path.exists():
            with safe_open(str(v1_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "shared_expert_gate.weight" in key and "weight_scale" not in key:
                        v1_gates[key] = f.get_tensor(key)

        # Read v2 shard, fix, write
        with safe_open(str(shard_name), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            tensors = {}
            for name in keys:
                t = f.get_tensor(name)
                if name in v1_gates and t.dtype != torch.bfloat16:
                    # Replace FP8 with v1 BF16
                    tensors[name] = v1_gates[name]
                    fixed += 1
                elif "shared_expert_gate.weight_scale" in name:
                    # Skip orphaned weight_scale
                    continue
                else:
                    tensors[name] = t

        save_file(tensors, str(out / shard_name.name), metadata={"format": "pt"})
        del tensors
        print(f"[{shard_name.name}] saved ({len(keys)} tensors)", flush=True)

    # Update index: remove shared_expert_gate.weight_scale entries
    idx = json.loads((v2 / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    to_remove = [k for k in wm if "shared_expert_gate.weight_scale" in k]
    for k in to_remove:
        del wm[k]
    (out / "model.safetensors.index.json").write_text(
        json.dumps(idx, indent=2, sort_keys=True) + "\n")

    print(f"fixed {fixed} shared_expert_gate tensors")
    print(f"removed {len(to_remove)} weight_scale index entries")
    print(f"output: {out}")


if __name__ == "__main__":
    import torch
    main()
