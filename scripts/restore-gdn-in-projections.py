#!/usr/bin/env python3
"""Build the public v1 checkpoint by restoring GDN in_proj_a/b to BF16.

The preceding FP8 companion build quantizes these projections. They are part
of Gated Delta Net's recurrent state update, so the final validated build keeps
their original BF16 values. This script is deliberately model-specific and
fail-closed: for the pinned checkpoint it must restore exactly 72 tensors and
remove exactly 72 corresponding FP8 scale tensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPECTED_TENSORS = 72


def is_projection_weight(name: str) -> bool:
    return name.endswith(".linear_attn.in_proj_a.weight") or name.endswith(
        ".linear_attn.in_proj_b.weight")


def is_projection_scale(name: str) -> bool:
    return name.endswith(".linear_attn.in_proj_a.weight_scale") or name.endswith(
        ".linear_attn.in_proj_b.weight_scale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="BF16 thin-v1 checkpoint")
    parser.add_argument("base", type=Path, help="thin-v2-gate checkpoint")
    parser.add_argument("out", type=Path, help="destination v1 checkpoint")
    args = parser.parse_args()
    source, base, out = (args.source.resolve(), args.base.resolve(),
                         args.out.resolve())

    if out.exists():
        raise SystemExit(f"destination already exists: {out}")
    out.mkdir(parents=True)

    for file in base.iterdir():
        if file.suffix != ".safetensors" and file.name != "THIN-MANIFEST.json":
            shutil.copy2(file, out / file.name)

    restored_names: list[str] = []
    removed_scales: list[str] = []
    for shard in sorted(base.glob("model-*.safetensors")):
        source_shard = source / shard.name
        if not source_shard.is_file():
            raise SystemExit(f"missing BF16 source shard: {source_shard}")

        with safe_open(str(source_shard), framework="pt", device="cpu") as handle:
            bf16 = {name: handle.get_tensor(name) for name in handle.keys()
                    if is_projection_weight(name)}
        if any(tensor.dtype != torch.bfloat16 for tensor in bf16.values()):
            raise SystemExit(f"non-BF16 GDN source weight in {source_shard}")

        print(f"[{shard.name}] restoring GDN in_proj_a/b", flush=True)
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if is_projection_scale(name):
                    removed_scales.append(name)
                    continue
                if is_projection_weight(name):
                    if name not in bf16:
                        raise SystemExit(f"missing BF16 source tensor: {name}")
                    if tensor.dtype != torch.float8_e4m3fn:
                        raise SystemExit(
                            f"expected FP8 base tensor for {name}, got {tensor.dtype}")
                    tensors[name] = bf16[name]
                    restored_names.append(name)
                else:
                    tensors[name] = tensor
        save_file(tensors, str(out / shard.name), metadata={"format": "pt"})

    if len(restored_names) != EXPECTED_TENSORS:
        raise SystemExit(
            f"restored {len(restored_names)} tensors, expected {EXPECTED_TENSORS}")
    if len(removed_scales) != EXPECTED_TENSORS:
        raise SystemExit(
            f"removed {len(removed_scales)} scales, expected {EXPECTED_TENSORS}")

    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    indexed_scales = [name for name in weight_map if is_projection_scale(name)]
    if set(indexed_scales) != set(removed_scales):
        raise SystemExit("index scale set does not match rewritten shard scale set")
    for name in indexed_scales:
        del weight_map[name]
    index.setdefault("metadata", {})["qwen38_awq_4x3090_v1"] = {
        "restored_gdn_in_proj_a_b_bf16": len(restored_names),
        "removed_gdn_in_proj_a_b_scales": len(removed_scales),
    }
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")

    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    quantization = config["quantization_config"]
    targets = quantization["config_groups"]["group_1"]["targets"]
    old = "(in_proj_z|in_proj_b|in_proj_a|out_proj)"
    new = "(in_proj_z|out_proj)"
    replaced = 0
    updated_targets = []
    for target in targets:
        if old in target:
            updated_targets.append(target.replace(old, new))
            replaced += 1
        else:
            updated_targets.append(target)
    if replaced != 1:
        raise SystemExit(f"updated {replaced} FP8 target patterns, expected 1")
    quantization["config_groups"]["group_1"]["targets"] = updated_targets
    ignore = quantization.setdefault("ignore", [])
    for name in restored_names:
        module = name.removesuffix(".weight")
        if module not in ignore:
            ignore.append(module)
    quantization["ignore"] = sorted(ignore)
    config["quantization_config"] = quantization
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    manifest = {
        "format": "qwen38-awq-4x3090-v1",
        "base": str(base),
        "bf16_source": str(source),
        "restored_gdn_in_proj_a_b_bf16": sorted(restored_names),
        "removed_gdn_in_proj_a_b_scales": sorted(removed_scales),
    }
    (out / "GDN-BF16-RESTORE-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PASS: restored {len(restored_names)} GDN projections to BF16")


if __name__ == "__main__":
    main()
