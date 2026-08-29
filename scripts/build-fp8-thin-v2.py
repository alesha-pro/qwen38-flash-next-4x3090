#!/usr/bin/env python3
"""Build thin-v2: FP8-quantize selected BF16 modules of the AWQ thin checkpoint.

Targets (all currently BF16, per-rank savings in parentheses):
  - hyper-connections (ReplicatedLinear, ~0.6 GiB/rank)
  - GDN linear_attn projections (sharded, ~0.48 GiB/rank)
  - shared experts + gates (sharded, ~0.055 GiB/rank)
  - lm_head (sharded, ~0.15 GiB/rank)

NOT touched: Vision tower, QSA self_attn, embeddings, MoE routed experts
(already INT4), norms, PLE, MTP.

Output: a new derived checkpoint directory; the v1 thin checkpoint is never
modified. Config gains a second compressed-tensors group with
format="float-quantized"; target modules are removed from `ignore`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FP8_MAX = 448.0  # float8_e4m3fn max normal


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr_bytes = f.read(n)
    return json.loads(hdr_bytes), 8 + n


def is_target(name: str) -> bool:
    # With the split-proj overlay, fused modules are split at runtime.
    # FP8 targets (companion parts):
    #   GDN: in_proj_z, in_proj_b, in_proj_a, out_proj
    #   Hyper: input_mix_weight_down, input_mix_weight_up
    #   MoE: shared_expert gate/up/down + gate
    # BF16 (quality-sensitive, not quantized):
    #   GDN: in_proj_qkv (attention projections)
    #   Hyper: block_inject_weight (tiny)
    #   lm_head (stage 1)
    if "hyper_connection" in name:
        if "block_inject" in name:
            return False
        return True
    if ".linear_attn." in name:
        if "in_proj_qkv" in name:
            return False
        if any(k in name for k in ("in_proj_z", "in_proj_b", "in_proj_a", "out_proj")):
            return True
        return False
    if ".mlp.shared_expert." in name or name.endswith("mlp.shared_expert_gate.weight"):
        return True
    if name == "lm_head.weight":
        return False  # stage 1: BF16
    return False


def quantize_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 E4M3 with a single float32 scale."""
    amax = w.abs().amax().item()
    if amax == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = torch.tensor(amax / FP8_MAX, dtype=torch.float32)
    q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="v1 thin checkpoint dir")
    parser.add_argument("destination", type=Path, help="v2 output dir (must not exist)")
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    if dst.exists():
        raise SystemExit(f"destination exists: {dst}")
    dst.mkdir(parents=True)

    # 1. Rewrite shards
    total_before = total_after = 0
    n_quantized = 0
    from safetensors import safe_open
    for shard in sorted(src.glob("model-*.safetensors")):
        print(f"[{shard.name}] processing ...", flush=True)
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for name in keys:
                t = f.get_tensor(name)
                if t.dtype == torch.bfloat16 and t.dim() == 2 and is_target(name):
                    q, scale = quantize_fp8(t)
                    out[name] = q
                    out[name.replace(".weight", ".weight_scale")] = scale
                    total_before += t.numel() * 2
                    total_after += q.numel() + 4
                    n_quantized += 1
                    del t
                else:
                    out[name] = t
        save_file(out, str(dst / shard.name), metadata={"format": "pt"})
        del out
        print(f"[{shard.name}] saved ({len(keys)} tensors)", flush=True)

    print(f"quantized {n_quantized} tensors: {total_before/2**30:.3f} GiB -> "
          f"{total_after/2**30:.3f} GiB (saved {(total_before-total_after)/2**30:.3f} GiB model-total)")

    # 2. Copy small files (index rewritten later)
    for f in ("config.json", "generation_config.json", "chat_template.jinja",
              "preprocessor_config.json", "video_preprocessor_config.json",
              "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        p = src / f
        if p.exists():
            shutil.copy2(p, dst / f)

    # 3. Rewrite index: add weight_scale entries for quantized tensors
    idx_path = src / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]
    new_wm: dict[str, str] = {}
    for name, shard in wm.items():
        new_wm[name] = shard
        if name.endswith(".weight") and is_target(name):
            new_wm[name.replace(".weight", ".weight_scale")] = shard
    idx["weight_map"] = new_wm
    idx.setdefault("metadata", {})
    idx["metadata"]["thin_checkpoint_v2_fp8_targets"] = n_quantized
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(idx, indent=2, sort_keys=True) + "\n")

    # 4. Rewrite config.json: add FP8 group, trim ignore list
    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg["quantization_config"]
    ignore = qcfg.get("ignore", [])
    new_ignore = []
    for pat in ignore:
        # Remove ignore entries ONLY for modules we actually FP8-quantize.
        # Keep: in_proj_qkv (BF16), block_inject (BF16), lm_head (stage 1 BF16),
        # all parents, norms, Vision, QSA self_attn, etc.
        if any(pat.endswith(s) for s in (
            ".linear_attn.in_proj_z",
            ".linear_attn.in_proj_b",
            ".linear_attn.in_proj_a",
            ".linear_attn.out_proj",
            ".hyper_connection.input_mix_weight_down",
            ".hyper_connection.input_mix_weight_up",
        )):
            continue
        if ".mlp.shared_expert." in pat or pat.endswith(".mlp.shared_expert_gate"):
            continue
        new_ignore.append(pat)
    qcfg["ignore"] = new_ignore
    qcfg["config_groups"]["group_1"] = {
        "format": "float-quantized",
        "weights": {
            "num_bits": 8, "type": "float", "symmetric": True,
            "strategy": "tensor", "dynamic": False,
            "observer": "minmax",
        },
        "input_activations": {
            "num_bits": 8, "type": "float", "symmetric": True,
            "strategy": "tensor", "dynamic": True,
            "observer": "minmax",
        },
        "targets": [
            "re:.*hyper_connection\\.(input_mix_weight_down|input_mix_weight_up)",
            "re:.*linear_attn\\.(in_proj_z|in_proj_b|in_proj_a|out_proj)",
            "re:.*mlp\\.shared_expert.*",
        ],
    }
    cfg["quantization_config"] = qcfg
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "quantized_tensors": n_quantized,
        "bytes_before": total_before,
        "bytes_after": total_after,
        "saved_gib_model_total": (total_before - total_after) / 2**30,
    }, indent=2))


if __name__ == "__main__":
    main()
