# SPDX-License-Identifier: Apache-2.0
"""Minimal repro sweep for reshape_and_cache_flash corruption at scale."""
import sys

import torch

sys.path.insert(0, "/work")
from e4m3 import decode_e4m3_torch  # noqa: E402

from vllm._custom_ops import reshape_and_cache_flash  # noqa: E402

KV, DIM = 2, 256
BS = 16

for num_blocks, ntok, scale_val in [
    (8, 20, 2.0),
    (8, 128, 0.012),
    (501, 8016, 2.0),
    (501, 8016, 0.012),
    (64, 1024, 0.012),
    (64, 512, 0.012),
]:
    k8 = torch.zeros(num_blocks, BS, KV, DIM, dtype=torch.uint8, device="cuda")
    v8 = torch.zeros_like(k8)
    x = torch.randn(ntok, KV, DIM, dtype=torch.bfloat16, device="cuda")
    s = torch.tensor(scale_val, dtype=torch.float32, device="cuda")
    slots = torch.arange(ntok, device="cuda", dtype=torch.long)
    reshape_and_cache_flash(x, x, k8, v8, slots, "fp8_e4m3", s, s)
    torch.cuda.synchronize()
    dec = decode_e4m3_torch(k8.view(-1, KV, DIM)[slots]) * scale_val
    orig = x.float()
    cos = torch.nn.functional.cosine_similarity(
        dec.flatten(), orig.flatten(), dim=0
    ).item()
    zf = (k8.view(-1, KV, DIM)[slots] == 0).float().mean().item()
    print(f"blocks={num_blocks} ntok={ntok} scale={scale_val}: cos={cos:.6f} "
          f"zeros_in_written_region={zf:.4f}")
