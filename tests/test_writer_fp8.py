# SPDX-License-Identifier: Apache-2.0
"""Phase 2 microtest: reshape_and_cache_flash FP8 E4M3 semantics on sm_86.

Boundary: GPU kernel test, 1x RTX 3090, vendor image, no model weights.

Proves or rejects, for the exact compiled writer in the pinned image:
1. layout: [num_blocks, block_size, num_kv_heads, head_size] uint8 cache,
   slot_mapping scatter, including cross-block slots;
2. scale direction: stored bytes represent x / k_scale (vLLM convention) and
   decode(byte) * k_scale reconstructs x;
3. saturation: |x / scale| > 448 clamps to the finite E4M3 max (no NaN/inf
   bytes produced from finite inputs);
4. roundtrip error distribution for realistic BF16 magnitudes;
5. the "fp8" string alias behaves identically to "fp8_e4m3".

Exits non-zero on any failed check so the Phase 3 reuse decision is gated.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/work")
from e4m3 import decode_e4m3_torch  # noqa: E402

from vllm._custom_ops import reshape_and_cache_flash  # noqa: E402

NUM_BLOCKS = 8
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 256  # QSA main-cache head dim
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_caches(dtype: torch.dtype):
    k = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype, device="cuda"
    )
    return k, torch.zeros_like(k)


def main() -> int:
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} sm_{props.major}{props.minor}")
    check("device is sm_86", (props.major, props.minor) == (8, 6))

    # --- 1. scale direction + layout with non-unit scales ----------------
    k_scale = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")

    key_cache, value_cache = make_caches(torch.uint8)
    num_tokens = 20  # crosses the block boundary at slot 16
    # Slots deliberately non-monotonic: 0..9 -> block 0, 10..19 -> block 3.
    slot_mapping = torch.cat(
        [torch.arange(0, 10), 3 * BLOCK_SIZE + torch.arange(0, 10)]
    ).to(device="cuda", dtype=torch.long)

    key = torch.randn(
        num_tokens, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    value = torch.randn_like(key)

    reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "fp8_e4m3", k_scale, v_scale
    )
    torch.cuda.synchronize()

    flat_k = key_cache.view(-1, HEAD_SIZE)
    flat_v = value_cache.view(-1, HEAD_SIZE)
    dec_k = decode_e4m3_torch(flat_k[slot_mapping])  # [tokens, kv*head]? no:
    # cache rows are [slot, kv_head, head]; flatten only block*token axis.
    dec_k = decode_e4m3_torch(
        key_cache.view(NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)[slot_mapping]
    )
    dec_v = decode_e4m3_torch(
        value_cache.view(NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)[
            slot_mapping
        ]
    )

    recon_div = dec_k * k_scale  # assume stored = x / scale
    recon_mul = dec_k / k_scale  # assume stored = x * scale
    err_div = (recon_div - key.float()).abs().max().item()
    err_mul = (recon_mul - key.float()).abs().max().item()
    print(f"max abs err if stored=x/scale: {err_div:.6f}")
    print(f"max abs err if stored=x*scale: {err_mul:.6f}")
    check("scale direction is stored=x/scale", err_div < err_mul)
    direction_div = err_div < err_mul
    scale_dir = "x/scale" if direction_div else "x*scale"

    # v scale independently (0.5): decode must use v_scale, not k_scale.
    recon_v = dec_v * (v_scale if direction_div else 1.0 / v_scale)
    err_v = (recon_v - value.float()).abs().max().item()
    print(f"value roundtrip max abs err with v_scale=0.5: {err_v:.6f}")
    check("v_scale applied independently", err_v < 0.2)

    # untouched slots must stay zero-initialized
    untouched = torch.ones(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.bool, device="cuda")
    untouched[slot_mapping] = False
    check(
        "untouched slots unchanged",
        key_cache.view(NUM_BLOCKS * BLOCK_SIZE, -1)[untouched].abs().sum().item()
        == 0.0,
    )

    # --- 2. saturation: finite inputs beyond +/-448*scale ----------------
    key_cache2, value_cache2 = make_caches(torch.uint8)
    big = torch.full(
        (4, NUM_KV_HEADS, HEAD_SIZE), 1e30, dtype=torch.bfloat16, device="cuda"
    )
    big[1] *= -1
    big[2] = 449.0 * k_scale  # just past the finite max
    big[3] = -449.0 * k_scale
    slots2 = torch.arange(4, device="cuda", dtype=torch.long)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    reshape_and_cache_flash(
        big, big, key_cache2, value_cache2, slots2, "fp8_e4m3", one, one
    )
    torch.cuda.synchronize()
    bytes2 = key_cache2.view(NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)[slots2]
    dec2 = decode_e4m3_torch(bytes2)
    nan_count = torch.isnan(dec2).sum().item()
    print(f"saturation rows decode: max={dec2.max().item()} min={dec2.min().item()} "
          f"nan_count={nan_count}")
    check("no NaN bytes from finite inputs", nan_count == 0)
    check(
        "clamped to finite E4M3 max 448",
        dec2.abs().max().item() == 448.0,
        f"observed |max|={dec2.abs().max().item()}",
    )

    # --- 3. roundtrip error distribution, realistic magnitudes -----------
    key_cache3, value_cache3 = make_caches(torch.uint8)
    x = torch.randn(
        64, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
    ) * 3.0  # sigma 3, typical pre-scale KV magnitudes
    slots3 = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device="cuda")[:64].long()
    reshape_and_cache_flash(
        x, x, key_cache3, value_cache3, slots3, "fp8_e4m3", one, one
    )
    torch.cuda.synchronize()
    dec3 = decode_e4m3_torch(
        key_cache3.view(NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)[slots3]
    )
    err = (dec3 - x.float()).abs()
    denom = x.float().abs().clamp_min(1e-3)
    rel = err / denom
    print(
        f"roundtrip scale=1: max_abs={err.max().item():.6f} "
        f"mean_abs={err.mean().item():.6f} "
        f"mean_rel={rel.mean().item():.4f} p99_rel={rel.quantile(0.99).item():.4f}"
    )
    # E4M3 round-to-nearest error bound is half-ulp; for |x| in [8,16) the ulp
    # is 1, so 0.5 is exactly the theoretical maximum, not a defect.
    check(
        "roundtrip within half-ulp bound at scale=1",
        err.max().item() <= 0.5 + 1e-6 and rel.mean() < 0.05,
    )

    # --- 4. "fp8" alias equals "fp8_e4m3" ---------------------------------
    key_cache4, value_cache4 = make_caches(torch.uint8)
    reshape_and_cache_flash(
        x, x, key_cache4, value_cache4, slots3, "fp8", one, one
    )
    torch.cuda.synchronize()
    check(
        '"fp8" alias produces identical bytes',
        torch.equal(key_cache3, key_cache4),
    )

    print(f"scale direction in use: stored = {scale_dir}")
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS (writer reusable for QSA FP8 cache)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
