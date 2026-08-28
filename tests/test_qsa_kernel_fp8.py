# SPDX-License-Identifier: Apache-2.0
"""Phase 2 microtest: QSA sparse kernel, BF16 baseline vs FP8 E4M3 branch.

Boundary: GPU kernel test, 1x RTX 3090, vendor image, no model weights.

Compares the patched kernel (overlays/qwen38/ops_qsa.py, mounted at
/work_overlay/ops_qsa.py) against the pristine vendor kernel loaded from the
installed package. Cases:

1. random K/V, multi-split dispatch (rows=1, topk=2048 -> 64 splits);
2. random K/V, single-split dispatch (topk=16 -> num_tiles=1 -> 1 split);
3. adversarial magnitudes: +-448*scale outliers, zeros, -0.0, subnormals,
   half-ulp boundary values;
4. invalid (-1) index tails and a fully-invalid row;
5. regression: patched module BF16 path must be bitwise-equal to vendor.

Reports cosine similarity, max abs error, max rel error per case and checks
for NaNs. Gate: cosine >= 0.999 and no unexplained NaNs (VALIDATION.md).
"""

from __future__ import annotations

import importlib.util
import sys

import torch

sys.path.insert(0, "/work")
from e4m3 import decode_e4m3_torch  # noqa: E402

from vllm._custom_ops import reshape_and_cache_flash  # noqa: E402

KV_HEADS = 2
HEAD_DIM = 256
Q_HEADS = 16  # group size 8, matching the model's 32 q heads at TP4
BLOCK_SIZE = 16
FAILURES: list[str] = []


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    fa, fb = a.float().flatten(1), b.float().flatten(1)
    zero_rows = fa.norm(dim=1) == 0
    if zero_rows.any():
        # Fully-invalid selections produce zero output in both runs; require
        # the FP8 run to also be (near-)zero there instead of cos(0, 0) = 0.
        if fb[zero_rows].abs().max().item() > 1e-3:
            return 0.0
        fa, fb = fa[~zero_rows], fb[~zero_rows]
    return torch.nn.functional.cosine_similarity(fa, fb, dim=-1).min().item()


def build_case(
    rows: int,
    topk: int,
    seq_lens: list[int],
    kv: str,
    adversarial: bool = False,
):
    """Build paged caches + selection for `rows` queries over `seq_lens`."""
    num_reqs = len(seq_lens)
    pages_per_req = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens]
    num_blocks = sum(pages_per_req)
    g = torch.Generator(device="cuda").manual_seed(1234 + rows + topk)

    # Physical pages shuffled so block tables are non-trivial.
    perm = torch.randperm(num_blocks, device="cuda", generator=g)
    block_table = torch.zeros(num_reqs, max(pages_per_req), dtype=torch.int32, device="cuda")
    pos = 0
    for r in range(num_reqs):
        block_table[r, : pages_per_req[r]] = perm[pos : pos + pages_per_req[r]].to(torch.int32)
        pos += pages_per_req[r]

    k_bf16 = torch.zeros(
        num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_bf16 = torch.zeros_like(k_bf16)
    flat = num_blocks * BLOCK_SIZE
    k_bf16.view(flat, KV_HEADS, HEAD_DIM).normal_(generator=g)
    v_bf16.view(flat, KV_HEADS, HEAD_DIM).normal_(generator=g)

    # Static per-tensor scales, percentile-based like real calibration.
    if adversarial:
        k_scale = (
            k_bf16.float().abs().flatten().quantile(0.999) / 448.0
        ).clamp_min(1e-8)
        v_scale = (
            v_bf16.float().abs().flatten().quantile(0.999) / 448.0
        ).clamp_min(1e-8)
        # Boundary patterns relative to the calibrated range: just inside and
        # just past saturation, zeros, -0.0, subnormal-magnitude and half-ulp
        # values. Deliberately no 1e30 outliers: with per-tensor scales a
        # single extreme value underflows every normal element to zero, which
        # is a calibration property, not a kernel behavior.
        k_flat = k_bf16.view(flat, KV_HEADS, HEAD_DIM)
        v_flat = v_bf16.view(flat, KV_HEADS, HEAD_DIM)
        k_adv = torch.tensor(
            [448.0, -448.0, 672.0, -672.0, 0.0, -0.0, 2**-9, 2**-10],
            device="cuda",
        ) * k_scale
        v_adv = torch.tensor(
            [447.9, -447.9, 672.0, -672.0, 0.0, -0.0, 2**-9, 2**-10],
            device="cuda",
        ) * v_scale
        k_flat[0, 0, :8] = k_adv.bfloat16()
        v_flat[1, 1, :8] = v_adv.bfloat16()
    else:
        k_scale = (k_bf16.float().abs().max() / 448.0).clamp_min(1e-8)
        v_scale = (v_bf16.float().abs().max() / 448.0).clamp_min(1e-8)
    k_scale = k_scale.float()
    v_scale = v_scale.float()

    # Write FP8 caches with the proven vendor writer.
    k_u8 = torch.zeros_like(k_bf16, dtype=torch.uint8)
    v_u8 = torch.zeros_like(v_bf16, dtype=torch.uint8)
    slots = torch.arange(flat, device="cuda", dtype=torch.long)
    # NOTE: the writer requires the 4-D [blocks, block_size, kv, dim] cache
    # layout; passing a flattened 3-D view silently corrupts the write.
    reshape_and_cache_flash(
        k_bf16.view(flat, KV_HEADS, HEAD_DIM),
        v_bf16.view(flat, KV_HEADS, HEAD_DIM),
        k_u8,
        v_u8,
        slots,
        "fp8_e4m3",
        k_scale.reshape(()),
        v_scale.reshape(()),
    )
    torch.cuda.synchronize()

    # Queries and top-k selection rows.
    q = torch.randn(rows, Q_HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
    token_to_req = torch.randint(0, num_reqs, (rows,), generator=g, device="cuda").int()
    logical_indices = torch.zeros(rows, topk, dtype=torch.int32)
    for i in range(rows):
        seq_len = seq_lens[token_to_req[i].item()]
        take = min(topk, seq_len)
        idx = torch.randperm(seq_len, generator=g, device="cuda")[:take].cpu()
        logical_indices[i, :take] = idx.int()
        logical_indices[i, take:] = -1  # invalid tail
    if rows >= 3:
        logical_indices[2, :] = -1  # fully invalid row
    logical_indices = logical_indices.cuda()

    return {
        "q": q,
        "k_bf16": k_bf16,
        "v_bf16": v_bf16,
        "k_u8": k_u8,
        "v_u8": v_u8,
        "k_scale": k_scale.reshape(()),
        "v_scale": v_scale.reshape(()),
        "logical_indices": logical_indices,
        "block_table": block_table,
        "token_to_req": token_to_req,
    }


def emulate_fp8_attention(case) -> torch.Tensor:
    """Quantization-aware reference: attention over the dequantized cache."""
    q = case["q"].float()
    k = decode_e4m3_torch(case["k_u8"].view(-1, KV_HEADS, HEAD_DIM))
    v = decode_e4m3_torch(case["v_u8"].view(-1, KV_HEADS, HEAD_DIM))
    k = (k * case["k_scale"]).view(-1, KV_HEADS, HEAD_DIM)
    v = (v * case["v_scale"]).view(-1, KV_HEADS, HEAD_DIM)
    group = Q_HEADS // KV_HEADS
    out = torch.zeros_like(q)
    for i in range(q.shape[0]):
        req = case["token_to_req"][i].item()
        idx = case["logical_indices"][i]
        toks = idx[idx >= 0].long()
        phys = (
            case["block_table"][req][toks // BLOCK_SIZE] * BLOCK_SIZE
            + toks % BLOCK_SIZE
        ).long()
        kk, vv = k[phys], v[phys]  # [n, kv, dim]
        kv_of_head = torch.arange(Q_HEADS, device=q.device) // group
        scores = torch.einsum("hd,nhd->hn", q[i], kk[:, kv_of_head])
        p = torch.softmax(scores * (HEAD_DIM**-0.5), dim=-1)
        out[i] = torch.einsum("hn,nhd->hd", p, vv[:, kv_of_head])
    return out


def run_case(vendor, patched, name: str, case: dict, vs_emu: bool = False) -> None:
    out_ref = vendor.qsa_sparse_paged_attention(
        case["q"],
        case["k_bf16"],
        case["v_bf16"],
        case["logical_indices"],
        case["block_table"],
        case["token_to_req"],
    )
    out_fp8 = patched.qsa_sparse_paged_attention(
        case["q"],
        case["k_u8"],
        case["v_u8"],
        case["logical_indices"],
        case["block_table"],
        case["token_to_req"],
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
    )
    torch.cuda.synchronize()
    ref, got = out_ref.float(), out_fp8.float()
    nan_ref = torch.isnan(ref).sum().item()
    nan_got = torch.isnan(got).sum().item()
    diff = (got - ref).abs()
    denom = ref.abs().max()
    cos = cosine(ref, got)
    print(
        f"case {name}: cosine_vs_bf16={cos:.6f} max_abs={diff.max().item():.6f} "
        f"rel_to_max={diff.max().item() / denom.item():.4f} "
        f"nan_ref={nan_ref} nan_fp8={nan_got}"
    )
    check(f"{name}: no NaNs", nan_ref == 0 and nan_got == 0)
    if vs_emu:
        # Saturating/adversarial content: gate against the quantization-aware
        # reference (must be near-exact); report BF16 delta for information.
        emu = emulate_fp8_attention(case)
        cos_emu = cosine(emu, got)
        print(f"case {name}: cosine_vs_emu={cos_emu:.6f}")
        check(
            f"{name}: cosine vs quantized reference >= 0.9999",
            cos_emu >= 0.9999,
            f"cosine={cos_emu:.6f}",
        )
        check(f"{name}: cosine vs bf16 >= 0.95", cos >= 0.95, f"cosine={cos:.6f}")
    else:
        check(f"{name}: cosine >= 0.999", cos >= 0.999, f"cosine={cos:.6f}")


def main() -> int:
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} sm_{props.major}{props.minor}")

    vendor = load_module(
        "qsa_ops_vendor",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/"
        "qwen3_8_flash_next/nvidia/ops/qsa.py",
    )
    patched = load_module("qsa_ops_patched", "/work_overlay/ops_qsa.py")

    # Regression: patched BF16 path must equal vendor bitwise.
    case = build_case(rows=4, topk=256, seq_lens=[5000, 3000], kv="bf16")
    out_vendor = vendor.qsa_sparse_paged_attention(
        case["q"], case["k_bf16"], case["v_bf16"],
        case["logical_indices"], case["block_table"], case["token_to_req"],
    )
    out_patched_bf16 = patched.qsa_sparse_paged_attention(
        case["q"], case["k_bf16"], case["v_bf16"],
        case["logical_indices"], case["block_table"], case["token_to_req"],
    )
    torch.cuda.synchronize()
    check(
        "patched BF16 path bitwise-equal to vendor",
        torch.equal(out_vendor, out_patched_bf16),
    )

    # 1. multi-split dispatch (rows*kv_heads=2 <= 4 -> target 64 splits)
    run_case(
        vendor, patched, "multi-split topk=2048",
        build_case(rows=1, topk=2048, seq_lens=[5000, 3000], kv="fp8"),
    )
    # 2. single-split dispatch (topk=16 -> num_tiles=1 -> num_splits=1)
    run_case(
        vendor, patched, "single-split topk=16",
        build_case(rows=1, topk=16, seq_lens=[5000, 3000], kv="fp8"),
    )
    # 3. mid-profile (rows=64 -> base_programs=128 -> 64-wide tiles, 8 splits)
    run_case(
        vendor, patched, "batch64 topk=2048",
        build_case(rows=64, topk=2048, seq_lens=[5000, 3000], kv="fp8"),
    )
    # 4. adversarial magnitudes
    run_case(
        vendor, patched, "adversarial topk=2048",
        build_case(rows=4, topk=2048, seq_lens=[5000, 3000], kv="fp8",
                   adversarial=True),
        vs_emu=True,
    )
    # 5. large-batch single-split (>512 programs -> target 1 split)
    run_case(
        vendor, patched, "batch512 topk=512",
        build_case(rows=512, topk=512, seq_lens=[5000, 3000], kv="fp8"),
    )

    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS (FP8 kernel matches BF16 within E4M3 tolerance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
