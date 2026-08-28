# SPDX-License-Identifier: Apache-2.0
"""Debug: isolate FP8 kernel divergence from quantization error.

Runs one tiny case through (a) vendor BF16 kernel, (b) patched FP8 kernel,
(c) pure-torch emulation of FP8 attention from the uint8 cache bytes.
If (b) diverges from (c), the kernel branch is buggy; if (b)==(c) but both
diverge from (a), the quantization/scale setup is wrong.
"""

from __future__ import annotations

import importlib.util
import sys

import torch

sys.path.insert(0, "/work")
from e4m3 import decode_e4m3_torch  # noqa: E402
from test_qsa_kernel_fp8 import build_case  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def emulate_fp8_attention(case) -> torch.Tensor:
    q = case["q"].float()
    if case.get("use_bf16_cache"):
        k = case["k_bf16"].float().view(-1, 2, 256)
        v = case["v_bf16"].float().view(-1, 2, 256)
    else:
        k = (
            decode_e4m3_torch(case["k_u8"].view(-1, 2, 256)) * case["k_scale"]
        ).view(-1, 2, 256)
        v = (
            decode_e4m3_torch(case["v_u8"].view(-1, 2, 256)) * case["v_scale"]
        ).view(-1, 2, 256)
    k = k.view(-1, BLOCK := 16, 2, 256).reshape(-1, 2, 256)  # [tokens, kv, dim]
    out = torch.zeros_like(q)
    group = 8
    for i in range(q.shape[0]):
        req = case["token_to_req"][i].item()
        idx = case["logical_indices"][i]
        valid = idx >= 0
        toks = idx[valid].long()
        phys = case["block_table"][req][toks // BLOCK] * BLOCK + toks % BLOCK
        kk = k[phys.long()]  # [n, kv, dim]
        vv = v[phys.long()]
                # head h uses kv head h // 8
        kv_of_head = torch.arange(16, device=q.device) // 8
        kk_h = kk[:, kv_of_head, :]  # [n, 16, dim]
        vv_h = vv[:, kv_of_head, :]
        scores = torch.einsum("hd,nhd->hn", q[i], kk_h) * (256 ** -0.5)
        p = torch.softmax(scores, dim=-1)  # [16, n]
        out[i] = torch.einsum("hn,nhd->hd", p, vv_h)
    return out


def main() -> int:
    vendor = load_module(
        "qsa_ops_vendor",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/"
        "qwen3_8_flash_next/nvidia/ops/qsa.py",
    )
    patched = load_module("qsa_ops_patched", "/work_overlay/ops_qsa.py")
    case = build_case(rows=1, topk=128, seq_lens=[5000, 3000], kv="fp8")
    dec = decode_e4m3_torch(case["k_u8"].view(-1)) * case["k_scale"]
    orig = case["k_bf16"].float().view(-1)
    print(f"direct roundtrip: max_abs={(dec - orig).abs().max().item():.6f} "
          f"cos={torch.nn.functional.cosine_similarity(dec, orig, dim=0).item():.6f}")
    print(f"k_scale={case['k_scale'].item():.6f} v_scale={case['v_scale'].item():.6f}")
    print(f"k_u8 zeros fraction: {(case['k_u8'] == 0).float().mean().item():.4f}")
    z = (case["k_u8"].view(-1, 2, 256) == 0).float()
    print(f"zeros per kv-head: {z.mean(dim=(0, 2)).tolist()}")
    print(f"zeros per dim half: first128={z[:, :, :128].mean().item():.4f} "
          f"last128={z[:, :, 128:].mean().item():.4f}")
    print(f"zeros even tokens={z[0::2].mean().item():.4f} odd tokens={z[1::2].mean().item():.4f}")
    nz = case["k_u8"].view(-1, 2, 256)[0]
    print(f"token0 head0 nonzero: {(nz[0] != 0).sum().item()}/256 "
          f"head1: {(nz[1] != 0).sum().item()}/256")

    out_ref = vendor.qsa_sparse_paged_attention(
        case["q"], case["k_bf16"], case["v_bf16"],
        case["logical_indices"], case["block_table"], case["token_to_req"],
    ).float()
    out_fp8 = patched.qsa_sparse_paged_attention(
        case["q"], case["k_u8"], case["v_u8"],
        case["logical_indices"], case["block_table"], case["token_to_req"],
        k_scale=case["k_scale"], v_scale=case["v_scale"],
    ).float()
    out_emu = emulate_fp8_attention(case)
    case["use_bf16_cache"] = True
    out_emu_bf16 = emulate_fp8_attention(case)
    case["use_bf16_cache"] = False

    def cs(a, b):
        return torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item()

    print(f"bf16_emu   vs bf16_ref : cos={cs(out_emu_bf16, out_ref):.6f}")
    print(f"torch_emu  vs bf16_ref : cos={cs(out_emu, out_ref):.6f}")
    print(f"fp8_kernel vs torch_emu: cos={cs(out_fp8, out_emu):.6f}")
    print(f"ref norm={out_ref.norm():.4f} fp8 norm={out_fp8.norm():.4f} "
          f"emu norm={out_emu.norm():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
