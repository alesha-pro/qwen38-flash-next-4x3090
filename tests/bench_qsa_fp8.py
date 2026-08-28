# SPDX-License-Identifier: Apache-2.0
"""Phase 2 benchmark: QSA sparse kernel latency, BF16 vs FP8 E4M3, sm_86.

Boundary: GPU kernel benchmark, 1x RTX 3090, vendor image. CUDA-event timing,
50 warmup + 200 iterations per configuration. Shapes cover decode
(rows 1/8/32, topk 2048) and a prefill-like tile (rows 256, topk 2048).
"""

from __future__ import annotations

import importlib.util
import sys

import torch

sys.path.insert(0, "/work")
from test_qsa_kernel_fp8 import build_case  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, warmup=50, iters=200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main() -> int:
    vendor = load_module(
        "qsa_ops_vendor",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/"
        "qwen3_8_flash_next/nvidia/ops/qsa.py",
    )
    patched = load_module("qsa_ops_patched", "/work_overlay/ops_qsa.py")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} sm_{props.major}{props.minor}")
    print(f"{'shape':<28} {'bf16_ms':>9} {'fp8_ms':>9} {'fp8/bf16':>9}")

    for rows, topk, seq_lens in [
        (1, 2048, [65536, 32768]),
        (8, 2048, [65536, 32768]),
        (32, 2048, [65536, 32768]),
        (256, 2048, [65536, 32768]),
    ]:
        case = build_case(rows=rows, topk=topk, seq_lens=seq_lens, kv="fp8")
        args = (
            case["q"],
            case["logical_indices"],
            case["block_table"],
            case["token_to_req"],
        )
        t_bf16 = bench(
            lambda: vendor.qsa_sparse_paged_attention(
                case["q"], case["k_bf16"], case["v_bf16"], *args[1:]
            )
        )
        t_fp8 = bench(
            lambda: patched.qsa_sparse_paged_attention(
                case["q"], case["k_u8"], case["v_u8"], *args[1:],
                k_scale=case["k_scale"], v_scale=case["v_scale"],
            )
        )
        print(
            f"rows={rows:<4} topk={topk:<6} seqs={seq_lens[0]//1024}K"
            f"{'':<6} {t_bf16:9.4f} {t_fp8:9.4f} {t_fp8 / t_bf16:8.2f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
