# SPDX-License-Identifier: Apache-2.0
"""Profile sweep for the FP8 QSA kernel on sm_86 (tuning aid, not a gate)."""
import importlib.util
import sys

import torch

sys.path.insert(0, "/work")
from test_qsa_kernel_fp8 import build_case  # noqa: E402
from bench_qsa_fp8 import bench, load_module  # noqa: E402

patched = load_module("qsa_ops_patched", "/work_overlay/ops_qsa.py")

for rows in (32, 256):
    case = build_case(rows=rows, topk=2048, seq_lens=[65536, 32768], kv="fp8")
    print(f"--- rows={rows} ---")
    for bn, ts, w in [(16, 64, 4), (16, 32, 4), (16, 16, 4), (16, 8, 4),
                      (16, 4, 4), (16, 2, 4), (16, 1, 4), (16, 8, 8),
                      (32, 32, 4), (32, 16, 4), (32, 8, 4), (32, 8, 8),
                      (64, 8, 4), (64, 4, 4)]:
        try:
            t = bench(
                lambda: patched.qsa_sparse_paged_attention(
                    case["q"], case["k_u8"], case["v_u8"],
                    case["logical_indices"], case["block_table"],
                    case["token_to_req"],
                    k_scale=case["k_scale"], v_scale=case["v_scale"],
                    _profile_override=(bn, ts, w),
                ),
                warmup=20, iters=100,
            )
            print(f"block_n={bn:<3} splits={ts:<3} warps={w}: {t:.4f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"block_n={bn:<3} splits={ts:<3} warps={w}: FAILED {type(e).__name__}")
