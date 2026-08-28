# SPDX-License-Identifier: Apache-2.0
"""Exhaustive E4M3 decode verification: all 256 byte patterns.

Boundary: CPU unit test inside the vendor image (no GPU required).
Compares the bit-math decoders in tests/e4m3.py against PyTorch's own
float8_e4m3fn conversion. Any mismatch aborts with a full diff table.

Named cases checked explicitly per VALIDATION.md:
- signed zero (0x00, 0x80);
- subnormals (exponent 0, mantissa 1..7 and signed);
- max finite (0x7E = +448, 0xFE = -448);
- NaN encodings (0x7F, 0xFF).
"""

from __future__ import annotations

import sys

import torch

from e4m3 import decode_e4m3_scalar, decode_e4m3_torch

NAMED_CASES = {
    0x00: "+0.0 (zero)",
    0x80: "-0.0 (signed zero)",
    0x01: "+2^-9 (min subnormal)",
    0x07: "+7*2^-9 (max subnormal)",
    0x08: "+2^-6 (min normal)",
    0x7E: "+448 (max finite)",
    0xFE: "-448 (min finite)",
    0x7F: "NaN encoding",
    0xFF: "NaN encoding (negative payload slot)",
}


def main() -> int:
    patterns = torch.arange(256, dtype=torch.uint8)
    expected = patterns.view(torch.float8_e4m3fn).to(torch.float32)
    got_vec = decode_e4m3_torch(patterns)

    failures: list[str] = []

    # Vectorized bit-math decode vs torch, all 256 patterns.
    for i in range(256):
        e = expected[i].item()
        g = got_vec[i].item()
        same = (e != e and g != g) or e == g  # NaN == NaN for this purpose
        if not same:
            failures.append(f"byte 0x{i:02X}: torch={e!r} decode_e4m3_torch={g!r}")

    # Scalar decoder vs torch, all 256 patterns.
    for i in range(256):
        e = expected[i].item()
        g = decode_e4m3_scalar(i)
        same = (e != e and g != g) or e == g
        if not same:
            failures.append(f"byte 0x{i:02X}: torch={e!r} decode_e4m3_scalar={g!r}")

    print("== exhaustive E4M3 decode: 256/256 patterns checked ==")
    print("== named cases ==")
    for byte, label in NAMED_CASES.items():
        e = expected[byte].item()
        g = decode_e4m3_scalar(byte)
        status = "OK" if ((e != e and g != g) or e == g) else "MISMATCH"
        print(f"0x{byte:02X} {label}: torch={e!r} scalar={g!r} [{status}]")

    # Sign bit of zero must survive (torch keeps -0.0).
    neg_zero = expected[0x80]
    assert neg_zero.item() == 0.0 and torch.signbit(neg_zero).item(), (
        "torch lost the sign of 0x80; decoder contract must keep -0.0"
    )
    g = decode_e4m3_scalar(0x80)
    assert g == 0.0 and str(g).startswith("-"), "scalar decoder lost -0.0"
    gv = decode_e4m3_torch(patterns)[0x80]
    assert gv.item() == 0.0 and torch.signbit(gv).item(), (
        "vectorized decoder lost -0.0"
    )
    print("signed-zero sign preservation: OK")

    if failures:
        print(f"== FAILURES: {len(failures)} ==")
        for line in failures:
            print(line)
        return 1
    print("RESULT: PASS (bit-exact over all 256 byte patterns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
