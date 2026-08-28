# SPDX-License-Identifier: Apache-2.0
"""Reference E4M3 (float8_e4m3fn) byte decoder.

Pure bit-math decode of the 256 possible E4M3 byte patterns to float32,
mirroring CUDA float8_e4m3fn / torch.float8_e4m3fn semantics:

- 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits;
- exponent 0 -> subnormals (value = mant * 2^-9), signed zero at 0x00/0x80;
- exponent 15 + mantissa 7 -> NaN (0x7F, 0xFF); there is no infinity;
- max finite magnitude is 448 (0x7E/0xFE).

The same formula is what the Triton in-kernel decoder implements on sm_86
(no native FP8 load path), so this file is the CPU ground truth for the
kernel test in Phase 2.
"""

from __future__ import annotations


def decode_e4m3_scalar(byte: int) -> float:
    """Decode one E4M3 byte (0..255) to a Python float.

    NaN patterns (0x7F, 0xFF) return float('nan'); the sign of NaN is not
    preserved, matching torch.float8_e4m3fn -> float32 conversion.
    """
    if not 0 <= byte <= 255:
        raise ValueError(f"E4M3 input must be a byte, got {byte}")
    sign = byte >> 7
    exp = (byte >> 3) & 0xF
    mant = byte & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0:
        value = mant * 2.0**-9  # subnormal: 0.mant * 2^-6
    else:
        value = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    return -value if sign else value


def decode_e4m3_torch(uint8_tensor):
    """Vectorized bit-math decode, tensorized for torch (CPU or CUDA).

    Uses only integer shifts/masks plus float ldexp-style scaling, identical
    in structure to the Triton kernel branch. Input: uint8 tensor. Output:
    float32 tensor. NaN where the pattern is S.1111.111.
    """
    import torch

    b = uint8_tensor.to(torch.int32)
    sign = (b >> 7) & 0x1
    exp = (b >> 3) & 0xF
    mant = b & 0x7

    # mantissa with implicit leading 1 for normals; scale exponent folded in:
    #   subnormal: mant       * 2^-9
    #   normal:    (8 + mant) * 2^(exp - 10)
    is_sub = exp == 0
    mantissa = torch.where(is_sub, mant, mant + 8).to(torch.float32)
    exponent = torch.where(is_sub, torch.full_like(exp, -9), exp - 10)
    value = mantissa * torch.exp2(exponent.to(torch.float32))

    is_nan = (exp == 0xF) & (mant == 0x7)
    value = torch.where(is_nan, torch.full_like(value, float("nan")), value)
    return torch.where(sign.bool(), -value, value)
