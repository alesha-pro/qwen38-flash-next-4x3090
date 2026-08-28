# Patch map

| Component | Owner | Integration rule |
|---|---|---|
| PLE layer and scale fix | W4A16 repo | Always mounted; never replace with KV copy |
| PLE worker and external source | W4A16 repo | Always mounted with manifest |
| gpu_worker warmup fix | W4A16 repo | Always mounted |
| multiproc executor PLE ordering | W4A16 repo | Always mounted |
| GDN/shared-expert compatibility | W4A16 repo | Always mounted |
| QSA dtype/scale plumbing | KV repo | Mount only for `KV_CACHE_DTYPE=fp8*` |
| QSA FP8 decode kernel | KV repo | Mount only for `KV_CACHE_DTYPE=fp8*` |
| QSA base hashes | KV repo | Verify against pinned image before GPU launch |
| Calibrated QSA scales | KV repo | Mandatory for FP8; no scale=1 fallback |

The combined launcher is locally owned by this integration repository. BF16
KV must not mount the QSA FP8 overlays, preserving a rollback control.

