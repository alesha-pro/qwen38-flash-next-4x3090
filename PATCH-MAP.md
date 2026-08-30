# Patch map

| Component | Owner | Integration rule |
|---|---|---|
| PLE layer and scale fix | integration base | Always mounted; never replace with KV copy |
| PLE worker and external source | integration base | Always mounted with manifest |
| gpu_worker warmup fix | integration base | Always mounted |
| multiproc executor PLE ordering | integration base | Always mounted |
| GDN/shared-expert compatibility | integration base | Always mounted |
| Split GDN/hyper model overlays | cyankiwi AWQ | Always mounted for AWQ 4x3090 v1 |
| shared-expert gate repair | cyankiwi AWQ | Build-time restore to BF16; mandatory |
| GDN `in_proj_a` / `in_proj_b` repair | cyankiwi AWQ | Build-time BF16 restore; mandatory |
| QSA dtype/scale plumbing | KV repo | Mount only for `KV_CACHE_DTYPE=fp8*` |
| QSA FP8 decode kernel | KV repo | Mount only for `KV_CACHE_DTYPE=fp8*` |
| QSA base hashes | KV repo | Verify against pinned image before GPU launch |
| Calibrated QSA scales | KV repo | Mandatory for FP8; no scale=1 fallback |

The combined launcher is locally owned by this integration repository. BF16
KV must not mount the QSA FP8 overlays, preserving a rollback control.
