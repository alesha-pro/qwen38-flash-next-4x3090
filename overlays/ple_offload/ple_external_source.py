# SPDX-License-Identifier: Apache-2.0
"""Fail-closed external PLE source resolution for the PLE offload worker.

Mounted into the vLLM image alongside the patched
``vllm/v1/ple_offload/worker.py``. Stdlib-only on purpose: it must run before
any model code and must fail closed on any inconsistency.

Environment:
  VLLM_PLE_MODEL_PATH   -- secondary checkpoint dir providing PLE tensors
                           (e.g. /ple-model with the RadixArk FP8 PLE)
  VLLM_PLE_MANIFEST     -- required manifest JSON pinning that source
  PLE_MANIFEST_FULL_HASH=1 -- audit mode: hash whole ngram payloads

Source of truth for the validation logic:
  /mnt/nvme2/projects/qwen38-quantize/scripts/ple_manifest.py
Keep both copies in sync.
"""
import copy
import hashlib
import json
import os
import struct

SCHEMA = "qwen38.external_ple.v1"
WINDOW = 16 * 1024 * 1024


class ExternalPleError(RuntimeError):
    """Raised when the external PLE source fails validation."""


def _read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(hlen)), 8 + hlen


def _find_revision(root):
    tdir = os.path.join(root, ".cache", "huggingface", "trees")
    if not os.path.isdir(tdir):
        return None
    fs = sorted(f for f in os.listdir(tdir) if f.endswith(".json"))
    return fs[-1][:-5] if fs else None


def _bf16_scalar(raw):
    bits = struct.unpack("<H", raw)[0]
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _payload(path, data_start, a, b):
    with open(path, "rb") as f:
        f.seek(data_start + a)
        return f.read(b - a)


def _windowed_hash(path, data_start, a, b):
    size = b - a
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(data_start + a)
        if size < 3 * WINDOW:  # tiny payload: full hash
            h.update(f.read(size))
        else:
            h.update(f.read(WINDOW))
            f.seek(data_start + a + size // 2 - WINDOW // 2)
            h.update(f.read(WINDOW))
            f.seek(data_start + b - WINDOW)
            h.update(f.read(WINDOW))
    return h.hexdigest()


def _collect_ple(root):
    idx_path = os.path.join(root, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    tensors = {}
    by_file = {}
    for n, s in wm.items():
        if ".ple." in n:
            by_file.setdefault(s, []).append(n)
    for s, names in sorted(by_file.items()):
        p = os.path.join(root, s)
        if not os.path.isfile(p):
            raise ExternalPleError(f"missing PLE file {s}")
        header, ds = read_header = _read_header(p)
        for n in names:
            e = header[n]
            tensors[n] = {"dtype": e["dtype"], "shape": e["shape"],
                          "file": s, "data_offsets": e["data_offsets"],
                          "_ds": ds}
    return tensors


def validate_external_ple(root, manifest_path, full_hash=False, logger=None):
    """Validate the external PLE source; raise ExternalPleError on failure."""
    log = logger.info if logger is not None else print
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        raise ExternalPleError(
            f"cannot read PLE manifest {manifest_path}: {e}") from e
    if manifest.get("schema") != SCHEMA:
        raise ExternalPleError(
            f"unknown PLE manifest schema {manifest.get('schema')!r}")
    violations = []
    rev = _find_revision(root)
    if rev != manifest.get("revision"):
        violations.append(
            f"revision mismatch: manifest={manifest.get('revision')} "
            f"on-disk={rev}")
    for fn, size in manifest.get("files", {}).items():
        p = os.path.join(root, fn)
        if not os.path.isfile(p):
            violations.append(f"missing file {fn}")
        elif os.path.getsize(p) != size:
            violations.append(f"size mismatch {fn}")
    tensors = _collect_ple(root)
    mt = manifest.get("tensors", {})
    if set(tensors) != set(mt):
        violations.append("PLE tensor set mismatch")
    else:
        for n, t in tensors.items():
            m = mt[n]
            if (t["dtype"], t["shape"], t["file"]) != (
                    m["dtype"], m["shape"], m["file"]):
                violations.append(f"tensor mismatch {n}")
    ms = manifest.get("scale")
    if not ms:
        violations.append("manifest has no scale entry")
    elif ms["name"] in tensors:
        t = tensors[ms["name"]]
        raw = _payload(os.path.join(root, t["file"]),
                       t["_ds"], *t["data_offsets"])
        if hashlib.sha256(raw).hexdigest() != ms["sha256"]:
            violations.append("scale payload hash mismatch")
        elif _bf16_scalar(raw) != ms["value"]:
            violations.append("scale value mismatch")
    for n, want in manifest.get("metadata_sha256", {}).items():
        t = tensors.get(n)
        if t is None:
            violations.append(f"metadata tensor {n} not present")
            continue
        raw = _payload(os.path.join(root, t["file"]), t["_ds"], *t["data_offsets"])
        if hashlib.sha256(raw).hexdigest() != want:
            violations.append(f"metadata hash mismatch {n}")
    for sid, want in manifest.get("ngram_window_sha256", {}).items():
        n = (f"model.language_model.layers.1.ple.ple_embedding"
             f".ngram_embedding.shard_{sid}.weight")
        t = tensors.get(n)
        if t is None:
            violations.append(f"ngram shard {sid} not present")
            continue
        p = os.path.join(root, t["file"])
        a, b = t["data_offsets"]
        got = (hashlib.sha256(_payload(p, t["_ds"], a, b)).hexdigest() if full_hash
               else _windowed_hash(p, t["_ds"], a, b))
        if got != want:
            violations.append(f"ngram shard {sid} hash mismatch")
    if violations:
        raise ExternalPleError(
            "external PLE source failed validation: "
            + "; ".join(violations[:20]))
    log("PleOffload: external PLE source %s validated against manifest "
        "(revision=%s, %d tensors, full_hash=%s)",
        root, rev, len(tensors), full_hash)
    return manifest


def resolve_external_ple_model_config(model_config, logger=None):
    """Return a model_config reading weights from VLLM_PLE_MODEL_PATH.

    Fail-closed: requires VLLM_PLE_MANIFEST, validates the source, then
    returns a shallow copy of model_config with only `.model` replaced so the
    loader streams PLE tensors from the secondary checkpoint while every
    other config attribute (hf_config, dtype, quantization) still describes
    the primary model.
    """
    ple_path = os.environ.get("VLLM_PLE_MODEL_PATH", "").strip()
    if not ple_path:
        return model_config
    manifest_path = os.environ.get("VLLM_PLE_MANIFEST", "").strip()
    if not manifest_path:
        raise ExternalPleError(
            "VLLM_PLE_MODEL_PATH is set but VLLM_PLE_MANIFEST is not; "
            "refusing to load an unpinned external PLE source")
    full_hash = os.environ.get("PLE_MANIFEST_FULL_HASH") == "1"
    validate_external_ple(ple_path, manifest_path, full_hash=full_hash,
                          logger=logger)
    external = copy.copy(model_config)
    external.model = ple_path
    if logger is not None:
        logger.info("PleOffload: PLE weights will be read from %s "
                    "(primary model %s unchanged)", ple_path,
                    model_config.model)
    return external
