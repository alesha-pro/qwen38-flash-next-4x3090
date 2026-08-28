#!/usr/bin/env python3
"""External PLE source manifest: generate and validate (CPU, read-only).

The manifest pins exactly which secondary checkpoint may feed PLE tensors to
the offload worker. Validation is fail-closed: any mismatch, missing file or
read error is a violation.

Schema `qwen38.external_ple.v1`:
  repo_id, revision                -- pinned HF source
  ple_prefix                       -- tensor name prefix filter
  files                            -- {name: size_bytes} of files carrying PLE tensors
  tensors                          -- {name: {dtype, shape, file}} for every PLE tensor
  scale                            -- {name, value, sha256}
  metadata_sha256                  -- full payload sha256 of every non-ngram PLE tensor
  ngram_window_sha256              -- {shard_id: sha256 of first/middle/last 16 MiB}
  reference                        -- provenance notes (BF16 base revision etc.)

Runtime validation performs: revision check, file size check, safetensors
header check (names/dtypes/shapes/files), scale value check, full payload
hash of the small metadata tensors, and sampled window hashes of the 128
ngram shards. Set env PLE_MANIFEST_FULL_HASH=1 to additionally hash whole
ngram payloads (slow audit mode).
"""
import hashlib
import json
import os
import struct
import sys
import time

SCHEMA = "qwen38.external_ple.v1"
WINDOW = 16 * 1024 * 1024
FULL_PAYLOAD_LIMIT = 64 * 1024 * 1024


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(hlen)), 8 + hlen


def tensor_payload(path, entry, data_start):
    a, b = entry["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + a)
        return f.read(b - a)


def bf16_scalar(raw):
    bits = struct.unpack("<H", raw)[0]
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def windowed_hash(path, entry, data_start):
    a, b = entry["data_offsets"]
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


def find_revision(root):
    tdir = os.path.join(root, ".cache", "huggingface", "trees")
    if not os.path.isdir(tdir):
        return None
    fs = sorted(f for f in os.listdir(tdir) if f.endswith(".json"))
    return fs[-1][:-5] if fs else None


def collect_ple(root):
    """Return (tensors dict, files dict) for all `.ple.` tensors."""
    idx_path = os.path.join(root, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    ple = {n: s for n, s in wm.items() if ".ple." in n}
    tensors = {}
    files = {}
    by_file = {}
    for n, s in ple.items():
        by_file.setdefault(s, []).append(n)
    for s, names in sorted(by_file.items()):
        p = os.path.join(root, s)
        files[s] = os.path.getsize(p)
        header, ds = read_header(p)
        for n in names:
            e = header[n]
            tensors[n] = {"dtype": e["dtype"], "shape": e["shape"],
                          "file": s, "data_offsets": e["data_offsets"],
                          "_ds": ds}
    return tensors, files


def build_manifest(root, repo_id, reference):
    revision = find_revision(root)
    tensors, files = collect_ple(root)
    scale = None
    metadata_sha256 = {}
    ngram_windows = {}
    for n, t in sorted(tensors.items()):
        p = os.path.join(root, t["file"])
        entry = {"data_offsets": t["data_offsets"]}
        if n.endswith("ngram_embedding.weight_scale"):
            raw = tensor_payload(p, entry, t["_ds"])
            scale = {"name": n, "value": bf16_scalar(raw),
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "dtype": t["dtype"]}
        elif ".ngram_embedding.shard_" in n:
            sid = int(n.rsplit("shard_", 1)[1].split(".")[0])
            ngram_windows[str(sid)] = windowed_hash(p, entry, t["_ds"])
        else:
            raw = tensor_payload(p, entry, t["_ds"])
            metadata_sha256[n] = hashlib.sha256(raw).hexdigest()
    for t in tensors.values():
        t.pop("_ds", None)
        t.pop("data_offsets", None)
    return {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_id": repo_id,
        "revision": revision,
        "ple_prefix": ".ple.",
        "files": files,
        "tensors": tensors,
        "scale": scale,
        "metadata_sha256": metadata_sha256,
        "ngram_window_sha256": ngram_windows,
        "ngram_window_method": "sha256(first|middle|last 16 MiB of payload)",
        "reference": reference,
    }


def validate_manifest(root, manifest, full_hash=False):
    """Return list of violation strings; empty means pass. Never raises."""
    v = []
    try:
        if manifest.get("schema") != SCHEMA:
            return [f"unknown schema {manifest.get('schema')!r}"]
        rev = find_revision(root)
        if rev != manifest.get("revision"):
            v.append(f"revision mismatch: manifest={manifest.get('revision')} "
                     f"on-disk={rev}")
        for fn, size in manifest.get("files", {}).items():
            p = os.path.join(root, fn)
            if not os.path.isfile(p):
                v.append(f"missing file {fn}")
            elif os.path.getsize(p) != size:
                v.append(f"size mismatch {fn}: manifest={size} "
                         f"on-disk={os.path.getsize(p)}")
        try:
            tensors, files = collect_ple(root)
        except Exception as e:  # fail closed on unreadable checkpoint
            return v + [f"cannot enumerate PLE tensors: {e}"]
        mt = manifest.get("tensors", {})
        if set(tensors) != set(mt):
            v.append(f"PLE tensor set mismatch: "
                     f"only-on-disk={sorted(set(tensors) - set(mt))[:5]} "
                     f"only-in-manifest={sorted(set(mt) - set(tensors))[:5]}")
        else:
            for n, t in tensors.items():
                m = mt[n]
                if (t["dtype"], t["shape"], t["file"]) != (
                        m["dtype"], m["shape"], m["file"]):
                    v.append(f"tensor mismatch {n}: manifest="
                             f"{(m['dtype'], m['shape'], m['file'])} on-disk="
                             f"{(t['dtype'], t['shape'], t['file'])}")
        # scale value
        ms = manifest.get("scale")
        if ms:
            t = tensors.get(ms["name"])
            if t is None:
                v.append(f"scale tensor {ms['name']} not present")
            else:
                raw = tensor_payload(os.path.join(root, t["file"]),
                                     {"data_offsets": t["data_offsets"]},
                                     t["_ds"])
                if hashlib.sha256(raw).hexdigest() != ms["sha256"]:
                    v.append("scale payload hash mismatch")
                elif bf16_scalar(raw) != ms["value"]:
                    v.append("scale value mismatch")
        else:
            v.append("manifest has no scale entry")
        # metadata payload hashes
        for n, want in manifest.get("metadata_sha256", {}).items():
            t = tensors.get(n)
            if t is None:
                v.append(f"metadata tensor {n} not present")
                continue
            raw = tensor_payload(os.path.join(root, t["file"]),
                                 {"data_offsets": t["data_offsets"]}, t["_ds"])
            if hashlib.sha256(raw).hexdigest() != want:
                v.append(f"metadata hash mismatch {n}")
        # ngram windows (or full audit hash)
        for sid, want in manifest.get("ngram_window_sha256", {}).items():
            n = (f"model.language_model.layers.1.ple.ple_embedding"
                 f".ngram_embedding.shard_{sid}.weight")
            t = tensors.get(n)
            if t is None:
                v.append(f"ngram shard {sid} not present")
                continue
            p = os.path.join(root, t["file"])
            if full_hash:
                raw = tensor_payload(p, {"data_offsets": t["data_offsets"]},
                                     t["_ds"])
                got = hashlib.sha256(raw).hexdigest()
            else:
                got = windowed_hash(p, {"data_offsets": t["data_offsets"]},
                                    t["_ds"])
            if got != want:
                v.append(f"ngram shard {sid} window hash mismatch")
    except Exception as e:  # fail closed
        v.append(f"validation error: {e}")
    return v


def main():
    cmd = sys.argv[1]
    if cmd == "generate":
        root, repo_id, out = sys.argv[2], sys.argv[3], sys.argv[4]
        reference = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
        m = build_manifest(root, repo_id, reference)
        with open(out, "w") as f:
            json.dump(m, f, indent=1)
        print(f"wrote {out}: revision={m['revision']} "
              f"tensors={len(m['tensors'])} files={len(m['files'])}")
    elif cmd == "validate":
        root, mpath = sys.argv[2], sys.argv[3]
        with open(mpath) as f:
            m = json.load(f)
        full = os.environ.get("PLE_MANIFEST_FULL_HASH") == "1"
        t0 = time.time()
        violations = validate_manifest(root, m, full_hash=full)
        if violations:
            print(f"FAIL ({len(violations)} violations):")
            for x in violations:
                print(" -", x)
            sys.exit(1)
        print(f"PASS: manifest matches {root} "
              f"({time.time() - t0:.1f}s, full_hash={full})")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
