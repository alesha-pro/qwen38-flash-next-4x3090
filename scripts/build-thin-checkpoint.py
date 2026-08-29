#!/usr/bin/env python3
"""Build and validate metadata for a PLE/MTP-free candidate checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

KEEP_SHARDS = {
    "model-00002-of-00004.safetensors": 50_025_607_344,
    "model-00003-of-00004.safetensors": 30_646_247_168,
}
COPY_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def excluded(name: str) -> bool:
    return ".ple." in name or name.startswith("mtp.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    original = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = original["weight_map"]
    filtered = {k: v for k, v in weight_map.items() if not excluded(k)}

    bad_shards = sorted(set(filtered.values()) - set(KEEP_SHARDS))
    if bad_shards:
        raise SystemExit(f"unexpected referenced shards after filtering: {bad_shards}")

    removed = sorted(set(weight_map) - set(filtered))
    ple_removed = sum(".ple." in name for name in removed)
    mtp_removed = sum(name.startswith("mtp.") for name in removed)
    if ple_removed != 137 or mtp_removed != 31:
        raise SystemExit(
            f"unexpected exclusion count: ple={ple_removed}, mtp={mtp_removed}"
        )

    for name in COPY_FILES:
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)

    output = dict(original)
    output["metadata"] = dict(original.get("metadata", {}))
    output["metadata"]["thin_checkpoint_physical_bytes"] = sum(KEEP_SHARDS.values())
    output["metadata"]["thin_checkpoint_excluded_ple_tensors"] = ple_removed
    output["metadata"]["thin_checkpoint_excluded_mtp_tensors"] = mtp_removed
    output["weight_map"] = filtered
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )

    for shard, expected in KEEP_SHARDS.items():
        path = destination / shard
        if path.exists() and path.stat().st_size != expected:
            raise SystemExit(
                f"size mismatch for {shard}: {path.stat().st_size} != {expected}"
            )

    manifest = {
        "upstream_revision": "01324cfa2c3f46948781fad30641ac360014e008",
        "kept_shards": KEEP_SHARDS,
        "physical_bytes": sum(KEEP_SHARDS.values()),
        "excluded": {"ple_tensors": ple_removed, "mtp_tensors": mtp_removed},
        "status": "metadata-built; runtime compatibility unproven",
    }
    (destination / "THIN-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

