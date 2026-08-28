#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "VnimanieAI/Qwen3.8-Flash-Next-W4A16"
MODEL_REVISION = "9236d703b25f25eb5c17e9640204f84fa1ce0c6e"
PLE_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
PLE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ple-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "manifests/ple-external-fp8-radixark.json").read_text()
    )
    token = os.environ.get("HF_TOKEN") or None

    print(f"Downloading {MODEL_REPO}@{MODEL_REVISION} to {args.model_dir}")
    snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION,
        local_dir=args.model_dir, token=token, ignore_patterns=["*.log"])

    allow = ["model.safetensors.index.json", *sorted(manifest["files"])]
    print(f"Downloading selected FP8 PLE files from {PLE_REPO}@{PLE_REVISION}")
    snapshot_download(
        repo_id=PLE_REPO, revision=PLE_REVISION,
        local_dir=args.ple_dir, token=token, allow_patterns=allow)

    shard_count = len(set(json.loads(
        (Path(args.model_dir) / "model.safetensors.index.json").read_text()
    )["weight_map"].values()))
    print(f"Download complete: {shard_count} primary shards and "
          f"{len(manifest['files'])} external PLE files")


if __name__ == "__main__":
    main()
