#!/usr/bin/env python3
"""Download and derive the exact cyankiwi AWQ thin-v2-fix runtime checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4"
MODEL_REVISION = "01324cfa2c3f46948781fad30641ac360014e008"
PLE_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
PLE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"

# Shard 1 contains only PLE tensors and shard 4 only MTP tensors. Both are
# deliberately replaced by the external PLE source / disabled MTP path.
UPSTREAM_SHARDS = (
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
)
MODEL_METADATA = (
    "model.safetensors.index.json",
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


def populated_checkpoint(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text())
        shards = set(index["weight_map"].values())
    except (KeyError, json.JSONDecodeError):
        return False
    return bool(shards) and all((path / shard).is_file() for shard in shards)


def docker_build(root: Path, staging: Path, model_parent: Path, image: str,
                 script: str, *args: str) -> None:
    if subprocess.run(["docker", "image", "inspect", image],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["docker", "pull", image], check=True)
    command = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{root}:/recipe:ro",
        "-v", f"{staging}:/staging",
        "-v", f"{model_parent}:/models",
        image, "python", f"/recipe/scripts/{script}", *args,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--ple-dir", required=True, type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    ple_dir = args.ple_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "manifests/ple-external-fp8-radixark.json").read_text()
    )
    token = os.environ.get("HF_TOKEN") or None

    if not populated_checkpoint(model_dir):
        staging = model_dir.parent / ".qwen38-awq-staging"
        if staging.exists():
            shutil.rmtree(staging)
        source, v1, v2 = staging / "source", staging / "thin-v1", staging / "thin-v2"
        print(f"Downloading {MODEL_REPO}@{MODEL_REVISION} selected shards to {source}")
        snapshot_download(
            repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=source,
            token=token, allow_patterns=[*MODEL_METADATA, *UPSTREAM_SHARDS],
        )
        subprocess.run(
            ["python3", str(root / "scripts" / "build-thin-checkpoint.py"),
             str(source), str(v1)],
            check=True,
        )
        # The filtered index refers to the two retained source shards. Hardlink
        # them into thin-v1 so this build does not create a second 80 GiB copy.
        for shard in UPSTREAM_SHARDS:
            os.link(source / shard, v1 / shard)
        image = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:qwen38-flash-next")
        docker_build(
            root, staging, model_dir.parent, image, "build-fp8-thin-v2.py",
            "/staging/thin-v1", "/staging/thin-v2",
        )
        docker_build(
            root, staging, model_dir.parent, image, "fix-shared-expert-gate.py",
            "/staging/thin-v1", "/staging/thin-v2", f"/models/{model_dir.name}",
        )
        (model_dir / "AWQ-V2FIX-MANIFEST.json").write_text(json.dumps({
            "upstream_repo": MODEL_REPO,
            "upstream_revision": MODEL_REVISION,
            "kept_upstream_shards": list(UPSTREAM_SHARDS),
            "excluded": {
                "model-00001-of-00004.safetensors": "PLE-only; external FP8 PLE",
                "model-00004-of-00004.safetensors": "MTP-only; not used",
            },
            "derived_recipe": "thin-v1 -> FP8 companions -> BF16 shared_expert_gate fix",
        }, indent=2, sort_keys=True) + "\n")
        if not populated_checkpoint(model_dir):
            raise SystemExit(f"derived checkpoint validation failed: {model_dir}")
        shutil.rmtree(staging)
        print(f"Derived AWQ thin-v2-fix checkpoint: {model_dir}")
    else:
        print(f"Reusing derived AWQ thin-v2-fix checkpoint: {model_dir}")

    allow = ["model.safetensors.index.json", *sorted(manifest["files"])]
    print(f"Downloading selected FP8 PLE files from {PLE_REPO}@{PLE_REVISION}")
    snapshot_download(
        repo_id=PLE_REPO, revision=PLE_REVISION,
        local_dir=ple_dir, token=token, allow_patterns=allow,
    )
    print("Download and derivation complete")


if __name__ == "__main__":
    main()
