#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")
    try:
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    except Exception as error:
        raise SystemExit(f"invalid {index_path}: {error}") from error
    missing = [name for name in shards
               if not (root / name).is_file() or os.path.getsize(root / name) == 0]
    if missing:
        raise SystemExit(f"missing or empty checkpoint shards: {missing}")
    print(f"PASS: primary checkpoint has all {len(shards)} indexed shards")


if __name__ == "__main__":
    main()
