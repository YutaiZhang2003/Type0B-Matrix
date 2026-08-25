#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file():
        raise SystemExit(f"missing file: {item['path']}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"checksum mismatch: {item['path']}")
print(f"verified {len(manifest['files'])} files")
