#!/usr/bin/env python3
"""Verify every source/document/audit/wheel file recorded in MANIFEST.json."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for name,entry in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe manifest path")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            failures.append(name+": missing or symlink")
            continue
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            failures.append(name+": digest mismatch")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(manifest['files'])} files verified for version {manifest['version']}")


if __name__ == "__main__":
    main()
