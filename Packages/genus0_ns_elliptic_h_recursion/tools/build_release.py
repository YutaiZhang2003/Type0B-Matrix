#!/usr/bin/env python3
"""Build the wheel and deterministic self-contained research handoff archive.

Requires setuptools>=61 and wheel already installed; no dependency/network
resolution is attempted. Runtime dependencies are not vendored.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-wheel",action="store_true",help="package an existing, already tested dist wheel")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = re.search(r'__version__ = "([^"]+)"',
                (root/"src/genus0_ns_elliptic_h_recursion/__init__.py").read_text()).group(1)
    distribution = "genus0_ns_elliptic_h_recursion"
    stem = f"{distribution}-{version}"
    dist = root/"dist"
    dist.mkdir(exist_ok=True)
    if not args.reuse_wheel:
        subprocess.run([sys.executable,"-m","pip","wheel","--no-index","--no-cache-dir",
                        "--no-deps","--no-build-isolation","--wheel-dir",str(dist),str(root)],check=True)
    wheel = dist/f"{stem}-py3-none-any.whl"
    if not wheel.is_file():
        raise RuntimeError("expected wheel missing")
    wheels = root/"wheels"
    wheels.mkdir(exist_ok=True)
    shutil.copy2(wheel,wheels/wheel.name)
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or any(part in ("dist","build","__pycache__",".git",".pytest_cache")
                 or part.endswith(".egg-info") for part in relative.parts)
                or path.name in ("MANIFEST.json",".DS_Store")
                or path.suffix in (".pyc",".pyo")):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink not permitted in handoff: {relative}")
        if relative.parts[0] == "wheels" and path.name != wheel.name:
            continue
        content = path.read_bytes()
        files[relative.as_posix()] = {"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content)}
    manifest = {"format":1,"package":distribution,"version":version,"files":files}
    (root/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    archive = dist/f"{stem}-handoff.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as output:
        for name in sorted([*files,"MANIFEST.json"]):
            info = zipfile.ZipInfo(f"{stem}/{name}",date_time=(2026,8,30,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info,(root/name).read_bytes(),compresslevel=9)
    sums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in (archive,wheel)]
    (dist/"SHA256SUMS").write_text("\n".join(sums)+"\n")
    print("\n".join(sums))
    print(f"Packaged {len(files)} manifested files in {archive.name}")


if __name__ == "__main__":
    main()
