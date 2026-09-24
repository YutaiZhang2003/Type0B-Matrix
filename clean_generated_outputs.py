#!/usr/bin/env python3
"""Remove generated numerical and build artifacts while retaining source and inputs.

Run without arguments for a count/size preview. Pass --apply to delete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
OUTPUT_DIRS = (
    'C++/bin',
    'C++/build',
    'C++/results',
    'Code/theta_fermion_ccy/results',
    'Code/c_Recursion/results',
    'Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/results',
    'Code/bosonic_c1_one_to_n_reference/reference_implementation/output',
)
SOURCE_SUFFIXES = {'.py', '.cpp', '.c', '.h', '.hpp', '.inc', '.sh'}
DATA_SUFFIXES = {'.npz', '.log', '.csv', '.out', '.err'}
BINARY_MAGIC = (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\x7fELF')


def under(path: Path, relative: str) -> bool:
    return path.is_relative_to(ROOT / relative)


def retain_json(path: Path) -> bool:
    if under(path, 'Data Set'):
        return path.name in {'config.json', 'manifest.json'} or 'vendor' in path.parts
    if under(path, 'Code'):
        if 'config' in path.parts or under(path, 'Code/config'):
            return True
        return path.name in {
            'baseline.json', 'MANIFEST.json', 'ENVIRONMENT.json',
            'manifest.json', 'nsrr_checked_kernel_manifest.json',
        }
    if under(path, 'Packages'):
        return ('tests' in path.parts or 'examples' in path.parts or
                path.name in {'MANIFEST.json', 'PROVENANCE.json'})
    return False


def candidates() -> tuple[list[Path], list[Path]]:
    directories = [ROOT / name for name in OUTPUT_DIRS if (ROOT / name).exists()]
    for directory in directories:
        if directory.name in {'bin', 'build'}:
            continue
        source = [p for p in directory.rglob('*') if p.is_file() and
                  p.suffix in SOURCE_SUFFIXES]
        if source:
            raise RuntimeError(f'Source files remain inside output directory {directory}: {source[:5]}')
    files = []
    for path in ROOT.rglob('*.json'):
        if any(path.is_relative_to(directory) for directory in directories):
            continue
        if not retain_json(path):
            files.append(path)
    for directory in (ROOT / 'Data Set', ROOT / 'C++/experiments'):
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if not path.is_file() or path in files:
                continue
            if under(path, 'Data Set') and 'vendor' in path.parts:
                continue
            if path.suffix in DATA_SUFFIXES or (under(path, 'C++/experiments') and
                                                  path.suffix == '.jsonl'):
                files.append(path)
    experiments = ROOT / 'C++/experiments'
    if experiments.exists():
        for path in experiments.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    if stream.read(4) in BINARY_MAGIC:
                        files.append(path)
    directories.extend(p for p in ROOT.rglob('__pycache__')
                       if '.git' not in p.parts and p.is_dir() and
                       not any(p.is_relative_to(directory) for directory in directories))
    return directories, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Delete the previewed outputs')
    args = parser.parse_args()
    directories, files = candidates()
    sizes = Counter()
    counts = Counter()
    total = 0
    for directory in directories:
        for path in directory.rglob('*'):
            if path.is_file():
                size = path.stat().st_size
                counts[str(directory.relative_to(ROOT))] += 1
                sizes[str(directory.relative_to(ROOT))] += size
                total += size
    for path in files:
        size = path.stat().st_size
        key = path.parts[len(ROOT.parts)]
        counts[key] += 1
        sizes[key] += size
        total += size
    for key in sorted(counts):
        print(f'{key}: {counts[key]} files, {sizes[key] / 2**20:.2f} MiB')
    print(f'Total: {sum(counts.values())} files, {total / 2**20:.2f} MiB')
    if args.apply:
        for directory in directories:
            shutil.rmtree(directory)
        for path in files:
            path.unlink(missing_ok=True)
        print('Removed generated outputs.')


if __name__ == '__main__':
    main()
