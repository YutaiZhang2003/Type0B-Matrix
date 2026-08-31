#!/usr/bin/env python3
"""Copy a stopped v1 store after verifying its coefficient-producing sources.

This does not migrate scalar integration checkpoints. Those retain the full
runtime/config fingerprint. Original databases and staged code are read-only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from fivepoint_runtime import coefficient_source_fingerprint


def method_ast(path, cls, name):
    module = ast.parse(path.read_text())
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(method, include_attributes=False)


def coefficient_sources(directory):
    """Follow every local import from the actual coefficient recursion module.

    Amplitude drivers/tests share the directory but do not produce these
    coefficients. Verify the transitive dependency closure, not those clients.
    """
    pending = ['ns_multipoint_c_recursion']
    sources = {}
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path = directory / (name + '.py')
        sources[name] = path.read_bytes()
        for node in ast.walk(ast.parse(sources[name])):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for imported in imports:
                if imported and (directory / (imported + '.py')).is_file():
                    pending.append(imported)
    return sources


def migrate(old_code, source, destination):
    old_code, source, destination = map(Path, (old_code, source, destination))
    current_code = Path(__file__).resolve().parents[2]
    # old_code is the staged Code directory, not the run root.
    old_recursion = coefficient_sources(old_code/'c_Recursion')
    new_recursion = coefficient_sources(current_code/'c_Recursion')
    if not old_recursion or old_recursion != new_recursion:
        raise ValueError('coefficient recursion sources differ; cannot migrate')
    relative = Path('higher_point_amplitude_attempts/type0b_ns_five_tachyon')
    old_runtime = old_code / relative / 'fivepoint_runtime.py'
    new_runtime = Path(__file__).with_name('fivepoint_runtime.py')
    for cls, name in (('CompactCBlock','_encode_number'), ('CompactCBlock','_prepare'),
                      ('CoefficientStore','get'), ('CoefficientStore','put')):
        if method_ast(old_runtime,cls,name) != method_ast(new_runtime,cls,name):
            raise ValueError(f'{cls}.{name} changed; cannot migrate')
    files = sorted((old_code/relative).glob('*.py'))
    files += sorted((old_code/'c_Recursion').glob('*.py'))
    files += sorted((old_code/'bosonic_c1_one_to_n_reference/reference_implementation/plumbing').glob('*.py'))
    digest=hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(old_code)).encode()); digest.update(path.read_bytes())
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    src=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        if src.execute("SELECT 1 FROM sqlite_master WHERE name='migration'").fetchone():
            raise ValueError('source already migrated; refuse ambiguous provenance')
        count=src.execute('SELECT count(*) FROM coefficients').fetchone()[0]
        dst=sqlite3.connect(destination)
        try:
            src.backup(dst)
            metadata=dict(legacy_source=digest.hexdigest(), coefficient_source=coefficient_source_fingerprint(),
                          source_path=str(source.resolve()), verified_rows=str(count),
                          verification='transitive coefficient dependencies and compilation/serialization ASTs identical',
                          verified_dependencies=json.dumps({name: hashlib.sha256(data).hexdigest()
                              for name, data in sorted(new_recursion.items())}))
            dst.execute('CREATE TABLE migration(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            dst.executemany('INSERT INTO migration VALUES (?,?)',metadata.items());dst.commit()
            if dst.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError('migrated SQLite integrity check failed')
        finally: dst.close()
    finally: src.close()
    return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-code',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(migrate(args.old_code,args.source,args.destination),indent=2))
