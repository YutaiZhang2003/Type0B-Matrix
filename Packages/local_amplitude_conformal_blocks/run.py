#!/usr/bin/env python3
"""Portable entry point. Run each scientific configuration in a fresh process."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))

def verify():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    for name, record in manifest['files'].items():
        p = ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Missing or changed package file: '+name)
    return {'verified_files': len(manifest['files']), 'convention': manifest['convention']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('evaluate', help='evaluate a JSON configuration')
    p.add_argument('configuration', type=Path)
    p.add_argument('--output', type=Path)
    p = subs.add_parser('examples', help='run the seven small examples')
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p = subs.add_parser('cross-check', help='recompute sphere crossing from included production banks')
    p.add_argument('--output', type=Path, default=Path('results/cross_channel.json'))
    subs.add_parser('verify', help='verify the complete package SHA256 manifest')
    p = subs.add_parser('test', help='run scientific and portability checks')
    p.add_argument('--output', type=Path, default=Path('results/validation.json'))
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(verify(), indent=2)); return
    if args.command == 'examples':
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for config in sorted((ROOT/'examples').glob('*.json')):
            subprocess.run([sys.executable, '-I', '-B', str(ROOT/'run.py'), 'evaluate', str(config),
                            '--output', str(args.output_dir/(config.stem+'.json'))], check=True)
        return
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    from runtime import engine
    if args.command == 'evaluate':
        result = engine.evaluate(json.loads(args.configuration.read_text()))
    elif args.command == 'cross-check':
        result = engine.cross_check()
    else:
        from runtime.validate import validate
        result = validate()
    output = getattr(args, 'output', None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'saved': str(output), 'kind': result.get('kind',args.command)}))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__': main()
