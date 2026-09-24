#!/usr/bin/env python3
"""Run the ten fixed-spin geometric-BPZ comparisons from fresh chiral blocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE = ROOT / 'C++/bin/partition'


def call(args, log):
    result = subprocess.run([str(EXECUTABLE), *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{log}: {result.stdout[-1500:]}')


def run_one(root, nodes, name, workers, digits):
    directory = root / name
    directory.mkdir(exist_ok=True)
    config = root / 'inputs' / f'{name}.json'
    config_arg = str(config)
    for channel, level in [('source', 5), ('target', 8)]:
        path = directory / channel
        call(['--config', config_arg,
              '--node-directory', str(nodes / channel),
              '--channel', channel, '--level', str(level), '--minimum-level', str(level),
              '--dps', str(digits), '--workers', str(workers),
              '--output', str(path)], directory / f'{channel}.log')
    call(['--config', config_arg, '--reduce-source', str(directory / 'source'),
          '--reduce-target', str(directory / 'target'),
          '--dps', str(digits), '--output', str(directory / 'comparison.json')],
         directory / 'reduce.log')
    comparison = json.loads((directory / 'comparison.json').read_text())
    row = comparison['comparisons'][0]
    geometry = json.loads((directory / 'geometry.json').read_text())
    spin_geometry = json.loads((directory / 'spin_geometry.json').read_text())
    return {
        'point_id': name,
        'source_eta_e': comparison['source_eta_e'][0],
        'target_eta_e': comparison['target_eta_e'][0],
        'source_Z': row['source_R_fixed'][0],
        'target_Z': row['target_NS_fixed_and_literal'][0],
        'free_frame_power': comparison['free_frame_power'],
        'fixed_spin_ratio': row['fixed_spin_ratios'][0],
        'source_seconds': json.loads((directory / 'source/run.json').read_text())['wall_seconds'],
        'target_seconds': json.loads((directory / 'target/run.json').read_text())['wall_seconds'],
        'period_residual': spin_geometry['period_residual'],
        'free_majorana_max_relative_error': geometry['maximum_complex_squared_relative_error'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--node-root', type=Path, required=True,
                        help='Generated source/target momentum-node input directories')
    parser.add_argument('--parallel-points', type=int, default=3)
    parser.add_argument('--workers-per-point', type=int, default=3)
    parser.add_argument('--digits', type=int, default=40)
    args = parser.parse_args()
    root = args.output_root.resolve()
    nodes = args.node_root.resolve()
    for channel in ('source', 'target'):
        if not (nodes / channel).is_dir():
            raise FileNotFoundError(nodes / channel)
    if args.parallel_points < 1 or args.workers_per_point < 1:
        raise ValueError('positive worker counts required')
    names = json.loads((root / 'inputs/manifest.json').read_text())['point_ids']
    if len(names) != 10:
        raise ValueError('expected ten fixed surfaces')
    for name in names:
        for suffix in ('geometry.json', 'spin_geometry.json'):
            if not (root / name / suffix).is_file():
                raise FileNotFoundError(root / name / suffix)
    start = time.monotonic()
    rows = {}
    with ThreadPoolExecutor(max_workers=args.parallel_points) as pool:
        futures = {pool.submit(run_one, root, nodes, name, args.workers_per_point, args.digits): name
                   for name in names}
        for future in as_completed(futures):
            name = futures[future]
            rows[name] = future.result()
            print(name, rows[name]['fixed_spin_ratio']['real'], flush=True)
    result = {'schema': 'partition-ten-geometric-fixed-result-v1',
              'level_source': 5, 'level_target': 8, 'digits': args.digits,
              'source_N': 7, 'target_N': 10, 'parallel_points': args.parallel_points,
              'workers_per_point': args.workers_per_point,
              'wall_seconds': time.monotonic() - start,
              'spin_status': 'one fixed marked [11|00] source spin transported to [00|00] target per surface',
              'rows': [rows[name] for name in names]}
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'complete': len(rows), 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
