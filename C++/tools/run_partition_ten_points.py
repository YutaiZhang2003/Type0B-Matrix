#!/usr/bin/env python3
"""Run the ten matched surfaces with the current fixed-spin block decomposition."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE = ROOT / 'C++/bin/partition'
GEOMETRY = ROOT / 'Data Set/nsrr_nsnsns_generic_10point_N4_20260904/config.json'


def period_transport_residual(matrix, source, target):
    """Compare the two archived period matrices through their homology map."""
    a = [[matrix[i][j] for j in range(2)] for i in range(2)]
    b = [[matrix[i][j + 2] for j in range(2)] for i in range(2)]
    c = [[matrix[i + 2][j] for j in range(2)] for i in range(2)]
    d = [[matrix[i + 2][j + 2] for j in range(2)] for i in range(2)]
    omega = [[complex(value) for value in row] for row in source]
    expected = [[complex(value) for value in row] for row in target]

    def mul(x, y):
        return [[sum(x[i][k] * y[k][j] for k in range(2)) for j in range(2)]
                for i in range(2)]

    numerator = mul(a, omega)
    denominator = mul(c, omega)
    for i in range(2):
        for j in range(2):
            numerator[i][j] += b[i][j]
            denominator[i][j] += d[i][j]
    determinant = denominator[0][0] * denominator[1][1] - denominator[0][1] * denominator[1][0]
    if abs(determinant) == 0:
        raise ValueError('singular period transport')
    inverse = [[denominator[1][1] / determinant, -denominator[0][1] / determinant],
               [-denominator[1][0] / determinant, denominator[0][0] / determinant]]
    transformed = mul(numerator, inverse)
    return max(abs(transformed[i][j] - expected[i][j]) for i in range(2) for j in range(2))


def call(args, log):
    result = subprocess.run([str(EXECUTABLE), *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{log}: {result.stdout[-1500:]}')


def run_one(root, nodes, configs, point, matrix, workers, digits):
    name = point['point_id']
    directory = root / name
    directory.mkdir(exist_ok=True)
    config = configs / f'{name}.json'
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
    residual = period_transport_residual(
        matrix, point['source']['omega'], point['target']['omega'])
    if residual >= 1e-12:
        raise AssertionError(f'{name}: source and target periods do not match ({residual})')
    return {
        'point_id': name,
        'fixed_spins': [
            {'source_eta_e': comparison['source_eta_e'][spin],
             'target_eta_e': comparison['target_eta_e'][spin],
             'source_Z': row['source_R_fixed'][spin],
             'target_Z': row['target_NS_fixed'][spin],
             'ratio': row['fixed_spin_ratios'][spin]}
            for spin in range(2)
        ],
        'free_frame_power': comparison['free_frame_power'],
        'source_seconds': json.loads((directory / 'source/run.json').read_text())['wall_seconds'],
        'target_seconds': json.loads((directory / 'target/run.json').read_text())['wall_seconds'],
        'archived_period_transport_residual': residual,
        'source_inverse_period_residual': point['source']['inverse_period_residual'],
        'target_inverse_period_residual': point['target']['inverse_period_residual'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--node-root', type=Path, required=True,
                        help='Generated source/target momentum-node input directories')
    parser.add_argument('--config-root', type=Path,
                        help='Prepared fixed-spin configurations (default: OUTPUT_ROOT/inputs)')
    parser.add_argument('--geometry-config', type=Path, default=GEOMETRY)
    parser.add_argument('--parallel-points', type=int, default=3)
    parser.add_argument('--workers-per-point', type=int, default=3)
    parser.add_argument('--digits', type=int, default=40)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    nodes = args.node_root.resolve()
    configs = (args.config_root or root / 'inputs').resolve()
    for channel in ('source', 'target'):
        if not (nodes / channel).is_dir():
            raise FileNotFoundError(nodes / channel)
    if args.parallel_points < 1 or args.workers_per_point < 1:
        raise ValueError('positive worker counts required')
    names = json.loads((configs / 'manifest.json').read_text())['point_ids']
    if len(names) != 10:
        raise ValueError('expected ten fixed surfaces')
    geometry = json.loads(args.geometry_config.read_text())
    points = {point['point_id']: point for point in geometry['points']}
    if set(names) != set(points):
        raise ValueError('prepared configurations do not match the geometry table')
    subprocess.run(['make', '-C', str(ROOT / 'C++'), 'bin/partition'], check=True)
    start = time.monotonic()
    rows = {}
    with ThreadPoolExecutor(max_workers=args.parallel_points) as pool:
        futures = {pool.submit(run_one, root, nodes, configs, points[name],
                               geometry['source_to_target'], args.workers_per_point, args.digits): name
                   for name in names}
        for future in as_completed(futures):
            name = futures[future]
            rows[name] = future.result()
            print(name, [spin['ratio']['real'] for spin in rows[name]['fixed_spins']], flush=True)
    errors = [abs(complex(float(spin['ratio']['real']), float(spin['ratio']['imag'])) - 1)
              for row in rows.values() for spin in row['fixed_spins']]
    result = {'schema': 'partition-ten-geometric-fixed-result-v1',
              'level_source': 5, 'level_target': 8, 'digits': args.digits,
              'source_N': 7, 'target_N': 10, 'parallel_points': args.parallel_points,
              'workers_per_point': args.workers_per_point,
              'wall_seconds': time.monotonic() - start,
              'spin_status': 'two marked spins per surface, each transported and evaluated separately',
              'maximum_ratio_deviation': max(errors),
              'rows': [rows[name] for name in names]}
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'complete': len(rows), 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
