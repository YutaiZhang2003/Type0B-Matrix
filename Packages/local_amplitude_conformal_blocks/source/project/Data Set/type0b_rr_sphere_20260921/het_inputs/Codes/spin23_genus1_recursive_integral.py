"""Sharded, regulated genus-one S -> VV integration with recursive blocks.

All nine integration coordinates are sampled jointly. Each energy and block
cutoff uses the same points. Completed node records are flushed immediately;
the reducer requires every node, including rejected geometry points as zeros.
No MQM prediction enters this calculation.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from spin23_genus1_corrected_integrand import evaluate_svv_integrand
from spin23_genus1_threepoint_integral import (
    Regulators, SPINS, complex_json, map_point, states_for_energy, summarize,
)
from spin23_type0b_reference import REFERENCE_COMMIT


@dataclass(frozen=True)
class ScanConfig:
    energies: tuple = ((.25, .35), (.3, .3), (1/3, 2/3), (.5, .5))
    total_twice_levels: tuple = (6, 8)
    replicates: int = 4
    samples_per_replicate: int = 32
    seed: int = 9023
    precision: int = 24
    regulators: Regulators = Regulators()

    def __post_init__(self):
        if not self.energies:
            raise ValueError('an energy grid is required')
        for p, q in self.energies:
            states_for_energy(p, q)
        cuts = self.total_twice_levels
        if not cuts or tuple(sorted(set(cuts))) != cuts or any(
            type(c) is not int or c < 0 or c > 8 or c % 2 for c in cuts
        ):
            raise ValueError('use increasing integer-level cutoffs, at most level 4')
        n = self.samples_per_replicate
        if type(n) is not int or n < 2 or n & (n - 1):
            raise ValueError('sample count must be a power of two, at least two')
        if type(self.replicates) is not int or self.replicates < 2:
            raise ValueError('at least two independent replicates are required')


def point_grid(config):
    return np.stack([
        qmc.Sobol(d=9, scramble=True, seed=config.seed + rep).random_base2(
            config.samples_per_replicate.bit_length() - 1)
        for rep in range(config.replicates)
    ])


def shard_assignment(config, task_count, grid=None):
    """Balance known geometry work without changing or discarding QMC points."""
    grid = point_grid(config) if grid is None else grid
    flags = [map_point(u, config.regulators).accepted for row in grid for u in row]
    order = sorted(range(len(flags)), key=lambda index: not flags[index])
    assignment = [0] * len(order)
    for rank, flat in enumerate(order):
        assignment[flat] = rank % task_count
    return assignment


def evaluate_node(point, config):
    """Corrected component ledger in sorted necklace order, with full measure."""
    values = np.zeros((len(config.energies), len(config.total_twice_levels), 3, 4), complex)
    if not point.accepted:
        return values
    for ie, energy in enumerate(config.energies):
        # map_point sorts the punctures. Their energy labels move with them.
        p, q = energy if point.permutation == (0, 1, 2) else energy[::-1]
        for il, cutoff in enumerate(config.total_twice_levels):
            result = evaluate_svv_integrand(
                p, q, tau=point.tau, points=point.points,
                maximum_twice_levels=cutoff, maximum_total_twice_level=cutoff,
                internal_momenta=point.internal_momenta,
                spectral_weight=point.spectral_weight, precision=config.precision,
                spin_labels=SPINS, block_backend='recursion',
            )
            for ispin, label in enumerate(SPINS):
                # Fixed-spin components are before i^3 and GSO; result.value
                # already includes them. Include them once in this ledger.
                values[ie, il, ispin] = [
                    -.5j * item.value * point.geometry_weight
                    for item in result.fixed_spin[label].components
                ]
            expected = result.value * point.geometry_weight
            if not np.allclose(values[ie, il].sum(), expected, rtol=2e-12, atol=1e-25):
                raise ArithmeticError('component ledger differs from the corrected total')
    if not np.all(np.isfinite(values)):
        raise ArithmeticError('non-finite node; the sample must not be discarded')
    return values


def source_record():
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'source_manifest.json'
    if manifest.exists():
        hashes = json.loads(manifest.read_text())
        for name, expected in hashes.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'deployed source changed: {name}')
        return hashes
    names = (
        'Codes/spin23_genus1_recursive_integral.py', 'Codes/spin23_genus1_corrected_integrand.py',
        'Codes/spin23_genus1_recursive_sewing.py', 'Codes/spin23_genus1_ns_c_recursion.py',
        'Codes/spin23_genus1_branch_recursion.py', 'Codes/spin23_genus1_virasoro_necklace.py',
        'Codes/spin23_genus1_virasoro_collision.py', 'Codes/spin23_genus1_threepoint_integral.py',
        'Codes/spin23_genus1_moduli_integral.py', 'Codes/spin23_genus1_svv_operator_order.py',
        'Codes/spin23_genus1_ns_pairing_audit.py', 'Codes/spin23_type0b_reference.py',
        'Codes/audit_spin23_genus1_threepoint_modularity.py',
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def run_shard(config, task_index, task_count, output):
    count = config.replicates * config.samples_per_replicate
    if not 0 <= task_index < task_count <= count:
        raise ValueError('invalid task partition')
    grid = point_grid(config)
    assignment = shard_assignment(config, task_count, grid)
    started = time.perf_counter()
    completed = 0
    with Path(output).open('x') as stream:
        def emit(record):
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
        emit(dict(kind='header', schema='spin23-recursive-svv-nodes-v1',
                  config=asdict(config), task_index=task_index, task_count=task_count,
                  reference_commit=REFERENCE_COMMIT, source_sha256=source_record(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  block_backend='recursion', cutoff_scheme='total',
                  node_partition='geometry_balanced_round_robin'))
        for flat in (index for index in range(count) if assignment[index] == task_index):
            rep, index = divmod(flat, config.samples_per_replicate)
            unit = grid[rep, index]
            point = map_point(unit, config.regulators)
            before = time.perf_counter()
            emit(dict(kind='started_node', replicate=rep, index=index, unit=unit.tolist()))
            try:
                values = evaluate_node(point, config)
            except Exception as exc:
                emit(dict(kind='failed_node', replicate=rep, index=index,
                          error_type=type(exc).__name__, error=str(exc)))
                raise
            completed += 1
            emit(dict(kind='node', replicate=rep, index=index, unit=unit.tolist(),
                      accepted=point.accepted, permutation=list(point.permutation),
                      tau=complex_json(point.tau), points=complex_json(point.points),
                      internal_momenta=list(point.internal_momenta),
                      geometry_weight=point.geometry_weight, spectral_weight=point.spectral_weight,
                      minimum_gap=point.minimum_gap, minimum_distance=point.minimum_distance,
                      weighted_components=complex_json(values),
                      runtime_seconds=time.perf_counter() - before))
            print(f'task {task_index}: node {completed}, replicate {rep}, sample {index}, '
                  f'accepted={point.accepted}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
        emit(dict(kind='completed_shard', nodes=completed, runtime_seconds=time.perf_counter()-started))


def _complex_array(value):
    if isinstance(value, dict):
        return complex(value['real'], value['imag'])
    return np.array([_complex_array(v) for v in value])


def _statistics(values):
    stats = summarize(values)
    return dict(mean=complex_json(stats['mean']), stderr_real=stats['stderr_real'].tolist(),
                stderr_imag=stats['stderr_imag'].tolist(),
                covariance_real_imag=stats['covariance_real_imag'].tolist())


def reduce_shards(paths):
    headers = []
    nodes = {}
    task_indices = set()
    assignment = None
    for path in paths:
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if not records or records[0].get('kind') != 'header' or records[-1].get('kind') != 'completed_shard':
            raise ValueError(f'incomplete shard: {path}')
        header = records[0]
        if header['task_index'] in task_indices:
            raise ValueError('duplicate shard')
        task_indices.add(header['task_index'])
        if headers and any(header[k] != headers[0][k] for k in (
            'schema', 'config', 'task_count', 'reference_commit', 'source_sha256',
            'block_backend', 'cutoff_scheme',
            'node_partition',
        )):
            raise ValueError('incompatible shard configurations or source revisions')
        headers.append(header)
        if assignment is None:
            cfg = dict(header['config'])
            cfg['energies'] = tuple(map(tuple, cfg['energies']))
            cfg['total_twice_levels'] = tuple(cfg['total_twice_levels'])
            cfg['regulators'] = Regulators(**cfg['regulators'])
            assignment = shard_assignment(ScanConfig(**cfg), header['task_count'])
        found = 0
        for record in records[1:-1]:
            if record['kind'] == 'failed_node':
                raise ValueError('a failed node cannot be omitted from the integral')
            if record['kind'] != 'node':
                continue
            key = (record['replicate'], record['index'])
            if key in nodes:
                raise ValueError('duplicate integration node')
            flat = key[0] * header['config']['samples_per_replicate'] + key[1]
            if not 0 <= flat < len(assignment) or assignment[flat] != header['task_index']:
                raise ValueError('node in the wrong shard')
            nodes[key] = record
            found += 1
        if found != records[-1]['nodes']:
            raise ValueError('inconsistent shard completion count')
    if not headers:
        raise ValueError('no shards supplied')
    header = headers[0]
    cfg = header['config']
    nr, ns = cfg['replicates'], cfg['samples_per_replicate']
    if task_indices != set(range(header['task_count'])) or set(nodes) != {
        (rep, index) for rep in range(nr) for index in range(ns)
    }:
        raise ValueError('missing shards or integration nodes')
    values = np.array([[_complex_array(nodes[rep, index]['weighted_components'])
                        for index in range(ns)] for rep in range(nr)])
    expected_shape = (nr, ns, len(cfg['energies']), len(cfg['total_twice_levels']), 3, 4)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError('invalid component ledger')
    raw = values.mean(axis=1)  # Keep rejected geometry samples in the denominator.
    amplitude = raw.sum(axis=(-1, -2))
    tree_factor = np.array([(p + q) * p * q for p, q in cfg['energies']])
    reduced = amplitude / tree_factor[None, :, None]
    return dict(
        schema='spin23-recursive-svv-regulated-integral-v1',
        status='regulated genus-one integral; boundary completion and physical continuation remain',
        dimension=9, block_backend='recursion', cutoff_scheme='total', config=cfg,
        reference_commit=header['reference_commit'], source_sha256=header['source_sha256'],
        spin_labels=list(SPINS), component_order='GGG, PPG, PGP, GPP in sorted necklace order',
        measures='d^2tau d^2z2 d^2z3 product(dP/pi); includes -i/2 exactly once',
        accepted=[sum(nodes[rep, index]['accepted'] for index in range(ns)) for rep in range(nr)],
        amplitude=_statistics(amplitude), tree_reduced=_statistics(reduced),
        tree_reduced_definition='raw/((p+q)*p*q); no MQM input or fitted energy factors',
        cutoff_differences=_statistics(np.diff(amplitude, axis=2)),
        tree_reduced_cutoff_differences=_statistics(np.diff(reduced, axis=2)),
        tree_reduced_energy_differences=_statistics(reduced[:, 1:] - reduced[:, :1]),
        energy_difference_reference_index=0, replicate_components=complex_json(raw),
        node_files=[str(p) for p in paths],
        statistical_errors='Independent scrambled Sobol replicate errors; do not include block or regulator error',
        remaining=['sampling and block convergence within the chosen level-4 cap',
                   'completion of omitted necklace charts and puncture collision regions',
                   'cusp treatment and Lorentzian continuation',
                   'matching external-state/background conventions to the physical S-matrix'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('--task-index', type=int, required=True)
    worker.add_argument('--task-count', type=int, default=16)
    worker.add_argument('--replicates', type=int, default=4)
    worker.add_argument('--samples', type=int, default=32)
    worker.add_argument('--seed', type=int, default=9023)
    worker.add_argument('--energy', action='append')
    worker.add_argument('--twice-levels', default='6,8')
    worker.add_argument('--tau2-max', type=float, default=2.)
    worker.add_argument('--gap', type=float, default=.12)
    worker.add_argument('--distance', type=float, default=.05)
    worker.add_argument('--output', type=Path, required=True)
    reducer = sub.add_parser('reduce')
    reducer.add_argument('shards', nargs='+', type=Path)
    reducer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output artifact')
    if args.command == 'worker':
        energies = tuple(tuple(map(float, x.split(','))) for x in args.energy) if args.energy else ScanConfig().energies
        config = ScanConfig(energies=energies, total_twice_levels=tuple(map(int, args.twice_levels.split(','))),
                            replicates=args.replicates, samples_per_replicate=args.samples, seed=args.seed,
                            regulators=Regulators(args.tau2_max, args.gap, args.distance))
        run_shard(config, args.task_index, args.task_count, args.output)
    else:
        result = reduce_shards(args.shards)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
