"""Bounded coefficient-only replay of rejected Ramond nodes.

The oracle evaluates fixed-weight Ward kernels and inverse Gram matrices at
b=1. It never calls the two-Virasoro recurrence. This script neither submits
jobs nor builds/merges production banks or integrates an amplitude.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import hashlib
import os
from pathlib import Path
import time

import numpy as np


def direct_tables(internal, external, word, cutoff=8):
    from spin23_genus1_amplitude import G_MINUS_HALF as G
    from spin23_genus1_ramond_point_sewing import vertex_pair
    from spin23_ramond_fast import _fast_edge_factor
    forms = tuple(f for f in product((0, 1), repeat=3) if sum(f) % 2 == sum(word) % 2)
    signs = tuple(product((-1, 1), repeat=3))
    levels = tuple(k for k in product(range(0, cutoff+1, 2), repeat=3) if sum(k) <= cutoff)
    values = np.zeros((len(forms), len(signs), len(levels)), complex)
    condition = 0.
    for ki, k in enumerate(levels):
        matrices = []
        for v in range(3):
            left = _fast_edge_factor(k[v-1], internal[v-1], 1, 1e11)
            right = _fast_edge_factor(k[v], internal[v], 1, 1e11)
            parts = vertex_pair(k[v-1], k[v], internal[v-1], internal[v],
                complex((1+external[v]**2)/2), G if word[v] else (), 1, 1, 1e11)
            parity = np.array([[(x.parity+y.parity+word[v]) % 2 for y in right.basis]
                               for x in left.basis])
            matrices.append((parts, parity))
            condition = max(condition, left.condition_number, right.condition_number)
        for fi, form in enumerate(forms):
            for si, sign in enumerate(signs):
                m = [parts[0 if s == -1 else 1]*(parity == f)
                     for (parts, parity), s, f in zip(matrices, sign, form)]
                values[fi, si, ki] = np.trace(m[0] @ m[1] @ m[2])
    return np.array(levels), values, forms, signs, condition


def comparison(actual, direct):
    levels, values, forms, signs = actual[:4]
    dl, dv, df, ds = direct[:4]
    assert forms == df and signs == ds
    lookup = {tuple(k): i for i, k in enumerate(dl)}
    target = np.stack([dv[..., lookup[tuple(k)]] for k in levels], axis=-1)
    error = abs(values-target)
    scales = np.maximum(1., np.max(abs(target), axis=-1))
    scaled = error/scales[..., None]
    worst = np.unravel_index(np.argmax(scaled), scaled.shape)
    return dict(maximum_absolute_error=float(error.max()),
        maximum_scaled_error=float(scaled.max()),
        worst_form=forms[worst[0]], worst_sign=signs[worst[1]],
        worst_level=levels[worst[-1]].tolist(),
        maximum_gram_condition=direct[-1])


def replay(task):
    from spin23_genus1_branch_recursion import self_dual_ramond_tables, checked_ramond_tables
    record, contours, samples, words, cutoff, confirm = task
    row = record['record']; internal = tuple(row['momentum'])
    external = (sum(row['energy']), *row['energy'])
    result = dict(file=record['file'], momentum=internal, energy=row['energy'], words=[])
    for word in words:
        started = time.perf_counter()
        direct = direct_tables(internal, external, word, cutoff)
        options = dict(internal_momenta=internal, external_momenta=external,
            external_descendants=word, maximum_twice_levels=cutoff,
            maximum_total_twice_level=cutoff)
        tests = []
        for radii in contours or (None,):
            before = time.perf_counter()
            try:
                if radii is None:
                    actual = checked_ramond_tables(**options)
                else:
                    # Diagnostic access to rejected arrays, never an acceptance override.
                    actual = self_dual_ramond_tables(**options, radius=radii[0],
                        check_radius=radii[1], samples=samples, tolerance=float('inf'))
                error = comparison(actual, direct)
                tests.append(dict(contours=radii, diagnostics=actual[-1], **error,
                    passed=error['maximum_scaled_error'] <= 2e-7 and
                        actual[-1]['scaled_finite_part_error'] <= 2e-7,
                    seconds=time.perf_counter()-before))
                if radii is None and confirm:
                    before = time.perf_counter()
                    finer = checked_ramond_tables(**options,
                        sample_counts=(2*actual[-1]['samples'], 4*actual[-1]['samples']))
                    fine_error = comparison(finer, direct)
                    difference = np.max(abs(finer[1]-actual[1]), axis=-1)
                    scale = np.maximum(1., np.max(abs(finer[1]), axis=-1))
                    refinement = float(np.max(difference/scale))
                    tests.append(dict(contours=None, diagnostics=finer[-1], **fine_error,
                        angular_refinement_error=refinement,
                        passed=fine_error['maximum_scaled_error'] <= 2e-7 and refinement <= 2e-7,
                        seconds=time.perf_counter()-before))
            except (ArithmeticError, ZeroDivisionError) as exc:
                tests.append(dict(contours=radii, passed=False, error=str(exc),
                                  seconds=time.perf_counter()-before))
        result['words'].append(dict(word=word, tests=tests, seconds=time.perf_counter()-started))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--names', nargs='*')
    parser.add_argument('--index', type=int, help='Replay exactly one record, for a validation array')
    parser.add_argument('--contours', nargs='*', help='Diagnostic radius pairs, e.g. .35,.4')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cutoff', type=int, default=8)
    parser.add_argument('--ggg-only', action='store_true')
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text())
    if args.names:
        records = [r for r in records if r['file'] in args.names]
        if len(records) != len(set(args.names)):
            raise ValueError('requested failure names not all present')
    if args.index is not None:
        if not 0 <= args.index < len(records):
            raise ValueError('record index out of range')
        records = [records[args.index]]
    words = ((1, 1, 1),) if args.ggg_only else ((1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0))
    contours = [tuple(map(float, item.split(','))) for item in args.contours or []]
    report = dict(scope='coefficient-only validation; no production submission',
        configuration=vars(args), results=[], job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), (Path(__file__).resolve().parents[1] / 'Codes/spin23_genus1_branch_recursion.py'))})
    def retain(row):
        report['results'].append(row)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
    tasks = [(record, contours, args.samples, words, args.cutoff, args.confirm) for record in records]
    if args.workers == 1:
        for task in tasks:
            retain(replay(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(replay, task) for task in tasks]):
                retain(future.result())
    if any(not test['passed'] for row in report['results'] for word in row['words'] for test in word['tests']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
