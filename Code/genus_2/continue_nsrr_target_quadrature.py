#!/usr/bin/env python3
"""Refine the remaining all-NS integral after the NSRR integral stabilizes.

Uses the unchanged worker and frozen numerical configuration. Source and
target quadrature orders are recorded independently; incomplete source
grids never enter a result. No extrapolation or normalization fit is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/genus_2'))
import numpy as np
import refine_nsrr_bilinear_quadrature as run


def context():
    config = run.read(run.OUT / 'config.json')
    summary = run.read(run.OUT / 'summary.json')
    assert summary['completed_orders'] == [3, 4, 5, 6, 7]
    for path, digest in {**config['implementation_sha256'], **config['baseline_sha256']}.items():
        assert run.sha(ROOT / path) == digest, path
    source_steps = []
    for earlier, later in zip(summary['results'][-3:-1], summary['results'][-2:]):
        step = max(abs(later['values'][k]['source_Z'] / earlier['values'][k]['source_Z'] - 1)
                   for k in run.VECTORS)
        source_steps.append(step)
    assert len(source_steps) == 2 and max(source_steps) < config['relative_stability_tolerance']
    source = run.read(run.OUT / 'order-N7.json')
    design = dict(source_N=7, source_stable_relative_steps=source_steps,
        target_orders=list(range(8, 13)), tolerance=config['relative_stability_tolerance'],
        required_successive_steps=2, source_L=5, target_R=12,
        frozen_config_sha256=run.sha(run.OUT / 'config.json'),
        source_integral_sha256=run.sha(run.OUT / 'order-N7.json'),
        driver_sha256=run.sha(Path(__file__)),
        policy='Refine the two integrals independently. Reuse only complete grids. The partial N8 source grid is excluded.',
        external_N9_tail='precompute-N9-tail.json')
    path = run.OUT / 'target_refinement_design.json'
    if path.exists():
        assert run.read(path) == design
    run.save(path, design)
    return config, summary, source, design


def integrate_target(config, n):
    digest = run.object_digest(config)
    matrices = []; hashes = {}; maximum_occupation = 0
    for i in range(n ** 3):
        path = run.OUT / 'target' / f'N{n}' / f'node-{i:04d}.json'
        row = run.read(path)
        assert (row['config_digest'], row['channel'], row['N'], row['index']) == (digest, 'target', n, i)
        assert row['global_nonconverged_calls'] == 0
        maximum_occupation = max(maximum_occupation, row['global_max_occupation_used'])
        matrices.append(run.matrix(row)); hashes[str(path.relative_to(ROOT))] = run.sha(path)
        p, measure = run.grid(config, 'target', n, i)
        assert max(abs(a - b) for a, b in zip(p, row['momenta'])) < 1e-13
        assert abs(measure / row['measure'] - 1) < 1e-12
    m = np.array([[run.csum(a[i, j] for a in matrices) for j in (0, 1)] for i in (0, 1)])
    scale = np.max(abs(m))
    assert np.max(abs(m - m.conjugate().T)) < 1e-12 * scale
    assert np.min(np.linalg.eigvalsh(m)) > -1e-12 * scale
    run.save(run.OUT / f'target-provenance-N{n}.json', hashes)
    return m, maximum_occupation


def result(config, source, n):
    target, maximum = integrate_target(config, n)
    source_matrix = np.array([[run.decode(z) for z in row] for row in source['source_density']])
    values = {}
    for label, vector in run.VECTORS.items():
        zs, zt = run.scalar(source_matrix, vector), run.scalar(target, vector)
        values[label] = dict(source_Z=zs, target_Z=zt, ratio=zs / zt / source['free_frame_power'])
    row = dict(source_N=7, target_N=n, target_nodes=n ** 3, values=values,
               source_density=source['source_density'], target_density=run.encoded_matrix(target),
               maximum_global_occupation=maximum, complete=True)
    run.save(run.OUT / f'target-refinement-N{n}.json', row)
    return row


def report(history, design):
    rows = []; steps = []; streak = 0
    for index, current in enumerate(history):
        maxima = dict(source_Z=0., target_Z=0., ratio=0.)
        for label, values in current['values'].items():
            row = dict(source_N=current['source_N'], target_N=current['target_N'], comparison=label,
                       **values, source_relative_step=None, target_relative_step=None, ratio_relative_step=None)
            if index:
                previous = history[index - 1]['values'][label]
                for key, field in (('source_Z', 'source_relative_step'), ('target_Z', 'target_relative_step'),
                                   ('ratio', 'ratio_relative_step')):
                    change = values[key] / previous[key] - 1
                    row[field] = change; maxima[key] = max(maxima[key], abs(change))
            rows.append(row)
        if index:
            streak = streak + 1 if max(maxima['target_Z'], maxima['ratio']) < design['tolerance'] else 0
        steps.append(dict(source_N=current['source_N'], target_N=current['target_N'],
                          maximum_relative_changes=maxima if index else None,
                          consecutive_stable_target_steps=streak))
    stable = streak >= design['required_successive_steps']
    output = dict(status='quadrature_stabilized' if stable else 'refinement_in_progress',
        source_N=7, source_stable_relative_steps=design['source_stable_relative_steps'],
        source_L=5, target_R=12, relative_tolerance=design['tolerance'],
        required_successive_steps=design['required_successive_steps'], results=history, steps=steps,
        scope='Central surface t=0.60, b=1.4. Source and target are refined independently. Empirical quadrature stability at fixed block cutoffs; no rigorous remainder bound.')
    run.save(run.OUT / 'target_refinement.json', output)
    run.write_csv(run.OUT / 'target_refinement.csv', rows)
    lines = ['# Convergence of the NSRR/all-NS momentum integrals', '',
        f'Status: **{output["status"]}**. Surface t=0.60, b=1.4, source L=5, target R=12.', '',
        'The source meets the two-step convergence criterion at N=7 and is then held fixed while the target '
        'is refined further. The two quadrature orders are stated separately. The pairing, spin transport, '
        'three-point constants, and channel primary factors are unchanged. Agreement requires normalized ratio one.', '',
        '| Source N | Target N | Ratio s=+ | Ratio s=- | Spin-sum ratio | Max target relative step |',
        '|---:|---:|---:|---:|---:|---:|']
    for row, step in zip(history, steps):
        v = row['values']; change = step['maximum_relative_changes']
        text = '—' if change is None else f'{change["target_Z"]:.3e}'
        lines.append(f'| {row["source_N"]} | {row["target_N"]} | {v["s=+1"]["ratio"]:.10f} | '
                     f'{v["s=-1"]["ratio"]:.10f} | {v["spin_sum"]["ratio"]:.10f} | {text} |')
    lines += ['', 'The stabilization criterion is two successive relative changes below 1e-4 (0.01%) '
        'for each integral, each sign, each resolved spin, and their sum. The last two source refinements give '
        + ', '.join(f'{v:.3e}' for v in design['source_stable_relative_steps']) + '.', '',
        'Only complete grids enter the table. N means nodes per momentum direction; the target uses N^3 nodes. '
        'An unused partial source N8 grid is excluded. The unchanged target worker is reused, including its '
        'global-series convergence requirement and independent momentum-measure check.', '',
        'See `target_refinement.json` and `target_refinement.csv` for both raw integrals, resolved spin ratios, '
        'and successive changes. `target_refinement_design.json` records the source stabilization and frozen inputs. '
        'The initial matched-grid comparison is in `README.md` and `summary.json`.', '',
        'Reproduce the remaining refinement with `python Code/genus_2/continue_nsrr_target_quadrature.py --workers 6`. '
        'The worker program and original configuration remain unchanged.', '']
    (run.OUT / 'CONVERGENCE.md').write_text('\n'.join(lines))
    return stable, steps[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    config, old, source, design = context()
    history = [dict(source_N=r['N'], target_N=r['N'], values=r['values']) for r in old['results']]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    script = ROOT / 'Code/genus_2/refine_nsrr_bilinear_quadrature.py'
    def dispatch(task):
        n, i = task
        with (run.OUT / 'logs' / f'continue-target-N{n}-{i:04d}.log').open('w') as stream:
            subprocess.run([sys.executable, str(script), '--output', str(run.OUT),
                            '--node', 'target', str(n), str(i)], env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    report(history, design)
    for n in design['target_orders']:
        external = set(run.read(run.OUT / design['external_N9_tail'])['indices']) if n == 9 else set()
        tasks = [(n, i) for i in range(n ** 3) if i not in external
                 and not (run.OUT / 'target' / f'N{n}' / f'node-{i:04d}.json').exists()]
        print(f'Target N={n}: {len(tasks)} remaining node evaluations; source fixed at N=7.', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(dispatch, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done % 50 == 0 or done == len(tasks):
                    print(f'Target N={n}: {done}/{len(tasks)} newly completed.', flush=True)
        # The separately scheduled tail batch owns these nodes. Do not race it.
        deadline = time.monotonic() + 600
        while any(not (run.OUT / 'target' / f'N{n}' / f'node-{i:04d}.json').exists() for i in external):
            if time.monotonic() > deadline:
                raise TimeoutError('External tail batch not complete; inspect its log before resuming')
            time.sleep(5)
        row = result(config, source, n); history.append(row)
        stable, step = report(history, design)
        print(json.dumps(dict(ratios={k: v['ratio'] for k, v in row['values'].items()}, **step)), flush=True)
        if stable:
            print('Both integrals meet the independent quadrature stability criterion.', flush=True)
            return
    print('Target refinement exhausted the design without meeting the criterion.', flush=True)


if __name__ == '__main__':
    main()
