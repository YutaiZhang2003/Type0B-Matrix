#!/usr/bin/env python3
"""Reproduce the matched N3..N7 stage and the independent target refinement.

Existing complete nodes are verified and reused. This entry point does
not schedule source N8 or higher, since the source stability is checked
before starting the independent target continuation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/genus_2'))
import refine_nsrr_bilinear_quadrature as run
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    config = run.prepare(run.OUT, maximum_order=10, tolerance=1e-4)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    script = ROOT / 'Code/genus_2/refine_nsrr_bilinear_quadrature.py'
    def work(task):
        channel, n, index = task
        subprocess.run([sys.executable, str(script), '--output', str(run.OUT), '--node',
                        channel, str(n), str(index)], env=env, check=True)
    history = []
    for n in range(3, 8):
        tasks = [(channel, n, index) for channel in ('source', 'target')
                 if not (channel == 'source' and n == 3 or channel == 'target' and n in (3, 4))
                 for index in range(n ** 3)
                 if not (run.OUT / channel / f'N{n}' / f'node-{index:04d}.json').exists()]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(work, task) for task in tasks]):
                future.result()
        history.append(run.reduce_order(run.OUT, config, n))
        run.report(run.OUT, config, history)
        print(f'Matched N={n} complete.', flush=True)
    # Recreate the reuse controls when starting from a fresh output directory.
    for filename, tasks in (
        ('target_reuse_check.json', ((4, 0),)),
        ('target_extra_reuse_checks.json', ((3, 13), (4, 21), (4, 63))),
    ):
        if (run.OUT / filename).exists():
            continue
        checks = []
        for n, index in tasks:
            fresh = run.target_node(config, n, index)
            path = run.BASE / 'target' / f'N{n}' / f'node-{index:03d}.json'
            old = next(r for r in run.read(path)['rows'] if r['t'] == .6 and r['order'] == 12)
            error = float(np.max(abs(run.matrix(fresh) - run.matrix(old))) / np.max(abs(run.matrix(old))))
            assert error < 1e-12
            checks.append(dict(N=n, index=index, relative_error=error,
                               input_sha256=run.sha(path), global_max_occupation_used=fresh['global_max_occupation_used']))
        run.save(run.OUT / filename, checks[0] if len(checks) == 1 else checks)
    # A fresh reproduction uses the regular continuation for the entire N9
    # grid. If resuming the original run, first finish any assigned tail nodes.
    plan_path = run.OUT / 'precompute-N9-tail.json'
    if not plan_path.exists():
        run.save(plan_path, dict(N=9, channel='target', indices=[],
                                purpose='No separate tail batch in this reproduction'))
    external = run.read(plan_path)['indices']
    tasks = [('target', 9, i) for i in external
             if not (run.OUT / 'target/N9' / f'node-{i:04d}.json').exists()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, task) for task in tasks]):
            future.result()
    subprocess.run([sys.executable, str(ROOT / 'Code/genus_2/continue_nsrr_target_quadrature.py'),
                    '--workers', str(args.workers)], env=env, check=True)


if __name__ == '__main__':
    main()
