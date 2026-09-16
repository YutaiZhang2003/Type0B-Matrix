#!/usr/bin/env python3
"""Check convergence of the entire complex integrated spin matrices."""
from pathlib import Path
import cmath
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/genus_2'))
import numpy as np
import refine_nsrr_bilinear_quadrature as run


def main():
    history = run.read(run.OUT / 'target_refinement.json')
    config = run.read(run.OUT / 'config.json')
    rows = []; hashes = {}; by_channel = {}
    for channel in ('source', 'target'):
        orders = list(range(3, 8)) if channel == 'source' else [r['target_N'] for r in history['results']]
        previous = None; records = []
        for n in orders:
            path = run.OUT / (f'order-N{n}.json' if n <= 7 else f'target-refinement-N{n}.json')
            hashes[str(path.relative_to(ROOT))] = run.sha(path)
            data = run.read(path)
            m = np.array([[run.decode(z) for z in row] for row in data[f'{channel}_density']])
            norm = np.linalg.norm(m)
            row = dict(channel=channel, N=n, norm=float(norm), off_diagonal_phase_rad=cmath.phase(m[0, 1]),
                       relative_matrix_step=None, off_diagonal_phase_step_rad=None)
            if previous is not None:
                row['relative_matrix_step'] = float(np.linalg.norm(m - previous) / np.linalg.norm(previous))
                row['off_diagonal_phase_step_rad'] = cmath.phase(m[0, 1] / previous[0, 1])
            assert np.linalg.norm(m - m.conjugate().T) < 1e-12 * norm
            v, w = run.VECTORS['s=+1'], run.VECTORS['s=-1']
            assert abs(v @ m @ v.conjugate() + w @ m @ w.conjugate() - np.trace(m)) < 1e-12 * norm
            rows.append(row); records.append(row); previous = m
        by_channel[channel] = records[-2:]
    stable = all(r['relative_matrix_step'] < config['relative_stability_tolerance']
                 for records in by_channel.values() for r in records)
    result = dict(full_complex_matrices_stable=stable, tolerance=config['relative_stability_tolerance'],
        rows=rows, last_two_steps=by_channel,
        phase_scope='Change in arg(K_01) within each channel under quadrature refinement. This is not an independent interacting spin-transport derivation.',
        script_sha256=run.sha(Path(__file__)), input_sha256=hashes)
    run.save(run.OUT / 'matrix_convergence.json', result)
    print(json.dumps(dict(full_complex_matrices_stable=stable, last_two_steps=by_channel), indent=2))


if __name__ == '__main__':
    main()
