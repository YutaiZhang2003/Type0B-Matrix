#!/usr/bin/env python3
"""Produce the final report only after both integrals and matrices stabilize."""
from pathlib import Path
import cmath
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/genus_2'))
import numpy as np
import refine_nsrr_bilinear_quadrature as run


def main():
    out = run.OUT
    study = run.read(out / 'target_refinement.json')
    matrix_check = run.read(out / 'matrix_convergence.json')
    assert study['status'] == 'quadrature_stabilized'
    assert matrix_check['full_complex_matrices_stable']
    config = run.read(out / 'config.json')
    design = run.read(out / 'target_refinement_design.json')
    assert run.sha(ROOT / 'Code/genus_2/continue_nsrr_target_quadrature.py') == design['driver_sha256']
    assert run.sha(out / 'config.json') == design['frozen_config_sha256']
    assert run.sha(out / 'order-N7.json') == design['source_integral_sha256']
    hashes = {**config['implementation_sha256'], **config['baseline_sha256']}
    for n in range(3, 8):
        hashes.update(run.read(out / f'provenance-N{n}.json'))
    for row in study['results']:
        if row['target_N'] >= 8:
            hashes.update(run.read(out / f'target-provenance-N{row["target_N"]}.json'))
    hashes.update(matrix_check['input_sha256'])
    for path, expected in hashes.items():
        assert run.sha(ROOT / path) == expected, path
    final = study['results'][-1]
    detailed = run.read(out / f'target-refinement-N{final["target_N"]}.json')
    matrices = {channel: np.array([[run.decode(z) for z in row] for row in detailed[f'{channel}_density']])
                for channel in ('source', 'target')}
    phase_difference = cmath.phase(matrices['source'][0, 1] / matrices['target'][0, 1])
    shape_difference = float(np.linalg.norm(matrices['source'] / np.trace(matrices['source'])
                                          - matrices['target'] / np.trace(matrices['target'])))
    diagnostics = {key: dict(**v, four_times_ratio_minus_one=4 * v['ratio'] - 1)
                   for key, v in final['values'].items()}
    result = dict(status='quadrature_stabilized_at_fixed_block_cutoffs', t=.6, b=config['b'],
        source_N=7, target_N=final['target_N'], source_L=5, target_R=12, values=diagnostics,
        last_source_steps=study['source_stable_relative_steps'], last_target_steps=study['steps'][-2:],
        matrix_steps=matrix_check['last_two_steps'], off_diagonal_phase_difference_rad=phase_difference,
        trace_normalized_matrix_difference_frobenius=shape_difference,
        interpretation='The low-N ratio was not stabilized. The refined central-surface ratio remains near one quarter with the original pairing and coefficients. No factor four is applied. This is quadrature convergence, not a full block-truncation or interacting-spin-transport validation.',
        scope='One surface, t=0.60. Other saved surfaces have not received this quadrature refinement.',
        excluded='The interrupted partial source N8 grid is not used.')
    run.save(out / 'final_result.json', result)
    for p in (out / 'target_refinement.json', out / 'matrix_convergence.json', out / 'final_result.json',
              out / 'target_refinement_design.json', Path(__file__),
              ROOT / 'Code/genus_2/reproduce_nsrr_bilinear_quadrature.py',
              ROOT / 'Code/genus_2/check_nsrr_quadrature_matrices.py',
              ROOT / 'Code/genus_2/audit_nsrr_bilinear_quadrature.py'):
        hashes[str(p.relative_to(ROOT))] = run.sha(p)
    run.save(out / 'final_provenance.json', hashes)
    lines = ['# Quadrature convergence of the new NSRR proposal', '',
        '**Both momentum integrals and their full complex spin matrices meet the two-step 0.01% stability criterion.** '
        'This statement is for the central saved surface t=0.60 at b=1.4, with source total descendant cutoff L=5 '
        'and target recursion twice-level R=12 (resummed global blocks).', '',
        r'The reported ratio is $\mathcal R=(Z_{\rm NSRR}/Z_{\rm NSNSNS})'
        r'(Z_{{\rm free},t}/Z_{{\rm free},s})^{1+2(b+b^{-1})^2}$. Agreement requires one.', '',
        'The source converges at N=7 and is then held fixed while the target is refined. '
        'N is the number of nodes per momentum direction, so each grid has N^3 nodes.', '',
        '| Source N | Target N | Ratio s=+ | Ratio s=- | Spin-sum ratio |',
        '|---:|---:|---:|---:|---:|']
    for row in study['results']:
        v = row['values']
        lines.append(f'| {row["source_N"]} | {row["target_N"]} | {v["s=+1"]["ratio"]:.9f} | '
                     f'{v["s=-1"]["ratio"]:.9f} | {v["spin_sum"]["ratio"]:.9f} |')
    lines += ['', '## Convergence evidence', '',
        '- Last two maximum relative source changes: '
        + ', '.join(f'{100 * z:.6f}%' for z in study['source_stable_relative_steps']) + '.',
        '- Last two maximum relative target changes: '
        + ', '.join(f'{100 * z["maximum_relative_changes"]["target_Z"]:.6f}%' for z in study['steps'][-2:]) + '.',
        '- These checks cover each sign, each resolved spin, and their sum, with each integral checked separately.',
        '- The full complex 2x2 integrated matrices also pass two successive relative Frobenius-norm changes below 1e-4; '
        '`matrix_convergence.json` records their norms and off-diagonal phases.',
        '- All accepted target global sums converge. Input hashes, complete grids, primary powers, and momentum measures are checked.', '',
        'These are empirical convergence observations at fixed block cutoffs, not rigorous remainder bounds. '
        'Only t=0.60 has been refined to this criterion.', '', '## Normalization and phase', '',
        'The original coefficients, pairing, free-field frame factor, and primary factors remain fixed. '
        'No multiplicative factor is fitted or applied. For comparison with the earlier normalization discussion, '
        'the diagnostic 4R-1 at the final grids is:', '',
        '| Combination | 4R-1 |', '|---|---:|']
    for label in ('s=+1', 's=-1', 'spin_sum'):
        lines.append(f'| {label} | {100 * diagnostics[label]["four_times_ratio_minus_one"]:+.6f}% |')
    lines += ['', 'Thus the coarse quadrature caused much of the smaller residual seen after a hypothetical factor four. '
        'The near-quarter value does not itself derive a correction to M.', '',
        f'The integrated off-diagonal phase difference is `{phase_difference:.9e}` radians in the tested spin basis. '
        f'The Frobenius distance between the trace-normalized matrices is `{shape_difference:.9e}`. '
        'These describe the integrated spin matrix K, not the three-point coefficient matrix M. '
        'Their remaining differences must still be assessed against block truncation and the proposed interacting spin transport.', '',
        'All channel-dependent q^h factors remain outside descendant blocks and M. Complex lift data are retained; '
        'the spin sum removes relative-phase interference.', '', '## Reproduction and data', '',
        '```sh', 'python Code/genus_2/reproduce_nsrr_bilinear_quadrature.py --workers 6',
        'python Code/genus_2/audit_nsrr_bilinear_quadrature.py',
        'python Code/genus_2/check_nsrr_quadrature_matrices.py',
        'python Code/genus_2/report_nsrr_quadrature_limit.py', '```', '',
        '- `final_result.json`: final values, convergence steps, and limitations.',
        '- `target_refinement.csv`: both channel integrals and all ratios at every completed order.',
        '- `matrix_convergence.json`: full complex matrix convergence.',
        '- `final_provenance.json`: verified input and node hashes.',
        '- `source/`, `target/`: numerical nodes. The unused partial source N8 grid is explicitly excluded.', '']
    (out / 'FINAL_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
