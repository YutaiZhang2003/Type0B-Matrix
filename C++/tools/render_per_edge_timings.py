"""Document the saved independent-plumbing-cutoff run; no numerical block evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'C++/results/per_edge_level10_2026-09-11'


def main():
    measurement_dir = FOLDER / 'optimized_full_pipeline_40dps'
    if not (measurement_dir / 'timings.json').exists():
        measurement_dir = FOLDER / 'full_pipeline_40dps'
    record = json.loads((measurement_dir / 'timings.json').read_text())
    runs = record['runs']
    lines = [
        '# Independent level-10 plumbing cutoffs', '',
        'Keep q1^(a/2) q2^l q3^m for 0 <= a <= 20 and 0 <= l,m <= 10. '
        'There are 2,541 monomials and 20,328 parity components, including the corner '
        'q1^10 q2^10 q3^10. This is not a total-level-10 truncation.', '',
        'Both sectors use 40 decimal digits (136 bits), b=7/5, '
        '(P1,P2,P3)=(11/23,13/29,17/31), p=f=0. Ordinary uses (+,+); inserted uses (+,-). '
        'Cases run sequentially as fresh processes with empty numerical caches. '
        'Compilation and validation are excluded from the measured wall times. '
        'Intermediate branching coefficients are stored and reused within each run.', '',
        'Before zero residues or zero branch prefactors are removed, the independent '
        'domains permit at most 42,861,104 global-seed summands for ordinary and '
        '1,740,820,224 for inserted. For a single edge with remaining budget L, the '
        'number of pairs (target, accumulated shift) is L+1+L(L-1)/2, since the shift '
        'is zero or at least two. Multiply these counts across edges and sum over '
        'branches, counting each transposed inserted product only once. These are '
        'work bounds, not elapsed-time estimates; actual counters are recorded below.', '',
        '## Measurements', '',
        '| Sector | Status | Internal total (s) | Full wall (s) |',
        '| --- | --- | ---: | ---: |',
    ]
    for r in runs:
        timing = r.get('timing_seconds', {})
        internal = f"{timing['total']:.3f}" if 'total' in timing else 'pending'
        wall = f"{r['wall_seconds']:.3f}" if 'wall_seconds' in r else 'pending'
        lines.append(f"| {r['mode']} | {r['status']} | "
                     f"{internal} | {wall} |")
    if measurement_dir.name == 'optimized_full_pipeline_40dps':
        lines += ['', 'The optimized measurements use complete inserted-vertex caches, '
                  'reused MPC temporary storage, and cached inverse denominators with '
                  'common residues factored outside the incoming sums. Precision and '
                  'retained coefficient domains are unchanged. See '
                  '../ccy_vertex_reuse_2026-09-11/README.md for directed validation.', '',
                  'The original ordinary run completed in 508.004 s. The original inserted '
                  'run was interrupted after 1,608.892 s for this restart; that duration is '
                  'not a completed block timing. Their records remain in full_pipeline_40dps. '
                  'The invalid_overlap_attempt directory contains an aborted overlapping '
                  'attempt and is excluded from every measurement and comparison.']
    completed = [r for r in runs if r['status'] == 'completed']
    if completed:
        lines += ['', '| Stage | ' + ' | '.join(r['mode'] for r in completed) + ' |',
                  '| --- | ' + ' | '.join('---:' for _ in completed) + ' |']
        for name, key in [('Initial mode actions', 'actions'),
                          ('Outer branching recurrence', 'outer_ward'),
                          ('Middle recurrence', 'middle'), ('Reduced CCY', 'ccy'),
                          ('Virasoro products', 'products'), ('Branch assembly', 'assembly'),
                          ('Direct fermion', 'auxiliary'), ('Sector division', 'division'),
                          ('Schottky vacuum and square', 'schottky'),
                          ('Vacuum restoration', 'restoration')]:
            lines.append('| ' + name + ' | ' +
                         ' | '.join(f"{r['timing_seconds'][key]:.3f}" for r in completed) + ' |')
        tracked = ('actions', 'outer_ward', 'middle', 'ccy', 'products', 'assembly',
                   'auxiliary', 'division', 'schottky', 'restoration')
        lines.append('| Other pipeline overhead | ' + ' | '.join(
            f"{r['timing_seconds']['total'] - sum(r['timing_seconds'][key] for key in tracked):.3f}"
            for r in completed) + ' |')
        lines.append('| Startup, serialization and exit | ' + ' | '.join(
            f"{r['wall_seconds'] - r['timing_seconds']['total']:.3f}"
            for r in completed) + ' |')
        lines += ['', 'All stage times are seconds. Internal total is timed independently; '
                  'other pipeline overhead is its difference from the individually timed '
                  'stages, including work and cache cleanup outside those timers. Wall '
                  'additionally includes startup, final serialization, cleanup and exit.', '',
                  '## Output and diagnostics', '']
        expected = {(a, b, b, d) for a in range(21) for b in range(11)
                    for d in range(0, 21, 2)}
        for r in completed:
            data = json.loads(Path(r['result']).read_text())
            keys = {tuple(row['exponents']) for row in data['coefficients']}
            assert keys == expected and len(data['coefficients']) == len(expected)
            assert data['q_level_cutoffs'] == [10, 10, 10]
            assert data['truncation'] == 'per-edge' and data['dps'] == 40
            assert all(len(row['values']) == 8 for row in data['coefficients'])
            c = data['counts']
            lines += [f"- {r['mode']}: complete 2,541-monomial box; "
                      f"{c['branches']:,} branch tuples; {c['virasoro_blocks']:,} Virasoro blocks; "
                      f"{c['ccy_seed_terms']:,} global-seed summands; "
                      f"{c['ccy_transitions']:,} CCY transitions. "
                      f"Maximum recorded recovery-sector residual: "
                      f"{data['diagnostics']['maximum_sector_residual']:.3e}."]
        lines += ['', 'Runs use sector-policy record. Residuals test consistency, not every '
                  'coefficient\'s accuracy. No new physical PBW block or independent full-box '
                  'reference was computed.']
    inspection_file = FOLDER / 'optimized_output_inspection.json'
    if measurement_dir.name == 'optimized_full_pipeline_40dps' and inspection_file.exists():
        inspection = json.loads(inspection_file.read_text())
        lines += ['', 'Inspection of the saved files confirms all 20,328 physical components '
                  'in each sector are finite and the complete requested monomial domains are '
                  'present. Source and executable hashes match those recorded at launch.']
        for r in inspection['runs']:
            comparison = r.get('saved_baseline_comparison')
            if comparison:
                lines += ['', 'Comparing all ordinary physical components with the already '
                          'saved pre-optimization 40-digit result gives maximum scaled difference '
                          f"{float(comparison['maximum_scaled_difference']):.3e} and maximum "
                          f"absolute difference {float(comparison['maximum_absolute_difference']):.3e}. "
                          'The scale is max(1, abs(reference), abs(current)); coefficients reach '
                          f"{float(r['maximum_coefficient_magnitude']):.3e}. This is a "
                          'same-precision implementation comparison, not independent accuracy '
                          'certification. The inserted full box has no completed pre-optimization '
                          'reference; its directed small-cutoff and individual-factor comparisons '
                          'are documented in ../ccy_vertex_reuse_2026-09-11/README.md. '
                          'This inspection only reads saved results and evaluates no blocks.']
    estimate_file = FOLDER / 'work_estimates.json'
    calibration_file = FOLDER / 'pbw_40digit_arithmetic_calibration.json'
    if estimate_file.exists():
        estimates = json.loads(estimate_file.read_text())
        latest = estimates['inserted_progress_estimates'][-1]
        lines += ['', '## Count-based estimates', '',
                  f"At {latest['completed_branch_tuples']} completed inserted tuples "
                  f"({latest['elapsed_seconds']:.1f} s elapsed), replaying the weighted branch "
                  f"domains estimated {latest['estimated_full_seconds']/3600:.2f} hours for the "
                  'full inserted 40-digit run with the former CCY implementation. This '
                  'historical forecast assumes similar seconds per seed term; it does not '
                  'describe the optimized run reported above.', '',
                  'The physical PBW box has 450,469 distinct Gram entries, the same as at total '
                  'level 10. It instead requests 1,318,075,412 three-point tensor entries and '
                  '254,809,434,024 dense-contraction complex multiplications. Its largest '
                  'single tensor has 8,665,664 entries. These counts do not evaluate any PBW block.', '']
    if calibration_file.exists():
        calibration = json.loads(calibration_file.read_text())
        lines += ['### PBW at matched 40-digit precision (136 bits)', '',
                  '| Saved model, recalibrated to 136 bits | Positive sector (hours) | Negative sector (hours) |',
                  '| --- | ---: | ---: |']
        for model in calibration['estimated_136_bit_pbw']:
            positive = model['positive_sector_seconds_range']
            negative = model['negative_sector_seconds_range']
            lines.append(f"| {model['model']} | {positive[0]/3600:.1f}–{positive[1]/3600:.1f} | "
                         f"{negative[0]/3600:.1f}–{negative[1]/3600:.1f} |")
        lines += ['', 'These are estimates for the existing Python PBW implementation. '
                  'Synthetic operations using its actual FLINT scalar and matrix wrappers '
                  'were timed at 136 and 384 bits; the measured precision ratios rescale '
                  'the previously saved PBW feature fits. No physical PBW block, Gram matrix, '
                  'or Ward identity was computed. The short arithmetic calibration took '
                  f"{calibration['benchmark_wall_seconds']:.2f} s while the inserted run continued.", '',
                  'Measured 136/384-bit ratios were about 0.91–0.92 for wrapped scalar '
                  'operations and 0.51–0.73 for matrix products. The estimates allow unchanged '
                  'Python/cache overhead, expose the spread between the lean and full feature '
                  'models, and assume sufficient memory. They do not certify full-block '
                  'performance: larger matrix shapes and the persistent Ward cache can change '
                  'the costs. Both sides use 136 bits, but their implementations differ '
                  '(C++ MPC versus Python FLINT). Raw timings and assumptions are in '
                  'pbw_40digit_arithmetic_calibration.json.']
    lines += ['', '## Implemented truncation', '',
              'Subtract primary levels separately from each edge limit. The ordinary CCY '
              'domains are three-dimensional boxes. The inserted domains are four-dimensional '
              'boxes; retain unequal middle levels inside each factor and assemble only equal '
              'final middle levels after primary shifts. Convolution and Schottky products '
              'truncate componentwise. Schottky walks are bounded by both total length and '
              'edge visits. Direct fermion occupation levels are independently bounded.', '',
              '## Cutoff validation', '',
              'At independent cutoff 2, both full physical pipelines were compared with all '
              '45 corresponding monomials in the saved total-level-10 40-digit results. '
              'Maximum scaled differences are 1.55e-36 (ordinary) and 6.00e-36 (inserted). '
              'Exact arithmetic checks additionally compare the Schottky boxes at cutoffs 2 '
              'and 3 and both fermion boxes at cutoff 2 with projections of their total-level '
              'expansions. A shifted-diagonal product is compared with the unrestricted '
              'product at a small asymmetric middle cutoff. All pass.', '',
              '## Reproduction', '',
              '    make -C C++',
              '    python3 C++/tools/time_current_pipelines.py --levels 10 --truncation per-edge --dps 40 --output /tmp/ramond_per_edge10', '',
              f'The {measurement_dir.name} directory contains commands, source/executable hashes, '
              'logs, stage timings, and all output coefficients.']
    (FOLDER / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
