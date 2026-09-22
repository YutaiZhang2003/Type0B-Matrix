#!/usr/bin/env python3
"""Vary block order with the validated NSRR/all-NS threshold integration fixed.

The default is the positive-half-line threshold Gaussian rule, with the saved
N=12 controls and N=7 full-order corrections. Only block orders are exposed as
numerical options. Geometry, weights, spin transport, normalization, precision,
global resummation, and vacuum cutoffs are inherited from the validated run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/'Data Set/nsrr_momentum_integration_20260911'
BASE = ROOT/'Data Set/nsrr_double_virasoro_N7_L5_20260911'
DEFAULT_OUTPUT = ROOT/'Data Set/nsrr_threshold_order_scan_20260911'

# Reuse the exact, already validated threshold quadrature and physical kernels.
# Keeping this runtime frozen also preserves the previous node fingerprints.
sys.path.insert(0, str(AUDIT))
from evaluator import scan  # noqa: E402
from momentum_rules import channel_rules  # noqa: E402
from analyze import audit_grid  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(value):
    return math.fsum(value) if isinstance(value, list) else value


def prepare(output, orders):
    if orders != sorted(set(orders)) or not orders or orders[0] != 5 or orders[-1] > 8:
        raise ValueError('Use distinct ascending block orders starting at 5, through at most 8.')
    base = read(BASE/'config.json')
    scan.validate_config(base)
    previous = read(AUDIT/'threshold_comparison.json')
    if previous['status'] != 'complete' or previous['fitted_normalization']:
        raise ValueError('The completed, fixed-normalization threshold baseline is required.')
    coarse = read(AUDIT/'full_threshold_half_N7/manifest.json')
    fine = read(AUDIT/'half_N12/manifest.json')
    for manifest, order in ((coarse, 7), (fine, 12)):
        assert manifest['order'] == order and manifest['scheme'] == 'half-gaussian'
        assert manifest['base_config_digest'] == scan.digest(base)
        assert manifest['point_ids'] == [p['point_id'] for p in base['points']]
        assert manifest['target_endpoint_cap'] == 8 and manifest['p_max'] is None
        for name, expected in manifest['implementation_sha256'].items():
            assert sha(AUDIT/name) == expected, name
        for channel in ('source', 'target'):
            rules = channel_rules(base, channel, order, 'half-gaussian')
            actual = [{'nodes': p.tolist(), 'weights': w.tolist()} for p, w in rules]
            assert actual == manifest['nodes_and_weights'][channel], (order, channel)
    assert coarse['source_levels'] == [3, 5] and coarse['target_twice_levels'] == [0, 10]
    assert coarse['native_control_precision'] == 40
    assert fine['source_levels'] == [1, 3] and fine['target_twice_levels'] == [0]
    expected_grid = copy.deepcopy(coarse)
    expected_grid['source_levels'] = [3, *orders]
    expected_grid['target_twice_levels'] = [0, *[2*level for level in orders]]
    actual_path = output/'grid_N7/manifest.json'
    if actual_path.exists() and read(actual_path) != expected_grid:
        raise ValueError('The existing grid changes a setting other than block order.')
    protected = [BASE/'config.json', BASE/'validation/audit_all_ns_depth.py',
                 AUDIT/'threshold_comparison.json']
    protected += [AUDIT/name for name in coarse['implementation_sha256']]
    protected += [AUDIT/directory/name for directory in ('half_N12', 'full_threshold_half_N7')
                  for name in ('manifest.json', 'summary.json')]
    config = {
        'schema': 'nsrr-threshold-fixed-grid-order-scan-v1',
        'orders': orders, 'expected_grid_manifest': expected_grid,
        'baseline_comparison': str((AUDIT/'threshold_comparison.json').relative_to(ROOT)),
        'frozen_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in protected},
        'fixed_integration': {
            'scheme': 'half-gaussian', 'fine_order': 12, 'correction_order': 7,
            'source_control_total_level': 3, 'target_control_null_twice_level': 0,
            'gaussian_decay_scale': 1., 'threshold_powers': {'source': [0, 0, 2], 'target': [2, 2, 2]},
            'momentum_domain': 'positive half-line; no finite cutoff',
        },
        'fixed_block_settings': {
            'source_native_precision': 40, 'target_working_precision': 50,
            'target_endpoint_cap': 8, 'target_vacuum_word_length': 7, 'target_vacuum_max_mode': 50,
            'target_middle_chain': 'analytic resummation',
        },
        'only_varied_quantities': ['source total q order', 'target null-recursion level'],
        'source_method': 'native C++ double Virasoro; no physical PBW',
        'target_method': 'collision-aware c recursion',
        'normalization_fitted': False,
        'estimator': 'I_threshold_N12(f_control) + I_threshold_N7(f_order - f_control)',
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output/'order_scan_config.json'
    if path.exists() and read(path) != config:
        raise ValueError('The frozen order-scan configuration changed.')
    scan.save(path, config)
    return config, base


def status(output):
    manifest_path = output/'grid_N7/manifest.json'
    if not manifest_path.exists():
        return {'status': 'not started'}
    manifest = read(manifest_path)
    result = {'elapsed_seconds': time.time()-manifest_path.stat().st_mtime, 'channels': {}}
    for channel, controls in (('source', (3, 5)), ('target', (0, 10))):
        files = sorted((output/'grid_N7'/channel).glob('node-*.json'))
        times, maximum = [], 0.
        for path in files:
            node = read(path)
            prior = read(AUDIT/'full_threshold_half_N7'/channel/path.name)
            assert node['momenta'] == prior['momenta'] and node['measure'] == prior['measure']
            times.append(node['wall_seconds'])
            for row, old in zip(node['rows'], prior['rows']):
                assert row['point_id'] == old['point_id']
                for level in controls:
                    maximum = max(maximum, abs(scalar(row['values'][str(level)])/
                                               scalar(old['values'][str(level)])-1))
        result['channels'][channel] = {
            'completed_nodes': len(files), 'required_nodes': manifest['order']**3,
            'maximum_baseline_relative_change': maximum if files else None,
            'median_node_seconds': statistics.median(times) if times else None,
        }
    result['status'] = 'complete' if (output/'summary.json').exists() else 'running'
    result['physical_spin_status'] = 'unverified_historical_prescription'
    return result


def write_report(output, summary):
    orders = summary['config']['orders']
    lines = [
        '# Block-order check with fixed threshold integration', '',
        '**Historical numerical diagnostic: its interacting fixed-spin sewing is unverified.** '
        'See ../nsrr_spin_tracking_20260911/README.md. Numerical order stability does not '
        'certify the spin identification.', '',
        'Only conformal-block order changes. The threshold half-Gaussian nodes and weights are '
        'identical at every order: N=12 for the saved controls and N=7 for the newly evaluated '
        'full-order corrections. Every source and target correction uses all 343 momentum nodes '
        'and all ten surfaces.', '',
        'NSRR uses strict total q order L. All-NS uses null-recursion level L (the code cutoff '
        'is R=2L), with endpoint occupation K=8 and the middle chain analytically resummed. '
        'These are different truncation definitions; global resummation, vacuum cutoffs, precision, '
        'geometry, spin projections, structure constants, and normalization stay fixed.', '',
        '```text',
        'Z_source(L) = I_N12[f_source,3] + I_N7[f_source,L - f_source,3]',
        'Z_target(L) = I_N12[f_target,0] + I_N7[f_target,R=2L - f_target,0]',
        '```', '',
        'The same source expansion computed through the largest requested order is truncated '
        'to every lower order. All-NS is evaluated at each requested recursion cutoff. '
        'The order-5 results and controls are checked against the previous threshold calculation '
        'at every momentum node.', '',
        '## Order dependence', '',
        'Entries below are maximum absolute relative changes in integrated Q over ten surfaces, '
        'expressed as percentages. They should approach zero as block order converges. They are '
        'not the discrepancy between the two channels.', '',
        '| Block-order step | NSRR change | All-NS change |',
        '|---|---:|---:|',
    ]
    for step in summary['order_steps']:
        lines.append(f'| {step["from_order"]} to {step["to_order"]} | '
                     f'{100*step["maximum_source_relative_change"]:.9f}% | '
                     f'{100*step["maximum_target_relative_change"]:.9f}% |')
    lines += ['', '## Cross-channel comparison', '',
              'The ratio is Q_NSRR/Q_all-NS and should equal **one**. The discrepancy is '
              '100*abs(ratio-1) and should equal **zero percent**. The next table gives the '
              'maximum discrepancy across ten surfaces, including separate order changes in each channel.', '',
              '| Order L | Raise NSRR only (all-NS fixed at 5) | Raise all-NS only (NSRR fixed at 5) | Both at L |',
              '|---|---:|---:|---:|']
    for row in summary['by_order']:
        lines.append(f'| {row["order"]} | {100*row["source_only_maximum_discrepancy"]:.9f}% | '
                     f'{100*row["target_only_maximum_discrepancy"]:.9f}% | '
                     f'{100*row["paired_maximum_discrepancy"]:.9f}% |')
    lines += ['', 'Paired-order ratios for every surface:', '',
              '| Surface | ' + ' | '.join(f'L={level}' for level in orders) + ' |',
              '|---|' + '---:|'*len(orders)]
    for point in summary['points']:
        lines.append('| ' + point['point_id'] + ' | ' + ' | '.join(
            f'{point["paired"][str(level)]["ratio"]:.10f}' for level in orders) + ' |')
    final = summary['by_order'][-1]
    first = summary['by_order'][0]
    lines += ['', f'Raising both block cutoffs from {orders[0]} to {orders[-1]} changes the largest '
              f'cross-channel discrepancy from {100*first["paired_maximum_discrepancy"]:.9f}% to '
              f'{100*final["paired_maximum_discrepancy"]:.9f}%. The final worst surface is '
              f'`{final["paired_worst_surface"]}`.', '',
              '## Validation and fixed inputs', '',
              f'The maximum pointwise change in the recomputed order-5 result/control is '
              f'{summary["validation"]["maximum_source_baseline_relative_change"]:.3e} for NSRR and '
              f'{summary["validation"]["maximum_target_baseline_relative_change"]:.3e} for all-NS. '
              'The run manifests differ from the baseline only in source and target block-order lists. '
              'Node coverage, exact saved momenta and weights, implementation hashes, native precision, '
              'PBW exclusion, positivity, and independently reconstructed integration sums are checked.', '',
              'The reported Q changes concern integrated physical amplitudes. They are numerical '
              'order-convergence diagnostics, not rigorous bounds on every individual holomorphic block. '
              'No momentum refinement or other parameter scan is performed in this check.', '',
              '[Full numerical results](summary.json) · [All source/target order pairs](comparison.csv) · '
              '[Q values at each order](channel_values.csv) · [Frozen configuration](order_scan_config.json)', '',
              'Run or resume from the repository root:', '', '```sh',
              'PYTHONPATH=Code:Code/genus_2:Code/full_ramond_block_runtime:Code/c_Recursion:Code/genus_2_cross_channel \\',
              '  /private/tmp/type0b-nsrr-smoke-venv/bin/python Code/genus_2/run_nsrr_threshold_order_scan.py',
              '```', '', 'Use `--reduce-only` to regenerate the report without block evaluations. '
              'Use `--status` for progress. The driver exposes no momentum or resummation options.', '']
    (output/'README.md').write_text('\n'.join(lines))


def reduce(output, config, base):
    for name, expected in config['frozen_input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    assert read(output/'grid_N7/manifest.json') == config['expected_grid_manifest']
    fine, fine_checks = audit_grid(str(AUDIT/'half_N12'), base)
    _, prior_checks = audit_grid(str(AUDIT/'full_threshold_half_N7'), base)
    current, current_checks = audit_grid(str(output/'grid_N7'), base)
    orders = config['orders']
    validation = {'maximum_source_baseline_relative_change': 0.,
                  'maximum_target_baseline_relative_change': 0.,
                  'grid_audits': {'fine_N12': fine_checks, 'prior_N7': prior_checks, 'current_N7': current_checks}}
    corrections = {}
    for channel, control, baseline_level in (('source', 3, 5), ('target', 0, 10)):
        sums = {(p['point_id'], level): [] for p in base['points'] for level in orders}
        for path in sorted((output/'grid_N7'/channel).glob('node-*.json')):
            node = read(path)
            old = read(AUDIT/'full_threshold_half_N7'/channel/path.name)
            assert node['momenta'] == old['momenta'] and node['measure'] == old['measure']
            if channel == 'source':
                assert node['checks']['native_precision_digits'] == 40
                assert node['checks']['native_pipeline_calls'] == 8
            for row, before in zip(node['rows'], old['rows']):
                pid = row['point_id']
                assert pid == before['point_id']
                for level in (control, baseline_level):
                    relative = abs(scalar(row['values'][str(level)])/scalar(before['values'][str(level)])-1)
                    validation['maximum_'+channel+'_baseline_relative_change'] = max(
                        validation['maximum_'+channel+'_baseline_relative_change'], relative)
                    assert relative < 1e-11, (channel, node['index'], pid, level, relative)
                low = scalar(row['values'][str(control)])
                for level in orders:
                    key = level if channel == 'source' else 2*level
                    sums[pid, level].append(node['measure']*(scalar(row['values'][str(key)])-low))
        corrections.update({(channel, pid, level): math.fsum(values)
                            for (pid, level), values in sums.items()})
    previous = {p['point_id']: p['updated'] for p in read(AUDIT/'threshold_comparison.json')['comparisons']}
    points, pair_rows, channel_rows = [], [], []
    for point in base['points']:
        pid = point['point_id']
        row = {'point_id': pid, 'source': {}, 'target': {}, 'paired': {}}
        for channel, control in (('source', 3), ('target', 0)):
            for level in orders:
                z = fine[channel, pid, control]+corrections[channel, pid, level]
                q = z/point[channel]['Z_free']**base['kappa']
                key = level if channel == 'source' else 2*level
                expected_z = fine[channel, pid, control]+current[channel, pid, key]-current[channel, pid, control]
                assert math.isfinite(z) and z > 0
                assert abs(z-expected_z) < 1e-13*z
                row[channel][str(level)] = {'Z': z, 'Q': q, 'correction_Z': corrections[channel, pid, level],
                                            'direct_N7_Z': current[channel, pid, key]}
                channel_rows.append([pid, channel, level, key, z, q])
                if level == 5:
                    assert abs(q/previous[pid][channel+'_Q']-1) < 1e-11
        for source_level, target_level in itertools.product(orders, repeat=2):
            s = row['source'][str(source_level)]['Q']
            t = row['target'][str(target_level)]['Q']
            ratio = s/t
            pair_rows.append({'point_id': pid, 'source_total_order': source_level,
                              'target_null_order': target_level, 'target_null_twice_order': 2*target_level,
                              'source_Q': s, 'target_Q': t, 'ratio': ratio,
                              'signed_discrepancy_percent': 100*(ratio-1)})
            if source_level == target_level:
                row['paired'][str(source_level)] = {'ratio': ratio, 'signed_discrepancy': ratio-1}
        points.append(row)
    by_order, steps = [], []
    for level in orders:
        subsets = {
            'source_only': [p for p in pair_rows if p['source_total_order']==level and p['target_null_order']==5],
            'target_only': [p for p in pair_rows if p['source_total_order']==5 and p['target_null_order']==level],
            'paired': [p for p in pair_rows if p['source_total_order']==p['target_null_order']==level],
        }
        item = {'order': level}
        for name, subset in subsets.items():
            worst = max(subset, key=lambda r: abs(r['ratio']-1))
            item[name+'_maximum_discrepancy'] = abs(worst['ratio']-1)
            item[name+'_worst_surface'] = worst['point_id']
        by_order.append(item)
    for before, after in zip(orders[:-1], orders[1:]):
        step = {'from_order': before, 'to_order': after}
        for channel in ('source', 'target'):
            values = {p['point_id']: p[channel][str(after)]['Q']/p[channel][str(before)]['Q']-1 for p in points}
            step[channel+'_signed_relative_changes'] = values
            step['maximum_'+channel+'_relative_change'] = max(abs(v) for v in values.values())
        steps.append(step)
    summary = {'schema': config['schema'], 'status': 'complete', 'config': config,
               'physical_spin_status': 'unverified_historical_prescription',
               'analysis_driver_sha256': sha(Path(__file__)), 'validation': validation,
               'points': points, 'by_order': by_order, 'order_steps': steps,
               'physical_PBW_used': False, 'fitted_normalization': False}
    scan.save(output/'summary.json', summary)
    with (output/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    with (output/'channel_values.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['point_id','channel','displayed_order','code_cutoff','Z','Q'])
        writer.writerows(channel_rows)
    write_report(output, summary)
    return {'status': 'complete', 'by_order': by_order, 'order_steps': steps, 'validation': validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[5, 6, 7, 8])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--reduce-only', action='store_true')
    modes.add_argument('--status', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.status:
        print(json.dumps(status(output), indent=2)); return
    if not args.reduce_only and not args.prepare_only:
        from spin_structure import UnverifiedSpinSewing
        raise UnverifiedSpinSewing(
            'This archived driver uses an unverified interacting spin prescription. '
            'Run Code/genus_2/audit_spin_tracked_comparison.py for the explicit spin ledger. '
            'A physical rerun requires the validated Ramond and all-NS vertex/pairing adapters. '
            '--status and --reduce-only remain available for historical numerical data.')
    config, base = prepare(output, args.orders)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'orders': args.orders, 'fixed_integration': config['fixed_integration']}, indent=2))
        return
    if not args.reduce_only:
        command = [sys.executable, str(AUDIT/'run_grid.py'), '--order', '7', '--scheme', 'half-gaussian',
                   '--source-levels', '3', *map(str, args.orders), '--target-twice-levels', '0',
                   *[str(2*v) for v in args.orders], '--native-dps', '40', '--workers', str(args.workers),
                   '--output', str(output/'grid_N7')]
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        subprocess.run(command, env=env, check=True)
    print(json.dumps(reduce(output, config, base), indent=2))


if __name__ == '__main__':
    main()
