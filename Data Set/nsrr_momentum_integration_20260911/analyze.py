#!/usr/bin/env python3
"""Audit saved grids and reproduce the fixed-order multilevel comparison.

This reads completed computations only. It launches no momentum evaluations.
"""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'nsrr_double_virasoro_N7_L5_20260911'


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def audit_grid(name, config):
    directory = ROOT / name
    manifest = read(directory / 'manifest.json')
    summary = read(directory / 'summary.json')
    assert summary['manifest'] == manifest
    assert manifest['base_config_digest'] == digest(config)
    assert manifest['target_endpoint_cap'] == 8
    assert manifest['point_ids'] == [p['point_id'] for p in config['points']]
    implementation = ROOT / 'validation/implementation_v1' if name.startswith('coarse_') else ROOT
    for filename, expected in manifest['implementation_sha256'].items():
        assert file_digest(implementation / filename) == expected, (name, filename)
    expected_digest = digest(manifest)
    n = manifest['order']
    totals, checks = {}, {'nodes': {}, 'fallback_nodes': 0, 'physical_PBW_used': False,
                          'maximum_native_residual': 0., 'summary_maximum_relative_difference': 0.}
    for channel in manifest['channels']:
        files = sorted((directory / channel).glob('node-*.json'))
        assert [p.name for p in files] == [f'node-{i:05d}.json' for i in range(n**3)]
        checks['nodes'][channel] = len(files)
        levels = manifest['source_levels'] if channel == 'source' else manifest['target_twice_levels']
        sums = {(p, level): [] for p in manifest['point_ids'] for level in levels}
        rules = manifest['nodes_and_weights'][channel]
        for index, idx in enumerate(itertools.product(range(n), repeat=3)):
            shard = read(files[index])
            assert shard['manifest_digest'] == expected_digest
            assert shard['channel'] == channel and shard['index'] == index
            assert shard['momenta'] == [rules[e]['nodes'][idx[e]] for e in range(3)]
            measure = math.prod(rules[e]['weights'][idx[e]] for e in range(3))
            assert abs(shard['measure'] / measure - 1) < 1e-15
            assert [r['point_id'] for r in shard['rows']] == manifest['point_ids']
            for row in shard['rows']:
                assert set(row['values']) == {str(level) for level in levels}
                for level in levels:
                    value = row['values'][str(level)]
                    value = sum(value) if isinstance(value, list) else value
                    assert math.isfinite(value) and value >= 0
                    sums[row['point_id'], level].append(shard['measure'] * value)
            if channel == 'source':
                assert not shard['checks']['physical_PBW_used']
                checks['fallback_nodes'] += int(shard['checks'].get('multiprecision_fallback', False))
                residual = max(v for c in shard['checks']['native_channels']
                               for v in c['diagnostics'].values())
                checks['maximum_native_residual'] = max(checks['maximum_native_residual'], residual)
            else:
                assert shard['checks']['endpoint_cap'] == 8
                assert shard['checks']['vacuum_word_length'] == 7
                assert shard['checks']['vacuum_max_mode'] == 50
        totals.update({(channel, p, level): math.fsum(values)
                       for (p, level), values in sums.items()})
    assert len(summary['rows']) == len(totals)
    assert len({(r['channel'], r['point_id'], r['level']) for r in summary['rows']}) == len(totals)
    for row in summary['rows']:
        actual = totals[row['channel'], row['point_id'], row['level']]
        relative = abs(actual / row['Z'] - 1)
        assert relative < 1e-14
        checks['summary_maximum_relative_difference'] = max(checks['summary_maximum_relative_difference'], relative)
    checks['manifest_sha256'] = file_digest(directory / 'manifest.json')
    checks['summary_sha256'] = file_digest(directory / 'summary.json')
    return totals, checks


def main():
    config = read(BASE / 'config.json')
    old = read(BASE / 'depth5_summary.json')
    assert old['status'] == 'complete' and not old['fitted_normalization']
    assert old['configuration']['source_total_level'] == 5
    assert old['configuration']['target_null_twice_level'] == 10
    assert old['configuration']['target_endpoint_cap'] == 8
    high = {(v['momentum_order'], v['point_id']): v for v in old['comparisons']}
    grids, validation = {}, {}
    for name in ('coarse_N4', 'coarse_N7', 'half_N9', 'half_N12'):
        grids[name], validation[name] = audit_grid(name, config)

    def estimate(point, fine, coarse, source_control=3):
        pid = point['point_id']
        result = {'fine_order': fine, 'correction_order': coarse,
                  'source_control_level': source_control, 'target_control_twice_level': 0}
        for channel, level in (('source', source_control), ('target', 0)):
            free = point[channel]['Z_free'] ** config['kappa']
            full = high[coarse, pid][channel + '_Q'] * free
            low_coarse = grids[f'coarse_N{coarse}'][channel, pid, level]
            low_fine = grids[f'half_N{fine}'][channel, pid, level]
            correction = full - low_coarse
            result[channel + '_Z'] = low_fine + correction
            result[channel + '_Q'] = result[channel + '_Z'] / free
            result[channel + '_correction_Z'] = correction
        result['source_over_target'] = result['source_Q'] / result['target_Q']
        result['relative_disagreement'] = result['source_over_target'] - 1
        return result

    rows = []
    for point in config['points']:
        pid = point['point_id']
        final = estimate(point, 12, 7)
        previous = estimate(point, 9, 7)
        correction_check = estimate(point, 12, 4)
        alternate_control = estimate(point, 12, 7, 1)
        row = {'point_id': pid, 'old_direct_N7': high[7, pid],
               'N9_with_N7_correction': previous, 'N12_with_N7_correction': final,
               'N12_with_N4_correction': correction_check,
               'N12_with_source_L1_control': alternate_control}
        for channel in ('source', 'target'):
            denominator = final[channel + '_Z']
            row[channel + '_fine_N9_to_N12_relative_change'] = (
                final[channel + '_Z'] - previous[channel + '_Z']) / denominator
            row[channel + '_correction_N4_to_N7_relative_change'] = (
                final[channel + '_Z'] - correction_check[channel + '_Z']) / denominator
        row['ratio_fine_N9_to_N12_absolute_change'] = final['source_over_target'] - previous['source_over_target']
        row['ratio_correction_N4_to_N7_absolute_change'] = final['source_over_target'] - correction_check['source_over_target']
        row['source_L1_vs_L3_control_relative_change'] = alternate_control['source_Z'] / final['source_Z'] - 1
        rows.append(row)

    metric_keys = [k for k in rows[0] if k.endswith('_change')]
    metrics = {'surfaces': len(rows), 'largest_momentum_order': 12,
               'maximum_relative_disagreement': max(abs(r['N12_with_N7_correction']['relative_disagreement']) for r in rows),
               'rms_relative_disagreement': math.sqrt(math.fsum(r['N12_with_N7_correction']['relative_disagreement']**2 for r in rows)/len(rows)),
               'worst_surface': max(rows, key=lambda r: abs(r['N12_with_N7_correction']['relative_disagreement']))['point_id']}
    metrics.update({'maximum_absolute_' + k: max(abs(r[k]) for r in rows) for k in metric_keys})
    result = {'schema': 'fixed-order-multilevel-cross-channel-v1', 'status': 'complete',
              'source_total_level': 5, 'target_null_twice_level': 10, 'target_endpoint_cap': 8,
              'target_vacuum_word_length': 7, 'target_vacuum_max_mode': 50,
              'quadrature': 'positive-half-line Gaussian, beta=(0,0,2) source and (2,2,2) target',
              'estimator': 'fine(low) + coarse(full-low)',
              'fine_orders': [9, 12], 'correction_orders': [4, 7],
              'source_low_control_total_level': 3, 'target_low_control_null_twice_level': 0,
              'normalization': 1., 'fitted_normalization': False,
              'base_config_sha256': file_digest(BASE / 'config.json'),
              'full_reference_sha256': file_digest(BASE / 'depth5_summary.json'),
              'uncertainty_policy': 'refinement differences are diagnostics, not rigorous error bounds',
              'metrics': metrics, 'grid_validation': validation, 'comparisons': rows}
    save(ROOT / 'comparison.json', result)
    save(ROOT / 'validation/grid_integrity.json', {'status': 'passed', 'grids': validation})
    with (ROOT / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['point_id', 'old_N7_signed_gap_percent', 'N9_signed_gap_percent',
                         'N12_signed_gap_percent', 'N12_source_Q', 'N12_target_Q',
                         'source_fine_relative_change', 'target_fine_relative_change',
                         'source_correction_relative_change', 'target_correction_relative_change'])
        for r in rows:
            final = r['N12_with_N7_correction']
            writer.writerow([r['point_id'], 100*r['old_direct_N7']['relative_disagreement'],
                             100*r['N9_with_N7_correction']['relative_disagreement'],
                             100*final['relative_disagreement'], final['source_Q'], final['target_Q'],
                             *[r[ch + suffix] for suffix in ('_fine_N9_to_N12_relative_change',
                                                            '_correction_N4_to_N7_relative_change')
                               for ch in ('source', 'target')]])
    print(json.dumps(metrics, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
