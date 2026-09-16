#!/usr/bin/env python3
"""Coefficient-aware convergence of the saved NSRR and all-NS calculations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ('Code', 'Code/genus_2', 'Code/genus_2_cross_channel', 'Code/c_Recursion',
                 'Code/double_virasoro/nsrr', 'Code/full_ramond_block_runtime',
                 'Data Set/nsrr_momentum_integration_20260911'):
    sys.path.insert(0, str(ROOT/relative))
OUT = ROOT/'Data Set/nsrr_block_convergence_20260911'
OLD = ROOT/'Data Set/nsrr_threshold_order_scan_20260911'
BASE = ROOT/'Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json'
INPUT_HASHES = {}


def read(path):
    INPUT_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def scalar(x):
    return math.fsum(x) if isinstance(x, list) else x


def number(x):
    return complex(float(x['real']), float(x['imag']))


def csum(values):
    values = list(values)
    return complex(math.fsum(x.real for x in values), math.fsum(x.imag for x in values))


def encoded(z):
    return {'real': z.real, 'imag': z.imag}


def grid_audit(summary):
    reference = {p['point_id']: p for p in summary['points']}
    manifest = read(OLD/'grid_N7/manifest.json')
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = {}
    for channel in ('source', 'target'):
        nodes = [read(p) for p in sorted((OLD/'grid_N7'/channel).glob('node-*.json'))]
        assert [n['index'] for n in nodes] == list(range(343))
        assert all(n['manifest_digest'] == manifest_digest and n['measure'] > 0 for n in nodes)
        for n in nodes:
            assert [r['point_id'] for r in n['rows']] == list(reference)
        records = []
        for point_index, pid in enumerate(reference):
            for lo, hi in ((5, 6), (6, 7), (7, 8)):
                a, b = map(str, (lo, hi) if channel == 'source' else (2*lo, 2*hi))
                terms, changes, relative = [], [], []
                for n in nodes:
                    values = n['rows'][point_index]['values']
                    first, last = values[a], values[b]
                    # Resolve the two target sectors before taking absolute values.
                    delta = (math.fsum(abs(x-y) for x, y in zip(first, last))
                             if isinstance(last, list) else abs(last-first))
                    terms.append(n['measure']*scalar(last))
                    changes.append(n['measure']*delta)
                    relative.append(delta/abs(scalar(last)))
                z = reference[pid][channel][str(hi)]['Z']
                direct = reference[pid][channel][str(hi)]['direct_N7_Z']
                assert abs(math.fsum(terms)/direct-1) < 1e-12
                j = max(range(343), key=lambda i: relative[i])
                records.append({'point_id': pid, 'from_order': lo, 'to_order': hi,
                                'absolute_change_before_integration': math.fsum(changes)/z,
                                'dominant_node': max(range(343), key=lambda i: terms[i]),
                                'largest_weighted_change_node': max(range(343), key=lambda i: changes[i]),
                                'largest_node_relative_change': relative[j],
                                'largest_node_relative_change_index': j,
                                'that_node_fraction_of_integral': terms[j]/z})
        result[channel] = {'momentum_nodes': 343, 'records': records,
            'maxima_by_step': [max((r for r in records if r['to_order'] == hi),
                                  key=lambda r: r['absolute_change_before_integration']) for hi in (6, 7, 8)],
            'maximum_unweighted_final_step': max((r for r in records if r['to_order'] == 8),
                                                 key=lambda r: r['largest_node_relative_change'])}
    return result


def source_coefficients(config):
    """Absolute q-weighted monomial sums, after the complete native recovery."""
    rows = []
    native_records = {}
    for path in sorted((OUT/'native').glob('node-*.json')):
        data = read(path)
        assert data['status'] == 'computed' and data['total_q_level'] == 10 and data['dps'] == 40
        node = int(path.name.split('_')[0].split('-')[1])
        channel = (data['f'], *data['etas'])
        native_records[node, channel] = data
        coefficients = {}
        for term in data['coefficients']:
            a, l, r, d = term['exponents']
            assert l == r
            coefficients[a, 2*l, d] = tuple(map(number, term['values']))
        assert len(coefficients) == 506
        for point in config['points']:
            q = tuple(map(complex, point['source']['q_values']))[::-1]
            powers = {e: math.prod(q[j]**(e[j]/2) for j in range(3)) for e in coefficients}
            for lift in ((1, 1, 1), (1, -1, 1)):
                terms, norms, maxima = [[] for _ in range(21)], [[] for _ in range(21)], [0.]*21
                for e, vector in coefficients.items():
                    c = csum(value*math.prod(lift[j]**((i>>j)&1) for j in range(3))
                             for i, value in enumerate(vector))
                    degree = sum(e)
                    term = c*powers[e]
                    terms[degree].append(term); norms[degree].append(abs(term))
                    maxima[degree] = max(maxima[degree], abs(c))
                shells = [csum(x) for x in terms]
                absolute = [math.fsum(x) for x in norms]
                values = {level: csum(shells[:2*level+1]) for level in range(3, 11)}
                scale = max(1., abs(values[10]))
                finite_tail = math.fsum(absolute[17:])/scale
                assert abs(values[10]-values[8])/scale <= finite_tail+2e-14
                rows.append({'node': node, 'channel': channel, 'point_id': point['point_id'],
                             'lifts_slots': lift, 'physical_momenta_slots': data['momenta'],
                             'normalization': 'max(1,abs(F10)); primary powers stripped at every order',
                             'values': {str(k): encoded(v) for k, v in values.items()},
                             'maximum_absolute_coefficient': max(maxima),
                             'shells': [{'total_level': k/2, 'signed_sum': encoded(shells[k]),
                                         'absolute_monomial_sum': absolute[k],
                                         'maximum_absolute_coefficient': maxima[k]} for k in range(21)],
                             'bands': {str(k): (absolute[2*k-1]+absolute[2*k])/scale for k in range(5, 11)},
                             'finite_8_to_10_absolute_bound': finite_tail,
                             'actual_8_to_10_change': abs(values[10]-values[8])/scale,
                             'summation_condition_number': math.fsum(absolute)/max(abs(values[10]), 1e-300)})
    assert set(n for n, _ in native_records) == {113, 170, 294, 336}
    assert len([1 for n, _ in native_records if n == 113]) == 8
    return rows, native_records


def selected_sewn_check(config, native_records):
    """Reproduce saved L<=8 and extend the same integrand at node 113 only."""
    from fast_constants import FastPositiveConstants
    from nsrr_cpp_backend import physical_rows
    from nsrr_factorized_sign_trial import evaluate_blocks
    from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
    from physical_nsrr_sewing import SOURCE_FIXED_SPIN_LIFTS, project_source_fixed_spin, contract_physical_blocks
    node = read(OLD/'grid_N7/source/node-00113.json')
    momenta = tuple(node['momenta'])
    components = {c: physical_rows(data) for (n, c), data in native_records.items() if n == 113}
    constants = FastPositiveConstants(config['b']).rr_ns_constants(momenta[1], momenta[0], momenta[2])
    records = []
    for point, saved in zip(config['points'], node['rows']):
        assert point['point_id'] == saved['point_id']
        values = {}
        for level in range(5, 11):
            amplitudes = {}
            for lift in SOURCE_FIXED_SPIN_LIFTS:
                plumbing = NSRRPlumbingInputs(tuple(map(complex, point['source']['q_values'])), lift, GEOMETRY_SECTORS)
                primary = plumbing.primary(config['b'], momenta)
                blocks = evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, level)
                amplitudes[lift] = {c: primary*v for c, v in blocks.items()}
            value = 4*contract_physical_blocks(project_source_fixed_spin(amplitudes), constants)['total']
            values[str(level)] = value
        error = max(abs(values[str(k)]/saved['values'][str(k)]-1) for k in (5, 6, 7, 8))
        assert error < 1e-11
        records.append({'point_id': point['point_id'], 'node': 113, 'integrands': values,
                        'maximum_saved_relative_change': error,
                        'step_changes': {f'{k-1}->{k}': values[str(k)]/values[str(k-1)]-1 for k in range(6, 11)}})
    return records


def main():
    config = read(BASE); previous = read(OLD/'summary.json')
    grid = grid_audit(previous)
    source, native = source_coefficients(config)
    selected_sewn = selected_sewn_check(config, native)
    prefactors = []
    for path in sorted((OUT/'prefactors').glob('node-*.json')):
        r = read(path)
        prefactors.append({'node': int(path.name.split('_')[0].split('-')[1]),
                           'maximum_raw_branching_coefficient': max(abs(number(x[k])) for x in r['rows'] for k in ('left', 'right')),
                           'maximum_normalized_prefactor': max(abs(number(x['normalized_prefactor'])) for x in r['rows'])})
    assert len(prefactors) == 4
    target = [read(p) for p in sorted((OUT/'all_ns').glob('node-*.json'))]
    assert len(target) == 2 and all(t['status'] == 'complete' for t in target)
    pbw = read(OUT/'pbw/summary.json')
    assert all(c['status'] == 'passed' for c in pbw['checks'])
    report = {'schema': 'coefficient-aware-block-convergence-v1', 'status': 'complete',
              'grid': grid, 'NSRR_chiral': source, 'selected_NSRR_sewn': selected_sewn,
              'raw_branching': prefactors, 'all_NS_chiral': target, 'PBW_crosscheck': pbw,
              'scope': 'Full saved N7 grid through L8; selected actual chiral coefficients/blocks through L10. No full-grid L9/L10 integral is claimed.',
              'frozen_parameters': {'fine_momentum_N': 12, 'correction_momentum_N': 7, 'b': 1.4,
                                    'source_dps': 40, 'target_dps': 50, 'target_endpoint_cap': 8,
                                    'target_vacuum_word_length': 7, 'target_vacuum_max_mode': 50},
              'interpretation': 'Numerical order-convergence evidence, not a rigorous infinite-series bound or proof of absolute spin calibration.',
              'source_sha256': {**INPUT_HASHES, str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    save('summary.json', report)
    lines = ['# Convergence with the actual coefficients included', '',
             'The numerical checks support convergence on these saved surfaces. Raw branching '
             'coefficients are indeed large, so small q alone is insufficient. This audit checks '
             'the full recovered coefficients, absolute q-weighted monomial sums, complex block '
             'values, and changes before momentum integration.', '',
             '## Full fixed momentum grid', '',
             'For each order step the table takes absolute differences at each momentum node '
             'before summing with the positive quadrature weights. For all-NS the two form '
             'sectors are resolved before the absolute value as well. The result is divided '
             'by that channel\'s integrated Z at the higher order. The same free normalization '
             'would cancel if Q_L were used. These are fractional changes, not percentages.', '',
             '| Order step | NSRR: sum w abs(delta integrand) / Z | All-NS: sum w sum_sector abs(delta integrand) / Z |',
             '|---|---:|---:|']
    for i, hi in enumerate((6, 7, 8)):
        lines.append(f"| {hi-1} to {hi} | {grid['source']['maxima_by_step'][i]['absolute_change_before_integration']:.8e} | "
                     f"{grid['target']['maxima_by_step'][i]['absolute_change_before_integration']:.8e} |")
    lines += ['', 'All 343 existing correction-grid nodes and all ten surfaces are included. '
              'No momentum nodes, weights, q values, lifts, central charge, or precision were '
              'changed. NSRR L is total q order; all-NS L is null-recursion level with endpoint '
              'K=8, its middle chain and vacuum still resummed.', '',
              '## Branching growth and absolute coefficient tails', '',
              '| Existing source node | Maximum raw branching coefficient through L10 | Maximum normalized prefactor |',
              '|---:|---:|---:|']
    lines += [f"| {p['node']} | {p['maximum_raw_branching_coefficient']:.6e} | {p['maximum_normalized_prefactor']:.6e} |" for p in prefactors]
    lines += ['', 'The normalized prefactor is the actual product of the two outer branching '
              'coefficients, the inserted middle coefficient, and inverse norms used in the '
              'native pipeline. Neither column alone is a tail estimate. Raw B depends on '
              'the primary normalization; the complete q-series coefficients are the relevant '
              'test of truncation. The production pipeline includes every branching primary '
              'that can contribute at its requested total q order; there is no smaller '
              'independent primary cutoff in these runs. Raising L also enlarges that support '
              'and the descendant budgets. The PBW comparison below independently checks '
              'the resulting low-order coefficients.', '',
              'Define A_L=sum_{L-1<|nu|<=L} abs(C_nu q^nu)/max(1,abs(F10)). '
              'It includes both half-integer shells at every integer step and discards '
              'cancellations between monomials. The following table uses the mixed HJS signs '
              '(+,-), f=0, source node 113, and surface generic_03, where the integrated '
              'NSRR order change is largest. Plumbing lifts are (+,-,+).', '',
              '| L | Absolute band A_L |', '|---:|---:|']
    example = next(r for r in source if r['node'] == 113 and r['channel'] == (0, 1, -1)
                   and r['point_id'] == 'generic_03' and r['lifts_slots'] == (1, -1, 1))
    lines += [f"| {k} | {example['bands'][str(k)]:.8e} |" for k in range(5, 11)]
    lines += ['', f"At that point max abs(C_nu)={example['maximum_absolute_coefficient']:.6e}; "
              f"the absolute finite tail from L8 to L10 is {example['finite_8_to_10_absolute_bound']:.6e}. "
              'The full JSON includes both source lifts, all ten surfaces, both ordinary and '
              'inserted modes at nodes 113, 170, 294, 336, and all eight HJS/form channels '
              'at node 113. Nodes 113/170 probe influential contributions; 294/336 probe '
              'large relative order changes in the suppressed momentum tail.', '',
              '## Independent PBW and higher-order checks', '',
              'At source node 113 a fresh direct physical PBW calculation through total level '
              '8 agrees coefficient by coefficient with the L10 double-Virasoro result '
              'restricted to those levels. C++ uses 40 decimal digits; PBW uses 384 bits. '
              'No double-Virasoro numerator or auxiliary block supplies the PBW reference.', '',
              '| Mode | Maximum scaled coefficient error through L8 |', '|---|---:|']
    lines += [f"| {c['mode']} | {c['maximum_scaled_error']} |" for c in pbw['checks']]
    lines += ['', 'With all eight source components at node 113, the saved L5--L8 integrands '
              'are reproduced before extending to L9/L10. The table gives maximum absolute '
              'relative changes over the tested surfaces; it is a selected-node check, '
              'not a recomputation of the full L9/L10 integral.', '',
              '| Step | NSRR sewn integrand, node 113 (ten surfaces) | All-NS complex block, node 114 (five surfaces, both sectors) |',
              '|---|---:|---:|']
    dominant_target = next(t for t in target if t['node'] == 114)
    for k in range(6, 11):
        src = max(abs(r['step_changes'][f'{k-1}->{k}']) for r in selected_sewn)
        tgt = max(s['relative_complex_change'] for r in dominant_target['rows'] for s in r['steps'] if s['to'] == k)
        lines.append(f'| {k-1} to {k} | {src:.8e} | {tgt:.8e} |')
    lines += ['', '## Limits of the conclusion', '',
              'Convergence is slower at large momentum; small integrated changes do not '
              'mean uniform accuracy for every block. The worst NSRR node-relative L7-to-L8 '
              'change is {:.6e}, at node {} on {}, whose weighted contribution is only '
              '{:.6e} of that surface\'s integral.'.format(
                  grid['source']['maximum_unweighted_final_step']['largest_node_relative_change'],
                  grid['source']['maximum_unweighted_final_step']['largest_node_relative_change_index'],
                  grid['source']['maximum_unweighted_final_step']['point_id'],
                  grid['source']['maximum_unweighted_final_step']['that_node_fraction_of_integral']), '',
              'These finite-order checks support numerical convergence at the reported '
              'scales. They do not prove a bound beyond L10 or uniformly over the continuous '
              'momentum domain. The absolute band bounds concern the tested finite tails. '
              'The data provide no indication that the roughly 0.86% cross-channel residual '
              'comes from the measured block-order truncation. Spin/basis calibration is '
              'a separate issue from this convergence test.', '',
              '[Machine-readable results, coefficients, paths, and input hashes](summary.json)', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'source_chiral_checks': len(source),
                      'all_NS_chiral_checks': sum(len(t['rows']) for t in target),
                      'largest_raw_branching_coefficient': max(p['maximum_raw_branching_coefficient'] for p in prefactors),
                      'maximum_saved_source_norm_reproduction_error': max(r['maximum_saved_relative_change'] for r in selected_sewn)}, indent=2))


if __name__ == '__main__':
    main()
