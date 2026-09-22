#!/usr/bin/env python3
"""Independently trace the saved theta plumbing square-root branches.

Re-invert straight paths in the reference marked period coordinates. Unwrap
the resulting q phases without the saved eta-to-spin helper, then compare
with the saved eta values and integer periods. No Liouville integral changes.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for relative in ('Code', 'Code/genus_2', 'Code/genus_2_cross_channel', 'Code/c_Recursion',
                 'Code/double_virasoro/nsrr', 'Code/full_ramond_block_runtime'):
    sys.path.insert(0, str(ROOT/relative))

import numpy as np

from fixed_spin_free_plumbing import charged_frame
from nsrr_human_note_geometry import SOURCE_REMARKING, action
from nsrr_nsnsns_theta_omega_scan import inverse_chart, omega_action
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from physical_nsrr_sewing import SOURCE_FIXED_SPIN_LIFTS
from spin_structure import SpinCharacteristic, ThetaLogBranch, theta_log_windings

BASE = ROOT/'Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json'
ANCHOR = ROOT/'Data Set/nsrr_nsnsns_offaxis_constant_scan_continued_N4_20260904/config.json'
OUTPUT = ROOT/'Data Set/nsrr_spin_tracking_20260911'


def load(path):
    return json.loads(path.read_text())


def matrix(value):
    return np.asarray([[complex(z) for z in row] for row in value])


def sample(state, omega, reference_branch, t):
    charge = charged_frame(state.q_values, max_mode=32).omega_charge
    branch = np.rint((charge-omega).real).astype(int)
    error = float(np.max(abs(charge-omega-branch)))
    continued_error = float(np.max(abs(charge+state.period_shift-omega-reference_branch)))
    if max(error, continued_error) > 2e-8:
        raise AssertionError(f'period continuation failed at {t}: {error}, {continued_error}')
    if state.windings != theta_log_windings(reference_branch, branch):
        raise AssertionError('phase unwrapping and independent integer periods disagree')
    return {'t': float(t), **state.record(), 'period_branch': branch.tolist(),
            'principal_period_residual': error, 'continued_period_residual': continued_error}


def cut_check(before, after, edge, reference_lifts):
    """Local free-block continuity across the cut bracketed by path samples."""
    lo = np.asarray([complex(*z) for z in before['q_values']])
    hi = np.asarray([complex(*z) for z in after['q_values']])
    fraction = -lo[edge].imag/(hi[edge].imag-lo[edge].imag)
    if not 0 <= fraction <= 1:
        raise AssertionError('winding jump has no negative-real-axis crossing')
    qcut = lo+fraction*(hi-lo)
    if qcut[edge].real >= 0:
        raise AssertionError('winding jumped away from the principal cut')
    epsilon = 1e-10*abs(qcut[edge])
    first, second = qcut.copy(), qcut.copy()
    side = 1 if lo[edge].imag > 0 else -1
    first[edge] = qcut[edge].real+side*1j*epsilon
    second[edge] = qcut[edge].real-side*1j*epsilon
    state = ThetaLogBranch(tuple(lo), tuple(before['log_windings'])).advance(tuple(first))
    following = state.advance(tuple(second))
    lift1 = state.principal_root_lifts(reference_lifts)
    lift2 = following.principal_root_lifts(reference_lifts)
    f1 = theta_physical_fermion_fredholm(first, lift1, max_mode=32).chiral_value
    f2 = theta_physical_fermion_fredholm(second, lift2, max_mode=32).chiral_value
    held = theta_physical_fermion_fredholm(second, lift1, max_mode=32).chiral_value
    error = float(abs(f2-f1)/abs(f1))
    if error > 1e-8:
        raise AssertionError('the continued free block jumps across a root cut')
    return {'edge': ('zero', 'one', 'infinity')[edge],
            't_bracket': [before['t'], after['t']],
            'test': 'local straight-q crossing inside the inverse-path bracket',
            'relative_imaginary_offset': 1e-10,
            'lifts_before': lift1, 'lifts_after': lift2,
            'compensated_relative_jump': error,
            'uncompensated_relative_jump': float(abs(held-f1)/abs(f1))}


def audit(output, steps):
    if steps < 2:
        raise ValueError('at least two path intervals are required')
    start_time = time.monotonic()
    config = load(BASE)
    anchor = next(p for p in load(ANCHOR)['points'] if p['point_id'] == 'center')
    continuation = config['analytic_continuation']
    assert anchor['target']['period_branch'] == continuation['reference_target_period_branch']
    assert anchor['target']['lifts'] == continuation['reference_target_lifts']
    reference_omega = matrix(anchor['omega_reference'])
    source_spin = SpinCharacteristic((1, 1), (0, 0))
    assert source_spin.transport(config['source_to_target']).pairs == ((0, 0), (0, 0))
    rows = []
    for point in config['points']:
        result = {'point_id': point['point_id']}
        destination = matrix(point['omega_reference'])
        for channel in ('source', 'target'):
            data, origin = point[channel], anchor[channel]
            reference_branch = np.asarray(origin['period_branch'])
            state = ThetaLogBranch(tuple(map(complex, origin['q_values'])))
            transform = (lambda omega: action(SOURCE_REMARKING, omega)) if channel == 'source' else omega_action
            reference_lifts = (SOURCE_FIXED_SPIN_LIFTS if channel == 'source'
                               else (tuple(continuation['reference_target_lifts']),))
            samples = [sample(state, transform(reference_omega), reference_branch, 0)]
            phase_steps, cuts = [], []
            for t in np.linspace(0, 1, steps+1)[1:]:
                omega = transform((1-t)*reference_omega+t*destination)
                chart = inverse_chart(omega, state.q_values)
                new_q = tuple(map(complex, chart['q_values']))
                phase_steps.extend(abs(cmath.phase(b/a)) for a, b in zip(state.q_values, new_q))
                state = state.advance(new_q, maximum_phase_step=np.pi/4)
                current = sample(state, omega, reference_branch, t)
                if channel == 'target':
                    for edge, (old, new) in enumerate(zip(samples[-1]['log_windings'], state.windings)):
                        if old != new:
                            cuts.append(cut_check(samples[-1], current, edge, reference_lifts[0]))
                samples.append(current)
            saved_q = tuple(map(complex, data['q_values']))
            q_error = float(max(abs(a/b-1) for a, b in zip(state.q_values, saved_q)))
            if q_error > 1e-8 or samples[-1]['period_branch'] != data['period_branch']:
                raise AssertionError('inverse path does not end at the saved plumbing chart')
            expected = tuple(state.principal_root_lifts(lift) for lift in reference_lifts)
            saved = SOURCE_FIXED_SPIN_LIFTS if channel == 'source' else (tuple(data['lifts']),)
            if expected != saved:
                raise AssertionError(f'{point["point_id"]} {channel}: saved lifts omit branch compensation')
            root_error = max(abs(e*cmath.sqrt(q)/(e0*s)-1)
                             for old, new in zip(reference_lifts, saved)
                             for e0, e, q, s in zip(old, new, state.q_values, state.square_roots))
            result[channel] = {
                'marked_characteristic': data['characteristic'],
                'reference_period_branch': reference_branch.tolist(),
                'reference_lifts': reference_lifts, 'saved_lifts': saved,
                'unwrapped_endpoint_lifts': expected,
                'endpoint_log_windings': state.windings,
                'saved_lifts_match_path': True,
                'maximum_phase_step_radians': max(phase_steps),
                'endpoint_q_relative_error': q_error,
                'lifted_root_identity_relative_error': float(root_error),
                'local_free_cut_checks': cuts, 'samples': samples}
            print(point['point_id'], channel, 'windings', state.windings,
                  'saved eta matches', expected == saved, flush=True)
        rows.append(result)
    all_charts = [p[c] for p in rows for c in ('source', 'target')]
    all_samples = [s for c in all_charts for s in c['samples']]
    all_cuts = [x for c in all_charts for x in c['local_free_cut_checks']]
    report = {
        'schema': 'theta-sqrt-continuation-audit-v1', 'status': 'passed',
        'path': 'Omega_reference(t)=(1-t)*Omega_anchor+t*Omega_saved, separately for every point',
        'path_intervals': steps, 'old_path_samples_were_saved': False,
        'interpretation': 'New explicit paths independently reproduce the previously saved endpoint continuation.',
        'edge_order': ['zero', 'one', 'infinity'], 'rows': rows,
        'all_saved_lifts_match': True, 'absolute_reference_spin_calibration_tested': False,
        'Liouville_recomputed': False, 'production_parameters_changed': False,
        'maximum_endpoint_q_relative_error': max(c['endpoint_q_relative_error'] for c in all_charts),
        'maximum_continued_period_residual': max(s['continued_period_residual'] for s in all_samples),
        'maximum_lifted_root_identity_error': max(c['lifted_root_identity_relative_error'] for c in all_charts),
        'maximum_compensated_free_jump': max(x['compensated_relative_jump'] for x in all_cuts),
        'uncompensated_free_jump_range': [min(x['uncompensated_relative_jump'] for x in all_cuts),
                                          max(x['uncompensated_relative_jump'] for x in all_cuts)],
        'input_and_implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (BASE, ANCHOR, Path(__file__), ROOT/'Code/genus_2/spin_structure.py')},
        'wall_seconds': time.monotonic()-start_time}
    output.mkdir(parents=True, exist_ok=True)
    (output/'sqrt_branch_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Saved plumbing square-root continuation: independently confirmed', '',
             'The saved eta choices already compensate the principal-square-root cuts. '
             'No additional eta flip is justified by this audit.', '',
             'I re-inverted straight paths in the original marked period coordinates from '
             'the saved center to each of the ten surfaces, in both charts. Each step uses '
             'collocation and a Schottky period check. I unwrapped q phases without calling '
             'the eta-to-spin helper and independently checked the charge-period branch. '
             'These are new verification paths; the old configuration saved endpoints and '
             'the continuation rule, not intermediate path samples.', '',
             'With Log_cont(q_e)=Log_pr(q_e)+2 pi i w_e, the root lift in principal '
             'coordinates is eta_e=eta_ref,e (-1)^w_e. The product eta_e sqrt_pr(q_e) '
             'equals eta_ref,e sqrt_cont(q_e). The marked spin stays fixed.', '',
             'Reference target lifts are (+,-,+), with B_ref=[[-1,-1],[-1,0]]. '
             'All tuples below use geometry order (zero, one, infinity).', '',
             '| Point | Source windings | Target windings | Saved target eta |',
             '|---|---|---|---|']
    lines += [f"| {p['point_id']} | {p['source']['endpoint_log_windings']} | "
              f"{p['target']['endpoint_log_windings']} | {p['target']['saved_lifts'][0]} |" for p in rows]
    lines += ['', f"All {len(all_charts)} paths reproduce the saved endpoint roots and signs. "
              f"Maximum endpoint q relative error: {report['maximum_endpoint_q_relative_error']:.3e}; "
              f"continued-period residual: {report['maximum_continued_period_residual']:.3e}.", '',
              f"In {len(all_cuts)} local free-block cut tests, at relative imaginary offsets 1e-10 "
              f"on either side, the largest compensated relative jump is "
              f"{report['maximum_compensated_free_jump']:.3e}. Holding eta fixed instead gives jumps "
              f"from {report['uncompensated_free_jump_range'][0]:.3e} to "
              f"{report['uncompensated_free_jump_range'][1]:.3e}.", '',
              'This confirms continuation relative to the chosen anchor. It does not '
              'identify the anchor Human/HJS block basis with an individual geometric-spin '
              'partition function. The earlier free-basis concern is a separate question; '
              'a failure of relative branch bookkeeping is excluded here. Source Ramond '
              'edges have no cut crossings on these paths, so this audit does not test a '
              'Ramond Clifford action at a crossing.', '',
              'No Liouville integral, momentum grid, block order, saved eta, or production '
              'configuration was changed.', '', '[Full path data and checks](sqrt_branch_audit.json)', '']
    (output/'sqrt_branch_audit.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--path-intervals', type=int, default=16)
    args = parser.parse_args()
    report = audit(args.output, args.path_intervals)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
