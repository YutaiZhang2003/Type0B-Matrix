#!/usr/bin/env python3
"""Build an explicit spin ledger and rerun its independent free controls.

This audits the saved ten geometries without altering any Liouville grid or
block cutoff. An unverified interacting boundary never produces a Q table.
"""
from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ('Code', 'Code/genus_2', 'Code/genus_2_cross_channel', 'Code/c_Recursion',
                 'Code/double_virasoro/nsrr', 'Code/full_ramond_block_runtime'):
    sys.path.insert(0, str(ROOT/relative))

import numpy as np

from fixed_spin_free_plumbing import charged_frame, charge_lattice_sum, fixed_spin_partition
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from spin_structure import (ALL_LIFTS, SpinCharacteristic, ThetaSpinFrame,
                            UnverifiedSpinSewing, remove_human_theta_quadratic_factor)

BASE = ROOT/'Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json'
DEFAULT_OUTPUT = ROOT/'Data Set/nsrr_spin_tracking_20260911'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def audit(output):
    config = json.loads(BASE.read_text())
    source_spin = SpinCharacteristic((1, 1), (0, 0))
    target_spin = source_spin.transport(config['source_to_target'])
    if target_spin != SpinCharacteristic((0, 0), (0, 0)):
        raise AssertionError('the requested target geometric spin changed')
    rows = []
    for point in config['points']:
        result = {'point_id': point['point_id']}
        for channel, spin in (('source', source_spin), ('target', target_spin)):
            data = point[channel]
            q = tuple(complex(z) for z in data['q_values'])
            frame = ThetaSpinFrame(spin, q, data['period_branch'])
            omega = [[complex(z) for z in row] for row in data['omega']]
            free = fixed_spin_partition(q, omega, spin.pairs,
                                       period_branch=data['period_branch'], max_mode=32, lattice_cutoff=5)
            result[channel] = {'frame': frame.record(), 'frame_digest': frame.digest,
                               'period_residual': free['period_residual'],
                               'Z_free': free['Z_free'],
                               'saved_free_relative_change': free['Z_free']/data['Z_free']-1,
                               'interacting_sewing_status': 'unverified'}
            if channel == 'target':
                charged = charged_frame(q, max_mode=32)
                controls = {lift: theta_physical_fermion_fredholm(q, lift, max_mode=32)
                            for lift in ALL_LIFTS}
                human = {lift: value.chiral_value for lift, value in controls.items()}
                spin_rows = []
                for beta in product((0, 1), repeat=2):
                    charge_spin = SpinCharacteristic((0, 0), beta)
                    lifts = charge_spin.all_ns_determinant_lifts()
                    converted = remove_human_theta_quadratic_factor(human, lifts)
                    raw = controls[lifts].determinant_values[0]
                    lattice = charge_lattice_sum(charged.omega_charge, charge_spin.pairs, cutoff=5)
                    expected = abs(charged.boson_chiral*lattice)
                    error = abs(abs(converted)**2/expected-1)
                    if error > 2e-11 or abs(converted-raw) > 2e-12:
                        raise ArithmeticError('geometric free-spin conversion failed')
                    spin_rows.append({'charge_characteristic': charge_spin.pairs,
                                      'marked_characteristic': charge_spin.charge_frame(-np.asarray(data['period_branch'])).pairs,
                                      'raw_determinant_lifts': lifts, 'converted_chiral': converted,
                                      'raw_chiral': raw, 'Majorana_expected': expected,
                                      'Majorana_relative_error': error})
                desired = next(r for r in spin_rows if r['charge_characteristic'] == frame.charge_spin.pairs)
                old = controls[tuple(data['lifts'])].nonchiral_value/free['Z_majorana']
                result[channel].update(free_spin_controls=spin_rows,
                                       saved_Human_lifts=data['lifts'],
                                       saved_free_ratio=old,
                                       corrected_free_ratio=abs(desired['converted_chiral'])**2/free['Z_majorana'])
        rows.append(result)
        print(point['point_id'], 'free spin corrected ratio', result['target']['corrected_free_ratio'], flush=True)
    requirements = {
        'source': ['derive the NR/RN local-coordinate and BPZ conversion of both physical Ramond vertices',
                   'transport simultaneous holomorphic and antiholomorphic spin operators through the restricted Ramond pairing',
                   'verify the resulting interacting form/parity contraction in a complete free-superfield limit'],
        'target': ['derive the physical all-NS nonchiral contraction in the geometric spin basis',
                   'verify the complete recursion, including its vacuum/global parity convolution, against that contraction'],
    }
    report = {'schema': 'spin-tracked-comparison-audit-v1',
              'status': 'blocked_by_unverified_interacting_sewing',
              'input_config_sha256': sha(BASE),
              'source_to_target': config['source_to_target'],
              'free_controls_status': 'passed', 'rows': rows,
              'maximum_corrected_free_error': max(r['Majorana_relative_error'] for p in rows
                                                 for r in p['target']['free_spin_controls']),
              'maximum_saved_free_error': max(abs(p['target']['saved_free_ratio']-1) for p in rows),
              'missing_interacting_certificates': requirements,
              'Liouville_recomputed': False,
              'frozen_numerical_settings': {
                  'fine_momentum_order': 12, 'correction_momentum_order': 7,
                  'source_total_block_order': 8, 'target_null_level': 8,
                  'target_endpoint_cap': 8, 'source_native_digits': 40, 'target_digits': 50},
              'implementation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                  (Path(__file__), ROOT/'Code/genus_2/spin_structure.py',
                   ROOT/'Code/genus_2/fixed_spin_free_plumbing.py',
                   ROOT/'Code/genus_2/physical_free_plumbing_resummation.py')},
              'cache_policy': 'Contracted old node values are not convertible; a verified rerun must save complex spin components before contraction.'}
    output.mkdir(parents=True, exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report, indent=2, default=encode)+'\n')
    lines = ['# Explicit spin tracking: free controls corrected; interacting rerun blocked', '',
             'The previous Liouville table is not a certified fixed-spin comparison. '
             'No replacement Liouville values are claimed by this audit.', '',
             '**Branch-continuation clarification:** The saved eta values already '
             'compensate the principal-square-root jumps. The independent '
             '[path audit](sqrt_branch_audit.md) reproduces all ten saved points '
             'in both channels. The free-basis check below does not diagnose a '
             'missed branch flip and does not justify applying another flip. '
             'It concerns the absolute reference spin/basis identification.', '',
             'Each chart now carries a binary marked characteristic, its integer period branch, '
             'the transported charge characteristic, edge sectors, edge order, propagation convention, '
             'and a digest binding those data to its q triple. The source [11|00] transports to '
             'the target [00|00] under the saved symplectic map.', '',
             'The all-NS free adapter resolves the complete complex lift vector, removes its '
             'explicit Human-Note quadratic parity factor, and selects the geometric determinant '
             'character. This is verified for all four all-NS spins on all ten target surfaces. '
             'It is not silently extended to interacting or Ramond sewing.', '',
             '| Point | Previous free-spin ratio | Corrected free-spin ratio |',
             '|---|---:|---:|']
    lines += [f"| {p['point_id']} | {p['target']['saved_free_ratio']:.12f} | {p['target']['corrected_free_ratio']:.12f} |" for p in rows]
    lines += ['', f"Maximum corrected error over all 40 spin checks: {report['maximum_corrected_free_error']:.3e}.", '',
              '## Work required before a physical Liouville rerun', '']
    for channel, items in requirements.items():
        lines += [f'**{channel.capitalize()}**', '']+[f'- {item}.' for item in items]+['']
    lines += ['The shared certificate guard rejects a free-only check, a bare eta label, '
              'a different chart, or an incomplete physical vertex/pairing validation. '
              'The old momentum grids and block cutoffs remain frozen. Their contracted '
              'integrands cannot reconstruct the missing complex spin components.', '',
              '[Machine-readable audit](audit.json)', '',
              '## Ramond free-superfield calibration', '',
              'Run `Code/genus_2/audit_nsrr_free_sewing_limit.py` for the independent '
              'order-2/order-3 PBW diagnostic (no production PBW or Liouville integration). '
              'The complete legacy vertex/reflection/spin dictionary, with its proposed '
              'free couplings stated explicitly, cancels a required leading NS half-level '
              'term. At the smaller diagnostic q triple the normalized local-kernel ratio '
              'is 0.992259276625 at order 2 and 0.992259269024 at order 3. '
              'The single even projected block has ratios 0.999999967722 and '
              '1.000000000004. This exposes a boundary-calibration problem; it does not '
              'establish the single even block as a universal interacting replacement.', '',
              'Writing x=sqrt(q_NS) and stripping the common primary factor, the exact '
              'ground/first-half-level Ward terms are F0=sqrt(2)(1+x/2) and '
              'F1=-i sqrt(2)(1-x/2). The legacy quarter-modulus cancels the linear term, '
              'whereas bosonization requires 2+x+conjugate(x)+O(|x|^2). '
              'This discrepancy concerns the combined vertex/BPZ/spin dictionary, '
              'not the correctness of the checked chiral PBW or double-Virasoro blocks.', '',
              '[Ramond calibration data](nsrr_free_limit.json)', '']
    (output/'README.md').write_text('\n'.join(lines))
    return report


def require_interacting(report):
    if report['status'] != 'verified':
        raise UnverifiedSpinSewing('The interacting spin-to-sewing adapters have not passed validation; no physical Q_L comparison can be generated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--require-interacting', action='store_true')
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps({k: report[k] for k in ('status', 'free_controls_status', 'maximum_corrected_free_error', 'Liouville_recomputed')}))
    if args.require_interacting:
        require_interacting(report)


if __name__ == '__main__':
    main()
