#!/usr/bin/env python3
"""A necessary free-superfield calibration of the legacy NSRR boundary.

Uses direct PBW only for diagnostic total orders two and three, not for a
Liouville integral. A free-model result is not promoted to a universal
interacting replacement matrix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ('Code', 'Code/genus_2', 'Code/genus_2_cross_channel', 'Code/c_Recursion',
                 'Code/double_virasoro/nsrr', 'Code/full_ramond_block_runtime'):
    sys.path.insert(0, str(ROOT/relative))

import sympy as sp
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples
from physical_nsrr_sewing import CHANNELS, contract_physical_blocks
from physical_free_plumbing_resummation import theta_charged_boson_resummation
from fixed_spin_free_plumbing import charged_frame, charge_lattice_sum


def audit():
    p_zero, p_one = sp.Rational(31, 100), sp.Rational(47, 100)
    c = sp.Rational(3, 2)
    h_ns = (p_one-p_zero)**2/2
    components = {}
    for f in (0, 1):
        oracle = HumanNSRRThetaOracle(central_charge=c, h_ns=h_ns,
            beta_r1=sp.I*p_one/sp.sqrt(2), beta_r2=sp.I*p_zero/sp.sqrt(2),
            form_parity=f, primary_parity=0, etas=(1, 1))
        components[f] = {e: oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                         for e in level_triples(6)}
    base = json.loads((ROOT/'Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json').read_text())
    original_q = tuple(complex(z) for z in next(p for p in base['points']
                          if p['point_id'] == 'generic_04')['source']['q_values'])
    rows = []
    for scale in (.02, .1):
        q = tuple(scale*z for z in original_q)
        qs = q[::-1]
        primary = math.prod(qs[k]**float(h) for k, h in enumerate(
            (h_ns, c/24+p_one**2/2, c/24+p_zero**2/2)))
        charged = theta_charged_boson_resummation(q, alpha_zero=float(p_zero),
                                                alpha_one=-float(p_one), max_mode=32)
        frame = charged_frame(q, max_mode=32)
        majorana = abs(frame.boson_chiral*charge_lattice_sum(frame.omega_charge, ((1, 1), (0, 0)), cutoff=5))
        expected = abs(charged.chiral_value)**2*majorana
        for order in (2, 3):
            values = {}
            for f in (0, 1):
                values[f] = sum(math.sqrt(2)*coefficient
                                * math.prod(qs[k]**(e[k]/2) for k in range(3))
                                for e, vector in components[f].items() if sum(e) <= 2*order
                                for p, coefficient in enumerate(vector) if not ((p >> 1) & 1))
            blocks = {channel: 0j for channel in CHANNELS}
            for f in (0, 1):
                blocks[f, 1, 1] = primary*values[f]
            normalized = contract_physical_blocks(blocks, (2, 0))['total']
            individual = abs(primary*values[0])**2
            rows.append({'q_scale': scale, 'total_PBW_order': order,
                         'free_reference': expected,
                         'legacy_local_kernel_value': normalized,
                         'legacy_local_kernel_ratio': normalized/expected,
                         'legacy_production_multiplier': 4,
                         'legacy_production_ratio': 4*normalized/expected,
                         'single_even_projected_block_ratio': individual/expected})
    # These are exact Ward checkpoints, not a numerical fit of a new kernel.
    # Put x=sqrt(q_NS). A=(beta_zero-beta_one)^2=-h_NS. After the saved
    # half-Ramond projection: F0=sqrt(2)(1+x/2), F1=-i sqrt(2)(1-x/2).
    return {'schema': 'nsrr-free-superfield-boundary-calibration-v1',
            'status': 'legacy_boundary_fails_this_free_field_calibration',
            'theory': 'one free scalar plus one Majorana; c=3/2',
            'signed_scalar_charges_geometry': [float(p_zero), -float(p_one), float(p_one-p_zero)],
            'HJS_beta_momenta_slots': [float(p_one), float(p_zero)],
            'charge_characteristic': [[1, 1], [0, 0]],
            'reference': '|charged Heisenberg pants block|^2 times |P theta[11|00]|',
            'vertex_dictionary_under_test': {'BRY_even': 2, 'BRY_odd': 0, 'HJS_plus': 1, 'HJS_minus': 0},
            'interpretation': 'A necessary test of the complete current vertex/reflection/spin dictionary. Its failure does not determine which interacting matrix replaces it.',
            'leading_chiral_terms': {'F0': 'sqrt(2)*(1+x/2)', 'F1': '-i*sqrt(2)*(1-x/2)', 'x': 'sqrt(q_NS)'},
            'leading_nonchiral_terms': {'legacy_local_kernel': '2+O(|x|^2)',
                                       'free_reference': '2+x+conjugate(x)+O(|x|^2)'},
            'leading_terms_convention': 'common primary factor stripped; Ramond descendant levels zero',
            'rows': rows, 'Liouville_parameters_changed': False, 'Liouville_recomputed': False,
            'replacement_interacting_kernel': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Data Set/nsrr_spin_tracking_20260911/nsrr_free_limit.json')
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
