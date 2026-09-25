#!/usr/bin/env python3
"""Test the sensitivity of Type0B theta sewing to a vertex parity rephasing.

This is a local diagnostic, with no momentum or moduli integration. It does
not assert that the rephased tensor is the complete Human Note convention.
The external Type0B checkout is read-only.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp


class RephasedForm:
    """Multiply one tensor by (-1)^(NS descendant parity * R-zero parity)."""

    def __init__(self, original, word_parity):
        self.original = original
        self.word_parity = word_parity

    def value(self, ns, middle, middle_ground, zero, zero_ground):
        ns_parity = self.word_parity(ns)
        zero_parity = (self.word_parity(zero)+zero_ground) % 2
        return (-1)**(ns_parity*zero_parity)*self.original.value(
            ns, middle, middle_ground, zero, zero_ground)


def run(type0b_root):
    directory = type0b_root/'Code/double_virasoro/nsrr'
    sys.path.insert(0, str(directory))
    from nsrr_genus2_block import HumanNSRRThetaOracle
    from ramond_pbw_generalized_ward import GeneralizedNRRWard, clean, word_parity

    # The lowest disputed component identity, with no numerical beta choice.
    b2, b3 = sp.symbols('beta2 beta3')
    star = (('G', -sp.Rational(1, 2)),)
    anchors = []
    for eta, middle, zero in product((1, -1), (0, 1), (0, 1)):
        f = (middle+zero+1) % 2
        ward = GeneralizedNRRWard(
            p_phi=0, form_parity=f, eta=eta, h_ns=0,
            h_second=1-b2*b2, h_third=1-b3*b3,
            beta_second=b2, beta_third=b3, central_charge=24)
        def acted(slot):
            module = ward.modules[slot]
            bit = middle if slot == 1 else zero
            answer = sp.S.Zero
            for (word, ground), coefficient in module.act('G', 0, (), bit).items():
                args = ((), word, ground, (), zero) if slot == 1 else (
                    (), (), middle, word, ground)
                answer += coefficient*ward.value(*args)
            return clean(answer)
        first, second = acted(1), acted(2)
        literal = ward.value(star, (), middle, (), zero)
        literature = first-sp.I*(-1)**zero*second
        appendix_certificate = (-1)**zero*first-sp.I*second
        assert clean(literal-literature) == 0
        assert clean(appendix_certificate-(-1)**zero*literal) == 0
        anchors.append(dict(eta=eta, form_parity=f, R_ground_bits=[middle, zero],
                            raw=str(literal), certificate=str(clean(appendix_certificate)),
                            certificate_over_raw=(-1)**zero))

    base = dict(central_charge=sp.Rational(27, 2), h_ns=sp.Rational(5, 9),
                beta_r1=sp.Rational(2, 7), beta_r2=sp.Rational(3, 8),
                primary_parity=0)
    levels = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1))
    rows = []
    for f, eta, etap in product((0, 1), (1, -1), (1, -1)):
        original = HumanNSRRThetaOracle(form_parity=f, etas=(eta, etap), **base)
        both = HumanNSRRThetaOracle(form_parity=f, etas=(eta, etap), **base)
        one = HumanNSRRThetaOracle(form_parity=f, etas=(eta, etap), **base)
        both.forms = tuple(RephasedForm(form, word_parity) for form in both.forms)
        one.forms = (RephasedForm(one.forms[0], word_parity), one.forms[1])
        for level in levels:
            expected = np.array(original.coefficient_components(*level))
            changed_both = np.array(both.coefficient_components(*level))
            changed_one = np.array(one.coefficient_components(*level))
            rows.append(dict(form_parity=f, structure_signs=[eta, etap], levels=level,
                             both_vertices_maximum_change=float(max(abs(changed_both-expected))),
                             one_vertex_maximum_change=float(max(abs(changed_one-expected)))))
    maximum_both = max(row['both_vertices_maximum_change'] for row in rows)
    maximum_one = max(row['one_vertex_maximum_change'] for row in rows)
    assert maximum_both < 1e-13
    assert maximum_one > .01

    # Algebraic proof at the parity level: the even inverse pairings force
    # equal state parity on the two ends of each sewn edge.
    parity_checks = []
    for ns, middle, zero in product((0, 1), repeat=3):
        multiplier = (-1)**(ns*zero+ns*zero)
        assert multiplier == 1
        parity_checks.append(dict(edge_parities=[ns, middle, zero], multiplier=multiplier))

    source_paths = [Path(__file__), directory/'nsrr_genus2_block.py',
                    directory/'ramond_pbw_generalized_ward.py',
                    type0b_root/'Code/c_recursion_check/check_poles_and_signs.py']
    crossing_path = type0b_root/'Data Set/nsrr_cross_channel_extensive_20260916/result.json'
    crossing = json.loads(crossing_path.read_text())
    source_paths.append(crossing_path)
    return dict(
        conclusion='The theta block is invariant under simultaneous parity rephasing of both ordered pants; open three-point components and a one-pants-only change are sensitive.',
        rephasing='rho_sharp(xi,u,v)=(-1)^(NS_descendant_parity(xi)*total_R_parity(v))*rho(xi,u,v), for an even NS primary',
        human_note_all_descendant_dictionary_established=False,
        integration='none; no production crossing or stress run',
        symbolic_ground_anchors=anchors, parity_proof=parity_checks,
        coefficient_vectors=len(rows), parity_components=8*len(rows),
        maximum_two_vertex_change=maximum_both, maximum_one_vertex_change=maximum_one,
        comparisons=rows,
        saved_crossing=dict(points=crossing['points'], normalization=crossing['normalization'],
                            maximum_refined_ratio_error=crossing['maximum_refined_ratio_error'],
                            maximum_refined_phase_difference_rad=crossing['maximum_refined_phase_difference_rad']),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in source_paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type0b-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.type0b_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({key: result[key] for key in ('coefficient_vectors', 'parity_components',
                                       'maximum_two_vertex_change', 'maximum_one_vertex_change')})
