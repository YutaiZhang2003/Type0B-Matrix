#!/usr/bin/env python3
"""Compare directly with Type0B code without modifying it or doing integrals."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import sympy as sp

from literature_component_blocks import ThreePoint
from literature_human_bpz import bpz_gram, bpz_three_point


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(root):
    sources = {
        'pbw': root/'Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py',
        'poles': root/'Code/c_recursion_check/check_poles_and_signs.py',
        'ordered_slots': root/'Code/c_recursion_check/check_all_mixed_slots.py',
        'note': root/'Human Notes/SCblock.tex',
    }
    pbw = load_module('type0b_pbw_reference', sources['pbw'])
    checks = {}
    for key in ('poles', 'ordered_slots'):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(sources[key]), run_name='__main__')
        checks[key] = dict(passed=True, output=output.getvalue().strip())
    slots = load_module('type0b_ordered_slot_reference', sources['ordered_slots'])

    # Exercise the actual PBW library in its documented one-descendant-leg
    # domain, also allowing the external NS primary's G_-1/2 component.
    ps = [sp.Rational(k, 100) for k in (31, 37, 43)]
    hn = [sp.Rational(1, 16)+p*p/2 for p in ps]
    br = [-sp.I*p/sp.sqrt(2) for p in ps]
    native = ThreePoint('NS', 'R', 'R', *map(float, ps))
    word = lambda v: tuple((kind, sp.Rational(n, 2)) for kind, n in v[0])
    count, nr_error, gram_error = 0, 0., 0.
    for f in (0, 1):
        for eta in (1, -1):
            ward = pbw.GeneralizedNRRWard(
                p_phi=0, form_parity=f, eta=eta, h_ns=hn[0],
                h_second=hn[1]+sp.Rational(1, 16),
                h_third=hn[2]+sp.Rational(1, 16),
                beta_second=br[1], beta_third=br[2], central_charge=3)
            for n in range(4):
                for ns in native.m3.basis(n, n % 2):
                    for m in (0, 2):
                        if m and n > 1:
                            continue
                        for k in (0, 1):
                            for r in native.m1.basis(m, k):
                                for middle in (0, 1):
                                    if (n+middle+k) % 2 != f:
                                        continue
                                    ours = bpz_three_point(native.nr(ns, middle, r), 'NR', f, eta)
                                    reference = complex(ward.value(word(ns), (), middle, word(r), r[1]))
                                    nr_error = max(nr_error, float(abs(ours-reference)))
                                    count += 1
    module = pbw.RamondPBWModule(hn[2]+sp.Rational(1, 16), br[2], sp.Integer(3))
    for level in (0, 2, 4):
        for k in (0, 1):
            basis = native.m1.basis(level, k)
            states = [pbw.RamondState(word(v), v[1]) for v in basis]
            direct = np.array([[complex(module.inner_product(a, b)) for b in states] for a in states])
            if direct.size:
                gram_error = max(gram_error, float(np.max(abs(direct-bpz_gram(native.m1, level, k)))))
    assert nr_error < 1e-12 and gram_error < 1e-12

    beta2, beta3 = sp.symbols('beta2 beta3')
    ns_rows = []
    for eta in (1, -1):
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            f = (a+b+1) % 2
            ward = pbw.GeneralizedNRRWard(
                p_phi=0, form_parity=f, eta=eta, h_ns=0,
                h_second=1-beta2**2, h_third=1-beta3**2,
                beta_second=beta2, beta_third=beta3, central_charge=24)
            plane = ward.value((('G', -sp.Rational(1, 2)),), (), a, (), b)
            g2, flipped2 = slots.g0(beta2, a)
            g3, flipped3 = slots.g0(beta3, b)
            # Assumed fixed-parity identity in check_poles_and_signs.py:303.
            # Agreement below does not derive it from contour deformation.
            fixed = ((-1)**b*g2*slots.ground(f, eta, flipped2, b)
                     -sp.I*g3*slots.ground(f, eta, a, flipped3))
            printed = ((1+sp.I*(-1)**f)/sp.sqrt(2)*(beta3-eta*beta2)
                       * slots.ground(1-f, -eta, a, b))
            residual = pbw.clean(fixed-printed)
            assert residual == 0
            assert pbw.clean(fixed-(-1)**b*plane) == 0
            ns_rows.append(dict(eta=eta, grounds=[a, b], f=f,
                                plane=str(plane), type0b_fixed_parity=str(pbw.clean(fixed)),
                                appendix=str(pbw.clean(printed)),
                                fixed_over_plane=str(pbw.clean(fixed/plane)), residual=str(residual)))
    r_rows = []
    bb, lam, bj = sp.symbols('b lambda beta_j', nonzero=True)
    for null_slot in (2, 3):
        for p_phi in (0, 1):
            for eta in (1, -1):
                result = pbw.contract_ramond_null(
                    r=2, s=1, p_phi=p_phi, eta=eta, b=bb,
                    lambda_i=lam, beta_j=bj, null_slot=null_slot)
                assert result['generalized_residual'] == 0
                r_rows.append(dict(slot=null_slot, primary_parity=p_phi, eta=eta,
                                   effective_eta=int(result['effective_eta']),
                                   coordinate_sign=int(result['coordinate_sign']),
                                   residual=str(result['generalized_residual'])))
    return dict(
        conclusion='Previous Appendix A correction claims withdrawn: raw plane BPZ and Type0B fixed-parity forms were conflated.',
        complete_human_note_dictionary_certified=False,
        fixed_identity_derivation_verified=False,
        certificate_scope='The fixed-parity Ward identity is an input to the Type0B certificate, not derived by it. Its ground-component rephasing relative to PBW is verified algebraically; the map from the Human Note definitions remains unproved.',
        integration='none; symbolic anchors and low-level PBW comparisons only',
        source_files={key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                      for key, path in sources.items()},
        existing_type0b_certificates=checks,
        raw_plane_comparison=dict(NR_components=count, max_NR_error=nr_error,
                                  max_R_Gram_error=gram_error),
        NS_null_fixed_parity_anchors=ns_rows,
        R_null_raw_plane_convention_checks=r_rows,
        RN_status='Type0B ordered-slot certificate passes; previous same-beta raw-bra conversion was not its full local-coordinate/dual dictionary.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type0b-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.type0b_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['conclusion'])
    print(result['raw_plane_comparison'])
    print('Both existing Type0B certificates and all 16 targeted null anchors passed.')
