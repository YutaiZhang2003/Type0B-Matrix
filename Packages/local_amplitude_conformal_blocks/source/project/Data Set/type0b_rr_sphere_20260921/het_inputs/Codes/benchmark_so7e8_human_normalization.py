"""Independent low-order checks against the Type0B human-note conventions.

This audit does NOT alter the production block generators. In particular,
raw mismatches are reported as mismatches, not absorbed into fitted signs.
The NS reference is the printed eight-entry table and global sl(2) formula
in Human Notes/SCblock.tex. R Gram references follow directly from its G0
action and bilinear contravariance. An optional independent Type0B NSRR
implementation is compared only at ground/level one.
"""

import argparse
from functools import lru_cache
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import sympy as sp

from ns_algebra.ns_sca import G as NG, L as NL, descendant_inner_product as ns_pair, fermion_parity
from ns_algebra.ns_three_point_tensor import ns_three_point
from ramond_algebra.ramond_sca import (
    G as RG, L as RL, PBWState, descendant_inner_product as r_pair,
    ground_action_matrix, ground_gram_matrix,
)
from ramond_algebra.nrr_three_point_tensor import (
    hjs_ground_tensor, rr_three_point_hjs, rr_three_point_from_ground_tensor,
)
from ramond_algebra.ns_rr_three_point_tensor import (
    hjs_ns_rr_polynomial_ground_tensor, ns_rr_three_point_from_ground_tensor,
)


ALPHA = (1 + sp.I) / sp.sqrt(2)
HALF = sp.Rational(1, 2)


def clean(value):
    return sp.factor(sp.cancel(sp.expand(value)))


def human_ns_seed(parities, weights):
    """Printed global table, EVEN intrinsic NS primaries, ordered (inf,1,0)."""
    h1, h2, h3 = weights
    return {
        (0, 0, 0): sp.S.One, (1, 0, 0): sp.S.One,
        (0, 1, 0): sp.S.One, (0, 0, 1): -sp.S.One,
        (1, 1, 0): h1 + h2 - h3,
        (1, 0, 1): h1 - h2 + h3,
        (0, 1, 1): h1 - h2 - h3,
        (1, 1, 1): -(h1 + h2 + h3 - HALF),
    }[tuple(parities)]


def human_ns_global(parities, l1_powers, weights):
    """Literal s_km and middle L_-1 formula following the human seed table."""
    h1, h2, h3 = (h + HALF * a for h, a in zip(weights, parities))
    k, ell, m = l1_powers
    skm = sum(
        sp.binomial(k, p) * sp.ff(2*h3+m-1, p) * sp.ff(m, p)
        * sp.rf(h3+h2-h1, m-p) * sp.rf(h1+h2-h3+p-m, k-p)
        for p in range(min(k, m) + 1))
    return clean(human_ns_seed(parities, weights) * skm
                 * sp.rf(h1+k-h2-ell+1-h3-m, ell))


def ns_terminal_convention_phase(words):
    """Conversion verified on the global table; not an all-level repair.

    f is the homogeneous form parity; F0 is the descendant parity at zero.
    The printed table differs by (-1)^(f F0). Intrinsic primaries here are
    even. Extending this to all Ward/residue conventions is separate work.
    """
    parities = tuple(fermion_parity(w) for w in words)
    return (-1) ** ((sum(parities) % 2) * parities[2])


def ns_checks():
    h1, h2, h3, c = sp.symbols("h1 h2 h3 c")
    weights = (h1, h2, h3)
    raw_bad, mapped_bad, seeds = [], [], []
    for parities, powers in product(product((0, 1), repeat=3), repeat=2):
        words = tuple((NL(-1),)*k + ((NG(-HALF),) if a else ())
                      for k, a in zip(powers, parities))
        expected = human_ns_global(parities, powers, weights)
        actual = ns_three_point(*words, h_infinity=h1, h_middle=h2, h_zero=h3, c=c)
        residual = clean(actual - expected)
        mapped_residual = clean(ns_terminal_convention_phase(words) * actual - expected)
        label = dict(G_parities=parities, L1_powers=powers)
        if residual != 0:
            raw_bad.append(dict(label, residual=str(residual)))
        if mapped_residual != 0:
            mapped_bad.append(dict(label, residual=str(mapped_residual)))
        if powers == (0, 0, 0):
            seeds.append(dict(parities=parities, human=str(expected), pbw=str(actual), match=residual == 0))
    norms = []
    for k, a in product(range(3), (0, 1)):
        word = (NL(-1),)*k + ((NG(-HALF),) if a else ())
        expected = sp.factorial(k) * sp.rf(2*h1, k+a)
        actual = ns_pair(word, word, h=h1, c=c)
        norms.append(dict(k=k, a=a, residual=str(clean(actual-expected))))
    return dict(global_comparisons=64, raw_mismatches=len(raw_bad),
                after_terminal_phase_mismatches=len(mapped_bad),
                raw_mismatch_details=raw_bad, mapped_mismatch_details=mapped_bad,
                primary_component_table=seeds, global_norms=norms,
                conclusion="NS Gram normalization agrees; raw ordered three-point convention does not")


def ramond_gram_checks():
    beta, c = sp.symbols("beta c", nonzero=True)
    h = c/24-beta**2
    ket = sp.diag(1, 1/(ALPHA*beta))
    hjs_dual = sp.diag(1, 1/(sp.I*ALPHA*beta))
    action = sp.Matrix([[0, sp.I*ALPHA*beta], [ALPHA*beta, 0]])
    poly = ground_gram_matrix(h, c)
    bilinear = (ket.T * poly * ket).applyfunc(clean)
    hjs = (hjs_dual.T * poly * ket).applyfunc(clean)
    action_residual = (ket.inv() * ground_action_matrix(h, c) * ket - action).applyfunc(clean)
    grams = []
    for parity in (0, 1):
        states = (PBWState((RL(-1),), parity), PBWState((RG(-1),), 1-parity))
        factors = tuple(1/(ALPHA*beta)**state.ground_parity for state in states)
        actual = sp.Matrix([[factors[i]*factors[j]*r_pair(left, right, h=h, c=c)
                             for j, right in enumerate(states)] for i, left in enumerate(states)])
        expected = sp.Matrix([[2*h*sp.I**parity, sp.Rational(3,2)*sp.I*ALPHA*beta],
                              [sp.Rational(3,2)*sp.I*ALPHA*beta, (2*h+c/4)*sp.I**(1-parity)]])
        grams.append(dict(parity=parity, reference=str(expected), residual=str((actual-expected).applyfunc(clean))))
    return dict(ket_basis="u_minus = exp(i*pi/4)*beta*w_minus",
                G0_residual=str(action_residual),
                bilinear_ground_gram=str(bilinear), hjs_dual_ground_gram=str(hjs),
                bilinear_G0_contravariance_residual=str((action.T*bilinear-bilinear*action).applyfunc(clean)),
                unit_metric_bilinear_G0_residual=str((action.T-action).applyfunc(clean)),
                level_one=grams,
                conclusion="polynomial PBW agrees after linear basis conversion; unit HJS Gram uses a different dual frame")


def linear_rr_tensor(infinity, middle_word, zero, *, h_ns, beta_infinity, beta_zero, c, structure_sign):
    """Diagnostic literal bilinear R--NS--R form with unit component seeds.

    Both Ramond changes of basis are linear. Unlike the production HJS
    wrapper, there is no extra i on the infinity conversion. This tests a
    convention map, not independent validity of the RR Ward contour system.
    """
    left, right = sp.diag(1, ALPHA*beta_infinity), sp.diag(1, ALPHA*beta_zero)
    ground = left*hjs_ground_tensor(structure_sign)*right
    result = rr_three_point_from_ground_tensor(
        infinity, middle_word, zero, h_infinity=c/24-beta_infinity**2,
        h_ns=h_ns, h_zero=c/24-beta_zero**2, c=c, ground_tensor=ground)
    return clean(result/(ALPHA*beta_infinity)**infinity.ground_parity
                 /(ALPHA*beta_zero)**zero.ground_parity)


def rr_dual_checks():
    c, b4, b1, hn = sp.Rational(27,2), 2*sp.I/5, 3*sp.I/7, sp.Rational(4,9)
    words = ((), (RL(-1),), (RG(-1),))
    raw_bad, converted_bad, total, ground_bad = 0, 0, 0, 0
    @lru_cache(None)
    def raw(infinity, middle, zero, sign):
        return rr_three_point_hjs(infinity, middle, zero, h_ns=hn,
                                  beta_infinity=b4, beta_zero=b1, c=c, structure_sign=sign)
    for w4, wn, w1, a4, a1, sign in product(words, ((),(NG(-HALF),)), words, (0,1), (0,1), (-1,1)):
        inf, zero = PBWState(w4,a4), PBWState(w1,a1)
        expected = linear_rr_tensor(inf,wn,zero,h_ns=hn,beta_infinity=b4,beta_zero=b1,c=c,structure_sign=sign)
        actual = raw(inf,wn,zero,sign)
        # Exact two-component basis rotation from K_inf^HJS=i K_inf^linear
        # on the odd ground state. This is NOT a fit to a correlator.
        translated = sp.I**a4 * ((1-sp.I)*actual + (1+sp.I)*raw(inf,wn,zero,-sign))/2
        raw_bad += clean(actual-expected) != 0
        converted_bad += clean(translated-expected) != 0
        if not w4 and not wn and not w1:
            ground_bad += clean(actual-expected) != 0
        total += 1
    return dict(comparisons=total, raw_vs_bilinear_mismatches=raw_bad,
                ground_component_mismatches=ground_bad,
                after_analytic_structure_rotation_mismatches=converted_bad,
                interpretation="basis/duality test with the same polynomial Ward reducer; not a crossing validation")


def independent_type0b_nsrr_checks(root):
    path = root / "Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py"
    spec = importlib.util.spec_from_file_location("_human_ramond_ward_audit_reference", path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    c, b2, b3, hn = sp.Rational(27,2), 2*sp.I/5, 3*sp.I/7, sp.Rational(4,9)
    h2, h3 = c/24-b2*b2, c/24-b3*b3
    words, bad, count = ((), (RL(-1),), (RG(-1),)), [], 0
    def convert(word):
        return tuple((mode.kind, sp.sympify(mode.index)) for mode in word)
    for wn,w2,w3,g2,g3,sign in product(((),(NG(-HALF),)),words,words,(0,1),(0,1),(-1,1)):
        f = (len(wn)+sum(mode.kind=="G" for mode in w2+w3)+g2+g3) % 2
        ward = reference.GeneralizedNRRWard(p_phi=0, form_parity=f, eta=sign,
            h_ns=hn,h_second=h2,h_third=h3,beta_second=b2,beta_third=b3,central_charge=c)
        expected = ward.value(convert(wn),convert(w2),g2,convert(w3),g3)
        actual = ns_rr_three_point_from_ground_tensor(wn,PBWState(w2,g2),PBWState(w3,g3),
            h_infinity=hn,h_middle=h2,h_zero=h3,c=c,
            ground_tensor=hjs_ns_rr_polynomial_ground_tensor(b2,b3,structure_sign=sign))
        actual /= (ALPHA*b2)**g2 * (ALPHA*b3)**g3
        residual = clean(actual-expected)
        if residual != 0:
            bad.append(dict(ns_word=str(wn),middle=str((w2,g2)),zero=str((w3,g3)),sign=sign,residual=str(residual)))
        count += 1
    return dict(comparisons=count,mismatches=len(bad),details=bad,
                reference=str(path.resolve()),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(type0b_root=None):
    ns, rr = ns_checks(), rr_dual_checks()
    result = dict(raw_human_convention_compatible=(ns["raw_mismatches"] == 0
                                                  and rr["raw_vs_bilinear_mismatches"] == 0),
                  production_changed=False, expensive_pbw_used=False, ns=ns,
                  ramond_gram=ramond_gram_checks(), rr_infinity_dual=rr)
    if type0b_root is not None:
        root = Path(type0b_root)
        note = root / "Human Notes/SCblock.tex"
        result["human_note"] = dict(path=str(note.resolve()),sha256=hashlib.sha256(note.read_bytes()).hexdigest())
        result["independent_type0b_nsrr"] = independent_type0b_nsrr_checks(root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type0b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.type0b_root)
    sources = (Path(__file__),) + tuple(Path(__file__).parent / name for name in (
        "ns_algebra/ns_sca.py", "ns_algebra/ns_three_point_tensor.py",
        "ramond_algebra/ramond_sca.py", "ramond_algebra/nrr_three_point_tensor.py",
        "ramond_algebra/ns_rr_three_point_tensor.py"))
    report["source_sha256"] = {str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
