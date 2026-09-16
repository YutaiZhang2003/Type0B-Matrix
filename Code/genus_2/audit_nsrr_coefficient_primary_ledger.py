#!/usr/bin/env python3
"""Track supplied NSRR constants, ordered vertex phases, and primary powers.

The unreduced vertex tensor is derived from the literature's ordered R+
and R- operators. The existing eight-channel genus-two matrix is exported
as a LEGACY CANDIDATE, not certified by this local calculation. All F's
below are the descendant-only blocks in Human Notes/SCblock.tex.
"""
from __future__ import annotations

import argparse
import cmath
import csv
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys

import sympy as s

ROOT = Path(__file__).resolve().parents[2]
for relative in ("Code/genus_2", "Code/double_virasoro/nsrr",
                 "Code/ramond_branching_recursion", "Code/full_ramond_block_runtime"):
    sys.path.insert(0, str(ROOT / relative))

from nsrr_plumbing_adapter import NSRRPlumbingInputs
from physical_nsrr_sewing import CHANNELS, physical_form_matrix
from ramond_pbw_generalized_ward import GeneralizedNRRWard
from compute_target import BranchWeights


def pair(z):
    z = complex(z)
    return [z.real, z.imag]


def matrix_strings(matrix):
    return [[str(s.simplify(z)) for z in row] for row in matrix.tolist()]


def ordered_vertex_ledger():
    """Keep all four parity indices until the remaining sewing is specified."""
    el, ol, er, or_ = s.symbols("E_L O_L E_R O_R")
    cl = {1: el/2, -1: ol/2}
    cr = {1: er/2, -1: or_/2}
    # Coefficients multiplying V_a tensor Vbar_b, before matrix-element
    # Koszul signs. Neither index denotes the physical R-family label.
    plus = s.diag(1, -s.I)/s.sqrt(2)
    minus = s.Matrix([[0, 1], [1, 0]])/s.sqrt(2)
    parity_pairs = tuple(itertools.product((0, 1), repeat=2))
    rows, tensors = [], {}
    for eta, etap in itertools.product((1, -1), repeat=2):
        k = eta*etap
        tensor = s.zeros(4)
        for row, (a, c) in enumerate(parity_pairs):
            for col, (b, d) in enumerate(parity_pairs):
                # Bilinear product of separately ordered pants constants;
                # no automatic conjugation of the second coefficient.
                phase = s.simplify(plus[a, b]*plus[c, d]
                                   + k*minus[a, b]*minus[c, d])
                tensor[row, col] = phase
                rows.append(dict(
                    eta_left=eta, eta_right=etap,
                    holo_vertex_parities=f"{a}{c}", anti_vertex_parities=f"{b}{d}",
                    left_three_form_coefficient=str(cl[eta]),
                    right_three_form_coefficient=str(cr[etap]),
                    ordered_vertex_factor=str(phase),
                    coefficient=str(s.expand(cl[eta]*cr[etap]*phase)),
                    includes_primary_powers=False,
                    status="ordered vertex product; remaining graded edges not contracted",
                ))
        expected = s.Matrix([[1, 0, 0, k], [0, -s.I, k, 0],
                             [0, k, -s.I, 0], [k, 0, 0, -1]])/2
        assert tensor == expected
        tensors[f"{eta},{etap}"] = matrix_strings(cl[eta]*cr[etap]*tensor)

    # Independent local normalization check: actual operators acting on the
    # normalized small-representation kets. The +i here includes the
    # (-1) from an odd operator crossing the odd ket factor.
    embedding = s.Matrix([[1, 0], [0, 1], [0, 1], [-s.I, 0]])/s.sqrt(2)
    projector = s.simplify(embedding*embedding.conjugate().T)
    primary_checks = {}
    for eta, etap in itertools.product((1, -1), repeat=2):
        vl = s.Matrix([[1, 0, 0, s.I], [0, eta, eta, 0]])/s.sqrt(2)
        vr = s.Matrix([[1, 0, 0, s.I], [0, etap, etap, 0]])/s.sqrt(2)
        assert s.simplify(vl*embedding) == s.diag(1, eta)
        full = s.simplify(s.trace(vl*projector*vr.conjugate().T))
        diagonal = s.simplify(s.trace(vl*(s.eye(4)/2)*vr.conjugate().T))
        assert full == 1+eta*etap
        assert diagonal == s.Rational(1+eta*etap, 2)
        primary_checks[f"{eta},{etap}"] = dict(
            full_family_sum=str(full),
            diagonal_part_of_projector_only=str(diagonal),
            off_diagonal_part=str(s.simplify(full-diagonal)),
        )
    ground = s.expand(sum(cl[t]*cr[u]*(1+t*u)
                          for t, u in itertools.product((1, -1), repeat=2)))
    assert ground == (el*er+ol*or_)/2
    return dict(
        physical_constants={"d_plus": "(E+O)/2", "d_minus": "(E-O)/2"},
        three_form_constants={"c_plus": "E/2", "c_minus": "O/2"},
        single_vertex_plus=matrix_strings(plus),
        single_vertex_minus_over_eta=matrix_strings(minus),
        parity_pair_order=[list(p) for p in parity_pairs],
        tensors=tensors,
        primary_family_sum=str(ground),
        primary_checks=primary_checks,
        projector=matrix_strings(projector),
        scope="Local primary/operator normalization. The Hermitian ground check does not choose the genus-two BPZ continuation or fixed-spin contraction.",
    ), rows


def legacy_matrix_ledger():
    """Export each coefficient without mistaking the candidate for a result."""
    constants = {1: "E", -1: "O"}
    rows = []
    for eta, etap in itertools.product((1, -1), repeat=2):
        actual = physical_form_matrix(eta, etap)
        k = eta*etap
        stripped = s.Matrix([[1, -s.I*k], [s.I*k, 1]])
        for f, g in itertools.product((0, 1), repeat=2):
            assert abs(complex(stripped[f, g]/4)-actual[f, g]) < 1e-15
            rows.append(dict(
                A=str((f, eta, etap)), B=str((g, eta, etap)),
                constant_product=f"{constants[eta]}_L*{constants[etap]}_R",
                coefficient_of_constant_product=str(stripped[f, g]/16),
                includes_primary_powers=False,
                status="legacy candidate; not inferred from ordered vertex tensor",
            ))
    return rows


def low_level_blocks():
    """Exact Ward computation, with exponents relative to the SCA primary."""
    h, c, b2, b3 = s.symbols("h c beta_2 beta_3", real=True)
    forms = {(f, t): GeneralizedNRRWard(
        p_phi=0, form_parity=f, eta=t, h_ns=h,
        h_second=c/24-b2**2, h_third=c/24-b3**2,
        beta_second=b2, beta_third=b3, central_charge=c)
        for f, t in itertools.product((0, 1), (1, -1))}
    rows = []
    for eta, etap, lifts, half, f in itertools.product(
            (1, -1), (1, -1), tuple(itertools.product((1, -1), repeat=3)),
            (0, 1), (0, 1)):
        word = (("G", -s.Rational(1, 2)),) if half else ()
        actual = 0
        for a, b in itertools.product((0, 1), repeat=2):
            actual += ((-1)**(half*a+half*b+a*b)
                       * lifts[0]**half*lifts[1]**a*lifts[2]**b
                       * forms[f, eta].value(word, (), a, (), b)
                       * forms[f, etap].value(word, (), a, (), b)
                       / ((2*h if half else 1)*s.I**(a+b)))
        k = eta*etap
        aa = (b3-eta*b2)*(b3-etap*b2)
        expected = {
            (0, 0): 1+k*lifts[1]*lifts[2],
            (0, 1): -s.I*(lifts[2]-k*lifts[1]),
            (1, 0): -lifts[0]*aa*(lifts[2]-k*lifts[1])/(2*h),
            (1, 1): -s.I*lifts[0]*aa*(1+k*lifts[1]*lifts[2])/(2*h),
        }[half, f]
        assert s.simplify(actual-expected) == 0
        rows.append(dict(
            f=f, eta_left=eta, eta_right=etap, lifts_slots=list(lifts),
            descendant_exponents_slots=[str(s.Rational(half, 2)), "0", "0"],
            full_exponents_slots=[f"h_NS+{s.Rational(half, 2)}", "h_R1", "h_R0"],
            coefficient=str(s.factor(expected)),
        ))
    return rows


def primary_ledger(config):
    """Different chart q's and edge weights; no pointwise crossing assertion."""
    b = float(config["b"])
    background = b+1/b
    momenta = (.52, .37, .21)  # Deliberately unequal edge labels.
    rows, maximum = [], 0.0
    for point in config["points"]:
        for channel in ("source", "target"):
            q = tuple(complex(z) for z in point[channel]["q_values"])
            sectors = ("R", "R", "NS") if channel == "source" else ("NS",)*3
            weights = tuple(background**2/8+p*p/2+(1/16 if sector == "R" else 0)
                            for p, sector in zip(momenta, sectors))
            logs = tuple(cmath.log(z) for z in q)
            log_primary = sum(h*ell for h, ell in zip(weights, logs))
            primary = cmath.exp(log_primary)
            if channel == "source":
                adapter = NSRRPlumbingInputs(q, (1, 1, 1), sectors)
                relative_error = abs(adapter.primary(b, momenta)/primary-1)
                maximum = max(maximum, relative_error)
                assert relative_error < 1e-14
                assert max(abs(x-y) for x, y in zip(
                    adapter.weights_slots(b, momenta), weights[::-1])) < 1e-14
            rows.append(dict(
                point=point["point_id"], channel=channel,
                momenta_geometry=list(momenta), sectors_geometry=list(sectors),
                q_geometry=list(map(pair, q)), weights_geometry=list(weights),
                weights_human_slots=list(weights[::-1]),
                log_q_geometry=list(map(pair, logs)), primary=pair(primary),
                log_primary=pair(log_primary), norm=abs(primary),
                phase_radians=cmath.phase(primary),
                branch="principal pointwise, matching existing adapter",
            ))
    # This is a SAME-q bookkeeping identity, not a modular/channel map.
    q = tuple(complex(z) for z in config["points"][0]["source"]["q_values"])
    rr_vs_nn = cmath.exp((cmath.log(q[0])+cmath.log(q[1]))/16)
    return dict(rows=rows, maximum_adapter_relative_error=maximum,
                scope="Each chart evaluated separately. The same test momenta in two charts do not identify their integration variables.",
                same_q_RRNS_over_NSNSNS_primary=pair(rr_vs_nn))


def branching_primary_ledger(b=1.4):
    momenta = (.21, .37, .52)  # Human slot order.
    weights = BranchWeights(b, tuple(1j*p for p in momenta))
    background = b+1/b
    rows = []
    for leg, sector in enumerate(("NS", "R", "R")):
        h_aux = Fraction(0) if sector == "NS" else Fraction(1, 16)
        h_sca = background**2/8+momenta[leg]**2/2+float(h_aux)
        labels = (Fraction(0), Fraction(1, 2), Fraction(-1, 2), Fraction(1)) if sector == "NS" else (
            Fraction(1, 4), Fraction(-1, 4), Fraction(3, 4), Fraction(-3, 4), Fraction(5, 4))
        for n in labels:
            copies = tuple(complex(weights.weight(leg, n, copy)) for copy in (0, 1))
            relative = 2*n*n-(Fraction(1, 8) if sector == "R" else 0)
            residual = abs(sum(copies)-h_sca-float(h_aux)-float(relative))
            assert residual < 2e-13, (sector, n, residual)
            rows.append(dict(
                slot=leg+1, sector=sector, branching_label=str(n),
                h_sca=h_sca, h_auxiliary=str(h_aux),
                virasoro_weights=list(map(pair, copies)),
                relative_q_power=str(relative), residual=residual,
            ))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"Data Set/nsrr_coefficient_primary_ledger_20260915")
    args = parser.parse_args()
    config_path = ROOT/"Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json"
    config = json.loads(config_path.read_text())
    vertex, vertex_rows = ordered_vertex_ledger()
    legacy_rows = legacy_matrix_ledger()
    low_levels = low_level_blocks()
    primary = primary_ledger(config)
    branching = branching_primary_ledger(config["b"])
    inputs = (Path(__file__), ROOT/"Human Notes/SCblock.tex", config_path,
              ROOT/"Code/genus_2/nsrr_plumbing_adapter.py",
              ROOT/"Code/genus_2/nsrr_resummed_sewing.py",
              ROOT/"Code/genus_2/physical_nsrr_sewing.py",
              ROOT/"Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py",
              ROOT/"Code/ramond_branching_recursion/compute_target.py")
    output = dict(
        schema="nsrr-coefficient-primary-ledger-v1",
        block_definition="F contains descendant powers only; P_A=exp(sum_i h_i,A Log(q_i)) is outside F and outside M",
        full_decomposition="sum_AB M_AB P_A Ptilde_B F_A Ftilde_B",
        vertex=vertex, low_level_blocks=low_levels, primary=primary,
        branching=branching,
        checks=dict(ordered_vertex_entries=len(vertex_rows),
                    exact_Ward_coefficients=len(low_levels),
                    chart_primary_records=len(primary["rows"]),
                    branching_weight_checks=len(branching)),
        physical_genus_two_M_status="Not certified: requires remaining restricted BPZ, ordered-slot, and fixed-spin contraction in the saved block basis.",
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"ledger.json").write_text(json.dumps(output, indent=2)+"\n")
    write_csv(args.output/"ordered_vertex_coefficients.csv", vertex_rows)
    write_csv(args.output/"legacy_candidate_M.csv", legacy_rows)
    print(json.dumps(output["checks"]))
    print(f"Maximum primary adapter relative error: {primary['maximum_adapter_relative_error']:.3e}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
