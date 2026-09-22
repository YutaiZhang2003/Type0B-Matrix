#!/usr/bin/env python3
"""Check the NSRR coefficient normalization and both all-NS sewing sectors.

The identity limit is evaluated in the original two-puncture plumbing
coordinates, with the torus multiplier derived from their Mobius maps.
Primary powers stay outside the blocks. No production kernel is changed.
"""
from __future__ import annotations

import argparse
import cmath
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ("Code", "Code/genus_2", "Code/c_Recursion",
                 "Code/genus_2_cross_channel", "Code/double_virasoro/nsrr"):
    sys.path.insert(0, str(ROOT/relative))

import mpmath as mp
import numpy as np
import sympy as s
from all_ns_reflected_sewing import (
    LIFTS, coefficient_matrix, lift_conversion, primary_from_weights,
    quadratic_parity, reflected_blocks,
)
from ns_genus2_symbolic_low_order import (
    ExactDirectThetaOracle, ExactNSDescendantThreeForm,
)
from nsrr_genus2_block import HumanNSRRThetaOracle
from nsrr_reflected_state_sewing import ReflectedNSRRStateSewing
from theta_partition import theta_sector_pair

OUTPUT = ROOT/"Data Set/ns_sewing_identity_limits_20260915"


def encode(z):
    z = complex(z)
    return [z.real, z.imag]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def all_ns_algebra():
    bits = tuple(itertools.product((0, 1), repeat=3))
    count = 0
    for n, m, p, pa in itertools.product(bits, repeat=4):
        x = tuple(a ^ b for a, b in zip(n, p))
        y = tuple(a ^ b for a, b in zip(m, pa))
        if sum(x) % 2 != sum(y) % 2:
            continue
        pair = theta_sector_pair(sum(n) % 2, holomorphic_primary_parities=p,
                                 antiholomorphic_primary_parities=pa)
        assert pair.antiholomorphic_sector == sum(m) % 2
        original = (quadratic_parity(tuple(a ^ b for a, b in zip(x, y)))
                    + sum(a*b for a, b in zip(x, y))) % 2
        decomposed = (quadratic_parity(x)+quadratic_parity(y)+pair.absolute_parity) % 2
        assert original == decomposed
        count += 1

    matrices = {}
    transform_checks = 0
    for a in (0, 1):
        u = s.Matrix(lift_conversion(a)).applyfunc(lambda z: s.Rational(str(z)))
        assert u*u == s.eye(4)
        for parity in bits:
            if sum(parity) % 2 != a:
                continue
            character = s.Matrix([l[0]**parity[0]*l[1]**parity[1] for l in LIFTS])
            assert u*((-1)**quadratic_parity(parity)*character) == character
            transform_checks += 1
        matrices[a] = [[str(z) for z in row] for row in u.tolist()]

    h = s.symbols("h_infinity h_one h_zero", positive=True)
    ward = ExactNSDescendantThreeForm(c=s.Rational(81, 5), weights=h)
    seeds = []
    for parity in bits:
        states = tuple((("G", -1),) if p else () for p in parity)
        rho = s.factor(ward.value(*states))
        norm = s.prod(2*w for w, p in zip(h, parity) if p)
        reflected = s.factor(rho**2/norm)
        literal = (-1)**quadratic_parity(parity)*reflected
        seeds.append(dict(parities_human_slots=parity,sector=sum(parity) % 2,
                          rho=str(rho),literal_coefficient=str(literal),
                          reflected_coefficient=str(reflected)))

    cl, cr = s.symbols("Ctilde_L Ctilde_R")
    assert s.expand(-(s.I*cl)*(s.I*cr)) == cl*cr
    return dict(exact_grading_checks=count,exact_lift_character_checks=transform_checks,
                U=matrices,seeds=seeds,
                odd_coefficient="-(i*Ctilde_L)*(i*Ctilde_R)=+Ctilde_L*Ctilde_R",
                result="The written all-NS grading and odd coefficient are internally consistent. A reflected/fixed-spin interpretation needs the lift-basis pairing as well.")


def torus_multiplier(q0, q1):
    """Log k continuous from small q0,q1; the principal log of k need not agree."""
    a = 1-q0-q1
    root = cmath.sqrt(a*a-4*q0*q1)
    if abs(root-a) > abs(-root-a):
        root = -root
    b = (a+root)/2
    logk = cmath.log(q0)+cmath.log(q1)-2*cmath.log(b)
    return cmath.exp(logk), logk, b


def identity_polynomial(h, sector, epsilon=1):
    """Exact torus character in sqrt(q0),sqrt(q1) through total level 3."""
    x, y, t = s.symbols("x y t")
    # B=(1-q0-q1+sqrt((1-q0-q1)^2-4*q0*q1))/2.
    delta = -(x*x+y*y)*t*t-x*x*y*y*t**4-x*x*y*y*(x*x+y*y)*t**6

    def power(exponent):
        return sum(s.binomial(exponent, n)*delta**n for n in range(4))

    if sector == "R":
        expression = s.sqrt(2)*(power(-2*h)+2*x*x*y*y*t**4*power(-2*h-2))
    elif sector == "NS":
        expression = (power(-2*h)+epsilon*x*y*t*t*power(-2*h-1)
                      +x*x*y*y*t**4*power(-2*h-2)
                      +2*epsilon*x**3*y**3*t**6*power(-2*h-3))
    else:
        raise ValueError(sector)
    expanded = s.Poly(s.expand(expression), t)
    truncated = sum(coefficient for (degree,), coefficient in expanded.terms() if degree <= 6)
    return s.Poly(s.expand(truncated), x, y)


def identity_limits():
    c = s.Rational(81, 5)
    beta = s.I*s.Rational(2, 5)
    hr = c/24-beta*beta
    hn = s.Rational(7, 10)
    expected_r = identity_polynomial(hr, "R")
    expected_ns = identity_polynomial(hn, "NS", epsilon=-1)
    rows = []
    rcoeff = {0: {}, 1: {}}
    for f in (0, 1):
        oracle = HumanNSRRThetaOracle(
            central_charge=c,h_ns=s.S.Zero,beta_r1=beta,beta_r2=beta,
            form_parity=f,primary_parity=0,etas=(1, 1))
        for total in range(4):
            for r1 in range(total+1):
                r0 = total-r1
                vector = oracle.coefficient_components(0, r1, r0)
                value = math.sqrt(2)*sum(z for parity, z in enumerate(vector) if not parity & 2)
                rcoeff[f][r0, r1] = value
                expected = complex(expected_r.coeff_monomial((2*r0, 2*r1)))*(-1j)**f
                error = abs(value-expected)/max(1, abs(value), abs(expected))
                assert error < 3e-12, (f, r0, r1, value, expected)
                rows.append(dict(sector="R",form_parity=f,twice_level_zero=2*r0,
                    twice_level_one=2*r1,block_real=value.real,block_imag=value.imag,
                    character_coefficient=str(expected_r.coeff_monomial((2*r0, 2*r1))),
                    expected_phase=encode((-1j)**f),scaled_error=error))
        print(f"Ramond identity limit: f={f}, all coefficients through total level 3 pass.",flush=True)

    # The Ward oracle uses inverse Gram matrices even in intermediate mode
    # reductions. Keep the identity weight symbolic until those cancellations
    # are complete; substituting zero first would invert a null Verma matrix.
    h_identity = s.Symbol("h_identity", positive=True)
    nsoracle = ExactDirectThetaOracle(c=c, weights=(h_identity, hn, hn))
    nscoeff = {}
    for total in range(0, 7, 2):
        for n1 in range(total+1):
            n0 = total-n1
            value = s.cancel(nsoracle.coefficient((0, n1, n0))).subs(h_identity, 0)
            expected = expected_ns.coeff_monomial((n0, n1))
            assert s.simplify(value-expected) == 0, (n0, n1, value, expected)
            nscoeff[n0, n1] = complex(value)
            rows.append(dict(sector="NS",form_parity=0,twice_level_zero=n0,
                twice_level_one=n1,block_real=float(value),block_imag=0.,
                character_coefficient=str(expected),expected_phase=encode(1),scaled_error=0.))

    numerical = []
    original = (.016*cmath.exp(.37j), .021*cmath.exp(-.22j))
    for scale in (.02, .1, 1.):
        q0, q1 = (scale*q for q in original)
        k, logk, b = torus_multiplier(q0, q1)
        for sector in ("R", "NS+", "NS-"):
            h = float(hr if sector == "R" else hn)
            primary = primary_from_weights((q0, q1, 1), (h, h, 0))
            if sector == "R":
                f0 = sum(z*q0**n0*q1**n1 for (n0, n1), z in rcoeff[0].items())
                f1 = sum(z*q0**n0*q1**n1 for (n0, n1), z in rcoeff[1].items())
                descendant = f0
                oscillator = math.prod((1+k**n)/(1-k**n) for n in range(1, 65))
                expected = math.sqrt(2)*cmath.exp(h*logk)*oscillator
                assert abs(f1+1j*f0)/max(1,abs(f0)) < 3e-13
                new_z = abs(primary*f0)**2  # E=2, O=0: E^2/4=1.
                old_local = abs(primary*(f0+1j*f1))**2/4
                assert abs(old_local/new_z-1) < 2e-13
            else:
                sign = 1 if sector == "NS+" else -1
                # Removing K changes the raw NS trace's sign to +lift0*lift1.
                lift = (1, sign, 1)
                literal = {l: sum(z*q0**(n0/2)*q1**(n1/2)*l[0]**(n0 % 2)*l[1]**(n1 % 2)
                                  for (n0, n1), z in nscoeff.items()) for l in LIFTS}
                descendant = reflected_blocks(literal, 0)[lift]
                oscillator = math.prod(
                    (1+sign*cmath.exp((n-.5)*logk))/(1-k**n) for n in range(1, 65))
                expected = cmath.exp(h*logk)*oscillator
            actual = primary*descendant
            ratio = actual/expected
            tolerance = 5e-11 if scale == .02 else (1e-8 if scale == .1 else 5e-5)
            assert abs(ratio-1) < tolerance, (sector, scale, ratio)
            numerical.append(dict(sector=sector,q_scale=scale,total_level=3,
                q_zero=encode(q0),q_one=encode(q1),torus_k=encode(k),log_k=encode(logk),
                primary=encode(primary),descendant_block=encode(descendant),
                sewn_chiral=encode(actual),torus_character_qL0=encode(expected),
                norm_ratio=abs(ratio),phase_difference_rad=cmath.phase(ratio),
                complex_relative_error=abs(ratio-1)))

    physical = ReflectedNSRRStateSewing(c=c,beta2=beta,beta3=beta)
    counts = []
    for n, m in itertools.product(range(3), repeat=2):
        basis, inverse = physical.ramond_basis(0, n, m)
        parity = np.diag([(-1)**(sum(k=="G" for k, _ in w)+sum(k=="G" for k, _ in wa)+e)
                          for w, wa, e in basis])
        assert np.max(abs(parity@inverse-inverse@parity)) < 2e-12
        assert np.trace(parity) == 0
        # Per-ground-family oscillator multiplicities from the NS/R PBW words.
        dn = len({w for w, _, _ in basis})
        dm = len({wa for _, wa, _ in basis})
        assert len(basis) == 2*dn*dm
        counts.append(dict(R_level=n,anti_R_level=m,physical_dimension=len(basis),
                           expected_dimension=2*dn*dm,parity_trace=int(np.trace(parity))))
    return dict(c=str(c),h_R=str(hr),h_NS=str(hn),coefficients=rows,numerical=numerical,
                physical_R_trace_counts=counts,
                normalization="d_+=d_-=1 -> E=2,O=0 -> E_L E_R/4=1; Fhat_R=sqrt(2)*character_per_ground_family.",
                limiting_R_ground_degeneracy=2,
                degeneracy_if_the_quarter_is_omitted=8,
                old_local_also_passes=True,
                scope="NS identity inserted on the pinched edge; null descendants are removed by retaining that edge at level zero. This is a module sewing/identity-insertion test, not an exchange of limits with the noncompact Liouville momentum integral.")


def majorana_torus():
    """Independent theta-series and one-Majorana Fock-product comparison."""
    rows = []
    with mp.workdps(60):
        def eta(tau):
            q = mp.exp(2j*mp.pi*tau)
            return mp.exp(1j*mp.pi*tau/12)*mp.fprod(1-q**n for n in range(1, 121))

        def theta(tau, alpha, beta):
            return mp.fsum(mp.exp(mp.pi*1j*tau*(n+alpha)**2+2j*mp.pi*(n+alpha)*beta)
                           for n in range(-24, 25))

        def bosonized(tau, alpha, beta):
            start = 1j*tau.imag
            root = mp.sqrt(theta(start, alpha, beta)/eta(start))
            for j in range(1, 97):
                point = start+tau.real*j/96
                candidate = mp.sqrt(theta(point, alpha, beta)/eta(point))
                root = candidate if abs(candidate-root) <= abs(-candidate-root) else -candidate
            return root

        points = [mp.mpc(".17", ".83"),mp.mpc("1.17", ".83"),mp.mpc("2.17", ".83"),
                  mp.mpc("-.23", "1.1"),mp.mpc(".42", ".57"),
                  mp.mpc("12.17", ".83"),mp.mpc("24.17", ".83")]
        values = {}
        for name, alpha, beta in (("NS+",mp.mpf(0),mp.mpf(0)),
                                  ("NS-",mp.mpf(0),mp.mpf(".5")),
                                  ("R+",mp.mpf(".5"),mp.mpf(0))):
            for tau in points:
                logq = 2j*mp.pi*tau
                if name == "R+":
                    fock = mp.sqrt(2)*mp.exp(logq/24)*mp.fprod(
                        1+mp.exp(n*logq) for n in range(1, 121))
                else:
                    sign = 1 if name == "NS+" else -1
                    fock = mp.exp(-logq/48)*mp.fprod(
                        1+sign*mp.exp((mp.mpf(n)-mp.mpf(".5"))*logq) for n in range(1, 121))
                reference = bosonized(tau, alpha, beta)
                ratio = fock/reference
                assert abs(ratio-1) < mp.mpf("1e-48"), (name, tau, ratio)
                values[name,str(tau)] = reference
                rows.append(dict(sector=name,tau=str(tau),norm_ratio=mp.nstr(abs(ratio),55),
                    phase_difference_rad=mp.nstr(mp.arg(ratio),12),
                    complex_relative_error=mp.nstr(abs(ratio-1),12)))
        r_multiplier = values["R+",str(points[1])]/values["R+",str(points[0])]
        ns_multiplier = values["NS+",str(points[2])]/values["NS+",str(points[0])]
        assert abs(r_multiplier-mp.exp(mp.pi*1j/12)) < mp.mpf("1e-48")
        assert abs(ns_multiplier-mp.exp(-mp.pi*1j/12)) < mp.mpf("1e-48")
        r_winding = values["R+",str(points[5])]/values["R+",str(points[0])]
        ns_winding = values["NS+",str(points[6])]/values["NS+",str(points[0])]
        assert abs(r_winding+1) < mp.mpf("1e-48")
        assert abs(ns_winding+1) < mp.mpf("1e-48")
        assert abs(r_winding**2-1) < mp.mpf("1e-48")
        assert abs(ns_winding**2-1) < mp.mpf("1e-48")
    return dict(rows=rows,ramond_T_multiplier="exp(i*pi/12)",
                NS_T_squared_multiplier="exp(-i*pi/12)",
                ramond_T_to_12_multiplier="-1; squared amplitude returns to itself",
                NS_T_to_24_multiplier="-1; squared amplitude returns to itself",
                R_with_parity_insertion="0, from the two opposite-parity ground families",
                primary_convention="These standard theta/eta characters include q^(-c/24). Convert the q^L0 sewing answer by that external factor; it is not part of the descendant block.",
                branch="Continue the square root from positive imaginary tau, preserving log(q)=2*pi*i*tau.")


def all_ns_free_control():
    from fixed_spin_free_plumbing import charged_frame, fixed_spin_chiral_partition
    from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
    path = ROOT/"Data Set/fixed_spin_free_NSrr_20260830/summary.json"
    saved = json.loads(path.read_text())
    point = next(p for p in saved["points"] if p["t"] == .6)["target_NSnsns"]
    q = tuple(map(complex, point["q_values"]))
    frame = charged_frame(q,max_mode=32)
    literal = {lift:theta_physical_fermion_fredholm(q,lift,max_mode=32).chiral_value
               for lift in LIFTS}
    converted = reflected_blocks(literal,0)
    rows = []
    for lift in LIFTS:
        spin = ((0,0),(int(lift[0]<0),int(lift[1]<0)))
        reference = fixed_spin_chiral_partition(q,frame.omega_charge,spin,
            period_branch=np.zeros((2,2),dtype=int),max_mode=32)["majorana_chiral"]
        ratio = converted[lift]/reference
        old = literal[lift]/reference
        assert abs(ratio-1) < 2e-11
        rows.append(dict(lifts_geometry=lift,characteristic_charge=spin,
            literal_HN_chiral=encode(literal[lift]),converted_chiral=encode(converted[lift]),
            bosonized_chiral=encode(reference),norm_ratio=abs(ratio),
            phase_difference_rad=cmath.phase(ratio),complex_relative_error=abs(ratio-1),
            literal_norm_ratio=abs(old),literal_phase_difference_rad=cmath.phase(old)))
    return dict(t=.6,q_geometry=list(map(encode,q)),rows=rows,
                scope="Fresh all-four-spin single-Majorana control in the charge marking. The free vacuum has only even three-form parity; the odd NS vertex coefficient is checked separately by the exact grading and Ward data.",
                saved_target_fixed_spin_certified=False)


def write_report(output, result):
    identity = result["identity_limits"]
    majorana = result["majorana_torus"]
    coefficient_error = max(r["scaled_error"] for r in identity["coefficients"])
    torus_error = max(float(r["complex_relative_error"]) for r in majorana["rows"])
    lines = [
        "# NSRR normalization, genus-one limits, and all-NS signs",
        "", f"Completed: {result['completed_at_utc']}", "",
        "## Result", "",
        r"The coefficient \(1/4\) passes the identity-module genus-one test in the saved two-lift block normalization. The single-Majorana torus comparison passes in both norm and continued phase. The all-NS relative three-point sign is consistent, but a literal single Human-Note lift still fails the genus-two fixed-spin free-field comparison.",
        "", "## Why the coefficient is one quarter", "",
        r"The supplied amplitudes are \(d_\pm=(E\pm O)/2\). Thus \(T=(E I+O\sigma_z)/2\), and the two oriented vertices give \(\operatorname{Tr}(T_LT_R)=(E_LE_R+O_LO_R)/2\). Each saved projected block has leading value \(\sqrt2\), so its coefficient must be \(E_LE_R/4\) or \(O_LO_R/4\). With unit-leading blocks the same coefficients would instead be \(E_LE_R/2\) and \(O_LO_R/2\).",
        "",
        r"For the identity, \(d_+=d_-=1\), hence \(E=2,O=0\). In the original plumbing coordinates, \(k=q_0q_1/B^2\), where \(B=(1-q_0-q_1+\sqrt{(1-q_0-q_1)^2-4q_0q_1})/2\to1\). The identity limit gives",
        "", r"\[",
        r"P_2\widehat F_0=\sqrt2\,k^{h_R}\mathcal C_R(k),\qquad",
        r"Z_R^{(1)}=\frac{2\cdot2}{4}|P_2\widehat F_0|^2=2|k^{h_R}\mathcal C_R(k)|^2,",
        r"\qquad\mathcal C_R(k)=\prod_{n\ge1}\frac{1+k^n}{1-k^n}.",
        r"\]", "",
        r"The two nonchiral Ramond ground states are counted once each, in the unprojected fixed-spin theory. Omitting the quarter would count eight. All primary powers remain outside the blocks: \(P_2=(q_0q_1)^{h_R}\) before reducing to the torus coordinate. Standard cylinder characters further require the external factor \(k^{-c/24}\).",
        "",
        "This is a normalized identity-module sewing test. It is not a limit interchanged with the noncompact Liouville momentum integral. The old local matrix before its extra factor four also passes this limit, because its distinguishing NS half-level contribution vanishes for the identity. The test therefore fixes this limiting normalization without proving the generic genus-two matrix.",
        "", "## Direct descendant and torus trace checks", "",
        f"- {len(identity['coefficients'])} coefficient comparisons through total plumbing level three; maximum scaled Ramond error {coefficient_error:.4e}. The NS comparisons are exact symbolic equalities.",
        f"- {len(identity['physical_R_trace_counts'])} physical Ramond trace counts, with independent left/right oscillator levels zero through two. Every count is twice the product of the oscillator multiplicities; every total-parity trace is zero.",
        "",
        r"The complex comparison uses \(q_0=s\,0.016e^{0.37i}\), \(q_1=s\,0.021e^{-0.22i}\), with the exact multiplier and continued \(\Log k=\Log q_0+\Log q_1-2\Log B\). The sewn series is truncated at total level three; the torus products retain 64 factors.",
        "",
        "| Sector | Scale s | Sewn/reference norm | Phase difference (rad) | Complex relative error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in identity["numerical"]:
        lines.append(f"| {row['sector']} | {row['q_scale']:g} | {row['norm_ratio']:.15f} | {row['phase_difference_rad']:.4e} | {row['complex_relative_error']:.4e} |")
    lines += [
        "", "The errors decrease consistently with the omitted total-level-four terms.",
        "", "## Single-Majorana phase", "",
        f"Independent lattice theta sums and fermion Fock products agree in {len(majorana['rows'])} complex comparisons at 60 decimal digits; maximum relative error {torus_error:.4e}. These cover seven moduli and all three nonzero torus spin sectors. The square roots are continued from positive imaginary tau.",
        "", r"\[",
        r"\chi_{R,+}(\tau+1)=e^{i\pi/12}\chi_{R,+}(\tau),\qquad",
        r"\chi_{NS,+}(\tau+2)=e^{-i\pi/12}\chi_{NS,+}(\tau).",
        r"\]", "",
        r"The longer continuations \(\tau\to\tau+12\) in R and \(\tau\to\tau+24\) in NS both multiply the single-Majorana amplitude by \(-1\), while the squared amplitude returns to itself. Both sign changes are explicitly checked. A square root chosen independently at the endpoint cannot retain this path information.",
        "", "## All-NS sectors", "",
        f"The audit passes {result['all_NS']['exact_grading_checks']} exact grading checks, {result['all_NS']['exact_lift_character_checks']} exact lift-character checks, and the involution checks for both lift matrices. It also records all eight lowest Ward seeds with symbolic weights.",
        "",
        r"The odd coefficient is \(-(i\widetilde C_L)(i\widetilde C_R)=+\widetilde C_L\widetilde C_R\). In the literal Human-Note block basis, local radial reflection requires a coherent four-lift pairing \(H_a=U_aF_a\), with different matrices for the even and odd three-form sectors. The coefficient matrix is \(M_{\sigma\tau}^{(a,\ell)}=B_{L,a}B_{R,a}U_{a,\ell\sigma}U_{a,\ell\tau}\); primary propagation is external.",
        "",
        "Fresh single-Majorana genus-two comparison at the saved t=0.60 target geometry, in the charge marking. Each literal and converted value is compared to the same spin characteristic in that row:",
        "",
        "| Lift (0,1,infinity) | Literal norm ratio | Literal phase (rad) | Converted norm ratio | Converted phase (rad) | Converted complex error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["all_NS_free"]["rows"]:
        label = "".join("+" if x > 0 else "-" for x in row["lifts_geometry"])
        lines.append(f"| {label} | {row['literal_norm_ratio']:.10f} | {row['literal_phase_difference_rad']:.10f} | {row['norm_ratio']:.15f} | {row['phase_difference_rad']:.3e} | {row['complex_relative_error']:.3e} |")
    lines += [
        "",
        "The free vacuum checks the even three-form sector. It has no odd coupling; the odd-sector evidence comes from the exact grading, Ward forms, and lift transform. This local conversion does not certify the full interacting NSRR-to-all-NS spin transport. The historical integrated all-NS target has not been re-integrated here.",
        "", "## Files and reproduction", "",
    ]
    links = [
        ("Full derivation and literature conventions", ROOT/"Machine Notes/Genus 2/NS_SEWING_IDENTITY_LIMITS_2026-09-15.md"),
        ("Audit script", Path(__file__).resolve()),
        ("All-NS reflection adapter", ROOT/"Code/genus_2/all_ns_reflected_sewing.py"),
    ]
    links += [(name, output/name) for name in (
        "summary.json", "identity_coefficients.csv", "identity_complex_comparison.csv",
        "majorana_torus.csv", "all_NS_free.csv", "all_NS_reflection_matrices.csv")]
    lines += [f"- [{label}](<{path}>)" for label, path in links]
    lines += [
        "", "Run from the repository root with the existing diagnostic environment:", "",
        "```sh",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\",
        "  /private/tmp/type0b-nsrr-smoke-venv/bin/python \\",
        "  Code/genus_2/audit_ns_sewing_identity_limits.py",
        "```", "",
        "The summary records the source and input SHA-256 hashes. This audit leaves production kernels and historical integrated data unchanged.",
        "",
    ]
    (output/"README.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    result=dict(schema="ns-sewing-identity-limits-v1",all_NS=all_ns_algebra())
    result["identity_limits"]=identity_limits()
    result["majorana_torus"]=majorana_torus()
    result["all_NS_free"]=all_ns_free_control()
    result["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
    result["production_kernels_modified"]=False
    result["genus_two_global_spin_transport_certified"]=False
    files=(Path(__file__),ROOT/"Code/genus_2/all_ns_reflected_sewing.py",
           ROOT/"Code/genus_2/nsrr_reflected_state_sewing.py",
           ROOT/"Code/genus_2/theta_partition.py",
           ROOT/"Code/c_Recursion/ns_genus2_symbolic_low_order.py",
           ROOT/"Code/c_Recursion/ns_human_convention.py",
           ROOT/"Code/double_virasoro/nsrr/nsrr_genus2_block.py",
           ROOT/"Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py",
           ROOT/"Code/genus_2/fixed_spin_free_plumbing.py",
           ROOT/"Code/genus_2/physical_free_plumbing_resummation.py",
           ROOT/"Data Set/fixed_spin_free_NSrr_20260830/summary.json",
           ROOT/"Human Notes/SCblock.tex")
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    result["sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    write_csv(output/"identity_coefficients.csv",result["identity_limits"]["coefficients"])
    write_csv(output/"identity_complex_comparison.csv",result["identity_limits"]["numerical"])
    write_csv(output/"majorana_torus.csv",result["majorana_torus"]["rows"])
    write_csv(output/"all_NS_free.csv",result["all_NS_free"]["rows"])
    matrices=[]
    for lift,a in itertools.product(LIFTS,(0,1)):
        for i,row in enumerate(coefficient_matrix(a,1,1,lift)):
            for j,value in enumerate(row):
                matrices.append(dict(target_lift=lift,sector=a,left_lift=LIFTS[i],
                    right_lift=LIFTS[j],coefficient=value.real,
                    constant_product="C_L*C_R" if a==0 else "Ctilde_L*Ctilde_R",
                    includes_primary=False))
    write_csv(output/"all_NS_reflection_matrices.csv",matrices)
    write_report(output, result)
    print(json.dumps(dict(output=str(output),
        grading_checks=result["all_NS"]["exact_grading_checks"],
        identity_coefficients=len(result["identity_limits"]["coefficients"]),
        maximum_identity_coefficient_error=max(r["scaled_error"] for r in result["identity_limits"]["coefficients"]),
        identity_comparison=result["identity_limits"]["numerical"],
        maximum_majorana_torus_error=max(float(r["complex_relative_error"]) for r in result["majorana_torus"]["rows"]),
        all_NS_free=result["all_NS_free"]["rows"]),indent=2))


if __name__=="__main__":
    main()
