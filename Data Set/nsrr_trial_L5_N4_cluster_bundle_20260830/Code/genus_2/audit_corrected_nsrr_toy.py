#!/usr/bin/env python3
"""Independent PBW validation and reduction of the corrected NSRR norm toy.

PBW is used here for validation only, never to fill the production data.
The audit cannot turn the diagonal norm into a physical partition function.
"""
import argparse
from html import escape
import math
from pathlib import Path

import numpy as np
import sympy as sp

import run_corrected_nsrr_toy as toy
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples
from theta_star_algebra import fwht


def audit(run_dir):
    result = toy.scan._load(run_dir/"summary.json")
    config = result["config"]
    toy.validate_config(config)
    shards = [toy.scan._load(run_dir/"shards"/f"node-{i:03d}.json") for i in range(35)]
    matrix_error, amplitude_error = 0., 0.
    for i, shard in enumerate(shards):
        toy.validate_shard(config, i, shard)
        for row in shard["values"]:
            for a, (f, eta) in enumerate(toy.CHANNELS):
                expected = toy.decode(shard["C_eta"][0 if eta == 1 else 1])*toy.decode(row["primary"])*toy.decode(row["blocks"][a])
                amplitude_error = max(amplitude_error, abs(expected-toy.decode(row["amplitudes"][a]))/max(abs(expected), 1e-280))
    for row in result["rows"]:
        selected = [(s, next(r for r in s["values"] if r["t"] == row["t"]
                            and r["level"] == row["level"] and r["lifts_geometry"] == row["lifts_geometry"]))
                    for s in shards if s["quadrature_order"] == row["quadrature_order"]]
        expected = np.zeros((4, 4), dtype=complex)
        for a in range(4):
            for b in range(4):
                terms = [s["measure"]*toy.decode(r["amplitudes"][a])*toy.decode(r["amplitudes"][b]).conjugate()
                         for s, r in selected]
                expected[a, b] = complex(math.fsum(z.real for z in terms), math.fsum(z.imag for z in terms))
        actual = np.array([[toy.decode(z) for z in r] for r in row["H"]])
        matrix_error = max(matrix_error, float(np.max(abs(expected-actual)))/row["D_diagnostic"])
        if row["physical_Z"] is not None or row["physical_Q"] is not None:
            raise ValueError("toy mislabelled as a physical partition")
    if amplitude_error > 1e-13 or matrix_error > 1e-13:
        raise ArithmeticError("independent reduction failed")
    pbw_rows = []
    bg = sp.Rational(7, 5)+sp.Rational(5, 7)
    for index in (0, 7, 8, 21, 34):
        shard = shards[index]
        p = tuple(sp.Rational(str(x)) for x in shard["momenta_slots"])
        max_error, checks = 0., 0
        for channel, (f, eta) in enumerate(toy.CHANNELS):
            oracle = HumanNSRRThetaOracle(
                central_charge=sp.Rational(3, 2)+3*bg**2,
                h_ns=bg**2/8+p[0]**2/2, beta_r1=sp.I*p[1]/sp.sqrt(2),
                beta_r2=sp.I*p[2]/sp.sqrt(2), form_parity=f, primary_parity=0, etas=(eta, eta))
            vectors = {e: oracle.coefficient_components(e[0], e[1]//2, e[2]//2) for e in level_triples(4)}
            for row in shard["values"]:
                point = next(p for p in config["points"] if p["t"] == row["t"])
                plumbing = toy.NSRRPlumbingInputs(tuple(complex(q) for q in point["q_geometry"]),
                                                  tuple(row["lifts_geometry"]), toy.GEOMETRY_SECTORS)
                k = toy.dv.spin_character_index(plumbing.lifts_slots)
                expected = toy.dv.evaluate_twice_level_series(
                    {e: fwht(v)[k] for e, v in vectors.items() if sum(e) <= 2*row["level"]}, plumbing.q_slots)
                actual = toy.decode(row["blocks"][channel])
                max_error = max(max_error, abs(actual-expected)/max(1., abs(expected)))
                checks += 1
        if max_error > 1e-9:
            raise ArithmeticError(f"independent PBW audit failed at node {index}")
        pbw_rows.append({"node": index, "quadrature_order": shard["quadrature_order"],
                         "block_evaluations_checked": checks, "maximum_scaled_block_error": max_error})
        print(f"validation-only PBW node={index}: {checks} blocks, scaled error {max_error:.3e}", flush=True)
    diagnostics = []
    for point in config["points"]:
        def select(n, level, lifts=(1, 1, 1)):
            return next(r for r in result["rows"] if r["t"] == point["t"] and r["quadrature_order"] == n
                        and r["level"] == level and r["lifts_geometry"] == list(lifts))
        fine, lowlevel, coarse = select(3, 2), select(3, 1), select(2, 2)
        diagnostics.append({"t": point["t"], "D_N3_L2": fine["D_diagnostic"],
                            "level_1_to_2_relative_change": fine["D_diagnostic"]/lowlevel["D_diagnostic"]-1,
                            "quadrature_2_to_3_relative_change": fine["D_diagnostic"]/coarse["D_diagnostic"]-1,
                            "maximum_lift_trace_relative_difference": max(abs(select(3, 2, lifts)["D_diagnostic"]/fine["D_diagnostic"]-1) for lifts in toy.LIFTS)})
    report = {"schema": "corrected-nsrr-diagonal-toy-audit-v1", "summary_digest": toy.scan._digest(result),
              "protected_kernel_hashes": toy.protected_hashes(), "PBW_production_calls": result["PBW_production_calls"],
              "maximum_independent_matrix_reduction_error_scaled_by_trace": matrix_error,
              "maximum_amplitude_reconstruction_relative_error": amplitude_error,
              "independent_PBW_validation": pbw_rows, "convergence_diagnostics": diagnostics,
              "physical_Z": None, "physical_Q": None,
              "interpretation": "Only the corrected equal-HJS chiral diagonal norm is evaluated. No physical Ramond projector, spin choice, or free normalization is inferred."}
    toy.scan.write_json(run_dir/"verification.json", report)
    plot(result, run_dir/"nsrr_toy_overview.svg")
    return report


def plot(result, path):
    rows = result["rows"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="675" viewBox="0 0 1160 675">',
           '<rect width="1160" height="675" fill="white"/><g font-family="Arial,sans-serif" fill="#253044">']
    def label(x, y, value, size=15, anchor="start"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')
    label(70, 36, "Corrected NSRR toy — a diagonal sewing diagnostic, not physical Z or Q", 23)
    label(70, 65, "b=1.4; 35 momentum nodes; branching recursion x two Virasoro c-recursions; zero PBW production calls")
    curves = []
    for n, level, color in ((3, 0, "#888888"), (3, 1, "#d67a20"), (3, 2, "#176dad"), (2, 2, "#75459b")):
        curves.append((f"N={n}, L={level}", color, [r for r in rows if r["quadrature_order"] == n
                                                    and r["level"] == level and r["lifts_geometry"] == [1, 1, 1]]))
    left, top, width, height = 125, 153, 390, 330
    values = [r["D_diagnostic"] for _, _, curve in curves for r in curve]
    low, high = min(values), max(values)
    margin = .08*(high-low)
    low, high = low-margin, high+margin
    def xy(t, value):
        return left+width*(t-.52)/.16, top+height*(high-value)/(high-low)
    label(left, top-27, "D = weighted squared-block sum", 17)
    for j in range(5):
        y = low+(high-low)*j/4
        yy = xy(.52, y)[1]
        svg.append(f'<path d="M {left} {yy} h {width}" stroke="#e1e6ee"/>')
        label(left-9, yy+4, f"{y:.3e}", 12, "end")
    for t in (.52, .56, .60, .64, .68):
        label(xy(t, low)[0], top+height+24, f"{t:.2f}", 13, "middle")
    for i, (name, color, curve) in enumerate(curves):
        coords = " ".join(f"{xy(r['t'],r['D_diagnostic'])[0]:.3f},{xy(r['t'],r['D_diagnostic'])[1]:.3f}" for r in curve)
        dash = ' stroke-dasharray="6 4"' if i == 3 else ''
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>')
        x, y = left+(i%2)*200, top+height+64+(i//2)*25
        svg.append(f'<path d="M {x} {y-4} h 24" stroke="{color}" stroke-width="3"{dash}/>')
        label(x+30, y, name, 13)
    label(left+width/2, top+height+46, "t = Re Omega_original,12", 14, "middle")
    left, top, width, height = 700, 153, 370, 330
    selected = [next(r for r in rows if r["t"] == .6 and r["quadrature_order"] == 3
                     and r["level"] == 2 and r["lifts_geometry"] == list(lifts)) for lifts in toy.LIFTS]
    totals = [(sum(r["diagonal_channels"][:2]), sum(r["diagonal_channels"][2:])) for r in selected]
    ymax = max(max(pair) for pair in totals)*1.08
    label(left-25, top-27, "Parity channels exchange under lift changes", 17)
    for j in range(5):
        y = ymax*j/4
        yy = top+height*(1-y/ymax)
        svg.append(f'<path d="M {left} {yy} h {width}" stroke="#e1e6ee"/>')
        label(left-9, yy+4, f"{y:.2e}", 12, "end")
    for i, pair in enumerate(totals):
        center = left+width*(i+.5)/4
        for f, (value, color) in enumerate(zip(pair, ("#176dad", "#d67a20"))):
            h = height*value/ymax
            svg.append(f'<rect x="{center-26+f*27}" y="{top+height-h}" width="24" height="{h}" fill="{color}"/>')
        label(center, top+height+25, str(toy.LIFTS[i][:2]), 13, "middle")
    label(left+width/2, top+height+48, "(R0,R1) lifts; NSinf lift fixed +1", 14, "middle")
    label(left, top+height+75, "blue: f=0", 14)
    label(left+165, top+height+75, "orange: f=1", 14)
    label(left, top+height+99, "t=0.60, N=3, L=2; sums coincide", 14)
    label(70, 624, "D sums only eta_left=eta_right channels with a unit diagnostic metric. No Ramond multiplicity or free factor is assumed.", 14)
    label(70, 649, "Opposite-HJS blocks and the physical nonchiral contraction remain unresolved; this is not a modular-invariance comparison.", 14)
    svg.append('</g></svg>')
    Path(path).write_text("\n".join(svg)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    for row in report["convergence_diagnostics"]:
        print(row)
