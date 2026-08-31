#!/usr/bin/env python3
"""Audit and plot the refined NSRR ANSATZ against a saved all-NS diagnostic.

Both nonchiral numerators remain spin-uncertified. Physical same-plumbing
free factors are recomputed independently, without fitting a normalization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import math
from pathlib import Path

import numpy as np
import sympy as sp

import refine_nsrr_factorized_sign_trial as refine
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples

trial = refine.trial
TARGET_BRANCH = ((-1, -1), (-1, 0))


def extreme_momentum_probe(config, output):
    """Check the largest-grid endpoint, not just the moderate-momentum benchmark."""
    n = config["quadrature_orders"][-1]
    rules = trial._rules(config["q_envelope"], n)
    momenta = tuple(float(rule[0][-1]) for rule in rules)
    constants = [trial.GenericSuperLiouvilleConstants(config["b"], dps=dps).rr_ns_constants(
        momenta[1], momenta[0], momenta[2]) for dps in (30, 45)]
    constant_error = max(abs(complex(a)/complex(b)-1) for a, b in zip(*constants))
    blocks, checks = refine.block_components(config["b"], momenta[::-1], config["max_level"])
    b = sp.Rational(str(config["b"]))
    bg = b+1/b
    p = [sp.Rational(str(v)) for v in momenta[::-1]]
    pbw_error = 0.
    for f in (0, 1):
        for eta in (1, -1):
            oracle = HumanNSRRThetaOracle(
                central_charge=sp.Rational(3, 2)+3*bg**2,
                h_ns=bg**2/8+p[0]**2/2,
                beta_r1=sp.I*p[1]/sp.sqrt(2), beta_r2=sp.I*p[2]/sp.sqrt(2),
                form_parity=f, primary_parity=0, etas=(eta, eta))
            for e in level_triples(2*config["max_level"]):
                expected = oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                for actual, target in zip(blocks[f, eta, eta][e], expected):
                    pbw_error = max(pbw_error, abs(complex(actual)-complex(target))/max(1., abs(complex(target))))
    if constant_error > 1e-10 or pbw_error > 2e-8:
        raise ArithmeticError("high-momentum precision probe failed")
    result = {"quadrature_order": n, "momenta_geometry": momenta,
              "structure_precision_digits": [30, 45], "structure_constant_relative_error": constant_error,
              "equal_sign_DV_vs_independent_PBW_scaled_error": pbw_error, "checks": checks}
    trial.save(output/"extreme_momentum_probe.json", result)
    return result


def audit(output):
    result = trial.load(output/"summary.json")
    config = result["config"]
    refine.validate_config(config)
    if hashlib.sha256(Path(config["geometry_path"]).read_bytes()).hexdigest() != config["geometry_sha256"]:
        raise ValueError("saved geometry provenance changed")
    shards = [trial.load(output/"shards"/f"node-{i:03d}.json") for i in range(len(trial.tasks(config)))]
    term_error = primary_error = reduction_error = ground_error = 0.
    reconstructed = {}
    for i, shard in enumerate(shards):
        trial.validate_shard(config, i, shard)
        n, node = trial.tasks(config)[i]
        indices = np.unravel_index(node, (n,)*3)
        rules = trial._rules(config["q_envelope"], n)
        p = [float(rules[e][0][indices[e]]) for e in range(3)]
        if p != shard["momenta_geometry"] or trial._measure(rules, indices) != shard["measure"]:
            raise ValueError("quadrature data changed")
        c = [trial.decode(z)/2 for z in shard["C_BRY"]]
        bg = config["b"]+1/config["b"]
        weights = [bg**2/8+p[e]**2/2+(1/16 if e < 2 else 0) for e in range(3)]
        for row in shard["rows"]:
            q = next(pt["q_geometry"] for pt in config["points"] if pt["t"] == row["t"])
            primary2 = math.exp(sum(2*h*math.log(abs(complex(z))) for h, z in zip(weights, q)))
            primary_error = max(primary_error, abs(primary2/abs(trial.decode(row["primary"]))**2-1))
            terms = []
            for j, (_, eta, ep) in enumerate(trial.CHANNELS):
                block = trial.decode(row["blocks"][j])
                # Independent contraction after (-1)^f i^(2f)=1, retaining
                # products of coefficients rather than their absolute squares.
                value = primary2*c[0 if eta == 1 else 1]*c[0 if ep == 1 else 1]*abs(block)**2
                terms.append(value)
                term_error = max(term_error, abs(value-trial.decode(row["weighted_terms"][j]))/
                                 max(abs(value), abs(trial.decode(row["total"])), 1e-280))
            total = sum(terms)
            term_error = max(term_error, abs(total-trial.decode(row["total"]))/max(abs(total), 1e-280))
            sign_control = sum(terms[:4])-sum(terms[4:])
            term_error = max(term_error, abs(sign_control-trial.decode(row["without_sewing_sign"]))/
                             max(abs(total), 1e-280))
            if row["level"] == 0 and row["lifts_geometry"] == [1, 1, 1]:
                exact = primary2*4*(c[0]+c[1])**2
                ground_error = max(ground_error, abs(total-exact)/max(abs(exact), 1e-280))
            key = n, row["t"], row["level"], tuple(row["lifts_geometry"])
            reconstructed.setdefault(key, []).append(shard["measure"]*total)
    for row in result["rows"]:
        key = row["quadrature_order"], row["t"], row["level"], tuple(row["lifts_geometry"])
        terms = reconstructed[key]
        expected = complex(math.fsum(z.real for z in terms), math.fsum(z.imag for z in terms))
        reduction_error = max(reduction_error, abs(expected-row["total"])/max(abs(expected), 1e-280))
    if max(term_error, primary_error, reduction_error, ground_error) > 2e-11:
        raise ArithmeticError("independent refinement reconstruction failed")
    if result["physical_Z"] is not None or result["physical_Q"] is not None:
        raise ValueError("trial labelled physical")
    lift_spread = anti_error = 0.
    for n in config["quadrature_orders"]:
        for point in config["points"]:
            for level in config["levels"]:
                values = [r["total"] for r in result["rows"] if r["quadrature_order"] == n
                          and r["t"] == point["t"] and r["level"] == level]
                lift_spread = max(lift_spread, (max(values)-min(values))/max(map(abs, values)))
    for row in result["rows"]:
        anti_error = max(anti_error, abs(trial.decode(row["formal_same_convention_tilde"])-row["without_sewing_sign"])/
                         max(abs(row["total"]), 1e-280))
    report = {"schema": "nsrr-trial-refinement-audit-v1", "summary_digest": trial.digest(result),
              "audit_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "maximum_term_error": term_error, "maximum_primary_error": primary_error,
              "maximum_reduction_error": reduction_error, "maximum_ground_normalization_error": ground_error,
              "maximum_lift_relative_spread": lift_spread,
              "formal_tilde_vs_without_sign_control_error_scaled_by_trial": anti_error,
              "protected_kernel_sha256": trial.protected_hashes(), "physical_Z": None, "physical_Q": None}
    trial.save(output/"verification.json", report)
    return result, report


def comparison(result, all_ns_path, output):
    config = result["config"]
    all_ns = trial.load(all_ns_path)
    geometry = trial.load(config["geometry_path"])
    if hashlib.sha256(Path(config["geometry_path"]).read_bytes()).hexdigest() != config["geometry_sha256"]:
        raise ValueError("saved geometry provenance changed")
    previous = trial.load(Path(config["baseline_config_path"]).parent/"summary.json")
    if all_ns["config"]["parameters"]["b"] != config["b"] or \
            all_ns["config"]["parameters"]["include_cosmological_prefactor"]:
        raise ValueError("b or cosmological normalization mismatch")
    kappa = 1+2*(config["b"]+1/config["b"])**2
    rows = []
    for point, source_point in zip(geometry["points"], config["points"]):
        t = point["t"]
        target = point["target_chart"]
        all_ns_point = next(p for p in all_ns["config"]["points"] if p["t"] == t)
        if target["q_values"] != all_ns_point["q_values"] or target["omega"] != all_ns_point["omega"]:
            raise ValueError("saved all-NS numerator does not use the target geometry")
        factors = []
        for chart, spin, branch in ((point["source_chart"], ((1, 1), (0, 0)), trial.SOURCE_BRANCH),
                                    (target, ((0, 0), (0, 0)), TARGET_BRANCH)):
            q = tuple(complex(z) for z in chart["q_values"])
            omega = np.asarray([[complex(z) for z in row] for row in chart["omega"]])
            fine = trial.fixed_spin_partition(q, omega, spin, period_branch=branch, max_mode=40)
            coarse = trial.fixed_spin_partition(q, omega, spin, period_branch=branch, max_mode=32)
            if abs(fine["Z_free"]/coarse["Z_free"]-1) > 1e-11:
                raise ArithmeticError("free oscillator cutoff did not converge")
            factors.append(fine)
        if abs(factors[0]["Z_free"]/source_point["Z_free_reference"]-1) > 1e-12:
            raise ValueError("NSRR reference free normalization changed")
        z_all_ns = next(r["target_Z"] for r in all_ns["rows"] if r["t"] == t)
        q_all_ns = z_all_ns/factors[1]["Z_free"]**kappa
        old = next(r for r in previous["diagnostics"] if r["t"] == t)
        row = {"t": t, "allNS_saved_numerator": z_all_ns, "allNS_Q_diagnostic": q_all_ns,
               "source_free": factors[0]["Z_free"], "target_free": factors[1]["Z_free"],
               "baseline_Q_L2_N3": old["Q_trial_reference"],
               "baseline_relative_difference": old["Q_trial_reference"]/q_all_ns-1}
        for n in config["quadrature_orders"]:
            for level in (config["max_level"]-1, config["max_level"]):
                r = next(r for r in result["rows"] if r["t"] == t and r["quadrature_order"] == n
                         and r["level"] == level and r["lifts_geometry"] == [1, 1, 1])
                row[f"Q_L{level}_N{n}"] = r["Q_trial_reference"]
                row[f"relative_difference_L{level}_N{n}"] = r["Q_trial_reference"]/q_all_ns-1
        fine = next(d for d in result["diagnostics"] if d["t"] == t)
        row.update(refined_Z_trial=fine["Z_trial"], refined_Q_trial=fine["Q_trial_reference"],
                   refined_relative_difference=fine["Q_trial_reference"]/q_all_ns-1,
                   last_level_relative_change=fine["level_relative_change"],
                   last_quadrature_relative_change=fine["quadrature_relative_change"])
        rows.append(row)
    report = {"schema": "nsrr-refinement-allNS-diagnostic-comparison-v1", "kappa": kappa,
              "refined_summary_digest": trial.digest(result), "allNS_summary_path": str(all_ns_path.resolve()),
              "allNS_summary_digest": trial.digest(all_ns), "allNS_numerator_unchanged": True,
              "allNS_free_denominator_replaced": True, "fitted_normalization": None,
              "meaning": "Normalization-only diagnostic: neither numerator's fixed-spin identification is certified",
              "allNS_accuracy": {"quadrature_order": all_ns["config"]["quadrature_order"],
                                  "recursion_order_twice_level": all_ns["config"]["recursion_order_twice_level"]},
              "rows": rows, "physical_modular_agreement": None}
    trial.save(output/"comparison.json", report)
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def plot(result, comparison_result, path):
    config, rows = result["config"], comparison_result["rows"]
    nf, nc = config["quadrature_orders"][-1], config["quadrature_orders"][-2]
    level = config["max_level"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="760" viewBox="0 0 1300 760">',
           '<rect width="1300" height="760" fill="#fbfcfe"/>',
           '<g font-family="Arial,sans-serif" fill="#253248">']
    def text(x, y, value, size=14, anchor="start", color=None):
        fill = f' fill="{color}"' if color else ''
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{fill}>{escape(str(value))}</text>')
    text(50, 42, "NSRR trial refinement and NSNSNS comparison diagnostic", 26)
    text(50, 74, f"b=1.4; NSRR L={level}, N={config['quadrature_orders']}; unchanged sewing and vertex assumptions", 16)
    text(50, 101, "NOT a certified fixed-spin modular test. Both quotients use physical same-plumbing free factors.", 15, color="#a33f39")
    designs = [
        [("allNS_Q_diagnostic", "#253248", "NSNSNS diagnostic", False),
         ("baseline_Q_L2_N3", "#949ba6", "NSRR old L=2, N=3", True),
         (f"Q_L{level}_N{nc}", "#c07c23", f"NSRR L={level}, N={nc}", True),
         ("refined_Q_trial", "#087c9d", f"NSRR L={level}, N={nf}", False)],
        [("baseline_relative_difference", "#949ba6", "Old L=2, N=3", True),
         (f"relative_difference_L{level}_N3", "#b474aa", f"L={level}, N=3", True),
         (f"relative_difference_L{level}_N{nc}", "#c07c23", f"L={level}, N={nc}", True),
         ("refined_relative_difference", "#087c9d", f"L={level}, N={nf}", False)]
    ]
    for panel in (0, 1):
        left, top, width, height = 105+640*panel, 174, 485, 340
        scale = 1e7 if panel == 0 else 100.
        values = [r[key]*scale for key, _, _, _ in designs[panel] for r in rows]
        if panel:
            values.append(0.)
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.08, .01)
        lo, hi = lo-pad, hi+pad
        t0, t1 = min(r["t"] for r in rows), max(r["t"] for r in rows)
        xy = lambda t, y: (left+width*(t-t0)/(t1-t0), top+height*(hi-y)/(hi-lo))
        text(left, top-30, "Reference-normalized Q (units of 10^-7)" if panel == 0
             else "NSRR / NSNSNS - 1 (%)", 19)
        for j in range(6):
            value = lo+(hi-lo)*j/5
            y = xy(t0, value)[1]
            svg.append(f'<path d="M{left},{y}h{width}" stroke="#dce3ec"/>')
            text(left-12, y+4, f"{value:.2f}", 13, "end")
        if panel:
            y = xy(t0, 0)[1]
            svg.append(f'<path d="M{left},{y}h{width}" stroke="#526074" stroke-dasharray="4 4"/>')
        for row in rows:
            text(xy(row["t"], lo)[0], top+height+25, f"{row['t']:.2f}", 13, "middle")
        for i, (key, color, label, dashed) in enumerate(designs[panel]):
            coords = " ".join(f"{xy(r['t'], r[key]*scale)[0]:.3f},{xy(r['t'], r[key]*scale)[1]:.3f}" for r in rows)
            dash = ' stroke-dasharray="6 4"' if dashed else ''
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
            for row in rows:
                x, y = xy(row["t"], row[key]*scale)
                svg.append(f'<circle cx="{x}" cy="{y}" r="3.2" fill="{color}"/>')
            x, y = left+(i%2)*250, 597+(i//2)*29
            svg.append(f'<path d="M{x},{y-5}h24" stroke="{color}" stroke-width="3"{dash}/>')
            text(x+31, y, label, 13)
        text(left+width/2, 566, "t = Re Omega_original,12", 15, "middle")
    text(50, 679, "Omega_original,11 = Omega_original,22 = i; Omega_original,12 = t + i/2. No fitted relative normalization.", 14)
    text(50, 707, "NSRR equal signs: branching + two Virasoro c-recursions. Mixed signs: explicit level-3 PBW diagnostic completion.", 14)
    text(50, 735, "NSNSNS numerator retained from R=16, N=5. Remaining physical spin/nonchiral-assembly questions are unchanged.", 14)
    svg.append('</g></svg>')
    path.write_text("\n".join(svg)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--all-ns-summary", type=Path, required=True)
    args = parser.parse_args()
    result, verification = audit(args.run_dir)
    precision = extreme_momentum_probe(result["config"], args.run_dir)
    diagnostic = comparison(result, args.all_ns_summary, args.run_dir)
    plot(result, diagnostic, args.run_dir/"nsrr_refinement_comparison.svg")
    print(verification)
    print(precision)
    for row in diagnostic["rows"]:
        print(row)
