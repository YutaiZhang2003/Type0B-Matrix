#!/usr/bin/env python3
"""Independently reassemble and plot the experimental NSRR sign trial."""
from __future__ import annotations

import argparse
from html import escape
import math
from pathlib import Path

import numpy as np

import nsrr_factorized_sign_trial as trial


def audit(output):
    result = trial.load(output/"summary.json")
    config = result["config"]
    trial.validate_config(config)
    shards = [trial.load(output/"shards"/f"node-{i:03d}.json") for i in range(len(trial.tasks(config)))]
    term_error, reduction_error, ground_error = 0., 0., 0.
    for i, shard in enumerate(shards):
        trial.validate_shard(config, i, shard)
        n, node = trial.tasks(config)[i]
        indices = np.unravel_index(node, (n,)*3)
        rules = trial._rules(config["q_envelope"], n)
        p = [float(rules[e][0][indices[e]]) for e in range(3)]
        if p != shard["momenta_geometry"] or trial._measure(rules, indices) != shard["measure"]:
            raise ValueError("quadrature node or measure mismatch")
        c = [trial.decode(z)/2 for z in shard["C_BRY"]]
        for row in shard["rows"]:
            primary2 = abs(trial.decode(row["primary"]))**2
            terms = []
            for j, (f, eta, ep) in enumerate(trial.CHANNELS):
                block = trial.decode(row["blocks"][j])
                # Independent expression AFTER the (-1)^f and i^(2f)
                # cancellation, with no call to the implementation's contract.
                value = primary2*c[0 if eta == 1 else 1]*c[0 if ep == 1 else 1]*abs(block)**2
                terms.append(value)
                term_error = max(term_error, abs(value-trial.decode(row["weighted_terms"][j]))/
                                 max(abs(value), abs(trial.decode(row["total"])), 1e-280))
            expected = sum(terms)
            term_error = max(term_error, abs(expected-trial.decode(row["total"]))/max(abs(expected), 1e-280))
            even, odd = sum(terms[:4]), sum(terms[4:])
            term_error = max(term_error, abs(even-odd-trial.decode(row["without_sewing_sign"]))/
                             max(abs(expected), 1e-280))
            if row["level"] == 0 and row["lifts_geometry"] == [1, 1, 1]:
                exact = primary2*4*(c[0]+c[1])**2
                ground_error = max(ground_error, abs(expected-exact)/max(abs(exact), 1e-280))
    for row in result["rows"]:
        terms = []
        for shard in shards:
            if shard["quadrature_order"] != row["quadrature_order"]:
                continue
            v = next(v for v in shard["rows"] if (v["t"], v["level"], v["lifts_geometry"])
                     == (row["t"], row["level"], row["lifts_geometry"]))
            terms.append(shard["measure"]*trial.decode(v["total"]))
        expected = complex(math.fsum(z.real for z in terms), math.fsum(z.imag for z in terms))
        reduction_error = max(reduction_error, abs(expected-row["total"])/max(abs(expected), 1e-280))
    if max(term_error, reduction_error, ground_error) > 1e-12:
        raise ArithmeticError("independent trial reduction failed")
    if result["physical_Z"] is not None or result["physical_Q"] is not None:
        raise ValueError("trial labelled physical")
    lift_spread, anti_control_error = 0., 0.
    for n in config["quadrature_orders"]:
        for point in config["points"]:
            for level in config["levels"]:
                selected = [r for r in result["rows"] if r["quadrature_order"] == n
                            and r["t"] == point["t"] and r["level"] == level]
                values = [r["total"] for r in selected]
                lift_spread = max(lift_spread, (max(values)-min(values))/max(map(abs, values)))
    for row in result["rows"]:
        anti_control_error = max(anti_control_error,
            abs(trial.decode(row["formal_same_convention_tilde"])-row["without_sewing_sign"])/
            max(abs(row["total"]), 1e-280))
    report = {"schema": "nsrr-factorized-sign-trial-audit-v1", "summary_digest": trial.digest(result),
              "maximum_term_error": term_error, "maximum_reduction_error": reduction_error,
              "maximum_ground_normalization_error": ground_error,
              "maximum_lift_relative_spread": lift_spread,
              "formal_tilde_vs_without_sign_control_error_scaled_by_trial": anti_control_error,
              "protected_kernel_sha256": trial.protected_hashes(),
              "physical_Z": None, "physical_Q": None}
    trial.save(output/"verification.json", report)
    return result, report


def probe(config, output):
    momenta_slots = (.31, .43, .57)
    components, checks = trial.block_components(config["b"], momenta_slots, 2)
    point = next(p for p in config["points"] if p["t"] == .6)
    q = tuple(complex(z) for z in point["q_geometry"])[::-1]
    rows = []
    for level in config["levels"]:
        blocks = trial.evaluate_blocks(components, q, (1, 1, 1), level)
        rows.append({"level": level, "blocks": [trial.encode(blocks[k]) for k in trial.CHANNELS]})
    checks["coefficient_conjugation_phase_error"] = max(
        abs(complex(v).conjugate()-(-1)**f*complex(v))/max(1., abs(v))
        for (f, _, _), vectors in components.items() for vector in vectors.values() for v in vector)
    value = {"b": config["b"], "t": .6, "momenta_slots": momenta_slots,
             "q_slots": [str(z) for z in q], "lifts_slots": [1, 1, 1],
             "channels": config["channels"], "rows": rows, "checks": checks,
             "meaning": "literal chiral NSRR blocks with primary powers stripped; not a partition"}
    trial.save(output/"probe_blocks_t060.json", value)
    return value


def plot(result, path):
    rows = result["rows"]
    selected = [r for r in rows if r["lifts_geometry"] == [1, 1, 1]]
    designs = [(3, 0, "#8c97a9", "N=3, L=0"), (3, 1, "#d58620", "N=3, L=1"),
               (3, 2, "#17699d", "N=3, L=2"), (2, 2, "#8563aa", "N=2, L=2")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="690" viewBox="0 0 1200 690">',
           '<rect width="1200" height="690" fill="#fbfcfe"/>',
           '<g font-family="Arial,sans-serif" fill="#253248">']
    def text(x, y, content, size=14, anchor="start", color=None):
        extra = f' fill="{color}"' if color else ''
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{extra}>{escape(str(content))}</text>')
    text(55, 41, "NSRR factorized sewing: explicit sign trial", 25)
    text(55, 71, "b=1.4; eight chiral components; graded sign and two odd-vertex phases retained", 15)
    text(55, 96, "EXPERIMENTAL ANSATZ — not a certified physical partition or modular comparison", 14, color="#a33f39")
    for panel, (field, title) in enumerate((("total", "Trial sewn numerator Z_trial"),
                                           ("Q_trial_reference", "Reference-normalized Q_trial"))):
        left, top, width, height = 135+570*panel, 165, 395, 310
        curves = [(color, label, [r for r in selected if r["quadrature_order"] == n and r["level"] == level])
                  for n, level, color, label in designs]
        values = [r[field] for _, _, curve in curves for r in curve]
        low, high = min(values), max(values)
        pad = max((high-low)*.08, max(abs(low), abs(high))*.01)
        low, high = low-pad, high+pad
        xy = lambda t, z: (left+width*(t-.52)/.16, top+height*(high-z)/(high-low))
        text(left, top-25, title, 19)
        for j in range(5):
            z = low+(high-low)*j/4
            y = xy(.52, z)[1]
            svg.append(f'<path d="M{left},{y}h{width}" stroke="#dce3ec"/>')
            text(left-12, y+4, f"{z:.3e}", 12, "end")
        for t in (.52, .56, .60, .64, .68):
            text(xy(t, low)[0], top+height+23, f"{t:.2f}", 13, "middle")
        for i, (color, label, curve) in enumerate(curves):
            coords = " ".join(f"{xy(r['t'],r[field])[0]:.3f},{xy(r['t'],r[field])[1]:.3f}" for r in curve)
            dash = ' stroke-dasharray="5 4"' if i == 3 else ''
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
            for r in curve:
                x, y = xy(r["t"], r[field])
                svg.append(f'<circle cx="{x}" cy="{y}" r="3.1" fill="{color}"/>')
            x, y = left+(i%2)*195, top+height+73+(i//2)*27
            svg.append(f'<path d="M{x},{y-4}h25" stroke="{color}" stroke-width="3"{dash}/>')
            text(x+32, y, label, 13)
        text(left+width/2, top+height+47, "t = Re Omega_original,12", 14, "middle")
    text(55, 615, "Q_trial = Z_trial / Z_free[11|00]^kappa; kappa = 1 + 2(b + 1/b)^2. Plumbing lifts shown: (+,+,+).", 14)
    text(55, 640, "The physical lift/spin dictionary and Ramond projection are not established by this trial.", 14)
    text(55, 664, "Equal signs: double Virasoro. Missing mixed-sign components: explicit level-2 PBW diagnostic completion.", 14)
    svg.append('</g></svg>')
    path.write_text("\n".join(svg)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result, report = audit(args.run_dir)
    sample = probe(result["config"], args.run_dir)
    plot(result, args.run_dir/"nsrr_factorized_sign_trial.svg")
    print(report)
    for channel, z in zip(trial.CHANNELS, sample["rows"][-1]["blocks"]):
        print(channel, trial.decode(z))
