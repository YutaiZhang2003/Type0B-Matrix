#!/usr/bin/env python3
"""Two distinct moduli-dependent diagnostics, with common numerator cutoffs.

Reuses validated N5 Liouville numerators, recomputes fixed/legacy free
factors, and plots the separately refined t=.60 point without splicing it
into the N5 curve. No production or archived data are modified.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

import check_nsrr_spin_quadrature as check
from fixed_spin_free_plumbing import fixed_spin_partition
from free_boson_plumbing import riemann_theta_constant_genus2
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from run_fixed_spin_free_check import SOURCE_BRANCH, TARGET_BRANCH, serializable


ROOT = check.ROOT
OUTPUT = ROOT / "Data Set/nsrr_moduli_difference_20260830"
MARKING = ROOT / "Data Set/full_fundamental_spin_marking_audit_20260830.json"


def assemble():
    config_path = check.DEFAULT_OUTPUT/"config.json"
    precision_path = check.DEFAULT_OUTPUT/"quadrature_summary.json"
    c = check.trial.load(config_path)
    check.validate(c)
    protected = check.fresh.protected_hashes()
    paths = {key: Path(value)/"summary.json" for key, value in c["references"].items()}
    refs = {key: check.trial.load(path) for key, path in paths.items()}
    if any(check.trial.digest(refs[key]) != c["reference_summary_digests"][key] for key in refs):
        raise ValueError("saved numerator/free references changed")
    precision = check.trial.load(precision_path)
    if precision["config_digest"] != check.trial.digest(c):
        raise ValueError("one-point refinement provenance mismatch")
    marking = check.trial.load(MARKING)
    kappa = refs["free"]["kappa"]
    rows = []
    for point in refs["free"]["points"]:
        t = point["t"]
        source = next(r for r in refs["source"]["rows"] if r["t"] == t and r["level"] == 3
                      and r["quadrature_order"] == 5 and r["lifts_geometry"] == [1, 1, 1])
        target = next(r for r in refs["target"]["rows"] if r["t"] == t)
        fd = next(r for r in marking["current"] if r["t"] == t)
        omega_fd = np.asarray([[complex(z) for z in row] for row in fd["omega_FD"]])
        spin_fd = tuple(tuple(row) for row in fd["spin_FD"])
        invariant_fd = np.linalg.det(2*omega_fd.imag)**.25*abs(
            riemann_theta_constant_genus2(omega_fd, spin_fd, tol=1e-15))
        free, diagnostics = {}, {}
        for channel, key, spin, branch in (
                ("source", "source_NSrr", ((1, 1), (0, 0)), SOURCE_BRANCH),
                ("target", "target_NSnsns", ((0, 0), (0, 0)), TARGET_BRANCH)):
            saved = point[key]
            q = tuple(map(complex, saved["q_values"]))
            chart = next(p for p in c[channel]["points"] if p["t"] == t)
            omega = np.asarray([[complex(z) for z in r] for r in chart["omega_source" if channel == "source" else "omega"]])
            results = [fixed_spin_partition(q, omega, spin, period_branch=branch, max_mode=m)
                       for m in (32, 40)]
            free[channel] = results[-1]
            change = abs(results[0]["Z_free"]/results[1]["Z_free"]-1)
            reproduction = abs(results[1]["Z_free"]/saved["Z_free"]-1)
            phi = results[1]["Z_free"]/results[1]["Z_boson"]**1.5
            fd_error = abs(phi/invariant_fd-1)
            if max(change, reproduction) > 1e-11 or fd_error > 1e-8:
                raise ArithmeticError("same-frame free/marking check failed")
            diagnostics[channel] = {"mode_32_to_40_change": change,
                                    "saved_free_reproduction_error": reproduction,
                                    "fundamental_domain_invariant_error": float(fd_error)}
        target_q = tuple(map(complex, point["target_NSnsns"]["q_values"]))
        legacy = [free["target"]["Z_boson"]*abs(theta_physical_fermion_fredholm(
            target_q, (1, -1, 1), max_mode=m).chiral_value)**2 for m in (32, 40)]
        legacy_change = abs(legacy[0]/legacy[1]-1)
        legacy_reproduction = abs(legacy[1]/target["target_Z_free"]-1)
        if max(legacy_change, legacy_reproduction) > 1e-11:
            raise ArithmeticError("legacy free reproduction failed")
        qr = source["total"]/free["source"]["Z_free"]**kappa
        qn = target["target_Z"]/free["target"]["Z_free"]**kappa
        if abs(qr/source["Q_trial_reference"]-1) > 1e-11:
            raise ArithmeticError("saved trial Q reproduction failed")
        free_ratio = legacy[1]/free["target"]["Z_free"]
        row = {"t": t, "Q_NSrr_trial_N5_L3": qr, "Q_NSnsns_N5_R16": qn,
               "Q_ratio_minus_one_percent": 100*(qr/qn-1),
               "Z_free_source_fixed": free["source"]["Z_free"],
               "Z_free_target_fixed": free["target"]["Z_free"],
               "Z_free_target_legacy": float(legacy[1]),
               "target_free_legacy_over_fixed_minus_one_percent": 100*(free_ratio-1),
               "target_only_legacy_denominator_Q_ratio_minus_one_percent": 100*((qr/qn)*free_ratio**kappa-1),
               "Z_NSrr_trial": source["total"], "Z_NSnsns": target["target_Z"],
               "spin_FD": fd["spin_FD"], "free_checks": diagnostics,
               "legacy_mode_32_to_40_change": legacy_change,
               "saved_legacy_free_reproduction_error": legacy_reproduction}
        rows.append(row)
    rows.sort(key=lambda r: r["t"])
    best = {ch: max((r for r in precision["rows"] if r["channel"] == ch), key=lambda r: r["N"])
            for ch in ("source", "target")}
    refined = {"t": precision["t"], "source_N": best["source"]["N"],
               "target_N": best["target"]["N"], "source_L": 3, "target_R": 16,
               "Q_NSrr_trial": best["source"]["Q_diagnostic"],
               "Q_NSnsns": best["target"]["Q_diagnostic"],
               "Q_ratio_minus_one_percent": 100*precision["finest_diagnostic_gap"]}
    if check.fresh.protected_hashes() != protected:
        raise ArithmeticError("protected kernels changed")
    return serializable({"schema": "nsrr-two-moduli-diagnostics-v1", "b": 1.4, "kappa": kappa,
                         "original_period_family": "Omega_11=Omega_22=i; Omega_12=t+i/2",
                         "common_cutoffs": {"source_L": 3, "source_N": 5, "target_R": 16, "target_N": 5},
                         "new_Liouville_integrals": 0, "new_free_mode_checks": [32, 40],
                         "rows": rows, "separate_refined_point": refined,
                         "physical_Q_NSrr": None,
                         "warning": "The free-factor and Q discrepancies are distinct. The target-only legacy-denominator column is a counterfactual, not a consistent alternative physical NSRR comparison. Numerical shifts are not rigorous error bounds.",
                         "provenance_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in [config_path, precision_path, MARKING, Path(__file__), *paths.values()]},
                         "protected_kernel_sha256": protected})


def svg_plot(data):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="900" viewBox="0 0 1120 900">',
           '<rect width="1120" height="900" fill="#fafbfe"/>',
           '<g font-family="Arial,Helvetica,sans-serif" fill="#263347">']
    def text(x, y, message, size=16, anchor="start", color="#263347"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(message))}</text>')
    def line(x1, y1, x2, y2, color="#e0e5ed", width=1, dash=False):
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 5"' if dash else '')+'/>')
    def marker(x, y, color, diamond=False):
        if diamond:
            svg.append(f'<path d="M {x} {y-7} L {x+7} {y} L {x} {y+7} L {x-7} {y} Z" fill="#fafbfe" stroke="{color}" stroke-width="2.5"/>')
        else:
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    text(85, 42, "Moduli dependence: free-factor choice and NSRR–NSNSNS comparison", 25)
    text(85, 76, "b = 1.4   ·   Ω₁₁ = Ω₂₂ = i   ·   Ω₁₂ = t + i/2 (original marking)", 18)
    text(85, 104, "Spin transport to a common fundamental domain is included. These are two different diagnostics.", 16, color="#59697d")
    left, width = 125, 870
    x = lambda t: left+width*(t-.52)/.16
    lo, hi, height = -6.3, .3, 240
    panels = [(170, "A. Target free-factor discrepancy", "100 × (Z_free,legacy / Z_free,fixed − 1)  [%]",
               "target_free_legacy_over_fixed_minus_one_percent", "#385fbc"),
              (530, "B. NSRR–NSNSNS quotient discrepancy", "100 × (Q_NSRR,trial / Q_NSNSNS − 1)  [%]",
               "Q_ratio_minus_one_percent", "#b45332")]
    for top, title, label, key, color in panels:
        y = lambda v: top+height*(hi-v)/(hi-lo)
        text(left-30, top-33, title, 20)
        text(left-30, top-10, label, 15, color="#59697d")
        for tick in range(-6, 1):
            line(left, y(tick), left+width, y(tick), dash=tick == 0)
            text(left-17, y(tick)+5, tick, 14, "end")
        line(left, top, left, top+height, "#8c9aae")
        line(left, top+height, left+width, top+height, "#8c9aae")
        coords = [(x(r["t"]), y(r[key])) for r in data["rows"]]
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="'+" ".join(f"{a},{b}" for a, b in coords)+'"/>')
        for row, (px, py) in zip(data["rows"], coords):
            marker(px, py, color)
            text(px, py-13, f'{row[key]:.2f}%', 15, "middle", color)
            line(px, top+height, px, top+height+5, "#8c9aae")
            text(px, top+height+25, f'{row["t"]:.2f}', 15, "middle")
        if key == "Q_ratio_minus_one_percent":
            fine = data["separate_refined_point"]
            fx, fy = x(fine["t"]), y(fine[key])
            marker(fx, fy, color, diamond=True)
            line(fx+9, fy+6, fx+58, fy+34, color)
            text(fx+65, fy+40, f'{fine[key]:.3f}%: N_source=6, N_target=7', 14, color=color)
        else:
            text(left, top+height+56, "Free mode 32 → 40 checked; same target plumbing frame and correctly transported characteristic.", 14, color="#59697d")
    text(125, 479, "Lower curve: common N=5; NSRR trial L=3 and all-NS recursion R=16. No fitted normalization.", 15, color="#59697d")
    text(560, 824, "t = Re Ω₁₂ (original marking)", 19, "middle")
    text(85, 857, "Five evaluated points; lines guide the eye. Diamond is a separate one-point refinement, not part of the N=5 curve.", 14, color="#59697d")
    text(85, 881, "No new Liouville integration. Physical NSRR nonchiral assembly remains a trial; no rigorous error bars are claimed.", 14, color="#59697d")
    svg.append('</g></svg>')
    return "\n".join(svg)+"\n"


def write(data, directory=OUTPUT):
    paths = [directory/name for name in ("summary.json", "moduli_differences.csv", "moduli_differences.svg")]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing moduli diagnostic")
    directory.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    columns = [k for k, v in data["rows"][0].items() if isinstance(v, (int, float))]
    with paths[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["rows"])
    paths[2].write_text(svg_plot(data))


if __name__ == "__main__":
    result = assemble()
    write(result)
    for row in result["rows"]:
        print(f't={row["t"]:.2f}: free difference {row["target_free_legacy_over_fixed_minus_one_percent"]:.6f}%; '
              f'Q difference {row["Q_ratio_minus_one_percent"]:.6f}%')
    print(OUTPUT)
