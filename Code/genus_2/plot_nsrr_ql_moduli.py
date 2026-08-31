#!/usr/bin/env python3
"""Q_L-only presentation of the saved, validated five-point comparison.

The archived two-panel diagnostic is preserved. No new integrations or
changes to sewing conventions are made by this presentation driver.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "Data Set/nsrr_moduli_difference_20260830"
REPORT = DIRECTORY / "summary.json"
OUTPUT = DIRECTORY / "ql_moduli_difference.svg"
GAP = "Q_ratio_minus_one_percent"


def load_verified():
    data = json.loads(REPORT.read_text())
    for hashes in ("provenance_sha256", "protected_kernel_sha256"):
        for name, expected in data[hashes].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Saved provenance changed: {name}")
    if data["common_cutoffs"] != dict(source_L=3, source_N=5, target_R=16, target_N=5):
        raise ValueError("Unexpected cutoffs for the common five-point curve")
    if [row["t"] for row in data["rows"]] != [.52, .56, .60, .64, .68]:
        raise ValueError("Unexpected original moduli")
    for row in data["rows"]:
        expected = 100 * (row["Q_NSrr_trial_N5_L3"] / row["Q_NSnsns_N5_R16"] - 1)
        if not math.isclose(row[GAP], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Inconsistent saved Q_L ratio")
    fine = data["separate_refined_point"]
    expected = 100 * (fine["Q_NSrr_trial"] / fine["Q_NSnsns"] - 1)
    if not math.isclose(fine[GAP], expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Inconsistent refined Q_L ratio")
    return data


def svg_plot(data):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" viewBox="0 0 1100 640">',
           '<rect width="1100" height="640" fill="#fafbfe"/>',
           '<g font-family="Arial,Helvetica,sans-serif" fill="#263347">']

    def text(x, y, message, size=16, anchor="start", color="#263347"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(message))}</text>')

    def line(x1, y1, x2, y2, color="#e0e5ed", dash=False):
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"'
                   + (' stroke-dasharray="5 5"' if dash else '') + '/>')

    text(70, 42, "Q_L comparison: NSRR vs NSNSNS", 27)
    text(70, 76, "b = 1.4   ·   Ω₁₁ = Ω₂₂ = i   ·   Ω₁₂ = t + i/2 (original marking)", 18)
    text(70, 106, "100 × (Q_L,NSRR trial / Q_L,NSNSNS − 1)  [%]", 18)
    text(70, 134, "Common cutoffs: NSRR L = 3; NSNSNS R = 16 (level 8); momentum quadrature N = 5.", 15, color="#59697d")

    left, top, width, height = 125, 165, 840, 330
    x = lambda t: left + width * (t - .52) / .16
    y = lambda v: top + height * (.3 - v) / 6.3
    color = "#b45332"
    for tick in range(-6, 1):
        line(left, y(tick), left + width, y(tick), "#9aa8ba" if tick == 0 else "#e0e5ed", tick == 0)
        text(left - 17, y(tick) + 5, tick, 14, "end")
    text(left + width, y(0) - 10, "Agreement", 14, "end", "#59697d")
    line(left, top, left, top + height, "#8c9aae")
    line(left, top + height, left + width, top + height, "#8c9aae")
    coords = [(x(row["t"]), y(row[GAP])) for row in data["rows"]]
    svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="'
               + " ".join(f"{px},{py}" for px, py in coords) + '"/>')
    for row, (px, py) in zip(data["rows"], coords):
        svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
        text(px, py - 15, f'{row[GAP]:.3f}%', 16, "middle", color)
        line(px, top + height, px, top + height + 5, "#8c9aae")
        text(px, top + height + 25, f'{row["t"]:.2f}', 15, "middle")
    fine = data["separate_refined_point"]
    fx, fy = x(fine["t"]), y(fine[GAP])
    svg.append(f'<path d="M {fx} {fy-7} L {fx+7} {fy} L {fx} {fy+7} L {fx-7} {fy} Z" fill="#fafbfe" stroke="{color}" stroke-width="2.5"/>')
    line(fx + 9, fy + 6, fx + 50, fy + 77, color)
    text(fx + 58, fy + 83, f'{fine[GAP]:.3f}%: separate N_NSRR = 6, N_NSNSNS = 7', 14, color=color)
    text(545, 558, "t = Re Ω₁₂ (original marking)", 19, "middle")
    text(70, 597, "Five evaluated points; line guides the eye. Diamond is a separate refinement, not part of the N = 5 curve.", 14, color="#59697d")
    text(70, 622, "Saved Q_L values, no new integrations. NSRR nonchiral assembly remains a trial; no rigorous error bars.", 14, color="#59697d")
    svg.append('</g></svg>')
    return "\n".join(svg) + "\n"


if __name__ == "__main__":
    data = load_verified()
    with OUTPUT.open("x") as stream:
        stream.write(svg_plot(data))
    for row in data["rows"]:
        print(f't={row["t"]:.2f}: Q_L relative difference {row[GAP]:.6f}%')
    print(OUTPUT)
