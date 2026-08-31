#!/usr/bin/env python3
"""Dependency-free SVG of the completed L3/L4/L5 trial comparison."""
import argparse
from html import escape
import json
from pathlib import Path


def plot(root):
    result = json.loads((root/"summary.json").read_text())
    rows = result["diagnostics"]
    order = result["config"]["quadrature_orders"][0]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1150" height="650" viewBox="0 0 1150 650">',
             '<rect width="1150" height="650" fill="#fbfcfe"/><g font-family="Arial,sans-serif" fill="#253248">']
    def text(x, y, value, size=14, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')
    text(50, 40, f"NSRR trial: block-cutoff convergence at N={order}", 26)
    text(50, 70, "b=1.4; same momentum grid and sewing assumptions at L=3,4,5; no fitted normalization", 15)
    text(50, 96, "Diagnostic comparison only: the physical spin / nonchiral assembly remains unverified.", 14)
    for panel in (0, 1):
        left, top, width, height = 100+550*panel, 165, 410, 325
        curves = [(f"L={l}", color, [r[f"Q_trial_L{l}_N{order}"]*1e7 if panel == 0
                   else (r[f"Q_trial_L{l}_N{order}"]/r["Q_allNS_diagnostic"]-1)*100 for r in rows])
                  for l, color in ((3, "#9a9faa"), (4, "#c48124"), (5, "#087c9d"))]
        if panel == 0:
            curves.insert(0, ("NSNSNS diagnostic", "#253248", [r["Q_allNS_diagnostic"]*1e7 for r in rows]))
        values = [z for _, _, curve in curves for z in curve]+([0] if panel else [])
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.08, .01)
        lo, hi = lo-pad, hi+pad
        xy = lambda t, z: (left+width*(t-.52)/.16, top+height*(hi-z)/(hi-lo))
        text(left, top-28, "Q in units of 10^-7" if panel == 0 else "NSRR / NSNSNS - 1 (%)", 19)
        for j in range(6):
            z = lo+(hi-lo)*j/5
            y = xy(.52, z)[1]
            parts.append(f'<path d="M{left},{y}h{width}" stroke="#dce3ec"/>')
            text(left-12, y+4, f"{z:.2f}", 12, "end")
        if panel:
            y = xy(.52, 0)[1]
            parts.append(f'<path d="M{left},{y}h{width}" stroke="#526074" stroke-dasharray="4 4"/>')
        for r in rows:
            text(xy(r["t"], lo)[0], 514, f"{r['t']:.2f}", 13, "middle")
        for i, (label, color, curve) in enumerate(curves):
            coords = " ".join(f"{xy(r['t'],z)[0]:.3f},{xy(r['t'],z)[1]:.3f}" for r, z in zip(rows, curve))
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            x, y = left+(i%2)*215, 570+(i//2)*25
            parts.append(f'<path d="M{x},{y-4}h23" stroke="{color}" stroke-width="3"/>')
            text(x+30, y, label, 13)
        text(left+width/2, 541, "t = Re Omega_original,12", 14, "middle")
    text(50, 633, "Equal signs: double Virasoro. Mixed signs: explicit PBW diagnostic completion through total chiral level 5.", 14)
    parts.append('</g></svg>')
    (root/"comparison.svg").write_text("\n".join(parts)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    plot(parser.parse_args().run_dir)
