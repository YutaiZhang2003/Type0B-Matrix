#!/usr/bin/env python3
"""Show the one-point refinement on the prior diagnostic Omega scan."""
from html import escape
from pathlib import Path

import check_nsrr_spin_quadrature as check


def plot(output=check.DEFAULT_OUTPUT):
    c = check.trial.load(output/"config.json")
    result = check.trial.load(output/"quadrature_summary.json")
    check.validate(c)
    refs = {k: check.trial.load(Path(v)/"summary.json") for k, v in c["references"].items()}
    if result["config_digest"] != check.trial.digest(c) or any(
            check.trial.digest(v) != c["reference_summary_digests"][k] for k, v in refs.items()):
        raise ValueError("plot input provenance mismatch")
    rows = []
    for p in refs["free"]["points"]:
        t = p["t"]
        target = next(r["target_Z"] for r in refs["target"]["rows"] if r["t"] == t)/p["target_NSnsns"]["Z_free"]**result["kappa"]
        source = next(r["Q_trial_reference"] for r in refs["source"]["rows"] if r["t"] == t and r["level"] == 3 and r["quadrature_order"] == 5 and r["lifts_geometry"] == [1, 1, 1])
        rows.append({"t": t, "source": source, "target": target, "gap_percent": 100*(source/target-1)})
    fine = {ch: max((r for r in result["rows"] if r["channel"] == ch), key=lambda r: r["N"]) for ch in ("source", "target")}
    new = {"t": c["t"], **{ch: fine[ch]["Q_diagnostic"] for ch in fine},
           "gap_percent": 100*result["finest_diagnostic_gap"]}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="810" viewBox="0 0 1080 810">',
           '<rect width="1080" height="810" fill="white"/><g font-family="Arial,sans-serif" fill="#243044">']
    def text(x, y, value, size=15, anchor="start", color=None):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'+(f' fill="{color}"' if color else '')+f'>{escape(str(value))}</text>')
    def line(x1, y1, x2, y2, color, width=1, dashed=False):
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 5"' if dashed else '')+'/>')
    def marker(x, y, color, diamond=False):
        if diamond:
            svg.append(f'<path d="M{x},{y-7} L{x+7},{y} L{x},{y+7} L{x-7},{y} Z" fill="white" stroke="{color}" stroke-width="2.5"/>')
        else:
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    blue, red = "#2864b7", "#bc4b2b"
    text(80, 40, "NSNSNS–NSRR: fixed-cutoff momentum refinement", 25)
    text(80, 68, "Diagnostic Q only: the Liouville numerator-to-spin identification remains unresolved.", 16)
    text(80, 96, "b = 1.4;  Omega_original(t) = [[i, t+0.5i], [t+0.5i, i]]; cosmological factor omitted", 14)
    line(80, 125, 111, 125, blue, 2); marker(96, 125, blue)
    text(122, 130, "NSNSNS: R16, N5")
    line(380, 125, 411, 125, red, 2); marker(396, 125, red)
    text(422, 130, "NSRR trial: L3, N5")
    marker(710, 125, "#243044", True)
    text(726, 130, "Diamonds: N7 / N6 at t = 0.60")
    left, width = 110, 870
    x = lambda t: left+width*(t-.52)/.16
    for panel, (top, height, lo, hi, ticks) in enumerate(((177, 280, 1.05, 3.20, (1.2, 1.6, 2.0, 2.4, 2.8, 3.2)),
                                                       (530, 180, -5.5, .3, (-5, -4, -3, -2, -1, 0)))):
        y = lambda v: top+height*(hi-v)/(hi-lo)
        text(left, top-17, "Q × 10⁷" if panel == 0 else "100 × (Q_NSRR / Q_NSNSNS − 1)   [%]", 17)
        for tick in ticks:
            line(left, y(tick), left+width, y(tick), "#e2e6ed", dashed=panel == 1 and tick == 0)
            text(left-14, y(tick)+5, tick, 14, "end")
        line(left, top, left, top+height, "#919aaa")
        line(left, top+height, left+width, top+height, "#919aaa")
        for row in rows:
            line(x(row["t"]), top+height, x(row["t"]), top+height+5, "#919aaa")
            text(x(row["t"]), top+height+23, f"{row['t']:.2f}", 14, "middle")
        keys = (("target", blue), ("source", red)) if panel == 0 else (("gap_percent", "#586275"),)
        for key, color in keys:
            values = [(x(r["t"]), y(r[key]*(1e7 if panel == 0 else 1))) for r in rows]
            svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(f'{a},{b}' for a, b in values)+'"/>')
            for a, b in values:
                marker(a, b, color)
            a, b = x(new["t"]), y(new[key]*(1e7 if panel == 0 else 1))
            marker(a, b, color, True)
            if panel == 1:
                text(a+18, b+24, f"Refined: {new[key]:.4f}%", 15)
    text(545, 760, "t = Re Ω₁₂ (original marking)", 18, "middle")
    text(80, 793, "Only t = 0.60 is newly refined. Successive grid changes are not rigorous integration error bounds.", 14)
    svg.append('</g></svg>')
    (output/"omega_comparison_refined_point.svg").write_text('\n'.join(svg)+'\n')
    check.trial.save(output/"plot_data.json", {"baseline_N5": rows, "refined_point": new,
                                              "new_grid_sizes": {k: v["N"] for k, v in fine.items()},
                                              "config_digest": check.trial.digest(c),
                                              "quadrature_summary_digest": check.trial.digest(result)})


if __name__ == "__main__":
    plot()
