#!/usr/bin/env python3
"""Plot the numerical norm/phase audit; requires matplotlib only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def plot(output):
    data = json.loads((output / "summary.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), layout="constrained")
    colors = {"source": "#2369a8", "target": "#ca6330"}
    for side, label in (("source", "NSRR"), ("target", "All NS")):
        rows = [r for r in data["maxima"] if r["side"] == side and r["factor"] == "majorana"]
        for ax, key in zip(axes[:2], ("norm_relative_error", "phase_absolute_error_radians")):
            ax.semilogy([r["level"] for r in rows], [r[key] for r in rows],
                        "o-", color=colors[side], label=label, linewidth=2)
            ax.set_xticks(data["levels"])
            ax.set_xlabel("Total oscillator cutoff L")
            ax.grid(axis="y", which="major", alpha=.22)
    axes[0].set_title("Single Majorana: magnitude")
    axes[0].set_ylabel(r"Maximum $\left|\,|M_{\rm Fock}/M_{\rm formula}|-1\right|$")
    axes[0].legend(frameon=False)
    axes[1].set_title("Single Majorana: phase")
    axes[1].set_ylabel(r"Maximum $|\arg(M_{\rm Fock}/M_{\rm formula})|$ (rad)")
    winding = data["ramond_eight_turn_loop"]["rows"]
    first = complex(*winding[0]["continued_majorana"])
    continued = [complex(*r["continued_majorana"]) / first for r in winding]
    principal = [complex(*r["principal_majorana"]) / first for r in winding]
    x = [r["turns"] for r in winding]
    axes[2].plot(x, [z.real for z in continued], "o-", color="#2369a8", label="Continued root = Fock")
    axes[2].plot(x, [z.real for z in principal], "s--", color="#ca6330", label="Principal root at endpoint")
    axes[2].axhline(0, linewidth=.7, color="gray")
    axes[2].set_xticks(range(0, 9, 2))
    axes[2].set_xlabel(r"Full turns of $q_0$ (Ramond edge)")
    axes[2].set_ylabel(r"$\mathrm{Re}\,[M(q)/M(q_{\rm start})]$")
    axes[2].set_title("Same squared value; opposite sign")
    axes[2].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Bosonization vs direct Fock sewing", fontsize=17, weight="semibold")
    fig.savefig(output / "norm_phase_comparison.png", dpi=180)
    fig.savefig(output / "norm_phase_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Data Set/free_bosonization_phase_20260915")
    plot(parser.parse_args().output)
