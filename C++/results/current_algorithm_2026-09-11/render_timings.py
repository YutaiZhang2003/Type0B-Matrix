#!/usr/bin/env python3
"""Render completed saved C++ timings into the new notes; computes no block."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
data = json.loads((HERE / "timings.json").read_text())
assert data["status"] == "completed", "Wait for all four fresh processes."
runs = {(r["level"], r["mode"]): r for r in data["runs"]}
ordered = [runs[n, mode] for n, mode in
           [(10, "ordinary"), (10, "inserted"), (15, "ordinary"), (15, "inserted")]]
assert all(r["status"] == "completed" and r["exit_code"] == 0 for r in ordered)
for source, expected in data["source_sha256"].items():
    assert hashlib.sha256((ROOT / "C++" / source).read_bytes()).hexdigest() == expected

lines = [
    r"These four fresh C++ measurements use 40-digit arithmetic (136 bits),",
    r"Apple clang 17 with \texttt{-O3}, and the macOS arm64 build with Accelerate.",
    r"The exact fermion and Schottky stages use GMP rationals.",
    r"All times in the next table are seconds. Indented rows are included in",
    r"branching; they must not be added to it again.",
    r"\begin{center}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"& \multicolumn{2}{c}{$N=10$} & \multicolumn{2}{c}{$N=15$}\\",
    r"Stage & $\eta\eta'=1$ & $\eta\eta'=-1$ & $\eta\eta'=1$ & $\eta\eta'=-1$\\",
    r"\midrule",
]
stages = [
    ("Branching", "branching"),
    (r"\quad Actions and support", "actions"),
    (r"\quad Outer Ward systems", "outer_ward"),
    ("Middle recurrence", "middle"),
    ("Reduced CCY blocks", "ccy"),
    ("Virasoro products", "products"),
    ("Branch assembly", "assembly"),
    ("Direct fermion factor", "auxiliary"),
    ("Sector convolution", "division"),
    ("Schottky vacuum", "schottky"),
    ("Vacuum restoration", "restoration"),
    ("Internal total", "total"),
]
for label, key in stages:
    if key == "total":
        lines.append(r"\midrule")
    lines.append(label + " & " + " & ".join(f'{r["timing_seconds"][key]:.3f}' for r in ordered) + r"\\")
lines += [
    "Full process & " + " & ".join(f'{r["wall_seconds"]:.3f}' for r in ordered) + r"\\",
    r"\bottomrule", r"\end{tabular}", r"\end{center}",
    r"The component timers leave a small amount of bookkeeping unassigned;",
    r"the internal total is measured independently rather than reconstructed",
    r"by adding rounded stage times. These are single-run observations,",
    r"not averaged benchmark distributions.",
    r"\begin{center}", r"\begin{tabular}{lrrrr}", r"\toprule",
    r"& $10,+$ & $10,-$ & $15,+$ & $15,-$\\", r"\midrule",
]
for label, key in [("Retained branch tuples", "branches"),
                   ("Virasoro engines evaluated", "virasoro_blocks"),
                   ("Transposed products reused", "transposed_products_reused")]:
    lines.append(label + " & " + " & ".join(f'{r["counts"][key]:,}' for r in ordered) + r"\\")
lines.append("Physical monomials & " + " & ".join(f'{r["coefficient_vectors"]:,}' for r in ordered) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}",
          r"Each physical monomial contains eight parity slots. The maximum",
          r"recorded residuals were:",
          r"\begin{center}", r"\begin{tabular}{lrrrr}", r"\toprule",
          r"& $10,+$ & $10,-$ & $15,+$ & $15,-$\\", r"\midrule"]
def scientific(x):
    mantissa, exponent = f"{x:.2e}".split("e")
    return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
for label, key in [("Action span", "maximum_action_residual"),
                   ("Outer Ward system", "maximum_ward_residual"),
                   ("Recovery parity sector", "maximum_sector_residual")]:
    lines.append(label + " & " + " & ".join(scientific(r["diagnostics"][key]) for r in ordered) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}",
          r"These diagnostics test consistency of the finite action solves and",
          r"the sector identity. They do not bound every physical coefficient's error.",
          r"The full coefficient arrays and unrounded timers are retained with",
          r"the source hashes, so this table can be regenerated without rerunning a block."]
target = ROOT / "Machine Notes/ramond_blocks_current_algorithm_timings.tex"
target.write_text("\n".join(lines) + "\n")
print(target)
