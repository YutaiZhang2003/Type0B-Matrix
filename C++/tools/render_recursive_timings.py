"""Render the saved recursive-branching and full-pipeline timing records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "C++/results/recursive_branching_2026-09-11"
NOTES = ROOT / "Machine Notes"


def table(headings, rows, alignment=None):
    md = ["| " + " | ".join(headings) + " |",
          "| " + " | ".join(["---"] * len(headings)) + " |"]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    alignment = alignment or "ll" + "r" * (len(headings) - 2)
    tex = [r"\begin{center}\small",
           r"\begin{tabular}{@{}" + alignment + r"@{}}",
           r"\toprule", " & ".join(headings) + r" \\", r"\midrule"]
    tex += [" & ".join(row) + r" \\" for row in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(md), "\n".join(tex)


def main():
    runs = json.loads((FOLDER / "level10_timings/timings.json").read_text())["runs"]
    rows = []
    for r in runs:
        rows.append([r["mode"].capitalize(), "Machine" if not r["dps"] else "40 digits"] +
                    [f"{r[k]:.4f}" for k in ("actions_seconds", "ward_seconds",
                                             "middle_seconds", "total_branching_seconds",
                                             "wall_seconds")])
    md_table, tex_table = table(["Mode", "Precision", "Actions", "Outer", "Middle", "Total", "Wall"], rows)
    md = [
        "# Recursive branching and level-10 timings", "",
        "The directly computed initial table is exactly the boundary box in Human Notes/SCblock.tex: "
        "NS n=0,+/-1/2, and both Ramond labels n=+/-1/4,+/-3/4. Higher coefficients are computed "
        "in increasing order of floor(abs(n1))+floor(abs(n2)-1/4)+floor(abs(n3)-1/4), and every "
        "result is stored for reuse. The bulk update is scalar; explicit boundary actions "
        "couple at most four unanchored coefficients at a time. There is no global outer Ward matrix.",
        "",
        "The second conformal Ward identity is used when the first local relation is insufficient. "
        "Additional mode actions are constructed lazily and cached. Ward rows, embedded weights "
        "and single-leg descendant factors are reused, and the requested vertex signs share "
        "each local elimination. Dependencies and used Ward identities are checked after each update.",
        "",
        "## Fresh level-10 branching benchmark", "",
        "Each case ran sequentially in a separate process with empty numerical caches. "
        "Parameters: b=7/5, (P1,P2,P3)=(11/23,13/29,17/31), p=f=0. Ordinary uses (+,+); "
        "inserted uses (+,-), matching production. Times are seconds; compilation is excluded.",
        "", md_table, "",
        "Total includes initial mode actions, direct boundary values, outer recursion, and "
        "the middle coefficients for the inserted mode. Wall also includes startup, final "
        "coefficient serialization, cleanup and exit. These branching-only runs do not compute CCY "
        "or full physical blocks. Initial action construction takes over 98% of these totals.",
        "",
        "Ordinary stores 792 outer coefficients: 96 direct boundary and 696 recursive values. "
        "Inserted stores 1,936 outer coefficients: 192 direct boundary and 1,744 recursive values, "
        "and evaluates 36 middle coefficients. No first-prefactor zero or second-identity fallback "
        "occurs at this parameter point.",
    ]
    tex = [
        "The following branching-only runs are separate fresh processes at the parameters",
        "listed above, run sequentially with numerical reuse only within each process.",
        "Ordinary requests one vertex sign; inserted requests both. Times are seconds.",
        tex_table,
        "Total includes initial mode actions, direct boundary values, outer recursion,",
        "and the inserted middle coefficients. Wall also includes startup, serialization,",
        "cleanup and exit. No CCY or physical block is computed in these four runs.",
        r"Initial action construction accounts for over 98\% of total branching time.",
        "Ordinary stores 792 outer values; inserted stores 1,936 outer values and evaluates",
        "36 middle values. Compilation is excluded.",
    ]
    full = []
    for dps in [0, 40]:
        p = FOLDER / f"full_pipeline_{dps}dps/timings.json"
        if p.exists():
            full += [(dps, r) for r in json.loads(p.read_text())["runs"]
                     if r["status"] == "completed"]
    if len(full) == 4:
        rows = []
        for dps, r in full:
            t = r["timing_seconds"]
            branch = t["branching"] + t["middle"]
            other = t["total"] - branch - t["ccy"]
            rows.append([r["mode"].capitalize(), "Machine" if not dps else "40 digits"] +
                        [f"{v:.4f}" for v in (branch, t["ccy"], other, t["total"], r["wall_seconds"])])
        m, t = table(["Mode", "Precision", "Branching", "CCY", "Other", "Total", "Wall"], rows)
        md += ["", "## Fresh full level-10 pipelines", "", m, "",
               "These are complete production runs: stored branching recursion, double-Virasoro "
               "CCY with Schottky, direct free-fermion factor, convolution and vacuum restoration. "
               "Branching includes the middle coefficients. Other contains products, assembly, "
               "auxiliary factor, division, Schottky and restoration. All use sector-policy record; "
               "completion alone is not an accuracy certificate. No new physical PBW block was run."]
        tex += [r"\subsubsection*{Fresh full level-10 pipelines}", t,
                "These complete production runs include CCY, Schottky, the direct fermion",
                "factor, convolution and vacuum restoration. Branching includes middle",
                "coefficients; Other contains the remaining stages. Each case is a fresh",
                r"process with \texttt{sector-policy record}; completion alone does not",
                "certify accuracy. No new physical PBW block was computed."]
        previous = json.loads((ROOT / "C++/results/current_algorithm_2026-09-11/timings.json").read_text())
        old = {r["mode"]: r for r in previous["runs"] if r["level"] == 10}
        comparison = []
        for dps, r in full:
            if dps != 40:
                continue
            prior = old[r["mode"]]
            comparison.append([r["mode"].capitalize(),
                               f"{prior['timing_seconds']['outer_ward']:.3f}",
                               f"{r['timing_seconds']['outer_ward']:.3f}",
                               f"{prior['wall_seconds']:.2f}", f"{r['wall_seconds']:.2f}"])
        m, t = table(["Mode", "Old outer", "New outer", "Old wall", "New wall"],
                     comparison, alignment="lrrrr")
        md += ["", "### Before and after recursive branching: level 10, 40 digits", "", m, "",
               "All times are seconds. Outer measures the outer-coefficient stage alone; wall "
               "measures the complete process. The former global solver is retained here only "
               "as a timing baseline. These are single runs on the same machine at the same "
               "parameters, measured in different sessions. Unchanged mode-action and CCY stages "
               "also ran faster; neither ratio isolates the effect of the recursion change. "
               "The baseline source is C++/results/current_algorithm_2026-09-11/timings.json."]
        tex += [r"\begin{samepage}", r"\subsubsection*{Before and after recursive branching}",
                "This comparison uses level 10 and 40-digit arithmetic; all times are seconds.", t,
                r"\end{samepage}",
                "Outer measures only the outer-coefficient stage; wall measures the complete",
                "process. The former global solver is retained only as a timing baseline.",
                "These single runs use the same machine and parameters but different sessions.",
                "Unchanged action and CCY stages also ran faster, so neither ratio isolates",
                "the effect of the recursion change. The baseline is saved in",
                r"\path{C++/results/current_algorithm_2026-09-11/timings.json}."]
    validation = FOLDER / "saved_high_precision_validation.json"
    if validation.exists():
        checks = json.loads(validation.read_text())["checks"]
        rows = [[r["mode"].capitalize(), "Machine" if not r["dps"] else "40 digits",
                 f"{float(r['maximum_scaled_difference']):.3e}", r["status"]]
                for r in checks]
        m, t = table(["Mode", "Precision", "Scaled difference", "Check"], rows)
        md += ["", "## Full-pipeline accuracy", "", m, "",
               "All 4,048 physical components through level 10 are compared against existing "
               "384-bit physical-result files. The ordinary reference is the saved restricted "
               "recovery result; the inserted reference is the saved physical PBW result. "
               "No PBW was recomputed. The scaled difference is abs(a-b)/max(1,abs(a),abs(b)). "
               "The criteria are 2e-8 at machine precision and 1e-20 at 40 digits.", "",
               "Both 40-digit runs pass. The machine runs do not meet the requested accuracy "
               "at level 10: about 6.97e-7 for ordinary and 1.20e-3 for inserted. A diagnostic "
               "run of the former global-branching pipeline has essentially the same large "
               "discrepancy (6.91e-7 and 1.20e-3). This is an existing machine-precision limitation, "
               "not evidence that the new 40-digit recursion is wrong. Machine runtimes are "
               "reported as performance measurements, not accuracy-qualified level-10 results. "
               "See saved_high_precision_validation.json and machine_precision_diagnosis.json."]
        tex += [r"\subsubsection*{Accuracy of the full pipelines}", t,
                "The comparison covers all 4,048 physical components against existing 384-bit",
                "physical-result files: restricted recovery for ordinary and direct PBW for",
                r"inserted. No PBW was recomputed. The scaled difference is",
                r"$|a-b|/\max(1,|a|,|b|)$; criteria are $2\times10^{-8}$ at machine precision",
                r"and $10^{-20}$ at 40 digits. Both 40-digit runs pass.",
                "The former global-branching pipeline exhibits essentially the same large",
                "machine-precision discrepancy. Machine times therefore measure performance;",
                "they are not accuracy-qualified level-10 results. Use the 40-digit runs",
                "for the validated full-pipeline measurements at this level."]
    md += [
        "", "## Directed validation", "",
        "- All 768 level-5 ordinary-support coefficients agree with the previous global solver: "
        "maximum scaled differences 1.67e-25 at the default point and 3.76e-25 at equal Ramond "
        "momenta, using 40 digits.",
        "- All 880 level-5 inserted-support coefficients agree within 1.59e-25 at 40 digits.",
        "- The equal-momentum case explicitly exercises vanishing first prefactors. Machine "
        "and 40-digit coefficients agree within 3.29e-9 (maximum scaled difference).",
        "- Each low-level run additionally checks 608 second Ward identities, covering both "
        "vertex signs and Ramond parities. These checks use stored coefficients, without CCY or PBW blocks.",
        "- The previously failing level-20 ordinary-support branching problem completes at 40 "
        "digits for both vertex signs: 4,432 stored coefficients, comprising 192 boundary and "
        "4,240 recursive values. Initial actions took 150.24 s; the outer recursion took "
        "0.697669 s. No vanishing prefactor or second-identity fallback occurred.",
        "",
        "Level-20 action reconstruction residuals are at most 8.51e-26. The maximum residual "
        "of the first Ward equations used for recursion is 1.71e-40; those equations determine "
        "the coefficients, so this checks arithmetic consistency rather than independently "
        "certifying output accuracy. The full level-20 physical pipelines remain stopped.",
        "",
        "## Reproduction", "",
        "    make -C C++ all bin/outer_ward_driver",
        "    python3 C++/tools/time_branching.py --level 10 --dps 0 40 --output /tmp/branching_level10",
        "    python3 C++/tools/time_current_pipelines.py --levels 10 --dps 40 --output /tmp/full_level10",
        "",
        "The directories level10_timings and full_pipeline_*dps contain commands, source hashes, "
        "stage timings and coefficient arrays. The diagnostic driver also saves a partial coefficient "
        "table if its outer recurrence fails; the accompanying log identifies the failed labels.",
    ]
    tex += [
        r"\subsubsection*{Level-20 branching diagnostic}",
        "A separate ordinary-support check at 40 digits requests both vertex signs and",
        "completes all 4,432 outer coefficients: 192 direct boundary values and 4,240",
        "recursive values. Initial actions take 150.24~s; outer recursion takes 0.698~s.",
        "No vanishing local prefactor or second-identity fallback occurs at this point.",
        r"Mode-action reconstruction residuals are below $8.51\times10^{-26}$.",
        "This is a branching-only validation, not a level-20 physical-block computation.",
        "All commands, hashes, timing records and computed arrays are saved under",
        r"\path{C++/results/recursive_branching_2026-09-11/}.",
    ]
    (FOLDER / "README.md").write_text("\n".join(md) + "\n")
    (NOTES / "ramond_blocks_current_algorithm_branching_timings.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
