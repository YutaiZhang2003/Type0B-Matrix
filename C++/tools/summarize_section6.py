#!/usr/bin/env python3
"""Assemble the completed Section 6 evidence; never recompute a block."""
import argparse
import json
from pathlib import Path

from section6 import atomic_json, source_hashes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "C++/results/section6_2026-09-23"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    directory = args.results
    read = lambda name: json.loads((directory / name).read_text())
    checks = []
    theta = read("theta_pbw5/run_manifest.json")
    assert theta["status"] == "completed" and len(theta["comparisons"]) == 16
    assert all(c["passed"] for c in theta["comparisons"])
    checks.append(dict(channel="theta NS-R-R", reference="physical PBW", truncation="per-edge", level=5,
        cases=16, multidegrees_per_case=396, dps=40, passed=True, fresh=True,
        max_scaled=max(float(c["max_scaled"]) for c in theta["comparisons"])))
    for channel in ("theta-ns", "glasses-ns", "glasses-ns-r", "glasses-r"):
        summary = read(f"graph_pbw5/{channel}_pbw-check_per-edge5/summary.json")
        cases = summary["cases"]
        precision = 40
        if channel == "glasses-r":
            retry = read("glasses_r_case4_60dps/summary.json")
            assert retry["failed_cases"] == 0
            original_metadata = read("graph_pbw5/glasses-r_pbw-check_per-edge5/metadata.json")
            retry_metadata = read("glasses_r_case4_60dps/metadata.json")
            assert original_metadata["cases"][4] == retry_metadata["cases"][0]
            cases = [retry["cases"][0] if c["case"] == 4 else c for c in cases]
            precision = "40; case 4 repeated at 60"
        for c in cases:
            assert all(c[k]["failed_components"] == 0 for k in
                       ("recovery", "forward", "unequal_split", "vanishing", "spin_evaluations"))
        checks.append(dict(channel=channel, reference="physical PBW", truncation="per-edge", level=5,
            cases=len(cases), multidegrees_per_case=summary["physical_multidegrees"], dps=precision,
            passed=True, fresh=True, max_scaled=max(c["recovery"]["max_scaled"] for c in cases)))
    tetra = read("tetrahedron_r_total5/summary.json")
    assert tetra["failed_cases"] == 0 and len(tetra["cases"]) == 64
    checks.append(dict(channel="tetrahedron, triangular R face", reference="physical PBW", truncation="total", level=5,
        cases=64, multidegrees_per_case=tetra["physical_multidegrees"], dps=40, passed=True, fresh=True,
        max_scaled=max(c["recovery"]["max_scaled"] for c in tetra["cases"])))
    ns = read("theta_ns10_comparison.json")
    assert ns["passed"] and ns["components"] == 9261
    checks.append(dict(channel="theta all NS", reference="SCA c-recursion", truncation="per-edge", level=10,
        multidegrees=ns["components"], dps=40, passed=True, fresh=True, max_scaled=float(ns["max_scaled"])))
    archive = read("archive_verification.json")
    forward_archive = read("ns_forward_archive_verification.json")
    assert archive["passed"] and forward_archive["passed"]
    checks.append(dict(channel="tetrahedron all NS", reference="SCA c-recursion", truncation="total", level=10,
        multidegrees=archive["archived_total10"]["components"], dps=40, passed=True, fresh=False,
        max_scaled=float(archive["archived_total10"]["max_scaled"]),
        evidence="Full archived level-10 coefficient files re-compared and hashes verified. Fresh production prefixes: DV total 3, batched NS recursion total 6."))
    directed = {name: read(name) for name in ("ns_forward_check.json", "ns_forward_seed_optimized_check.json",
                 "schottky_check.json", "graph_dv_comparison.json")}
    assert all(result["passed"] for result in directed.values())
    partition = read("partition_audit.json")
    result = dict(status="coefficient_tests_passed_with_recorded_precision_refinement",
        manuscript_edited=False, cutoff_policy="Genus two: independent edges 5/10. Tetrahedron: total levels 5/10.",
        coefficient_tests=checks, directed_implementation_checks=directed,
        precision_refinement=read("precision_refinement.json"),
        timing_context="Validation groups and compilation overlapped. These elapsed times are not controlled sequential speed benchmarks.",
        superseded_runs="The first graph suite stopped at its 40-digit glasses threshold failure; the failing case was repeated at 60 digits. The first NS theta level-10 seed was intentionally stopped before coefficient evaluation and restarted with trimmed Schottky expansions. Completed DV data were retained. Original logs and manifests are unchanged.",
        cross_channel_status=partition["status"], partition_comparison=partition,
        final_source_sha256=source_hashes())
    atomic_json(directory / "validation_summary.json", result)
    lines = ["# Section 6 validation", "", "All requested coefficient comparisons are covered with the agreed cutoff policy. Neither manuscript was edited.", "",
             "| Channel | Reference | Cutoff | Precision | Maximum scaled error |", "| --- | --- | --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c['channel']} | {c['reference']} | {c['truncation']} {c['level']} | {c['dps']} | {c['max_scaled']:.3e} |")
    lines += ["", "The tetrahedron all-NS level-10 result is the existing completed run, fully re-compared with output hashes checked. The new frontend reproduces its DV prefix through total level 3 and the new batched SCA reference reproduces its coefficients through total level 6. It was not rerun at level 10.", "",
        "One glasses case initially gave `5.719e-18` at 40 digits, above the `1e-18` release threshold. Repeating both DV and PBW at 60 digits reduced the discrepancy to `3.009e-38`. The original failure remains recorded; see `precision_refinement.json`.", "",
        "The fresh all-NS theta level-10 check contains all 9,261 multidegrees. Its maximum absolute discrepancy is `5.108e-7`; the maximum error scaled by `max(1,abs(DV),abs(SCA))` is `7.318e-19`. The large absolute discrepancy occurs for large coefficients; the report retains the worst scaled coefficient and both values.", "",
        "The all-NS theta DV run took 966.742 s; the optimized SCA reference took 131.407 s, including 11.180 s for the Schottky seed. Validation jobs and compilation overlapped, so these are elapsed validation times rather than controlled performance benchmarks.", "",
        "## Nonchiral comparison", "", "The partition-function equality claim is not established by this release. The completed quadrature study, with source N7 and target N10 at fixed block cutoffs, satisfies its two-step 0.01% stability criterion for both integrals and complex spin matrices. The two sign ratios are `0.250052595` and `0.249813244`; the spin-sum ratio is `0.249928019`. Multiplying the full NS-R-R amplitude by four is explicitly an assumption. The subsequent source-level-8 study with this assumption gives spin-sum ratio `0.999709394` and a largest resolved-spin discrepancy of about `0.0933%`. The old roughly 1% result used a superseded nonchiral contraction. Correction: the initial release report read an intermediate summary and incorrectly described the completed quadrature as unfinished.", "",
        "`partition_audit.json` recomputes these saved ratios without fitting or modifying them. The production frozen-grid assembler and normalization metadata are included in the source release; this audit does not claim a new momentum integration.", "",
        "## Reproduction", "", "See [the release guide](../../SECTION6.md). `validation_summary.json` contains the detailed scope, precision escalation, directed checks, and final source hashes. Per-run coefficient files and logs remain in this directory; the source archive includes compact reports, not numerical caches.", ""]
    (directory / "README.md").write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "coefficient_groups": len(checks), "cross_channel_status": result["cross_channel_status"]}, indent=2))


if __name__ == "__main__":
    main()
