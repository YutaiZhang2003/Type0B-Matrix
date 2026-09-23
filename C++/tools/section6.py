#!/usr/bin/env python3
"""Reproduce Section 6 with C++ numerical kernels and explicit cutoff domains.

Python launches processes, validates coverage, compares decimal output and writes
provenance. It does not evaluate a conformal block. No persisted numerical cache
is loaded by a fresh run. Run `section6.py --help` for commands.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

CPP = Path(__file__).resolve().parents[1]
REPO = CPP.parent
CONVENTIONS = "product_bpz_residue_2026-09-22"
ZERO = (Decimal(0), Decimal(0))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = [CPP / "Makefile", Path(__file__)]
    for directory in ("include/ramond", "include/scblocks", "src", "drivers"):
        paths.extend(p for p in (CPP / directory).iterdir()
                     if p.suffix in (".hpp", ".inc", ".cpp"))
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def norm(value):
    return (value[0] * value[0] + value[1] * value[1]).sqrt()


def compare_values(left, right, tolerance):
    if left.keys() != right.keys():
        raise ValueError("coefficient coverage differs between the two methods")
    absolute = scaled = Decimal(0)
    failed = 0
    worst = None
    with localcontext() as context:
        context.prec = 100
        for key, x in left.items():
            y = right[key]
            delta = norm((x[0] - y[0], x[1] - y[1]))
            error = delta / max(Decimal(1), norm(x), norm(y))
            absolute = max(absolute, delta)
            if error > scaled:
                scaled = error
                worst = {"index": key, "left": list(map(str, x)), "right": list(map(str, y))}
            failed += error > Decimal(tolerance)
    return {"passed": failed == 0, "components": len(left), "tolerance": tolerance,
            "failed_components": failed, "max_absolute": str(absolute),
            "max_scaled": str(scaled), "worst": worst}


def graph_rows(path, field, metadata):
    values = {}
    edges = len(metadata["ramond_edges"])
    level = metadata["level"]
    box = metadata["truncation"] == "per-edge"
    expected = ((2 * level + 1) ** edges if box else math.comb(2 * level + edges, edges))
    for line in Path(path).open():
        record = json.loads(line)
        k = tuple(record["level2"])
        if len(k) != edges or min(k) < 0 or (max(k) if box else sum(k)) > 2 * level:
            raise ValueError(f"coefficient outside the stated domain: {k}")
        parity = sum((n % 2) << e for e, n in enumerate(k))
        case = record["case"]
        expected_f = [sum(k[e] for e in slot) % 2 for slot in metadata["vertex_slots"]]
        if metadata["cases"][case]["f"] != expected_f:
            raise ValueError(f"wrong vertex form parity at {k}")
        row = record[field]
        if len(row) > 1 or (row and row[0][0] != parity):
            raise ValueError(f"wrong edge parity at {k}")
        key = (case,) + k
        if key in values:
            raise ValueError(f"duplicate coefficient: {key}")
        values[key] = (Decimal(row[0][1]), Decimal(row[0][2])) if row else ZERO
    if len(values) != expected:
        raise ValueError(f"incomplete coefficient domain: {len(values)} != {expected}")
    return values


def compare_graph(dv, ns, output, tolerance="1e-18"):
    dv, ns = Path(dv), Path(ns)
    a = json.loads((dv / "metadata.json").read_text())
    b = json.loads((ns / "metadata.json").read_text())
    if {k: v for k, v in a.items() if k != "method"} != {k: v for k, v in b.items() if k != "method"}:
        raise ValueError("parameters, precision, convention or truncation differ")
    if a["conventions"] != CONVENTIONS or any(a["ramond_edges"]):
        raise ValueError("comparison requires current-convention all-NS results")
    result = compare_values(graph_rows(dv / "coefficients.jsonl", "double_virasoro", a),
                            graph_rows(ns / "coefficients.jsonl", "c_recursion", b), tolerance)
    result.update(channel=a["channel"], level=a["level"], truncation=a["truncation"], dps=a["dps"],
                  double_virasoro_timing=json.loads((dv / "summary.json").read_text()),
                  ns_recursion_timing=json.loads((ns / "summary.json").read_text()))
    atomic_json(output, result)
    return result


def theta_rows(data, level):
    values = {}
    for record in data["coefficients"]:
        k = tuple(record["exponents"])
        if len(k) != 4 or k[1] != k[2] or k[3] % 2 or min(k) < 0:
            raise ValueError(f"invalid theta exponent: {k}")
        if max(k[0], 2*k[1], k[3]) > 2*level or len(record["values"]) != 8:
            raise ValueError("theta cutoff or parity coverage is wrong")
        for parity, value in enumerate(record["values"]):
            key = k + (parity,)
            if key in values:
                raise ValueError("duplicate theta coefficient")
            values[key] = Decimal(value["real"]), Decimal(value["imag"])
    if len(values) != 8*(2*level+1)*(level+1)**2:
        raise ValueError("incomplete theta coefficient domain")
    return values


class Run:
    def __init__(self, directory, suite, dps):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "run_manifest.json"
        if self.path.exists():
            raise ValueError("run directory already has a manifest; choose a fresh directory")
        self.start = time.perf_counter()
        self.report = {"status": "running", "suite": suite, "dps": dps,
                       "platform": platform.platform(), "conventions": CONVENTIONS,
                       "source_sha256": source_hashes(), "stages": [], "comparisons": [],
                       "binary_sha256": {name: hashlib.sha256((CPP/"bin"/name).read_bytes()).hexdigest()
                                         for name in ("ramond", "pbw", "graph_blocks")},
                       "protocol": "Sequential fresh C++ processes; no numerical cache loaded; build excluded."}
        self.save()

    def save(self):
        self.report["wall_seconds"] = time.perf_counter() - self.start
        atomic_json(self.path, self.report)

    def execute(self, name, command, allow_comparison_failure=False):
        stage = {"name": name, "command": list(map(str, command)), "status": "running",
                 "executable_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()}
        self.report["stages"].append(stage)
        self.save()
        print("Starting", name, flush=True)
        start = time.perf_counter()
        with (self.directory / (name + ".log")).open("w") as log:
            child = subprocess.Popen(stage["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            stage["pid"] = child.pid
            self.save()
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                stage.update(status="interrupted", wall_seconds=time.perf_counter()-start)
                self.report["status"] = "interrupted"
                self.save()
                raise
        stage.update(status="completed" if code == 0 else "failed", exit_code=code,
                     wall_seconds=time.perf_counter()-start)
        self.save()
        print(name, stage["status"], f"{stage['wall_seconds']:.3f} s", flush=True)
        if code and not (code == 1 and allow_comparison_failure):
            self.report["status"] = "failed"
            self.save()
            raise RuntimeError(f"{name} failed; see {self.directory / (name + '.log')}")
        return code

    def graph(self, channel, method, level, truncation):
        name = f"{channel}_{method}_{truncation}{level}"
        directory = self.directory / name
        code = self.execute(name, [CPP/"bin/graph_blocks", "--channel", channel, "--method", method,
                           "--level", str(level), "--truncation", truncation,
                           "--dps", str(self.report["dps"]), "--output", directory],
                           allow_comparison_failure=method == "pbw-check")
        if code:
            summary = json.loads((directory/"summary.json").read_text())
            retry_dps = self.report["dps"] + 20
            failed = [c["case"] for c in summary["cases"]
                      if any(c[key]["failed_components"] for key in
                             ("forward", "recovery", "vanishing", "unequal_split", "spin_evaluations"))]
            if not failed:
                raise RuntimeError("PBW driver failed without a failing comparison")
            for case in failed:
                retry = self.directory / f"{name}_case{case}_{retry_dps}dps"
                self.execute(retry.name, [CPP/"bin/graph_blocks", "--channel", channel, "--method", method,
                    "--level", str(level), "--truncation", truncation, "--case-index", str(case),
                    "--dps", str(retry_dps), "--output", retry])
                self.report["comparisons"].append({"channel": channel, "case": case,
                    "status": "passed_at_higher_precision", "initial_dps": self.report["dps"],
                    "validation_dps": retry_dps, "initial_output": str(directory), "retry_output": str(retry)})
            self.save()
        return directory

    def ns_pair(self, channel, level, truncation):
        dv = self.graph(channel, "dv", level, truncation)
        ns = self.graph(channel, "ns", level, truncation)
        result = compare_graph(dv, ns, self.directory / f"{channel}_comparison_{truncation}{level}.json")
        self.report["comparisons"].append(result)
        self.save()
        if not result["passed"]:
            raise RuntimeError(f"{channel} comparison failed: {result['max_scaled']}")

    def theta(self, level):
        for p, f, eta, mode in itertools.product(range(2), range(2), (1, -1), ("ordinary", "inserted")):
            stem = f"theta-nsrr_p{p}_f{f}_eta{eta}_{mode}"
            paths = {}
            for algorithm, executable in (("dv", "ramond"), ("pbw", "pbw")):
                paths[algorithm] = self.directory / f"{stem}_{algorithm}.json"
                command = [CPP/"bin"/executable, "--mode", mode, "--p", str(p), "--f", str(f),
                           "--eta", str(eta), "--level", str(level), "--dps", str(self.report["dps"]),
                           "--json", paths[algorithm]]
                if algorithm == "dv":
                    command += ["--truncation", "per-edge"]
                self.execute(f"{stem}_{algorithm}", command)
            data = {name: json.loads(path.read_text()) for name, path in paths.items()}
            if any(x.get("conventions") != CONVENTIONS for x in data.values()):
                raise ValueError("theta convention metadata does not match")
            result = compare_values(theta_rows(data["dv"], level), theta_rows(data["pbw"], level), "1e-18")
            result.update(channel="theta-nsrr", p=p, f=f, eta=eta, mode=mode, level=level, truncation="per-edge")
            self.report["comparisons"].append(result)
            self.save()
            if not result["passed"]:
                raise RuntimeError(f"{stem} comparison failed: {result['max_scaled']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a fresh requested group of checks")
    run.add_argument("--suite", choices=("theta-pbw", "graph-pbw", "theta-ns", "tetrahedron-ns", "paper"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--dps", type=int, default=40)
    compare = sub.add_parser("compare-ns", help="Compare existing production outputs without recomputing blocks")
    compare.add_argument("--dv", type=Path, required=True)
    compare.add_argument("--ns", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare-ns":
        result = compare_graph(args.dv, args.ns, args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.dps < 30:
        parser.error("Section 6 graph checks require at least 30 decimal digits")
    run = Run(args.output, args.suite, args.dps)
    try:
        if args.suite in ("theta-pbw", "paper"):
            run.theta(5)
        if args.suite in ("graph-pbw", "paper"):
            for channel in ("theta-ns", "glasses-ns", "glasses-ns-r", "glasses-r"):
                run.graph(channel, "pbw-check", 5, "per-edge")
            run.graph("tetrahedron-r", "pbw-check", 5, "total")
        if args.suite in ("theta-ns", "paper"):
            run.ns_pair("theta-ns", 10, "per-edge")
        if args.suite in ("tetrahedron-ns", "paper"):
            run.ns_pair("tetrahedron-ns", 10, "total")
        run.report["status"] = "completed"
    except BaseException:
        run.report["status"] = "failed"
        run.save()
        raise
    run.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
