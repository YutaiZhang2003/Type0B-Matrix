#!/usr/bin/env python3
"""Time the complete direct-definition auxiliary block in a fresh process.

This benchmark performs no comparison to another block.  Coefficients are
saved after extracting the common Q/sqrt(2), so the arithmetic can be exact.
The four-variable grading retains the position of the inserted fermion.
"""

import time

PROCESS_STARTED = time.perf_counter()
CPU_STARTED = time.process_time()

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=10)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.level < 0:
        parser.error("level must be nonnegative")

    import_started = time.perf_counter()
    from direct_fermion import DirectFermion
    import_seconds = time.perf_counter() - import_started

    def progress(record):
        print(json.dumps(record), flush=True)

    construction_started = time.perf_counter()
    provider = DirectFermion()
    series = provider.normalized_series(args.level, progress=progress)
    construction_seconds = time.perf_counter() - construction_started
    construction_cpu_seconds = time.process_time() - CPU_STARTED

    serialization_started = time.perf_counter()
    coefficients = [
        {"exponents": list(key), "values": [str(value) for value in vector]}
        for key, vector in sorted(series.items(), key=lambda item: (sum(item[0]), item[0]))
    ]
    here = Path(__file__).resolve().parent
    sources = (here / "direct_fermion.py", Path(__file__).resolve())
    report = {
        "status": "computed_without_crosschecks",
        "method": "direct fermion Fock-state sewing with the full ordered Theta psi(1) insertion",
        "total_physical_q_level": args.level,
        "normalization": "full auxiliary block divided by Q/sqrt(2)",
        "grading": {
            "exponent_order": ["a", "l", "r", "d"],
            "monomial": "q1^(a/2) q2_left^l q2_right^r q3^(d/2)",
            "cutoff": f"a+l+r+d <= {2*args.level}; d even",
            "split_relation": "q2_left=u*sqrt(q2), q2_right=sqrt(q2)/u",
        },
        "parity_order": "p_NS + 2*p_R2 + 4*p_R3",
        "arithmetic": "exact rational after extracting Q/sqrt(2)",
        "construction_seconds": construction_seconds,
        "cpu_seconds_through_construction": construction_cpu_seconds,
        "provider_import_seconds": import_seconds,
        "process_seconds_through_construction": serialization_started-PROCESS_STARTED,
        "coefficient_vectors": len(coefficients),
        "nonzero_scalar_coefficients": sum(bool(value) for vector in series.values() for value in vector),
        "diagnostics": provider.diagnostics,
        "environment": {
            "python": sys.version,
            "platform": platform.system(),
            "machine": platform.machine(),
        },
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "coefficients": coefficients,
    }
    # Encoding is included in the serialization time; file I/O is reported
    # separately.  No existing block, cache, or result file is read here.
    payload = json.dumps(report, indent=2)
    report["serialization_seconds"] = time.perf_counter()-serialization_started
    args.json.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    args.json.write_text(payload + "\n")
    report["initial_file_write_seconds"] = time.perf_counter()-write_started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_resident_memory_mib"] = peak/(1024**2 if sys.platform == "darwin" else 1024)
    report["process_seconds_through_initial_save"] = time.perf_counter()-PROCESS_STARTED
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "coefficients"}), flush=True)


if __name__ == "__main__":
    main()
