#!/usr/bin/env python3
"""Fresh, sequential branching-only benchmarks with production vertex signs."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=10)
    parser.add_argument("--dps", type=int, nargs="+", default=[0, 40])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = Path(__file__).resolve().parents[1]
    executable = native / "bin/outer_ward_driver"
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "branching only: mode actions, outer coefficients, and inserted middle coefficients",
        "level": args.level,
        "parameters": {"b": "7/5", "P1": "11/23", "P2": "13/29", "P3": "17/31", "p": 0, "f": 0},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "fresh_process_per_case": True,
        "sequential": True,
        "numerical_cache_loaded": False,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(native)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((native / "include/ramond").glob("*.hpp"))
        },
        "runs": [],
    }
    for dps in args.dps:
        for inserted, mode in [(0, "ordinary"), (1, "inserted")]:
            name = f"{mode}_level{args.level}_{dps}dps"
            output = (args.output / f"{name}.json").resolve()
            log = args.output / f"{name}.log"
            command = [str(executable), str(args.level), str(dps), str(inserted),
                       "0", str(output), "1"]
            start = time.perf_counter()
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            wall = time.perf_counter() - start
            data = json.loads(output.read_text()) if output.exists() else {}
            record = {k: v for k, v in data.items() if k != "coefficients"}
            record.update(mode=mode, dps=dps, command=command, exit_code=result.returncode,
                          wall_seconds=wall, coefficient_count=len(data.get("coefficients", [])),
                          coefficient_file=output.name, log_file=log.name)
            summary["runs"].append(record)
            (args.output / "timings.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{name}: {record.get('total_branching_seconds', 'failed')} s branching; "
                  f"{wall:.6f} s process wall", flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
