#!/usr/bin/env python3
"""Run the ten-point blind h-dominant torus-three-point scan safely.

The driver reuses any completed point whose numerical design matches exactly.
In particular, the existing t=0.75 result is referenced in place rather than
recomputed.  At most two point jobs run concurrently by default, and threaded
numerical libraries are pinned to one thread to prevent oversubscription.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


T_VALUES = tuple(round(0.05 + 0.10 * index, 2) for index in range(10))
DEFAULT_OUTPUT_DIR = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "hdominant_scan10_p8_h8l3_q030_007_n256_r4_v1"
)
DEFAULT_REUSED_T075 = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "channel_atlas_hdominant_t075_p8_h8l3_q030_007_n256_r4_v1.json"
)
DEFAULT_LEGACY_SCAN = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "equal_split_imaginary_t_scan10_p12_n256_v1/worldsheet_scan_manifest.json"
)


def _tag(t_value: float) -> str:
    return f"t{int(round(100.0 * float(t_value))):03d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(record: dict[str, float]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def _expected_design() -> dict[str, object]:
    return {
        "momentum_order": 8,
        "necklace_order": 8,
        "necklace_low_order": 3,
        "necklace_backend": "regulated-h-recursion",
        "necklace_c_regulator": 0.025,
        "necklace_qhat_threshold": 0.30,
        "necklace_second_qhat_threshold": 0.07,
        "ope_order": 6,
        "ope_loop_order": 4,
        "ope_low_loop_order": 2,
        "evaluation_order_cap": None,
        "sobol_power": 8,
        "replicates": 4,
        "position_alpha": 0.3,
        "patch_epsilon": 0.15,
        "triple_patch_epsilon": 0.10,
        "tail_integrated_directly_to_infinity": True,
    }


def validate_point(path: Path, t_value: float) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("matrix_model_present") is not False:
        raise ValueError(f"{path}: point is not target-blind")
    if abs(float(data["t"]) - float(t_value)) > 1.0e-12:
        raise ValueError(f"{path}: t mismatch")
    design = data["design"]
    for key, expected in _expected_design().items():
        actual = design.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1.0e-12:
                raise ValueError(f"{path}: design mismatch for {key}")
        elif actual != expected:
            raise ValueError(f"{path}: design mismatch for {key}")
    mean = _complex(data["mean"])
    standard_error = _complex(data["rqmc_standard_error"])
    if not all(
        value == value
        for value in (mean.real, mean.imag, standard_error.real, standard_error.imag)
    ):
        raise ValueError(f"{path}: non-finite estimator")
    return data


def point_output(output_dir: Path, t_value: float) -> Path:
    return output_dir / f"{_tag(t_value)}.json"


def point_command(
    output_dir: Path,
    t_value: float,
    *,
    coefficient_workers: int,
) -> list[str]:
    tag = _tag(t_value)
    return [
        "nice",
        "-n",
        "5",
        sys.executable,
        "plumbing/smoke_genus1_three_point_channel_atlas.py",
        "--t",
        f"{t_value:.2f}",
        "--momentum-order",
        "8",
        "--necklace-order",
        "8",
        "--necklace-low-order",
        "3",
        "--necklace-backend",
        "regulated-h-recursion",
        "--c-regulator",
        "0.025",
        "--coefficient-workers",
        str(int(coefficient_workers)),
        "--necklace-qhat-threshold",
        "0.30",
        "--necklace-second-qhat-threshold",
        "0.07",
        "--ope-order",
        "6",
        "--ope-loop-order",
        "4",
        "--ope-low-loop-order",
        "2",
        "--sobol-power",
        "8",
        "--replicates",
        "4",
        "--seed",
        "17051301",
        "--alpha",
        "0.3",
        "--patch-epsilon",
        "0.15",
        "--triple-patch-epsilon",
        "0.10",
        "--tail-proposal-exponent",
        "1.5",
        "--dps",
        "24",
        "--bank-cache",
        str(output_dir / f"{tag}_ope_banks.npz"),
        "--necklace-bank-cache",
        str(output_dir / f"{tag}_necklace_banks.npz"),
        "--prepare-necklace-first",
        "--output",
        str(point_output(output_dir, t_value)),
    ]


def _run_point(
    output_dir: Path,
    t_value: float,
    *,
    coefficient_workers: int,
) -> Path:
    output_path = point_output(output_dir, t_value)
    if output_path.is_file():
        validate_point(output_path, t_value)
        print(f"reusing completed {_tag(t_value)}: {output_path}", flush=True)
        return output_path

    command = point_command(
        output_dir,
        t_value,
        coefficient_workers=coefficient_workers,
    )
    log_path = output_dir / f"{_tag(t_value)}.log"
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND " + " ".join(command) + "\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{_tag(t_value)} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    validate_point(output_path, t_value)
    print(f"completed {_tag(t_value)}: {output_path}", flush=True)
    return output_path


def _legacy_rows(path: Path) -> dict[float, dict[str, object]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[float, dict[str, object]] = {}
    for t_value, point in zip(data["t_values"], data["points"]):
        rows[round(float(t_value), 12)] = {
            "I_1,3": point["I_1,3"],
            "rqmc_standard_error": point["rqmc_standard_error"],
            "role": "legacy cross-design diagnostic only; not combined with estimator",
        }
    return rows


def write_manifest(
    output_dir: Path,
    sources: dict[float, Path],
    *,
    reused_t075: Path,
    legacy_scan: Path,
    complete: bool,
) -> Path:
    legacy = _legacy_rows(legacy_scan)
    rows = []
    for t_value in T_VALUES:
        key = round(t_value, 12)
        source = sources.get(key)
        if source is None:
            rows.append({"t": t_value, "status": "pending"})
            continue
        data = validate_point(source, t_value)
        rows.append(
            {
                "t": t_value,
                "status": "complete",
                "reuse_kind": (
                    "pre-existing exact-design point"
                    if source.resolve() == reused_t075.resolve()
                    else "new exact-design point"
                ),
                "source_path": str(source),
                "source_sha256": _sha256(source),
                "I_1,3": data["mean"],
                "rqmc_standard_error": data["rqmc_standard_error"],
                "replicate_values": data["replicate_values"],
                "channel_components": data["channel_components"],
                "atlas_diagnostics": data["atlas_diagnostics"],
                "necklace_diagnostics": data["necklace_diagnostics"],
                "legacy_prior_data": legacy.get(key),
            }
        )
    payload = {
        "calculation": "blind ten-point h-dominant genus-one three-point scan",
        "matrix_model_present": False,
        "complete": bool(complete),
        "t_values": list(T_VALUES),
        "design": _expected_design(),
        "reuse_policy": (
            "reuse only exact-design completed estimators; retain the earlier "
            "p12 scan solely as a cross-design diagnostic"
        ),
        "points": rows,
    }
    path = output_dir / "worldsheet_scan_manifest.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    csv_path = output_dir / "worldsheet_t_dependence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "minus_imag_I_1_3", "rqmc_standard_error", "status"])
        for row in rows:
            if row["status"] == "complete":
                writer.writerow(
                    [
                        row["t"],
                        -float(row["I_1,3"]["imag"]),
                        float(row["rqmc_standard_error"]["imag"]),
                        row["reuse_kind"],
                    ]
                )
            else:
                writer.writerow([row["t"], "", "", "pending"])
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--reused-t075", type=Path, default=DEFAULT_REUSED_T075)
    result.add_argument("--legacy-scan", type=Path, default=DEFAULT_LEGACY_SCAN)
    result.add_argument("--max-concurrent", type=int, default=2)
    result.add_argument("--coefficient-workers", type=int, default=2)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.max_concurrent < 1 or args.coefficient_workers < 1:
        raise ValueError("worker counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_point(args.reused_t075, 0.75)

    sources: dict[float, Path] = {
        round(0.75, 12): args.reused_t075,
    }
    for t_value in T_VALUES:
        candidate = point_output(args.output_dir, t_value)
        if candidate.is_file():
            validate_point(candidate, t_value)
            sources[round(t_value, 12)] = candidate
    write_manifest(
        args.output_dir,
        sources,
        reused_t075=args.reused_t075,
        legacy_scan=args.legacy_scan,
        complete=len(sources) == len(T_VALUES),
    )

    pending = [
        t_value
        for t_value in T_VALUES
        if round(t_value, 12) not in sources
    ]
    print(
        f"scan status: {len(sources)}/{len(T_VALUES)} complete; "
        f"{len(pending)} pending",
        flush=True,
    )
    if args.dry_run:
        for t_value in pending:
            print(
                "DRY " + " ".join(
                    point_command(
                        args.output_dir,
                        t_value,
                        coefficient_workers=args.coefficient_workers,
                    )
                ),
                flush=True,
            )
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        future_to_t = {
            executor.submit(
                _run_point,
                args.output_dir,
                t_value,
                coefficient_workers=args.coefficient_workers,
            ): t_value
            for t_value in pending
        }
        for future in as_completed(future_to_t):
            t_value = future_to_t[future]
            try:
                source = future.result()
                sources[round(t_value, 12)] = source
            except Exception as error:
                failures.append(f"{_tag(t_value)}: {error}")
                print(f"FAILED {failures[-1]}", file=sys.stderr, flush=True)
            write_manifest(
                args.output_dir,
                sources,
                reused_t075=args.reused_t075,
                legacy_scan=args.legacy_scan,
                complete=len(sources) == len(T_VALUES),
            )

    if failures:
        raise RuntimeError("; ".join(failures))
    manifest = write_manifest(
        args.output_dir,
        sources,
        reused_t075=args.reused_t075,
        legacy_scan=args.legacy_scan,
        complete=True,
    )
    print(f"completed blind scan: {manifest}", flush=True)


if __name__ == "__main__":
    main()
