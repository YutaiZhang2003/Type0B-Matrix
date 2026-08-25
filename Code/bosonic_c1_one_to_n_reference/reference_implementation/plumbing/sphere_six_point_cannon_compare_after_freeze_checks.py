#!/usr/bin/env python3
"""Checks for the post-freeze-only sphere 1->5 comparison path."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import sphere_six_point_cannon_blind as campaign
    import sphere_six_point_cannon_blind_checks as blind_checks
    import sphere_six_point_cannon_blind30_3h_checks as blind30_checks
    import sphere_six_point_cannon_compare_after_freeze as comparison
except ImportError:  # pragma: no cover
    from plumbing import sphere_six_point_cannon_blind as campaign
    from plumbing import sphere_six_point_cannon_blind_checks as blind_checks
    from plumbing import sphere_six_point_cannon_blind30_3h_checks as blind30_checks
    from plumbing import sphere_six_point_cannon_compare_after_freeze as comparison


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plumbing" / "sphere_six_point_cannon_compare_after_freeze.py"


def check_campaign(checks: object, expected_point_count: int) -> None:
    with tempfile.TemporaryDirectory(prefix="sphere6_postfreeze_check_") as temporary:
        run_dir = Path(temporary)
        design = run_dir / "design"
        shards = run_dir / "shards"
        assembled = run_dir / "assembled"
        campaign.prepare_design(checks.CONFIG, design)
        rows = campaign.read_manifest(design / "manifest.csv")
        shards.mkdir()
        for row in rows:
            payload = (
                checks._synthetic_production(row)
                if row["task_kind"] == "production"
                else checks._synthetic_systematics(row)
            )
            campaign.atomic_write_json(
                shards / f"task_{int(row['task_id']):04d}.json", payload
            )
        campaign.assemble_worldsheet(
            checks.CONFIG, design / "manifest.csv", shards, assembled
        )
        campaign.validate_and_freeze(
            checks.CONFIG, design / "manifest.csv", assembled
        )
        scan, _checksum = comparison.verified_frozen_scan(run_dir)
        assert scan["point_count"] == expected_point_count
        output_dir = run_dir / "comparison"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads((output_dir / "matrix_model_comparison.json").read_text())
        assert result["comparison_performed_after_freeze"] is True
        assert result["point_count"] == expected_point_count
        assert (
            output_dir
            / f"sphere_one_to_five_cannon_blind{expected_point_count}_comparison.png"
        ).exists()

        frozen_path = assembled / "worldsheet_scan_frozen.json"
        frozen = json.loads(frozen_path.read_text())
        frozen["points"][0]["Q5_worldsheet"]["real"] += 1.0
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
        try:
            comparison.verified_frozen_scan(run_dir)
        except RuntimeError as error:
            assert "checksum" in str(error)
        else:  # pragma: no cover
            raise AssertionError("post-freeze tampering was not detected")
    print(
        f"post-freeze {expected_point_count}-point comparison barrier and outputs passed"
    )


def main() -> None:
    check_campaign(blind_checks, 50)
    check_campaign(blind30_checks, 30)
    print("post-freeze comparison tamper detection passed for both campaigns")


if __name__ == "__main__":
    main()
