#!/usr/bin/env python3
"""Build a hashed, independently runnable c=1 string 1->n code bundle.

The bundle is intentionally a copy: source files in the working repository are
never moved or rewritten.  Local Python imports are followed transitively so
the copied computation does not rely on modules outside the bundle.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUMBING = ROOT / "plumbing"
BUNDLE_NAME = "c1_one_to_n_amplitudes_standalone_20260825_v1"
DEFAULT_OUTPUT = ROOT / "standalone" / BUNDLE_NAME

# These are the computation families in the two current machine notes.  The
# closure below adds shared Liouville, block, and numerical dependencies.
PRIMARY_PATTERNS = (
    "audit_genus0_one_to_two_amplitude*.py",
    "audit_c1_sphere_topology_normalization*.py",
    "ccy_sphere_*.py",
    "sphere_four_point_*.py",
    "sphere_five_point_*.py",
    "sphere_six_point_*.py",
    "merge_sphere_four_point_*.py",
    "merge_sphere_six_point_*.py",
    "*genus1_two_point*.py",
    "*genus1_three_point*.py",
    "torus_two_point_blocks*.py",
    "torus_three_point_blocks*.py",
)

DOCUMENTATION = (
    "plumbing/sphere_one_to_n_amplitudes_machine_note.md",
    "plumbing/sphere_one_to_n_amplitudes_machine_note.tex",
    "plumbing/torus_one_to_n_amplitudes_note.tex",
    "plumbing/sphere_five_point_numerical_recipe.md",
    "plumbing/sphere_n_point_momentum_integration.md",
    "plumbing/genus1_two_point_hybrid_recursion_report.md",
    "plumbing/genus1_two_point_sewing_normalization.md",
    "plumbing/genus1_two_point_momentum_sampling_report_v1.md",
    "plumbing/genus1_two_point_threshold_laplace_report_v1.md",
    "output/pdf/sphere_one_to_n_amplitudes_note.pdf",
    "output/pdf/torus_one_to_n_amplitudes_note.pdf",
)

CURRENT_CLUSTER_AND_CONFIG = (
    "plumbing/config/sphere_six_point_1to5_cannon_blind30_3h_v1.json",
    "plumbing/cluster/sphere_six_point_1to5_blind_array.slurm",
    "plumbing/cluster/sphere_six_point_1to5_blind_assemble.slurm",
    "plumbing/cluster/sphere_six_point_1to5_blind_freeze.slurm",
    "plumbing/cluster/stage_submit_sphere_six_point_1to5_blind30_3h.sh",
    "plumbing/cluster/submit_sphere_six_point_1to5_blind30_boundary_fix_v2.sh",
    "plumbing/cluster/genus1_three_point_hdominant_array.slurm",
    "plumbing/cluster/genus1_three_point_hdominant_assemble.slurm",
    "plumbing/cluster/stage_submit_genus1_three_point_hdominant_scan.sh",
)

# Sphere result trees are small and some current fits verify hashes inherited
# from their earlier blind cohorts, so preserve the complete process trees.
# For the torus calculations, include only the current audited datasets plus
# the exact-design t=0.75 shard reused by the three-point scan.
RESULT_PATHS = (
    "plumbing/results/genus0_one_to_two_amplitude",
    "plumbing/results/sphere_four_point_1to3",
    "plumbing/results/sphere_five_point_1to4",
    "plumbing/results/sphere_six_point_1to5",
    "plumbing/results/genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2",
    "plumbing/results/genus1_two_point_worldsheet/direct_current_t045_n256_v1.json",
    "plumbing/results/genus1_three_point_worldsheet/hdominant_scan10_p8_h8l3_q030_007_n256_r4_v1",
    "plumbing/results/genus1_three_point_worldsheet/channel_atlas_hdominant_t075_p8_h8l3_q030_007_n256_r4_v1.json",
)

CORE_CHECKS = (
    "plumbing/audit_genus0_one_to_two_amplitude_checks.py",
    "plumbing/liouville_torus_checks.py",
    "plumbing/ccy_sphere_four_point_checks.py",
    "plumbing/ccy_sphere_five_point_checks.py",
    "plumbing/ccy_sphere_six_point_checks.py",
    "plumbing/ccy_sphere_six_point_star_checks.py",
    "plumbing/sphere_five_point_liouville_checks.py",
    "plumbing/sphere_six_point_atlas_checks.py",
    "plumbing/torus_two_point_blocks_checks.py",
    "plumbing/genus1_two_point_worldsheet_checks.py",
    "plumbing/genus1_three_point_worldsheet_checks.py",
    "plumbing/genus1_three_point_channel_atlas_checks.py",
    "plumbing/genus1_three_point_matrix_comparison_checks.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_imports(path: Path) -> set[str]:
    """Return top-level import names that may resolve to plumbing/*.py."""
    modules: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            modules.update(item.name.split(".")[0] for item in node.names)
    return modules


def dependency_closure() -> tuple[set[Path], set[Path]]:
    primary: set[Path] = set()
    for pattern in PRIMARY_PATTERNS:
        primary.update(path for path in PLUMBING.glob(pattern) if path.is_file())

    closure: set[Path] = set()
    queue = sorted(primary)
    while queue:
        path = queue.pop(0)
        if path in closure:
            continue
        closure.add(path)
        if path.suffix == ".py":
            for module in local_imports(path):
                candidate = PLUMBING / f"{module}.py"
                if candidate.is_file() and candidate not in closure:
                    queue.append(candidate)
            # A dependency's regression check is useful even when it is not
            # imported.  Put it back through the same queue so its own local
            # imports are also followed.
            check = path.with_name(f"{path.stem}_checks.py")
            if check.is_file() and check not in closure:
                queue.append(check)
        markdown = path.with_suffix(".md")
        if markdown.is_file() and markdown not in closure:
            queue.append(markdown)
    return primary, closure


def add_path(files: set[Path], path: Path) -> None:
    if path.is_file():
        files.add(path)
    elif path.is_dir():
        files.update(item for item in path.rglob("*") if item.is_file())


def gather_files() -> tuple[set[Path], set[Path]]:
    primary, files = dependency_closure()
    files.add(Path(__file__).resolve())
    for relative in DOCUMENTATION + CURRENT_CLUSTER_AND_CONFIG + RESULT_PATHS:
        add_path(files, ROOT / relative)
    return primary, files


def category(path: Path, primary: set[Path]) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if path in primary:
        if "sphere" in path.name or "genus0" in path.name:
            return "genus_zero_primary_code"
        return "genus_one_primary_code"
    if relative.startswith("plumbing/results/"):
        return "frozen_numerical_artifact"
    if relative.startswith("plumbing/cluster/") or relative.startswith("plumbing/config/"):
        return "cluster_or_configuration"
    if path.suffix in {".md", ".tex", ".pdf"}:
        return "documentation"
    if "checks" in path.stem:
        return "check_code"
    if path.suffix == ".py":
        return "transitive_code_dependency"
    return "supporting_artifact"


def copy_files(bundle: Path, files: set[Path]) -> None:
    for source in sorted(files):
        relative = source.relative_to(ROOT)
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_requirements(bundle: Path) -> None:
    (bundle / "requirements.txt").write_text(
        "# Python >=3.10. Minimum package versions used by the numerical code.\n"
        "numpy>=1.24\n"
        "scipy>=1.10\n"
        "mpmath>=1.2\n"
        "matplotlib>=3.7\n"
        "pillow>=9.0\n"
        "sympy>=1.11\n"
    )


def write_readme(bundle: Path, primary_count: int, code_count: int) -> None:
    readme = f"""# Standalone c=1 string 1->n amplitude computations

This directory is a self-contained copy of the current genus-zero and
genus-one `1 -> n` worldsheet computations.  It preserves the target-blind
worldsheet/post-freeze comparison boundary used in the machine notes.

It contains {primary_count} process-specific Python files and {code_count}
Python files after following every local import transitively.  Nothing in the
bundle imports code from the parent StringMC checkout.

## Current calculations

| Genus | Process | Current packaged result |
|---|---|---|
| 0 | sphere `1 -> 2` | normalization anchor and topology audit |
| 0 | sphere `1 -> 3` | blind thirty-point imaginary-ray scan and affine fit |
| 0 | sphere `1 -> 4` | blind thirty-point scan, quadratic fit, and residue continuation |
| 0 | sphere `1 -> 5` | blind thirty-point paired order-eight estimator and cubic fit |
| 1 | torus `1 -> 1` | frozen fifty-point hybrid-recursion scan and BRY fit |
| 1 | torus `1 -> 2` | frozen ten-point h-recursion-dominant channel-atlas scan |

## Install and verify

Use Python 3.10 or newer (Python 3.11 or 3.12 is recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_core_checks.sh
python verify_manifest.py
```

`run_core_checks.sh` can use another interpreter through
`PYTHON_BIN=/path/to/python`.  The checks run from the bundle root with
`PYTHONPATH` restricted to this bundle's `plumbing/` directory.

## Reproduction

See [RUNBOOK.md](RUNBOOK.md) for the worldsheet-only stages, frozen fits,
post-freeze comparisons, representative smoke commands, and cluster entry
points.  Production scans are intentionally not launched by the check suite:
their sample counts range from hours to a cluster campaign.

## Integrity

`MANIFEST.json` gives a SHA-256 digest and category for every packaged file.
`verify_manifest.py` verifies those hashes.  `CHECK_RESULTS.txt` records the
exact core checks run while this snapshot was built.
"""
    (bundle / "README.md").write_text(readme)


def write_runbook(bundle: Path) -> None:
    runbook = r"""# Reproduction runbook

Run every command below from the bundle root.  The frozen data are included so
the inexpensive analysis and comparison stages do not require rerunning the
Monte Carlo integrations.

## 1. Sphere normalization and block kernels

```bash
python plumbing/audit_genus0_one_to_two_amplitude.py
python plumbing/audit_genus0_one_to_two_amplitude_checks.py
python plumbing/ccy_sphere_four_point_checks.py
python plumbing/ccy_sphere_five_point_checks.py
python plumbing/ccy_sphere_six_point_checks.py
python plumbing/ccy_sphere_six_point_star_checks.py
```

## 2. Sphere 1 -> 3

The expensive target-blind production driver is
`plumbing/sphere_four_point_worldsheet_scan.py`.  With the packaged frozen
inputs, rerun the target-free fit and only then its separate comparison:

```bash
python plumbing/sphere_four_point_imaginary_ray_fit.py
python plumbing/sphere_four_point_30point_matrix_comparison.py
```

## 3. Sphere 1 -> 4

The worldsheet kernel is `plumbing/sphere_five_point_equal_energy.py`; the
thirty-point target-blind extension is
`plumbing/sphere_five_point_30point_worldsheet_extension.py`.  Reproduce the
frozen fit and later comparison with:

```bash
python plumbing/sphere_five_point_30point_worldsheet_fit.py
python plumbing/sphere_five_point_30point_audit_summary.py
python plumbing/sphere_five_point_30point_matrix_comparison.py
```

## 4. Sphere 1 -> 5

The production driver is `plumbing/sphere_six_point_worldsheet_scan.py` and
the completed Cannon orchestration is in
`plumbing/sphere_six_point_cannon_blind.py`.  The current paired order-eight
analysis and its deliberately separate comparison are:

```bash
python plumbing/sphere_six_point_order8_current_fit.py
python plumbing/sphere_six_point_order8_current_comparison.py
```

## 5. Torus 1 -> 1

Run the kernel/block checks first:

```bash
python plumbing/torus_two_point_blocks_checks.py
python plumbing/genus1_two_point_worldsheet_checks.py
```

The target-blind production scan is
`plumbing/run_genus1_two_point_imaginary_scan.py`.  Reproduce the post-freeze
BRY fit from the packaged fifty-point scan with:

```bash
python plumbing/fit_genus1_two_point_bry_scan.py \
  --scan-dir plumbing/results/genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2
```

## 6. Torus 1 -> 2

The full chain is the three-point block, worldsheet kernel, channel atlas,
stratified bulk/direct-tail integrator, and h-dominant scan driver:

```text
torus_three_point_blocks.py
genus1_three_point_worldsheet.py
genus1_three_point_channel_atlas.py
smoke_genus1_three_point_channel_atlas.py
run_genus1_three_point_hdominant_scan.py
```

Run the kernel checks with:

```bash
python plumbing/genus1_three_point_worldsheet_checks.py
python plumbing/genus1_three_point_channel_atlas_checks.py
```

Revalidate the blind ten-point freeze, recompute the target-free shape fit,
freeze it, and only afterward regenerate the BRY-normalized comparison:

```bash
python plumbing/analyze_genus1_three_point_hdominant_scan.py \
  --scan-dir plumbing/results/genus1_three_point_worldsheet/hdominant_scan10_p8_h8l3_q030_007_n256_r4_v1 \
  --reused-t075 plumbing/results/genus1_three_point_worldsheet/channel_atlas_hdominant_t075_p8_h8l3_q030_007_n256_r4_v1.json
```

This last command requires Matplotlib to regenerate the comparison plot.  It
checks every blind input hash before writing the target-free analysis.

## Cluster scripts

The packaged Slurm files preserve the exact current Cannon orchestration, but
site paths and account/partition settings must be adapted before submission on
another cluster.  They are not needed for local kernel checks or postprocessing.
"""
    (bundle / "RUNBOOK.md").write_text(runbook)


def write_check_runner(bundle: Path) -> None:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        'BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"',
        'export PYTHONPATH="${BUNDLE_ROOT}/plumbing:${BUNDLE_ROOT}"',
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/stringmc-matplotlib}"',
        'mkdir -p "${MPLCONFIGDIR}"',
        'cd "${BUNDLE_ROOT}"',
    ]
    for path in CORE_CHECKS:
        lines.extend((f'echo "RUN {path}"', f'"${{PYTHON_BIN}}" "{path}"'))
    lines.append('echo "ALL CORE CHECKS PASSED"')
    path = bundle / "run_core_checks.sh"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def write_manifest_verifier(bundle: Path) -> None:
    source = '''#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file():
        raise SystemExit(f"missing file: {item['path']}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"checksum mismatch: {item['path']}")
print(f"verified {len(manifest['files'])} files")
'''
    path = bundle / "verify_manifest.py"
    path.write_text(source)
    path.chmod(0o755)


def environment_record() -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "mpmath", "matplotlib", "PIL", "sympy"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed in build interpreter"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def run_checks(bundle: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHON_BIN"] = sys.executable
    environment["MPLCONFIGDIR"] = "/tmp/stringmc-matplotlib"
    result = subprocess.run(
        ["bash", "run_core_checks.sh"],
        cwd=bundle,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    report = (
        f"exit_code={result.returncode}\n"
        f"python={sys.executable}\n\n"
        f"{result.stdout}"
    )
    (bundle / "CHECK_RESULTS.txt").write_text(report)
    if result.returncode != 0:
        raise RuntimeError("core bundle checks failed; see CHECK_RESULTS.txt")


def write_manifest(bundle: Path, primary: set[Path]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    primary_relatives = {path.relative_to(ROOT).as_posix() for path in primary}
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.json":
            continue
        relative = path.relative_to(bundle).as_posix()
        source_path = ROOT / relative
        if relative in primary_relatives:
            file_category = category(source_path, primary)
        elif source_path.exists():
            file_category = category(source_path, primary)
        elif relative == "CHECK_RESULTS.txt":
            file_category = "verification_record"
        else:
            file_category = "bundle_metadata"
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": file_category,
            }
        )
    manifest = {
        "status": "c1_one_to_n_standalone_bundle_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_version": BUNDLE_NAME,
        "file_count": len(records),
        "python_file_count": sum(item["path"].endswith(".py") for item in records),
        "categories": {
            name: sum(item["category"] == name for item in records)
            for name in sorted({str(item["category"]) for item in records})
        },
        "files": records,
    }
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")

    primary, files = gather_files()
    output.mkdir(parents=True)
    copy_files(output, files)
    write_requirements(output)
    write_readme(output, len(primary), sum(path.suffix == ".py" for path in files))
    write_runbook(output)
    write_check_runner(output)
    write_manifest_verifier(output)
    (output / "ENVIRONMENT.json").write_text(
        json.dumps(environment_record(), indent=2) + "\n"
    )
    if arguments.skip_checks:
        (output / "CHECK_RESULTS.txt").write_text(
            "exit_code=not_run\nchecks were skipped by explicit builder option\n"
        )
    else:
        run_checks(output)
    manifest = write_manifest(output, primary)

    archive: Path | None = None
    if not arguments.no_archive:
        archive = Path(
            shutil.make_archive(
                str(output), "zip", root_dir=output.parent, base_dir=output.name
            )
        )
    print(
        json.dumps(
            {
                "bundle": str(output),
                "archive": str(archive) if archive else None,
                "file_count": manifest["file_count"],
                "python_file_count": manifest["python_file_count"],
                "archive_sha256": sha256_file(archive) if archive else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
