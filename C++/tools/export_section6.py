#!/usr/bin/env python3
"""Create a source archive for the Section 6 C++ code and its test commands.

No binaries, manuscript files, virtual environments or numerical caches are
included. A SHA-256 manifest records every packaged input. Frozen momentum
bundles are supplied separately to the optional partition-function frontend.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "section6_superconformal_blocks"


def files():
    paths = set()
    for directory in ("C++/include/ramond", "C++/include/scblocks", "C++/src"):
        paths.update(p for p in (ROOT/directory).iterdir() if p.suffix in (".hpp", ".cpp", ".inc"))
    relative = [
        "C++/Makefile", "C++/README.md", "C++/SECTION6.md", "C++/PARTITION.md", "C++/PARTITION_NORMALIZATION.md",
        "C++/drivers/partition_main.cpp", "C++/tests/partition_sewing.cpp", "C++/tests/partition_blocks.cpp",
        "C++/tests/partition_spin_geometry.py",
        "C++/drivers/pbw_main.cpp", "C++/drivers/graph_main.cpp", "C++/drivers/graph_pbw_check.inc",
        "C++/drivers/graph_dv.inc",
        "C++/tools/resummed_main.cpp", "C++/tools/section6.py", "C++/tools/export_section6.py",
        "C++/tools/summarize_section6.py",
        "C++/tests/graph_ns_forward.cpp", "C++/tests/graph_schottky.cpp", "C++/tools/section6_verify_archive.py",
        "C++/tools/section6_partition.py", "C++/tools/prepare_partition_inputs.py",
        "C++/tools/estimate_tetrahedron_box.py",
        "Code/full_ramond_block_runtime/nsrr_resummed_backend.py",
        "Code/full_ramond_block_runtime/GLOBAL_RESUMMATION.md",
        
         
         
        
        "Code/c_Recursion/generic_super_liouville_structure_constants.py",
        "Code/c_Recursion/super_liouville_structure_constants.py",
        "Code/c_Recursion/test_nsrr_identity_normalization.py",
        "Code/genus_2_cross_channel/liouville_torus.py",
        "C++/results/partition_normalization_resolution_2026-09-24/comparison.json",
        "C++/results/partition_normalization_resolution_2026-09-24/identity_residue_tests.json",
        "C++/results/partition_normalization_resolution_2026-09-24/validation.json",
        "C++/results/partition_normalization_resolution_2026-09-24/source/run.json",
        "C++/results/partition_normalization_resolution_2026-09-24/target/run.json",
        "C++/experiments/coherent_conventions_2026-09-22/README.md",
        "C++/experiments/mercedes_all_ns_level10_2026-09-16/results_L10.json",
        "C++/experiments/mercedes_all_ns_level10_2026-09-16/timing_dv_L10.json",
        "C++/experiments/mercedes_all_ns_level10_2026-09-16/timing_ns_L10.json",
        "C++/experiments/mercedes_all_ns_level10_2026-09-16/manifest_L10.json",
        "C++/experiments/total_level6_2026-09-16/mercedes_L6_results.json",
        "Data Set/nsrr_bilinear_quadrature_20260915/summary.json",
        "Data Set/nsrr_bilinear_quadrature_20260915/final_result.json",
        "Data Set/nsrr_bilinear_quadrature_20260915/FINAL_REPORT.md",
        "Data Set/nsrr_provisional_factor4_20260916/README.md",
        "Data Set/nsrr_provisional_factor4_20260916/result.json",
    ]
    relative += [f"C++/results/partition_paper_pairing_2026-09-24/{name}" for name in
                 ("comparison.json", "validation.json", "spin_geometry_test.json", "source/run.json", "target/run.json")]
    # Geometry test dependencies are diagnostics; production remains native C++.
    relative += ["Code/genus_2/spin_structure.py", "Code/genus_2/fixed_spin_free_plumbing.py",
                 "Code/genus_2_cross_channel/free_majorana_pair_of_pants.py",
                 "Code/genus_2_cross_channel/free_boson_pair_of_pants.py",
                 "Code/genus_2_cross_channel/free_boson_plumbing.py",
                 "Code/genus_2/physical_free_plumbing_resummation.py",
                 "Code/genus_2_cross_channel/genus2_vacuum_blocks.py", "Code/genus_2_cross_channel/plumbing_algorithms.py"]
    paths.update(ROOT/p for p in relative)
    # Validation reports, not cached inputs or large coefficient tables.
    results = ROOT/"C++/results/section6_2026-09-23"
    for pattern in ("**/run_manifest.json", "**/summary.json", "**/metadata.json", "*comparison.json",
                    "partition_audit.json", "validation_summary.json", "archive_verification.json",
                    "ns_forward_check.json", "ns_forward_seed_optimized_check.json", "ns_forward_archive_verification.json", "ns_forward_implementation.json",
                    "schottky_check.json", "theta_ns10_comparison.json", "precision_refinement.json", "README.md", "graph_dv_comparison.json"):
        paths.update(results.glob(pattern))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = files()
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=f"{PREFIX}/{p.relative_to(ROOT)}", recursive=False)
        generated = {
            "SOURCE_MANIFEST.json": json.dumps(manifest, indent=2) + "\n",
            "README.md": "# Higher-genus superconformal blocks\n\n"
                "Build: `make -C C++ section6`. Start with [C++/SECTION6.md](C++/SECTION6.md).\n\n"
                "The numerical cores are C++17. Python launches and compares them.\n"
                "The tetrahedron tests use total levels 5 and 10; genus-two tests\n"
                "use independent-edge levels 5 and 10. Full coefficients are generated\n"
                "by the test commands. Packaged reports identify completed validations.\n\n"
                "The native nonchiral partition comparison uses identity-normalized\n"
                "NSRR coefficients and the literal paper blocks, including their\n"
                "Koszul sign; see C++/PARTITION.md for the current result. Its supplied\n"
                "momentum/geometry bundles remain separate inputs.\n\n"
                "Historical result paths in reports identify provenance in the full\n"
                "working repository. This source bundle excludes large coefficient\n"
                "archives and unrelated historical tests. Use the Section 6 commands\n"
                "rather than the older Makefile check targets.\n",
            "requirements-partition.txt": "numpy\nscipy\nmpmath\n",
        }
        for name, text in generated.items():
            payload = text.encode()
            info = tarfile.TarInfo(f"{PREFIX}/{name}")
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    temporary.replace(args.output)
    print(json.dumps({"archive": str(args.output), "source_files": len(manifest),
                      "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
