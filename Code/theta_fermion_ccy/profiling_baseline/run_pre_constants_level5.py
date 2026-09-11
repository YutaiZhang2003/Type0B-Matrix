"""Timing-only rerun of the archived pre-constant-reuse implementation.

The full production pipeline is unchanged. Only compute_target is loaded
from its archived source, and the manifest records that actual source path.
No coefficient comparison is performed by this driver.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
archive = HERE / "compute_target_pre_constants.py"
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("compute_target", archive)
module = importlib.util.module_from_spec(spec)
sys.modules["compute_target"] = module
spec.loader.exec_module(module)

import pipeline

original_hashes = pipeline.source_hashes


def archived_source_hashes():
    result = original_hashes()
    del result["Code/ramond_branching_recursion/compute_target.py"]
    result[str(archive.relative_to(ROOT))] = hashlib.sha256(archive.read_bytes()).hexdigest()
    result[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return result


pipeline.source_hashes = archived_source_hashes

if __name__ == "__main__":
    pipeline.main()
