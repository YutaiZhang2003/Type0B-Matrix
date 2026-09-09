"""Verify the user's protected numerical kernels have not been edited."""
import hashlib
import json
from pathlib import Path
import unittest


class CheckedKernelBoundaryTests(unittest.TestCase):
    def test_protected_kernel_hashes(self):
        here=Path(__file__).resolve().parent
        root=here.parents[1]
        manifest=json.loads((here/"nsrr_checked_kernel_manifest.json").read_text())
        for relative,expected in manifest["sha256"].items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(),expected)


if __name__ == "__main__":
    unittest.main()
