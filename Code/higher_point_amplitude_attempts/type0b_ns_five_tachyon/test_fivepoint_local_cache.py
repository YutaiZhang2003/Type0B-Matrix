"""Storage location must not change coefficients or lose committed updates."""
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from fivepoint_local_cache import staged_coefficient_cache
from fivepoint_runtime import CoefficientStore


class LocalCacheTests(unittest.TestCase):
    def test_copy_in_is_exact_and_copy_back_keeps_new_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); persistent = root / "persistent" / "tables.sqlite"
            original = CoefficientStore(persistent)
            value = [[[0, 0], "1.23456789012345678901234567890123456789", "-0.125"]]
            original.put("old", value); original.close()
            with staged_coefficient_cache(persistent, root / "scratch") as (local, record):
                self.assertNotEqual(local, persistent)
                store = CoefficientStore(local)
                self.assertEqual(store.get("old"), value)
                store.put("new", [["2.0"]]); store.close()
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    CoefficientStore(persistent)
            self.assertTrue(record["copied_back"])
            self.assertFalse(local.exists())
            restored = CoefficientStore(persistent)
            self.assertEqual(restored.get("old"), value)
            self.assertEqual(restored.get("new"), [["2.0"]]); restored.close()

    def test_exception_and_termination_preserve_committed_tables(self):
        for terminate in (False, True):
            with self.subTest(terminate=terminate), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); persistent=root/"tables.sqlite"
                original_handler=signal.getsignal(signal.SIGTERM)
                with self.assertRaises(SystemExit if terminate else RuntimeError):
                    with staged_coefficient_cache(persistent, root/"scratch") as (local, record):
                        store=CoefficientStore(local)
                        store.put("committed", [42]); store.close()
                        if terminate:
                            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                        raise RuntimeError("interrupted evaluator")
                self.assertEqual(signal.getsignal(signal.SIGTERM), original_handler)
                restored=CoefficientStore(persistent)
                self.assertEqual(restored.get("committed"), [42]); restored.close()

    def test_failed_copy_back_keeps_previous_database_and_recovery_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); persistent=root/"tables.sqlite"
            original=CoefficientStore(persistent); original.put("old", [1]); original.close()
            from fivepoint_local_cache import _snapshot
            def snapshot(source, destination):
                if destination == persistent:
                    raise OSError("simulated storage failure")
                return _snapshot(source, destination)
            with mock.patch("fivepoint_local_cache._snapshot", side_effect=snapshot):
                with self.assertRaisesRegex(OSError, "storage failure"):
                    with staged_coefficient_cache(persistent, root/"scratch") as (local, record):
                        store=CoefficientStore(local); store.put("new", [2]); store.close()
            self.assertTrue(local.exists())
            restored=CoefficientStore(persistent)
            self.assertEqual(restored.get("old"), [1]); self.assertIsNone(restored.get("new")); restored.close()


if __name__ == "__main__":
    unittest.main()
