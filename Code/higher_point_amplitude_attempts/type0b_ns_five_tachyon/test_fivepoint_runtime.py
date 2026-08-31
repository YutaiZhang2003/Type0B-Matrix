"""Numerical equivalence, memory ownership, persistence, and restart tests."""

import cmath
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mpmath

from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from fivepoint_runtime import BoundedLRU, CoefficientStore, CompactCBlock, SampleCheckpoint
import type0b_ns_five_tachyon as fivepoint


def block_arguments():
    return dict(central_charge=13.5,
                external_weights=(0.53, 0.61, 0.62, 0.63, 0.64),
                internal_weights=(0.73, 0.81), vertex_sectors=(0, 0, 0),
                working_precision=45)


class FivePointRuntimeTests(unittest.TestCase):
    def test_compact_series_preserves_parities_masks_and_log_lifts(self):
        for sectors in ((0, 0, 0), (1, 0, 1)):
            args = {**block_arguments(), "vertex_sectors": sectors}
            old = NSSphereLinearCRecursion(**args)
            new = CompactCBlock(**args)
            for q, minima, total in (
                ((0.1 + 0.02j, 0.2 - 0.03j), (0, 0), 16),
                ((1e-5, 0.2), (2, 0), 12),
                ((-0.1 + 0j, 0.25 + 0.02j), (0, 2), 16),
            ):
                logs = tuple(cmath.log(z).conjugate() for z in q)
                kwargs = dict(max_total_twice_level=total, minimum_twice_levels=minima,
                              q_log_values=logs)
                expected = old.series_value(q, (8, 8), **kwargs)
                observed = new.series_value(q, (8, 8), **kwargs)
                with mpmath.workdps(45):
                    self.assertLess(abs(expected - observed), mpmath.mpf("1e-40"))
                self.assertFalse(new._coefficient_cache)
                self.assertLessEqual(len(new.final_coefficients), 25)
            count = new.compiled_coefficients
            new.series_value((0.2, 0.3), (8, 8), max_total_twice_level=16)
            self.assertEqual(count, new.compiled_coefficients)

    def test_compact_fourpoint_and_empty_mask(self):
        args = {**block_arguments(), "external_weights": (0.53, 0.61, 0.62, 0.63),
                "internal_weights": (0.73,), "vertex_sectors": (1, 1)}
        old, new = NSSphereLinearCRecursion(**args), CompactCBlock(**args)
        with mpmath.workdps(45):
            self.assertLess(abs(old.series_value((0.2,), (8,)) -
                                new.series_value((0.2,), (8,))), mpmath.mpf("1e-40"))
        self.assertEqual(new.series_value((0.2,), (0,)), 0)
        self.assertFalse(new._coefficient_cache)

    def test_disk_roundtrip_is_lossless_and_does_not_recurse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficients.sqlite"
            store = CoefficientStore(path)
            block = CompactCBlock(coefficient_store=store, **block_arguments())
            expected = block.series_value((0.12, 0.23), (8, 8))
            coefficients = dict(block.final_coefficients)
            store.close()
            store = CoefficientStore(path)
            restored = CompactCBlock(coefficient_store=store, **block_arguments())
            with mock.patch.object(restored, "_coefficient", side_effect=AssertionError("recompiled")):
                observed = restored.series_value((0.12, 0.23), (8, 8))
            self.assertEqual(expected, observed)
            self.assertEqual(coefficients, restored.final_coefficients)
            changed = CompactCBlock(coefficient_store=store, **{
                **block_arguments(), "working_precision": 50,
            })
            changed.series_value((0.12, 0.23), (2, 2))
            self.assertGreater(changed.compiled_coefficients, 0)
            store.close()

    def test_scratch_is_released_on_compilation_failure(self):
        block = CompactCBlock(**block_arguments())
        with mock.patch.object(block, "_coefficient", side_effect=RuntimeError("interrupted")):
            block._coefficient_cache[("scratch",)] = 3
            with self.assertRaises(RuntimeError):
                block.series_value((0.1, 0.2), (8, 8))
        self.assertFalse(block._coefficient_cache)

    def test_store_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tables.sqlite"
            store = CoefficientStore(path)
            with self.assertRaisesRegex(RuntimeError, "already owned"):
                CoefficientStore(path)
            store.close()
            replacement = CoefficientStore(path)
            replacement.close()

    def test_lru_is_bounded_and_get_updates_recency(self):
        cache = BoundedLRU(2)
        cache["a"], cache["b"] = 1, 2
        self.assertEqual(cache.get("a"), 1)
        cache["c"] = 3
        self.assertNotIn("b", cache)
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(cache.peak_entries, 2)
        for i in range(100):
            cache[i] = i
        self.assertEqual(len(cache), 2)

    def test_checkpoint_rejects_mismatch_and_nonfinite_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            checkpoint = SampleCheckpoint(path, "signature")
            checkpoint.evaluate("sample", lambda: 2 + 3j)
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                SampleCheckpoint(path, "different")
            with self.assertRaises(ArithmeticError):
                checkpoint.evaluate("bad", lambda: complex("nan"))
            restored = SampleCheckpoint(path, "signature")
            self.assertEqual(restored.evaluate("sample", lambda: 0), 2 + 3j)
            self.assertNotIn("bad", restored.values)

    def test_interrupted_integral_resumes_without_repeating_finished_samples(self):
        kernel = fivepoint.BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.25 + 0.02j,) * 4, block_backend="c",
            global_max_twice_levels=(0, 0), global_max_total_twice_level=0,
            momentum_orders=(2, 3), momentum_maximum=1,
            structure_precision=15, block_working_precision=30,
        )
        kwargs = dict(collar_radius=0.01, bulk_sobol_power=1, face_sobol_power=1,
                      replicates=2, compute_corner_contribution=False)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(kernel, "boundary_face_finite_part_density", return_value=2 + 1j), \
                mock.patch.object(kernel, "boundary_corner_face_counterterm_density", return_value=0j):
            path = Path(directory) / "samples.json"
            checkpoint = SampleCheckpoint(path, "test")
            with mock.patch.object(fivepoint, "_leading_local_forest_remainder_integrand",
                                   side_effect=[3 + 2j, 3 + 2j, RuntimeError("stop")]):
                with self.assertRaises(RuntimeError):
                    fivepoint._integrate_leading_local_finite_part_qmc(
                        kernel, checkpoint=checkpoint, **kwargs)
            restored = SampleCheckpoint(path, "test")
            with mock.patch.object(fivepoint, "_leading_local_forest_remainder_integrand",
                                   return_value=3 + 2j) as bulk:
                resumed = fivepoint._integrate_leading_local_finite_part_qmc(
                    kernel, checkpoint=restored, **kwargs)
                self.assertEqual(bulk.call_count, 2)
                expected = fivepoint._integrate_leading_local_finite_part_qmc(kernel, **kwargs)
            self.assertEqual(resumed.estimates, expected.estimates)
            self.assertEqual(resumed.bulk_estimates, expected.bulk_estimates)
            self.assertEqual(resumed.face_estimates, expected.face_estimates)
            self.assertEqual(len(restored.values), 8)


if __name__ == "__main__":
    unittest.main()
