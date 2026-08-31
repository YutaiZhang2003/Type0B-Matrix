"""Production CLI guards and default routing for the all-c amplitude policy."""

from contextlib import redirect_stderr
import importlib
import inspect
import io
from pathlib import Path
import sys
import unittest

FIVE_POINT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "higher_point_amplitude_attempts"
    / "type0b_ns_five_tachyon"
)
sys.path.insert(0, str(FIVE_POINT_DIRECTORY))

from bry_one_to_three import BRYOneToThreeBenchmark
from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from sphere_four_point import BRYNSFourPointCorrelator
from sphere_multipoint import BRYNSSphereMultipointCorrelator
from type0b_ns_five_tachyon import BRYNSFiveTachyonIntegrand
from type0b_sphere_four_point_hybrid import (
    Type0BFixedContourFourPointElliptic,
    Type0BSphereFourPointHybrid,
)


class NSAmplitudeCOnlyPolicyTests(unittest.TestCase):
    def test_shared_kernel_defaults_are_c_recursion(self):
        for kernel in (
            BRYOneToThreeBenchmark,
            BRYNSFourPointCorrelator,
            BRYNSSphereMultipointCorrelator,
            BRYNSFiveTachyonIntegrand,
            Type0BFixedContourFourPointElliptic,
            Type0BSphereFourPointHybrid,
        ):
            with self.subTest(kernel=kernel.__name__):
                self.assertEqual(
                    inspect.signature(kernel).parameters["block_backend"].default,
                    "c",
                )

    def test_production_cli_backends_accept_only_c(self):
        modules = (
            ("bry_one_to_three", "build_parser"),
            ("evaluate_type0b_ns_five_tachyon", "_parser"),
            ("evaluate_type0b_ns_five_tachyon_path", "_parser"),
            ("evaluate_type0b_ns_five_tachyon_minimal_subtraction_path", "_parser"),
            ("evaluate_type0b_ns_five_tachyon_one_divisor_path", "_parser"),
            ("evaluate_type0b_ns_five_tachyon_physical_i_epsilon", "_parser"),
        )
        for module_name, parser_name in modules:
            with self.subTest(module=module_name):
                parser = getattr(importlib.import_module(module_name), parser_name)()
                # Parsing does not evaluate or create an amplitude output.
                arguments = (
                    ["--output", "/tmp/ns-all-c-policy-not-written.json"]
                    if any(action.dest == "output" for action in parser._actions)
                    else []
                )
                self.assertEqual(parser.parse_args(arguments).block_backend, "c")
                self.assertEqual(
                    parser.parse_args(arguments + ["--block-backend", "c"]).block_backend,
                    "c",
                )
                for backend in ("h", "hybrid"):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        parser.parse_args(arguments + ["--block-backend", backend])
                    self.assertEqual(error.exception.code, 2)

    def test_collar_reference_cli_is_also_c_only(self):
        from evaluate_type0b_ns_five_tachyon_physical_i_epsilon import _parser

        parser = _parser()
        arguments = ["--output", "/tmp/ns-all-c-policy-not-written.json"]
        self.assertEqual(parser.parse_args(arguments).face_collar_reference_backend, "c")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(arguments + ["--face-collar-reference-backend", "h"])
        self.assertEqual(error.exception.code, 2)

    def test_multipoint_default_uses_c_for_bulk_and_corner(self):
        kernel = BRYNSSphereMultipointCorrelator(
            momenta=(0.1, 0.2, 0.3, 0.4, 0.5),
            points=(0.0, 0.05, 0.1, 1.0, 2.0),
            max_twice_levels=(0, 0),
        )
        for region in ("bulk", "corner"):
            self.assertEqual(kernel._selected_block_backend(region), "c")
            block = kernel._block(
                kernel.frame((0, 1, 2, 3, 4)),
                (0.7, 0.8),
                (0, 0, 0),
                block_region=region,
            )
            self.assertIsInstance(block, NSSphereLinearCRecursion)


if __name__ == "__main__":
    unittest.main()
