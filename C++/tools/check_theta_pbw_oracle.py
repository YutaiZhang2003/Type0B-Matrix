#!/usr/bin/env python3
"""Compare current theta PBW and double-Virasoro blocks with a local theta oracle.

The two outputs use the same f and vertex-sign labels. The C++ block includes
the BPZ plumbing factor; the oracle's linear-pairing block does not.
The oracle is the implementation of Human Notes/SCblock.tex in
double_virasoro/nsrr/nsrr_genus2_block.py. It is not one of Yutai's
independent sphere four-point or torus two-point calculations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "C++"
CODE = ROOT / "Packages/local_amplitude_conformal_blocks/source/project/Code"
sys.path[:0] = [str(CODE / "double_virasoro/nsrr"), str(CODE / "c_Recursion")]
from nsrr_genus2_block import HumanNSRRThetaOracle  # noqa: E402


def bpz_factor(parity: int) -> complex:
    return (-1j) ** parity.bit_count() * (-1 if parity & 1 else 1)


def number(value: dict[str, str]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def compare(level: int) -> dict[str, object]:
    # The build step is essential: an older executable once produced a false
    # apparent eta relabeling in the odd form.
    subprocess.run(["make", "-C", str(CPP), "bin/pbw", "bin/ramond"],
                   check=True, capture_output=True)
    maximum = 0.0
    maximum_without_bpz = 0.0
    maximum_algorithm = 0.0
    worst: tuple[object, ...] | None = None
    cases = 0
    components = 0
    with tempfile.TemporaryDirectory(prefix="theta-pbw-oracle-") as directory:
        for primary_parity in (0, 1):
            for mode in ("ordinary", "inserted"):
                for form_parity in (0, 1):
                    for eta in (1, -1):
                        output = Path(directory) / f"p{primary_parity}_{mode}_f{form_parity}_eta{eta}.json"
                        subprocess.run(
                            [str(CPP / "bin/pbw"), "--mode", mode, "--level", str(level),
                             "--dps", "0", "--p", str(primary_parity), "--f", str(form_parity),
                             "--eta", str(eta), "--json", str(output)],
                            check=True, capture_output=True,
                        )
                        record = json.loads(output.read_text())
                        algorithm_output = Path(directory) / "algorithm.json"
                        subprocess.run(
                            [str(CPP / "bin/ramond"), "--mode", mode,
                             "--truncation", "per-edge", "--level", str(level),
                             "--dps", "0", "--p", str(primary_parity),
                             "--f", str(form_parity), "--eta", str(eta),
                             "--json", str(algorithm_output)],
                            check=True, capture_output=True,
                        )
                        algorithm_rows = {
                            tuple(row["exponents"]): row["values"]
                            for row in json.loads(algorithm_output.read_text())["coefficients"]
                        }
                        b = sp.Rational(record["b"])
                        momenta = tuple(sp.Rational(x) for x in record["momenta"])
                        q = b + 1 / b
                        vertex_signs = (eta, eta if mode == "ordinary" else -eta)
                        oracle = HumanNSRRThetaOracle(
                            central_charge=sp.Rational(3, 2) + 3 * q * q,
                            h_ns=q * q / 8 - momenta[0] ** 2 / 2,
                            beta_r1=momenta[1] / sp.sqrt(2),
                            beta_r2=momenta[2] / sp.sqrt(2),
                            form_parity=form_parity,
                            primary_parity=primary_parity,
                            etas=vertex_signs,
                        )
                        for row in record["coefficients"]:
                            ns_twice, r1_level, second_copy, r2_twice = row["exponents"]
                            if second_copy != r1_level or r2_twice % 2:
                                raise AssertionError("unexpected C++ exponent convention")
                            reference = oracle.coefficient_components(
                                ns_twice, r1_level, r2_twice // 2
                            )
                            actual_algorithm = algorithm_rows[tuple(row["exponents"])]
                            for parity, (want, encoded) in enumerate(zip(reference, row["values"])):
                                got = number(encoded)
                                maximum_algorithm = max(
                                    maximum_algorithm,
                                    abs(number(actual_algorithm[parity]) - got),
                                )
                                error = abs(got - bpz_factor(parity) * want)
                                maximum_without_bpz = max(maximum_without_bpz, abs(got - want))
                                if error > maximum:
                                    maximum = error
                                    worst = (primary_parity, mode, form_parity, vertex_signs,
                                             ns_twice, r1_level, r2_twice // 2, parity)
                                components += 1
                        cases += 1
    if maximum >= 1e-12 or maximum_algorithm >= 1e-11 or maximum_without_bpz <= 1e-3:
        raise AssertionError(
            f"theta convention mismatch: {maximum=}, {maximum_algorithm=}, "
            f"{maximum_without_bpz=}, {worst=}"
        )
    return {
        "cases": cases,
        "components": components,
        "independent_edge_level": level,
        "maximum_absolute_error_after_bpz": maximum,
        "maximum_absolute_error_algorithm_vs_pbw": maximum_algorithm,
        "maximum_absolute_difference_without_bpz": maximum_without_bpz,
        "worst_case": worst,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()
    if args.level < 0:
        parser.error("--level must be nonnegative")
    print(json.dumps(compare(args.level), indent=2))
