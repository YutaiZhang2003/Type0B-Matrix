#!/usr/bin/env python3
"""Audit the odd NSRR basis change and the saved theta-channel contraction.

The Clifford calculation is local and exact.  The numerical comparison uses
the September 15 physical bilinear in its original form basis, with the
identity-normalized three-point coefficients.  Relabeling its odd forms is
reported separately; neither numerical option is fitted to the target.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def ground_matrices():
    i = sp.I
    # Rows are |++>, |+->, |-+>, |-->, columns the two physical R families.
    embedding = sp.Matrix([[1, 0], [0, 1], [0, 1], [-i, 0]]) / sp.sqrt(2)
    u = (1 - i) / sp.sqrt(2)
    right = sp.Matrix([[1, 0], [0, sp.conjugate(u)], [0, u], [-i, 0]]) / sp.sqrt(2)
    left = sp.Matrix([[1, 0], [0, -u], [0, -sp.conjugate(u)], [-i, 0]]) / sp.sqrt(2)
    graded_metric = sp.diag(1, -1, -1, -1)
    assert sp.simplify(left.T * graded_metric * right) == sp.eye(2)
    result = {}
    for convention, odd_sign in (("plus_i_eta", 1), ("minus_i_eta", -1)):
        for form in (0, 1):
            for eta in (1, -1):
                holomorphic = (sp.diag(1, eta) if form == 0 else
                               sp.Matrix([[0, 1], [odd_sign * i * eta, 0]]))
                antiholomorphic = sp.conjugate(holomorphic)
                matrix = sp.Matrix(2, 2, lambda left, right: sp.simplify(sum(
                    embedding[2 * a + b, left] * embedding[2 * c + d, right]
                    * (-1) ** (b * c) * holomorphic[a, c] * antiholomorphic[b, d]
                    for a in (0, 1) for b in (0, 1)
                    for c in (0, 1) for d in (0, 1))))
                result[convention, form, eta] = matrix
    for eta in (1, -1):
        even = result["minus_i_eta", 0, eta]
        assert result["minus_i_eta", 1, eta] == -i * even
        assert result["plus_i_eta", 1, eta] == -i * result["plus_i_eta", 0, -eta]
    output = {f"{name}:f{form}:eta{eta:+d}": str(matrix)
              for (name, form, eta), matrix in result.items()}
    output["restricted_R_ground_Gram"] = str(left.T * graded_metric * right)
    return output


def ward_parity_transport():
    """Check the sign change in the NSRR supercurrent Ward coefficient.

    In the old contour frame it is i(-1)^(p_NS+p_R0); in the paper frame it
    is i(-1)^p_R1.  A form of parity f has p_NS+p_R1+p_R0=f.
    """
    checked = 0
    for f in (0, 1):
        for ns in (0, 1):
            for r1 in (0, 1):
                r0 = (f + ns + r1) % 2
                old = sp.I * (-1) ** (ns + r0)
                current = sp.I * (-1) ** r1
                assert old == (-1) ** f * current
                # Acting with G in NS or R1 reverses this twist; acting in
                # R0 leaves it unchanged, exactly transporting the Ward rule.
                twist = (-1) ** (f * (ns + r1))
                assert twist * (-1) ** f == (-1) ** (f * (ns + r1 + 1))
                checked += 1
    return {"parity_assignments_checked": checked,
            "odd_form_transport": "(-1)^(parity_NS + parity_R_at_1)"}


def complex_number(value):
    return mp.mpc(value["real"], value["imag"])


def saved_comparison(folder: Path):
    comparison = json.loads((folder / "comparison.json").read_text())
    row = next(item for item in comparison["comparisons"]
               if item["source_level"] == 5 and item["target_level"] == 8)
    target = [complex_number(value) for value in row["target_Z"]]
    frame = complex_number(comparison["free_frame_power"])
    files = sorted(glob.glob(str(folder / "source" / "node-*.json")))
    if len(files) != 343:
        raise ValueError("Expected the complete N=7 source grid")
    totals = {}
    for relabel in (False, True):
        for sign in (1, -1):
            totals[relabel, sign] = mp.mpc(0)
    marked = [mp.mpc(0), mp.mpc(0)]
    odd_with_swapped_constants = mp.mpc(0)
    maximum_spin_identity_error = mp.mpf(0)
    for path in files:
        node = json.loads(Path(path).read_text())
        row5 = next(item for item in node["rows"] if item["level"] == 5)
        blocks = {(item["f"], item["eta"]):
                  [complex_number(value) for value in item["values"]]
                  for item in row5["F"]}
        constants = [complex_number(x) for x in node["C_f_eta"][0]]
        primary = complex_number(node["primary"])
        propagation = complex_number(node["measure"]) * primary * mp.conj(primary)
        for relabel in (False, True):
            for sign in (1, -1):
                for index, eta in enumerate((1, -1)):
                    # The projected basis uses the two lifts (+,+,+), (+,-,+).
                    even = sum(blocks[0, eta][k] for k in (0, 1)) / mp.sqrt(2)
                    odd_eta = -eta if relabel else eta
                    odd = sum(blocks[1, odd_eta][k] for k in (0, 1)) / mp.sqrt(2)
                    # Exact Clifford reduction: (C_left C_right/2)|F0+sF1|^2.
                    totals[relabel, sign] += (
                        propagation * constants[index] ** 2 / 2
                        * abs(even + sign * odd) ** 2)
        for index, eta in enumerate((1, -1)):
            even_first = sum(blocks[0, eta][k] for k in (0, 1)) / mp.sqrt(2)
            even_second = sum(blocks[0, eta][k] for k in (2, 3)) / mp.sqrt(2)
            odd_first = sum(blocks[1, eta][k] for k in (0, 1)) / mp.sqrt(2)
            maximum_spin_identity_error = max(maximum_spin_identity_error,
                                              abs(1j * odd_first - even_second))
            marked[0] += propagation * constants[index] ** 2 * abs(even_first) ** 2
            marked[1] += propagation * constants[index] ** 2 * abs(odd_first) ** 2
            odd_with_swapped_constants += (
                propagation * constants[1 - index] ** 2 * abs(odd_first) ** 2)
    out = []
    for relabel in (False, True):
        for sign in (1, -1):
            value = totals[relabel, sign]
            out.append({"odd_eta_relabelled": relabel, "s": sign,
                        "source": mp.nstr(value, 18),
                        "ratio_to_marked_11_00": mp.nstr(value / target[2] / frame, 16),
                        "ratio_to_marked_11_11": mp.nstr(value / target[0] / frame, 16)})
    fixed_ratios = [marked[0] / target[2] / frame,
                    marked[1] / target[0] / frame]
    return {"source_grid": 343, "source_level": 5, "target_level": 8,
            "target_source_order": ["[11|00]", "[11|11]"], "rows": out,
            "marked_spin_projection": {
                "first": mp.nstr(marked[0], 20),
                "second": mp.nstr(marked[1], 20),
                "fixed_spin_ratios": [mp.nstr(x, 20) for x in fixed_ratios],
                "second_ratio_if_odd_constants_swapped":
                    mp.nstr(odd_with_swapped_constants / target[0] / frame, 20),
                "spin_sum_ratio": mp.nstr(sum(marked) / (target[2] + target[0]) / frame, 20),
                "maximum_i_odd_equals_second_even_error":
                    mp.nstr(maximum_spin_identity_error, 10)},
            "note": "The s rows are tube-character superpositions, so their "
                    "ratios to individual marked spins are diagnostics, not crossing "
                    "tests. The marked-spin projection is the physical comparison. "
                    "The eta-relabelled rows additionally assume the current saved "
                    "f=1 blocks represent a single-vertex change of odd-form basis."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path,
                        default=Path("C++/results/partition_no_M_2026-09-24"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mp.mp.dps = 50
    result = {"ground_matrices": ground_matrices(),
              "ward_parity_transport": ward_parity_transport(),
              "saved_cross_channel": saved_comparison(args.folder)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
