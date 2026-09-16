#!/usr/bin/env python3
"""Compare the fixed-spin free formula with finite Fock sewing at saved q.

This is a diagnostic only. It does not change Liouville quadrature, block
orders, saved eta labels, or production denominators. NS Majorana states
are sewn directly using Wick/Pfaffian coefficients. The NSRR fermion check
uses the bosonized Dirac charge lattice, not a Ramond super-Virasoro block.
"""
from __future__ import annotations

import argparse
import cmath
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "Code", ROOT / "Code/genus_2_cross_channel"):
    sys.path.insert(0, str(directory))

from fixed_spin_free_plumbing import direct_charged_fock_sum, fixed_spin_partition
from free_boson_pair_of_pants import (
    heisenberg_gram_norm, integer_partitions, theta_heisenberg_plumbing_partition,
)
from free_majorana_pair_of_pants import majorana_three_point, ns_fermion_states_at_twice_level
from spin_structure import SpinCharacteristic


LEVELS = (6, 10, 14)
CHARGES = ((0.0, 0.0), (0.27, -0.63), (0.5, -0.5), (0.5, 1.5))


def ns_majorana_coefficients(max_level):
    """Raw fixed-spin Wick sewing, before selecting the root signs.

    The product of the two pants coefficients is rho**2. Bra-reversal
    signs cancel in this product. No Human-Note parity filter is applied.
    Keys are twice-levels in geometry order (0, 1, infinity).
    """
    cutoff = 2 * max_level
    states = [ns_fermion_states_at_twice_level(n) for n in range(cutoff + 1)]
    coefficients = {}
    for n_inf in range(cutoff + 1):
        for n_one in range(cutoff + 1 - n_inf):
            for n_zero in range(cutoff + 1 - n_inf - n_one):
                coefficient = sum(
                    majorana_three_point(*triple) ** 2
                    for triple in itertools.product(states[n_inf], states[n_one], states[n_zero])
                )
                if coefficient:
                    coefficients[(n_zero, n_one, n_inf)] = coefficient
    return coefficients


def charged_boson_coefficients(charges, max_level):
    """Direct charged-current Wick coefficients, without a determinant.

    Unlike the lattice oracle, this evaluates arbitrary continuous loop
    charges. Fractional primary powers are restored by the caller.
    """
    a, b = np.asarray(charges, dtype=float).T

    def single(field):
        slot, mode = field
        return b if slot == 0 else ((-1) ** (mode - 1) * a if slot == 1 else -b)

    def pair(left, right):
        (s, m), (t, n) = left, right
        if s == t:
            return 0
        if (s, t) == (0, 1):
            return m * math.comb(m - 1, n - 1) if m >= n else 0
        if (s, t) == (0, 2):
            return m if m == n else 0
        if (s, t) == (1, 2):
            return (-1) ** (m - 1) * n * math.comb(n + m - 1, m - 1)
        raise AssertionError("unordered current slots")

    @lru_cache(maxsize=None)
    def wick(fields):
        if not fields:
            return np.ones(len(charges))
        first, rest = fields[0], fields[1:]
        value = single(first) * wick(rest)
        for j, other in enumerate(rest):
            contraction = pair(first, other)
            if contraction:
                value = value + contraction * wick(rest[:j] + rest[j + 1:])
        return value

    coefficients = {}
    for n_inf in range(max_level + 1):
        for n_one in range(max_level + 1 - n_inf):
            for n_zero in range(max_level + 1 - n_inf - n_one):
                coefficient = np.zeros(len(charges))
                for states in itertools.product(*(integer_partitions(n) for n in (n_inf, n_one, n_zero))):
                    fields = tuple((s, m) for s, state in enumerate(states) for m in state)
                    rho = wick(fields)
                    norm = math.prod(heisenberg_gram_norm(state) for state in states)
                    coefficient += rho * rho / norm
                coefficients[(n_zero, n_one, n_inf)] = coefficient
    wick.cache_clear()
    return coefficients


def evaluate(coefficients, variables, cutoff):
    return sum(coefficient * math.prod(z ** n for z, n in zip(variables, powers))
               for powers, coefficient in coefficients.items() if sum(powers) <= cutoff)


def serializable(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def run(config_path, output):
    started = time.monotonic()
    payload = config_path.read_bytes()
    config = json.loads(payload)
    charged_coefficients = charged_boson_coefficients(CHARGES, max(LEVELS))
    # A separate, charge-free implementation checks every vacuum coefficient.
    vacuum_coefficients = theta_heisenberg_plumbing_partition(
        1, 1, 1, max_total_level=max(LEVELS)).level_contributions
    vacuum_coefficient_error = max(
        abs(values[0] - vacuum_coefficients.get(powers, 0))
        for powers, values in charged_coefficients.items())
    if vacuum_coefficient_error > 1e-10:
        raise ArithmeticError("the two direct boson vacuum expansions disagree")
    majorana_coefficients = ns_majorana_coefficients(max(LEVELS))
    rows = []
    for point in config["points"]:
        for side in ("source", "target"):
            chart = point[side]
            q = tuple(complex(z) for z in chart["q_values"])
            omega = np.asarray([[complex(z) for z in line] for line in chart["omega"]])
            exact = fixed_spin_partition(
                q, omega, chart["characteristic"], period_branch=chart["period_branch"],
                max_mode=32, lattice_cutoff=5)
            spin = SpinCharacteristic.from_pairs(exact["characteristic_charge"])
            p = exact["boson_chiral"]
            dirac_exact = p * exact["dirac_charge_sum"]
            charges = np.asarray(CHARGES)
            a, b = charges.T
            logs = np.log(np.asarray(q))
            primary = np.exp(.5 * (a*a*logs[0] + b*b*logs[1] + (a+b)**2*logs[2]))
            charge_period = np.asarray(exact["omega_charge"])
            charged_exact = p * np.exp(1j * math.pi * np.einsum("ni,ij,nj->n", charges, charge_period, charges))
            sweeps = []
            for level in LEVELS:
                charged_direct = primary * evaluate(charged_coefficients, q, level)
                p_direct = complex(charged_direct[0])
                lattice_direct = direct_charged_fock_sum(
                    q, spin.pairs, total_level=level, lattice_cutoff=5)
                dirac_direct = lattice_direct["dirac_chiral"]
                row = {
                    "level": level,
                    "boson_vacuum_relative_error": abs(p_direct / p - 1),
                    "charged_boson_relative_errors": abs(charged_direct / charged_exact - 1),
                    "dirac_chiral_relative_error": abs(dirac_direct / dirac_exact - 1),
                    "dirac_chiral_fock": dirac_direct,
                    "majorana_nonchiral_relative_error": abs(abs(dirac_direct) / exact["Z_majorana"] - 1),
                    "fock_triples": lattice_direct["fock_triples"],
                    "charge_pairs": lattice_direct["charge_pairs"],
                }
                # This checks all oscillator/lattice factors of Z_free;
                # the analytic continuous-charge Gaussian is shared.
                assembled = exact["loop_gaussian"] * abs(p_direct)**2 * abs(dirac_direct)
                row["Z_free_with_shared_gaussian"] = assembled
                row["Z_free_with_shared_gaussian_relative_error"] = abs(assembled / exact["Z_free"] - 1)
                if spin.alpha == (0, 0):
                    lifts = spin.all_ns_determinant_lifts()
                    roots = tuple(eta * cmath.sqrt(z) for eta, z in zip(lifts, q))
                    majorana = complex(evaluate(majorana_coefficients, roots, 2 * level))
                    row["direct_NS_majorana_chiral"] = majorana
                    row["direct_NS_majorana_squared_complex_relative_error"] = abs(majorana**2 / dirac_exact - 1)
                    row["direct_NS_majorana_norm_relative_error"] = abs(abs(majorana)**2 / exact["Z_majorana"] - 1)
                    row["direct_NS_Z_free_relative_error"] = abs(
                        exact["loop_gaussian"] * abs(p_direct * majorana)**2 / exact["Z_free"] - 1)
                sweeps.append(row)
            rows.append({
                "point_id": point["point_id"], "side": side,
                "q_values": q, "spin_frame": exact["spin_frame"],
                "formula_Z_free": exact["Z_free"],
                "formula_dirac_chiral": dirac_exact,
                "saved_denominator_relative_error": abs(exact["Z_free"] / chart["Z_free"] - 1),
                "lattice_cutoff_4_to_5_relative_change": exact["lattice_relative_change"],
                "sweep": sweeps,
            })
        print(f"checked {point['point_id']}", flush=True)

    maxima = []
    for side in ("source", "target"):
        for level in LEVELS:
            selected = [s for r in rows if r["side"] == side for s in r["sweep"] if s["level"] == level]
            keys = [key for key in selected[0] if key.endswith("relative_error")]
            maxima.append({"side": side, "level": level,
                           **{key: max(s[key] for s in selected) for key in keys},
                           "arbitrary_charged_boson_maximum_relative_error": max(float(max(s["charged_boson_relative_errors"])) for s in selected)})
    if max(m["dirac_chiral_relative_error"] for m in maxima if m["level"] == max(LEVELS)) > 1e-7:
        raise ArithmeticError("direct free Fock expansion has not approached the formula")
    if max(m["direct_NS_majorana_squared_complex_relative_error"] for m in maxima
           if m["side"] == "target" and m["level"] == max(LEVELS)) > 1e-7:
        raise ArithmeticError("direct Majorana Wick sewing has not approached the formula")

    dependencies = [Path(__file__), Path(__file__).with_name("fixed_spin_free_plumbing.py"),
                    Path(__file__).with_name("physical_free_plumbing_resummation.py"),
                    Path(__file__).with_name("spin_structure.py"),
                    ROOT / "Code/genus_2_cross_channel/free_boson_pair_of_pants.py",
                    ROOT / "Code/genus_2_cross_channel/free_majorana_pair_of_pants.py"]
    result = {
        "schema": "fixed-spin-free-q-expansion-audit-v1",
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        "levels": LEVELS, "arbitrary_boson_charges": CHARGES,
        "arithmetic": "complex128 / float64; integer NS Wick coefficients",
        "conventions": {
            "edge_order": "0, 1, infinity", "propagator": "q^L0",
            "fock_cutoff": "sum of three oscillator levels <= L; charge-dependent primary powers retained exactly",
            "NS_direct_cutoff": "sum of three Majorana half-integer levels <= L",
            "spin": "saved marked characteristic transported by its saved integer period branch",
            "NSRR_check": "bosonized Dirac charged-Heisenberg sewing, followed by Z_Majorana=abs(Z_Dirac,chiral)",
            "Gaussian": "analytic continuous-charge Gaussian is shared in the assembled Z_free comparison; charged boson blocks independently tested at four charges",
            "Liouville": "no Liouville parameters, grids, values, or saved lifts changed",
        },
        "vacuum_coefficient_crosscheck_maximum_absolute_error": vacuum_coefficient_error,
        "maximum_saved_denominator_relative_error": max(r["saved_denominator_relative_error"] for r in rows),
        "maximum_lattice_cutoff_change": max(r["lattice_cutoff_4_to_5_relative_change"] for r in rows),
        "maxima": maxima, "points": rows, "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(serializable(result), indent=2, allow_nan=False) + "\n")
    lines = ["# Fixed-spin free-CFT q-expansion check", "",
             "All ten saved surfaces, both theta charts. The formula is compared with direct finite Fock-state sums using the same local coordinates, primary powers, and transported marked spin.", "",
             "| Chart | Oscillator cutoff L | Boson P relative error | Dirac P theta relative error | Free Z relative error (shared analytic Gaussian) |", "|---|---:|---:|---:|---:|"]
    for m in maxima:
        lines.append(f"| {m['side']} | {m['level']} | {m['boson_vacuum_relative_error']:.6e} | {m['dirac_chiral_relative_error']:.6e} | {m['Z_free_with_shared_gaussian_relative_error']:.6e} |")
    lines += ["", "Errors are maxima over ten surfaces. L truncates the sum of oscillator levels on the three edges; charged primary powers are not truncated. The source NSRR test uses half-integer bosonized Dirac charges. This is independent of the period/theta resummation, but is not an independent unbosonized Ramond Majorana construction.", "",
              "The all-NS Majorana is also computed directly from fermion Fock states and squared Pfaffian pants coefficients, without a theta constant or bosonization in that direct calculation:", "",
              "| L | max absolute relative error of D_Fock^2 / (P theta) | Free Z relative error using direct Majorana (shared analytic Gaussian) |", "|---:|---:|---:|"]
    for m in maxima:
        if m["side"] == "target":
            lines.append(f"| {m['level']} | {m['direct_NS_majorana_squared_complex_relative_error']:.6e} | {m['direct_NS_Z_free_relative_error']:.6e} |")
    lines += ["", "The boson comparison additionally checks four continuous charge pairs before their analytic Gaussian integration. The assembled free Z comparison shares the exact analytic Gaussian, so it is not a separate numerical integration test. No interacting sewing convention is certified by this free-theory audit.", "",
              "Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /private/tmp/type0b-nsrr-smoke-venv/bin/python Code/genus_2/audit_fixed_spin_free_q_expansion.py`", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(serializable({"maxima": maxima, "elapsed_seconds": result["elapsed_seconds"]}), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "Data Set/fixed_spin_free_q_expansion_20260912")
    args = parser.parse_args()
    run(args.config, args.output)
