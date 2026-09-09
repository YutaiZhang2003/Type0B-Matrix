#!/usr/bin/env python3
"""Audit the Aug25 all-NS certificate without changing any sewing kernel.

Numerators are read, never recomputed or refitted. Fixed-spin free fields
are identified by charged Heisenberg sewing and four independent Majorana
determinants, not by the production helper's spin labels.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from fixed_spin_free_plumbing import charged_frame, charge_lattice_sum, fixed_spin_partition
from physical_free_plumbing_resummation import (
    glasses_charged_boson_resummation, glasses_physical_fermion_fredholm,
    theta_physical_fermion_fredholm,
)
from ns_genus2_partition import _spin_characteristic_from_lifts, _transport_spin_characteristic
from recompute_all_ns_reference import protected_hashes
from run_fixed_spin_free_check import SOURCE_BRANCH, TARGET_BRANCH, serializable


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json"
CERTIFICATE = ROOT / "Data Set/ns_genus2_human_note_fivepoint_certificate_2026-08-25.json"
CURRENT = ROOT / "Data Set/nsrr_spin_quadrature_t060_20260830/config.json"
OUTPUT = ROOT / "Data Set/previous_all_ns_free_spin_audit_20260830.json"
LIFTS = tuple((a, b, 1) for a, b in itertools.product((1, -1), repeat=2))
ZERO_SPIN = ((0, 0), (0, 0))
GLASSES_CHARGE_CYCLE_REVERSAL = np.diag([1, -1])


def load(path):
    return json.loads(path.read_text())


def omega_array(values):
    return np.asarray([[complex(z) for z in row] for row in values])


def frame(channel, q, mode):
    if channel == "theta":
        result = charged_frame(q, max_mode=mode)
        omega, product = result.omega_charge, result.boson_chiral
    elif channel == "glasses":
        logs = np.log(np.asarray(q))
        values = []
        for a, b in ((1, 0), (0, 1), (1, 1)):
            result = glasses_charged_boson_resummation(
                q, alpha_left=a, alpha_right=b, max_mode=mode)
            values.append(.5*(a*a*logs[0]+b*b*logs[1])+result.charged_exponent)
        u, v, w = values
        cross = (w-u-v)/2
        omega = np.array([[u, cross], [cross, v]])/(1j*np.pi)
        product = result.vacuum_chiral
    else:
        raise ValueError(channel)
    if np.linalg.eigvalsh(omega.imag)[0] <= 0:
        raise ArithmeticError("charge period has nonpositive imaginary part")
    gaussian = float(np.linalg.det(2*omega.imag)**(-.5))
    return omega, product, gaussian*abs(product)**2


def parity_components(raw):
    """Invert the four edge-sign characters; parity order is (0,1,infinity)."""
    return {
        "000": sum(raw)/4,
        "110": sum(a*b*d for (a, b, _), d in zip(LIFTS, raw))/4,
        "101": sum(a*d for (a, _, _), d in zip(LIFTS, raw))/4,
        "011": sum(b*d for (_, b, _), d in zip(LIFTS, raw))/4,
    }


def channel_audit(point, channel, lifts, mode=32):
    q = tuple(map(complex, point["q_values"][channel]))
    marked = omega_array(point["omega"][channel])
    omega, product, boson = frame(channel, q, mode)
    # Fixed analytic cycle dictionaries, the same for every historical point.
    # In particular there is NO integer translation in this theta marking.
    cycle = np.eye(2, dtype=int) if channel == "theta" else GLASSES_CHARGE_CYCLE_REVERSAL
    period_error = float(np.max(abs(omega-cycle@marked@cycle.T)))
    if period_error > 1e-8:
        raise ArithmeticError("historical charged/marked period dictionary failed")
    rows, raw = [], []
    for lift in LIFTS:
        beta = (int(lift[0] < 0), int(lift[1] < 0))
        spin = ((0, 0), beta)
        if channel == "theta":
            result = theta_physical_fermion_fredholm(q, lift, max_mode=mode)
            determinant, used = result.determinant_values[0], result.chiral_value
        else:
            result = glasses_physical_fermion_fredholm(q, lift, max_mode=mode)
            determinant = used = result.chiral_value
        theta = charge_lattice_sum(omega, spin, cutoff=10)
        error = abs(determinant**2/(product*theta)-1)
        if error > 1e-10:
            raise ArithmeticError("independent Majorana/bosonization check failed")
        raw.append(determinant)
        rows.append({"lifts": lift, "fixed_spin_marked": spin,
                     "old_helper_label": _spin_characteristic_from_lifts(channel, q, lift),
                     "raw_D": determinant, "legacy_F": used,
                     "complex_bosonization_relative_error": float(error),
                     "Z_free_fixed": float(boson*abs(product*theta)),
                     "Z_free_legacy": float(boson*abs(used)**2)})
    chosen = next(r for r in rows if r["lifts"] == tuple(lifts))
    desired = rows[0]  # beta=00 in both charge and historical marked frames.
    result = {"q": q, "omega_marked": marked.tolist(), "omega_charge": omega.tolist(),
              "charge_cycle_matrix": cycle.tolist(), "integer_period_branch": [[0, 0], [0, 0]],
              "period_residual": period_error, "mode": mode, "boson_chiral": product,
              "Z_boson": float(boson), "four_NS_controls": rows,
              "legacy_lifts": tuple(lifts), "desired_fixed_spin": ZERO_SPIN,
              "Z_free_legacy": chosen["Z_free_legacy"], "Z_free_fixed": desired["Z_free_fixed"],
              "legacy_over_fixed_minus_one": chosen["Z_free_legacy"]/desired["Z_free_fixed"]-1}
    if channel == "theta":
        sectors = parity_components(raw)
        residual = abs(chosen["legacy_F"]-(raw[0]-2*sectors["101"]))
        basis_error = np.max(abs(np.array([r["legacy_F"] for r in rows])
                                -(np.ones((4, 4))/2-np.eye(4))@np.asarray(raw)))
        if max(residual, basis_error) > 1e-12:
            raise ArithmeticError("free parity decomposition failed")
        result.update(parity_components=sectors,
                      F_selected_equals_D0000_minus_2_S101_error=float(residual),
                      free_basis_identity_error=float(basis_error))
    return result


def stress_test(q):
    """Vary geometry, not Liouville data, and retain the charge marking."""
    rows = []
    for scale in (1, 10, 100, 1000, 10000):
        varied = (q[0]*scale, q[1], q[2]*scale)
        omega, product, _ = frame("theta", varied, 32)
        controls = [theta_physical_fermion_fredholm(varied, lift, max_mode=32) for lift in LIFTS]
        sectors = parity_components([v.determinant_values[0] for v in controls])
        fixed = product*charge_lattice_sum(omega, ZERO_SPIN, cutoff=10)
        error = abs(controls[0].determinant_values[0]**2/fixed-1)
        if error > 1e-10:
            raise ArithmeticError("stress fixed-spin identity failed")
        rows.append({"endpoint_q_scale": scale, "q": varied,
                     "omega_charge": omega.tolist(), "S101": sectors["101"],
                     "legacy_over_fixed_minus_one": float(abs(controls[1].chiral_value)**2/abs(fixed)-1),
                     "bosonization_relative_error": float(error)})
    return rows


def current_spin_pair_control(config):
    """Changing both matched free spins must leave their frame ratio unchanged."""
    reference = load(Path(config["references"]["free"])/"summary.json")
    matrix = np.asarray(reference["source_to_target"], dtype=int)
    pairs = []
    for spin in (((1, 1), (0, 0)), ((1, 1), (1, 1))):
        target_spin = _transport_spin_characteristic(matrix, spin)
        values = {}
        for channel, characteristic, branch in (
                ("source", spin, SOURCE_BRANCH), ("target", target_spin, TARGET_BRANCH)):
            point = config[channel+"_point"]
            q = tuple(map(complex, point["q_geometry" if channel == "source" else "q_values"]))
            marked = omega_array(point["omega_source" if channel == "source" else "omega"])
            free = fixed_spin_partition(q, marked, characteristic, period_branch=branch, max_mode=32)
            values[channel] = free["Z_free"]
            if channel == "target" and spin[1] == (0, 0):
                raw = [theta_physical_fermion_fredholm(q, lift, max_mode=32).determinant_values[0]
                       for lift in LIFTS]
                f = theta_physical_fermion_fredholm(q, point["lifts"], max_mode=32).chiral_value
                parity = parity_components(raw)
                if abs(f-(raw[2]+2*parity["110"])) > 1e-12:
                    raise ArithmeticError("current target parity-sector identity failed")
                target_control = {"period_branch": branch, "numerator_lifts": point["lifts"],
                                  "fixed_spin_unfiltered_lifts": LIFTS[2], "S110": parity["110"],
                                  "F_selected": f, "D_fixed": raw[2],
                                  "F_selected_equals_D_fixed_plus_2_S110_error": float(abs(f-raw[2]-2*parity["110"])),
                                  "Z_free_filtered": float(free["Z_boson"]*abs(f)**2),
                                  "Z_free_fixed": free["Z_free"],
                                  "legacy_over_fixed_minus_one": float(free["Z_boson"]*abs(f)**2/free["Z_free"]-1)}
        pairs.append({"source_spin": spin, "target_spin": target_spin,
                      "Z_free_source": values["source"], "Z_free_target": values["target"],
                      "source_over_target_free": values["source"]/values["target"]})
    kappa = 1+2*(config["source"]["b"]+1/config["source"]["b"])**2
    effect = (pairs[0]["source_over_target_free"]/pairs[1]["source_over_target_free"])**kappa
    if abs(effect-1) > 1e-8:
        raise ArithmeticError("matched free-spin change altered the quotient ratio")
    return {"t": config["t"], "kappa": kappa, "source_to_target": matrix.tolist(),
            "matched_even_pairs": pairs, "paired_denominator_change_Q_ratio_multiplier": effect,
            "target_filtered_control": target_control,
            "interpretation": "Paired physical free-spin changes do not cure the Liouville mismatch. Numerator spin/assembly must be identified independently."}


def run():
    protected = protected_hashes()
    config, certificate = load(CONFIG), load(CERTIFICATE)
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != certificate["config_sha256"]:
        raise ValueError("historical certificate/config mismatch")
    matrix = np.asarray(config["provenance"]["symplectic_matrix_glasses_to_theta_after_branch"])
    if _transport_spin_characteristic(matrix, ZERO_SPIN) != ZERO_SPIN:
        raise ArithmeticError("historical marked spin transport failed")
    numerators = {r["point_id"]: r for r in certificate["fivepoint_R24_N10"]}
    rows = []
    for point in config["points"]:
        old = numerators[point["id"]]
        charts = {}
        for channel in ("theta", "glasses"):
            charts[channel] = channel_audit(point, channel, config["physical_lifts"][channel])
            low = channel_audit(point, channel, config["physical_lifts"][channel], mode=24)
            high = charts[channel]
            high["mode_24_to_32_legacy_relative_change"] = abs(low["Z_free_legacy"]/high["Z_free_legacy"]-1)
            high["mode_24_to_32_fixed_relative_change"] = abs(low["Z_free_fixed"]/high["Z_free_fixed"]-1)
            saved = old["channels"][channel]
            high["saved_free_reproduction_error"] = abs(high["Z_free_legacy"]/saved["physical_free_one_superfield"]-1)
            if max(high["saved_free_reproduction_error"], high["mode_24_to_32_fixed_relative_change"],
                   high["mode_24_to_32_legacy_relative_change"]) > 1e-10:
                raise ArithmeticError("historical free reproduction or mode stability failed")
            high["historical_Z_L"] = saved["human_note_numerator"]
            high["counterfactual_Q_fixed_denominator_only"] = saved["human_note_numerator"]/high["Z_free_fixed"]**9
        theta, glasses = charts["theta"], charts["glasses"]
        scalar_ratio = theta["Z_boson"]/glasses["Z_boson"]
        fixed_error = (theta["Z_free_fixed"]/glasses["Z_free_fixed"])/scalar_ratio**1.5-1
        legacy_error = (theta["Z_free_legacy"]/glasses["Z_free_legacy"])/scalar_ratio**1.5-1
        if abs(fixed_error) > 1e-8:
            raise ArithmeticError("historical free modular/frame identity failed")
        corrected_ratio = theta["counterfactual_Q_fixed_denominator_only"]/glasses["counterfactual_Q_fixed_denominator_only"]
        rows.append({"point_id": point["id"], "channels": charts,
                     "historical_R24_N10_Q_ratio_minus_one": old["relative_difference"],
                     "counterfactual_fixed_denominator_Q_ratio_minus_one": corrected_ratio-1,
                     "fixed_free_modular_frame_residual": float(fixed_error),
                     "legacy_free_modular_frame_residual": float(legacy_error)})
    result = {"schema": "previous-all-ns-free-spin-audit-v1",
              "scope": "Diagnostic only: no Liouville integral, block, coefficient, or prescription changed",
              "provenance": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (CONFIG, CERTIFICATE, CURRENT, Path(__file__))},
              "historical_kappa": 9, "source_to_target_glasses_to_theta": matrix.tolist(),
              "historical_coefficient_convention": certificate["coefficient_convention"],
              "historical_rows": rows,
              "thin_tube_stress_test": stress_test(tuple(map(complex, config["points"][0]["q_values"]["theta"]))),
              "current_spin_pair_control": current_spin_pair_control(load(CURRENT)),
              "protected_kernel_sha256": protected,
              "reference": "https://arxiv.org/html/1007.5203#S5.SS4 (bosonization, equation 78)",
              "physical_Q_NSrr": None}
    if protected != protected_hashes():
        raise ArithmeticError("protected kernels changed")
    return serializable(result)


if __name__ == "__main__":
    result = run()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["historical_rows"]:
        print(row["point_id"], "theta legacy/fixed - 1",
              row["channels"]["theta"]["legacy_over_fixed_minus_one"],
              "historical Q gap", row["historical_R24_N10_Q_ratio_minus_one"],
              "denominator-only Q gap", row["counterfactual_fixed_denominator_Q_ratio_minus_one"])
    print("Current paired-spin Q-ratio multiplier:",
          result["current_spin_pair_control"]["paired_denominator_change_Q_ratio_multiplier"])
    print(OUTPUT)
