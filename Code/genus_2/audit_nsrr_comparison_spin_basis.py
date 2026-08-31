#!/usr/bin/env python3
"""Independent spin-basis control for the existing comparison assembly.

This records an obstruction, not a proposed modification of the Human Note.
The exact free-fermion change of basis is NOT applied to Liouville blocks.
"""
from __future__ import annotations

import copy
from itertools import product
from pathlib import Path

import numpy as np

import check_nsrr_spin_quadrature as check
from fixed_spin_free_plumbing import (
    charged_frame, charge_lattice_sum, characteristic_in_charge_frame,
    fixed_spin_partition,
)
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from run_fixed_spin_free_check import SOURCE_BRANCH, TARGET_BRANCH, serializable
from ns_genus2_partition import _spin_characteristic_from_lifts, _transport_spin_characteristic

LIFTS = tuple((a, b, 1) for a, b in product((1, -1), repeat=2))


def raw_spin(lifts, branch):
    e0, e1, einf = lifts
    charge = ((0, 0), (int(e0*einf < 0), int(e1*einf < 0)))
    return characteristic_in_charge_frame(charge, -np.asarray(branch))


def leading_pair_signs(lifts, quadratic=False):
    e0, e1, einf = lifts
    factor = -1 if quadratic else 1
    return tuple(factor*z for z in (e0*e1, e0*einf, e1*einf))


def audit(c):
    check.validate(c)
    free = check.trial.load(Path(c["references"]["free"])/"summary.json")
    if check.trial.digest(free) != c["reference_summary_digests"]["free"]:
        raise ValueError("free reference changed")
    matrix = np.asarray(free["source_to_target"], dtype=int)
    source_spin, target_spin = ((1, 1), (0, 0)), ((0, 0), (0, 0))
    if _transport_spin_characteristic(matrix, source_spin) != target_spin:
        raise ArithmeticError("marked spin transport failed")
    U = np.ones((4, 4))/2-np.eye(4)
    charts = {}
    for channel, branch in (("source", SOURCE_BRANCH), ("target", TARGET_BRANCH)):
        point = c[channel+"_point"]
        q = tuple(map(complex, point["q_geometry" if channel == "source" else "q_values"]))
        omega = np.asarray([[complex(z) for z in row] for row in point["omega_source" if channel == "source" else "omega"]])
        frame = charged_frame(q, max_mode=32)
        period_error = float(np.max(abs(frame.omega_charge-omega-np.asarray(branch))))
        boson = frame.loop_gaussian*abs(frame.boson_chiral)**2
        rows = []
        raw, filtered = [], []
        for lift in LIFTS:
            result = theta_physical_fermion_fredholm(q, lift, max_mode=32)
            d, f = result.determinant_values[0], result.chiral_value
            charge_spin = ((0, 0), (int(lift[0] < 0), int(lift[1] < 0)))
            theta = charge_lattice_sum(frame.omega_charge, charge_spin)
            error = abs(d*d/(frame.boson_chiral*theta)-1)
            raw.append(d)
            filtered.append(f)
            rows.append({"lifts": lift, "unfiltered_marked_spin": raw_spin(lift, branch),
                         "old_helper_label": _spin_characteristic_from_lifts("theta", q, lift),
                         "raw_D": d, "filtered_F": f, "bosonization_relative_error": error,
                         "Z_free_unfiltered": boson*abs(d)**2,
                         "Z_free_filtered": boson*abs(f)**2})
        transform_error = float(np.max(abs(np.asarray(filtered)-U@np.asarray(raw))))
        if period_error > 1e-8 or transform_error > 1e-12 or max(r["bosonization_relative_error"] for r in rows) > 1e-10:
            raise ArithmeticError("independent free spin-basis check failed")
        desired = source_spin if channel == "source" else target_spin
        fixed = fixed_spin_partition(q, omega, desired, period_branch=branch, max_mode=32)
        if abs(fixed["Z_free"]/c["Z_free_"+channel]-1) > 1e-12:
            raise ArithmeticError("fixed denominator changed")
        charts[channel] = {"period_branch": branch, "charge_period_error": period_error,
                           "rows": rows, "F_equals_U_D_max_absolute_error": transform_error,
                           "desired_marked_spin": desired, "Z_free_desired": fixed["Z_free"]}
        if channel == "target":
            chosen = next(r for r in rows if list(r["lifts"]) == point["lifts"])
            matching = next(r for r in rows if r["unfiltered_marked_spin"] == desired)
            charts[channel].update(
                numerator_lifts=point["lifts"], chosen_free_control=chosen,
                fixed_spin_unfiltered_lifts=matching["lifts"],
                selected_filtered_over_desired_free=chosen["Z_free_filtered"]/fixed["Z_free"],
                # This quantifies normalization leverage, NOT a candidate fix.
                denominator_only_Q_multiplier=(chosen["Z_free_filtered"]/fixed["Z_free"])**(1+2*(c["source"]["b"]+1/c["source"]["b"])**2),
                conclusion="The selected HN lift label is not independently a fixed-spin identification: its free control is U D, not a single D.")
        else:
            alternatives = []
            for beta in product((0, 1), repeat=2):
                spin = ((1, 1), beta)
                value = fixed_spin_partition(q, omega, spin, period_branch=branch, max_mode=32)
                alternatives.append({"source_marked_spin": spin, "target_marked_spin": _transport_spin_characteristic(matrix, spin),
                                     "Z_free": value["Z_free"], "odd_spin": value["has_fermion_zero_mode"]})
            charts[channel]["physical_spin_controls"] = alternatives
    # Leading sqrt(q_i q_j) coefficients: no real lift relabel can undo K.
    leading = [{"lifts": lift, "raw_pair_signs": leading_pair_signs(lift),
                "filtered_pair_signs": leading_pair_signs(lift, True)} for lift in LIFTS]
    if any(np.prod(r["raw_pair_signs"]) != 1 or np.prod(r["filtered_pair_signs"]) != -1 for r in leading):
        raise ArithmeticError("quadratic-sign obstruction failed")
    source_summary = check.trial.load(Path(c["references"]["source"])/"summary.json")
    source_rows = [r for r in source_summary["rows"] if r["t"] == c["t"] and r["level"] == 3 and r["quadrature_order"] == 5]
    z = [r["total"] for r in source_rows]
    if len(z) != 4:
        raise ValueError("missing source lift test")
    report = {"schema": "comparison-spin-basis-audit-v1", "t": c["t"],
              "config_digest": check.trial.digest(c), "source_to_target": matrix.tolist(),
              "transported_source_spin": _transport_spin_characteristic(matrix, source_spin),
              "charts": charts, "lift_order": LIFTS, "free_even_sector_U": U.tolist(),
              "U_squared_identity_error": float(np.max(abs(U@U-np.eye(4)))),
              "leading_pair_sign_obstruction": leading,
              "NSRR_trial_N5_Z_by_literal_lift": z,
              "NSRR_trial_lift_relative_spread": max(abs(v/z[0]-1) for v in z),
              "physical_NSrr_projector": None,
              "conclusion": "Denominator spin transport passes. The target's old label-only audit does not establish the numerator spin; its free control is a nontrivial spin-basis combination. The NSRR trial has no verified physical spin projector. No correction is inferred from U alone.",
              "protected": check.fresh.protected_hashes()}
    return serializable(report)


def audit_fivepoint(c):
    """Repeat the independent free spin check, without new Liouville integrals."""
    free = check.trial.load(Path(c["references"]["free"])/"summary.json")
    reports = []
    for point in free["points"]:
        local = copy.deepcopy(c)
        local["t"] = point["t"]
        for channel, free_key in (("source", "source_NSrr"), ("target", "target_NSnsns")):
            local[channel+"_point"] = next(p for p in local[channel]["points"] if p["t"] == point["t"])
            local["Z_free_"+channel] = point[free_key]["Z_free"]
        reports.append(audit(local))
    return {"scope": "spin controls on the five saved surfaces; quadrature sweep is only at t=.60",
            "parent_config_digest": check.trial.digest(c), "points": reports}


if __name__ == "__main__":
    c = check.trial.load(check.DEFAULT_OUTPUT/"config.json")
    result = audit(c)
    check.trial.save(check.DEFAULT_OUTPUT/"spin_basis_audit.json", result)
    check.trial.save(check.DEFAULT_OUTPUT/"spin_basis_fivepoint.json", audit_fivepoint(c))
    for channel in ("source", "target"):
        chart = result["charts"][channel]
        print(channel, "period error", chart["charge_period_error"], "basis error", chart["F_equals_U_D_max_absolute_error"])
    print(result["conclusion"])
