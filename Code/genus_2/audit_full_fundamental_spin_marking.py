#!/usr/bin/env python3
"""Trace q -> native/atlas period -> common fundamental domain with spins.

Read-only with respect to production and all previous results. The original
StringMC overlap record is used to reconstruct the historical atlas word;
current q values are independently mapped back by holomorphic one-forms.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import audit_previous_all_ns_free_spin as previous
from bolza_torus_plumbing_reach import enumerate_symplectic_words, transform_omega
from genus2_siegel_fundamental_domain import gottschling_min_margin
from genus2_table_fundamental_reduction import b_shift_matrix, symplectic_inverse
from liouville_genus2_modular_check import named_transform
from plumbing_algorithms import (
    solve_theta_collocation, solve_glasses_collocation,
    schottky_theta_period_matrix_cross_ratio,
)
from fixed_spin_free_plumbing import charged_frame, charge_lattice_sum, characteristic_in_charge_frame
from free_boson_plumbing import riemann_theta_constant_genus2
from ns_genus2_partition import _transport_spin_characteristic as transport
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm, glasses_physical_fermion_fredholm
from recompute_all_ns_reference import protected_hashes
from run_fixed_spin_free_check import SOURCE_BRANCH, TARGET_BRANCH, serializable


ROOT = previous.ROOT
OVERLAP = Path("/Users/yutaizhang/Desktop/Project/StringMC/plumbing/results/genus2_plumbing_moduli_samples/q06_search_N256/overlap_samples.csv")
GEOMETRY = ROOT / "Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/source_geometry_audit.json"
OUTPUT = ROOT / "Data Set/full_fundamental_spin_marking_audit_20260830.json"
ZERO_SPIN = ((0, 0), (0, 0))
# The independently integrated B paths need not equal the Schottky log paths.
# These geometry-only differences were measured once on the five saved q's;
# they are explicit inputs, never chosen by a fermion amplitude or spin fit.
OLD_COLLOCATION_MINUS_SAVED = {
    "o0243-periodmatched": ((0, 0), (0, 0)),
    "o0127-periodmatched": ((0, 0), (0, 0)),
    "o0015-periodmatched": ((0, 0), (0, -1)),
    "o0167-periodmatched": ((0, 0), (0, -1)),
    "o0239-periodmatched": ((0, 0), (0, -1)),
}


def word_matrix(word):
    result = np.eye(4, dtype=int)
    for token in reversed(word.split()):
        if token == "I":
            continue
        inverse = token.endswith("^-1")
        tr = named_transform(token[:-3] if inverse else token)
        matrix = np.block([[tr.a, tr.b], [tr.c, tr.d]])
        if inverse:
            matrix = symplectic_inverse(matrix)
        result = matrix @ result
    return result


@lru_cache(maxsize=1)
def reduction_words():
    words = enumerate_symplectic_words(5)
    return words, np.asarray([m for _, m in words], dtype=int)


def reduce_fd(omega):
    words, matrices = reduction_words()
    values = ((matrices[:, :2, :2] @ omega + matrices[:, :2, 2:])
              @ np.linalg.inv(matrices[:, 2:, :2] @ omega + matrices[:, 2:, 2:]))
    values = (values+values.transpose(0, 2, 1))/2
    accepted = np.flatnonzero(gottschling_min_margin(values) >= -1e-10)
    if not len(accepted):
        raise ArithmeticError("no fundamental representative in bounded search")
    index = int(accepted[0])
    word, matrix = words[index]
    symplectic_inverse(matrix)  # exact integer certificate
    return word, matrix, values[index]


def theta(omega, spin):
    return riemann_theta_constant_genus2(omega, spin, tol=1e-15)


def invariant(omega, spin):
    """Z_(X+psi)/(Z_X)^(3/2), in the da, h=a^2/2 convention."""
    return float(np.linalg.det(2*omega.imag)**.25*abs(theta(omega, spin)))


def step(name, matrix, omega, spin):
    symplectic_inverse(matrix)
    mapped, changed = transform_omega(matrix, omega), transport(matrix, spin)
    factor = abs(np.linalg.det(matrix[2:, :2]@omega+matrix[2:, 2:]))**.5
    error = abs(abs(theta(mapped, changed))/(factor*abs(theta(omega, spin)))-1)
    if error > 1e-9:
        raise ArithmeticError(f"theta covariance failed at {name}: {error}")
    return {"name": name, "matrix": matrix.tolist(), "omega_before": omega.tolist(),
            "spin_before": spin, "omega_after": mapped.tolist(), "spin_after": changed,
            "theta_absolute_modular_factor": float(factor), "theta_covariance_relative_error": float(error)}


def all_even_covariance(matrix, omega):
    errors = []
    for bits in itertools.product((0, 1), repeat=4):
        if (bits[0]*bits[2]+bits[1]*bits[3]) % 2 == 0:
            errors.append(step("all-even-control", matrix, omega, (bits[:2], bits[2:]))["theta_covariance_relative_error"])
    return max(errors)


def forward(channel, q, marked, branch=((0, 0), (0, 0))):
    if channel == "theta":
        result = solve_theta_collocation(*q, basis_order=32, samples_per_seam=160)
        omega = result.omega
    else:
        result = solve_glasses_collocation(*q, basis_order=40, samples_per_seam=240,
                                          schottky_word_len=2)
        omega = result.omega_b_annular
    branch = np.asarray(branch, dtype=int)
    to_marked = b_shift_matrix(-branch)
    aligned = transform_omega(to_marked, omega)
    error = float(np.max(abs(aligned-marked)))
    if error > 1e-8:
        raise ArithmeticError(f"q-to-native-period mismatch {channel}: {error}")
    return {"omega_forward_raw": omega.tolist(), "omega_forward": aligned.tolist(),
            "raw_collocation_minus_saved_branch": branch.tolist(),
            "raw_collocation_to_saved": to_marked.tolist(),
            "native_period_residual": error,
            "seam_residual": float(result.max_seam_residual)}


def overlap_omega(row, channel):
    def z(ij):
        return complex(float(row[f"{channel}_omega_{ij}_real"]), float(row[f"{channel}_omega_{ij}_imag"]))
    return np.array([[z("11"), z("12")], [z("12"), z("22")]])


def historical_rows():
    config = previous.load(previous.CONFIG)
    with OVERLAP.open() as stream:
        overlap = {row["overlap_id"]: row for row in csv.DictReader(stream)}
    rows = []
    for point in config["points"]:
        record = overlap[point["id"].split("-")[0]]
        glass, native = (previous.omega_array(point["omega"][ch]) for ch in ("glasses", "theta"))
        word = point["provenance"]["theta_word"]
        if word != record["theta_word"]:
            raise ValueError("atlas word differs from original overlap record")
        W = word_matrix(word)
        B = np.asarray(point["provenance"]["theta_integer_branch"], dtype=int)
        branch = b_shift_matrix(B)
        M = branch@W
        if not np.array_equal(M, config["provenance"]["symplectic_matrix_glasses_to_theta_after_branch"]):
            raise ArithmeticError("branch-composed historical matrix mismatch")
        atlas = transform_omega(W, glass)
        atlas_error = float(np.max(abs(atlas-overlap_omega(record, "theta"))))
        if atlas_error > 1e-10 or np.max(abs(atlas+B-native)) > 1e-10:
            raise ArithmeticError("historical atlas/branch period relation failed")
        fd_word, Ngl, fd = reduce_fd(glass)
        Nth = Ngl@symplectic_inverse(M)
        a = step("glasses-native to atlas-theta", W, glass, ZERO_SPIN)
        b = step("atlas-theta to native-theta (integer branch)", branch, atlas, a["spin_after"])
        c = step("native-theta to common FD", Nth, native, b["spin_after"])
        g = step("native-glasses to common FD", Ngl, glass, ZERO_SPIN)
        if c["spin_after"] != g["spin_after"]:
            raise ArithmeticError("historical spins do not match in common FD")
        channels = {}
        for channel, marked, N in (("theta", native, Nth), ("glasses", glass, Ngl)):
            q = tuple(map(complex, point["q_values"][channel]))
            collocation_branch = OLD_COLLOCATION_MINUS_SAVED[point["id"]] if channel == "theta" else ((0, 0), (0, 0))
            geometry = forward(channel, q, marked, collocation_branch)
            raw_spin = transport(b_shift_matrix(collocation_branch), ZERO_SPIN)
            geometry["raw_collocation_spin"] = raw_spin
            geometry["raw_collocation_to_saved_spin_step"] = step(
                "independent collocation B paths to saved native marking",
                np.asarray(geometry["raw_collocation_to_saved"]),
                previous.omega_array(geometry["omega_forward_raw"]), raw_spin)
            if geometry["raw_collocation_to_saved_spin_step"]["spin_after"] != ZERO_SPIN:
                raise ArithmeticError("collocation path correction has wrong spin")
            if channel == "theta":
                schottky = schottky_theta_period_matrix_cross_ratio(*q, max_word_len=9)
                geometry["Schottky_word9_period"] = schottky.tolist()
                geometry["Schottky_word9_native_residual"] = float(np.max(abs(schottky-marked)))
                if geometry["Schottky_word9_native_residual"] > 1e-8:
                    raise ArithmeticError("historical Schottky q-to-period test failed")
            fd_error = float(np.max(abs(transform_omega(N, previous.omega_array(geometry["omega_forward"]))-fd)))
            omega_charge, P, boson = previous.frame(channel, q, 32)
            fn = theta_physical_fermion_fredholm if channel == "theta" else glasses_physical_fermion_fredholm
            F = fn(q, config["physical_lifts"][channel], max_mode=32).chiral_value
            fixed_free = boson*abs(P*charge_lattice_sum(omega_charge, ZERO_SPIN, cutoff=10))
            legacy_free = boson*abs(F)**2
            fd_value = invariant(fd, g["spin_after"])
            fixed_error = fixed_free/boson**1.5/fd_value-1
            legacy_error = legacy_free/boson**1.5/fd_value-1
            if abs(fixed_error) > 1e-8 or fd_error > 1e-8:
                raise ArithmeticError("historical forward-to-FD free/period test failed")
            channels[channel] = {"q": q, "forward": geometry, "native_to_FD": N.tolist(),
                                 "forward_to_common_FD_period_residual": fd_error,
                                 "all_ten_even_spin_covariance_error": all_even_covariance(N, marked),
                                 "fixed_free_invariant_over_FD_minus_one": float(fixed_error),
                                 "legacy_free_invariant_over_FD_minus_one": float(legacy_error)}
        rows.append({"point_id": point["id"], "atlas_word": word, "branch": B.tolist(),
                     "original_atlas_record_residual": atlas_error, "atlas_branch_composition": M.tolist(),
                     "marking_steps": [a, b, c], "glasses_direct_FD_step": g,
                     "FD_word_from_glasses": fd_word, "omega_FD": fd.tolist(),
                     "FD_domain_margin": float(gottschling_min_margin(fd)),
                     "spin_FD": g["spin_after"], "channels": channels})
    return rows


def current_rows():
    geometry = previous.load(GEOMETRY)
    R = np.asarray(geometry["source_remarking"])
    M = np.asarray(geometry["source_to_target"])
    M0 = M@R
    source_spin = ((1, 1), (0, 0))
    original_spin = transport(symplectic_inverse(R), source_spin)
    rows = []
    for point in geometry["points"]:
        t = point["t"]
        original = np.array([[1j, t+.5j], [t+.5j, 1j]])
        fd_word, N0, fd = reduce_fd(original)
        original_step = step("original period to FD", N0, original, original_spin)
        fd_spin = original_step["spin_after"]
        source_step = step("original period to re-marked NSRR source", R, original, original_spin)
        target_step = step("original period to all-NS target", M0, original, original_spin)
        channels = {}
        for channel, chart, local_spin, from_original, B in (
                ("source", point["source_chart"], source_spin, R, SOURCE_BRANCH),
                ("target", point["target_chart"], ZERO_SPIN, M0, TARGET_BRANCH)):
            q = tuple(map(complex, chart["q_values"]))
            marked = previous.omega_array(chart["omega"])
            if np.max(abs(transform_omega(from_original, original)-marked)) > 1e-12:
                raise ArithmeticError("current saved marked period mismatch")
            N = N0@symplectic_inverse(from_original)
            native_step = step(f"{channel} marked to common FD", N, marked, local_spin)
            if native_step["spin_after"] != fd_spin:
                raise ArithmeticError("current source/target FD spins differ")
            geometry_check = forward("theta", q, marked)
            frame = charged_frame(q, max_mode=32)
            charge_spin = characteristic_in_charge_frame(local_spin, B)
            back = b_shift_matrix(-np.asarray(B))
            charge_step = step(f"{channel} charge-log branch to native marked period", back, frame.omega_charge, charge_spin)
            if charge_step["spin_after"] != local_spin:
                raise ArithmeticError("charge characteristic branch failed")
            fd_from_charge = transform_omega(N@back, frame.omega_charge)
            error = float(np.max(abs(fd_from_charge-fd)))
            if error > 1e-8:
                raise ArithmeticError("current charge period does not reach common FD")
            free_phi = invariant(frame.omega_charge, charge_spin)
            expected_phi = invariant(fd, fd_spin)
            fixed_error = free_phi/expected_phi-1
            if abs(fixed_error) > 1e-8:
                raise ArithmeticError("current fixed free does not agree in FD")
            control = {"q": q, "native_spin": local_spin, "charge_spin": charge_spin,
                       "charge_period_branch": B, "forward": geometry_check,
                       "native_to_FD": N.tolist(), "charge_to_FD": (N@back).tolist(),
                       "charge_to_FD_period_residual": error,
                       "marking_steps": [charge_step, native_step],
                       "all_ten_even_spin_covariance_error": all_even_covariance(N, marked),
                       "fixed_free_invariant_over_FD_minus_one": float(fixed_error),
                       "wrong_untransported_native_spin_in_FD_relative_error": invariant(fd, local_spin)/expected_phi-1}
            if channel == "target":
                F = theta_physical_fermion_fredholm(q, (1, -1, 1), max_mode=32).chiral_value
                legacy_phi = np.linalg.det(2*frame.omega_charge.imag)**.25*abs(F)**2/abs(frame.boson_chiral)
                control["legacy_free_invariant_over_correct_FD_minus_one"] = float(legacy_phi/expected_phi-1)
            channels[channel] = control
        if not np.array_equal(np.asarray(channels["target"]["native_to_FD"])@M,
                              np.asarray(channels["source"]["native_to_FD"])):
            raise ArithmeticError("common FD matrices do not compose exactly")
        rows.append({"t": t, "original_spin": original_spin,
                     "original_to_source_step": source_step, "original_to_target_step": target_step,
                     "original_to_FD_step": original_step, "FD_word_from_original": fd_word,
                     "omega_FD": fd.tolist(), "FD_domain_margin": float(gottschling_min_margin(fd)),
                     "spin_FD": fd_spin, "channels": channels})
    return rows


def run():
    before = protected_hashes()
    result = {"schema": "full-fundamental-spin-marking-audit-v1",
              "scope": "Full marking audit; no changes to blocks, numerator, coefficients or production free factors",
              "provenance_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (OVERLAP, previous.CONFIG, GEOMETRY, Path(__file__))},
              "historical": historical_rows(), "current": current_rows(),
              "protected_kernel_sha256": before,
              "interpretation": "The saved branch-composed period/spin markings pass, including reduction to a common fundamental domain. The remaining legacy-vs-fixed free amplitude difference survives this transport and must not be described as an omitted fundamental-domain marking."}
    if before != protected_hashes():
        raise ArithmeticError("protected kernels changed")
    return serializable(result)


if __name__ == "__main__":
    result = run()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for section in ("historical", "current"):
        for row in result[section]:
            print(section, row.get("point_id", row.get("t")), "FD spin", row["spin_FD"],
                  "FD margin", row["FD_domain_margin"])
    print(OUTPUT)
