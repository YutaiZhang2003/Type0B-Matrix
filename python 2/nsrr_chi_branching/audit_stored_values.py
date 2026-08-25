#!/usr/bin/env python3
"""Exact audits of the chi-path formula against every stored NS--R--R value."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys
import time

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
GRID_DIR = ROOT / "python 2" / "ramond_three_point_grid"
for directory in (THIS_DIR, GRID_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as stored  # noqa: E402
import nsrr_chi_formula as formula  # noqa: E402


DISCRETE = tuple(itertools.product((0, 1), (0, 1), (0, 1), (1, -1)))
MASTERS = tuple(itertools.product((0, 1), (1, -1)))


def exact_residual(calculated, expected):
    return sp.factor(sp.cancel(calculated - expected))


def audit_norms():
    checked = 0
    for branch_label in stored.GRID_R_LEVELS:
        for parity in (0, 1):
            for sample in stored.SAMPLES:
                b_value, _, momentum, _ = sample
                direct = stored.ramond_branch.branch_norm(
                    branch_label,
                    parity,
                    substitutions={
                        stored.ramond_branch.Q: b_value + 1 / b_value,
                        stored.ramond_branch.P: momentum,
                    },
                )[3]
                closed = formula.ramond_norm(
                    branch_label, parity, b_value, momentum
                )
                residual = exact_residual(closed, direct)
                if residual != 0:
                    raise AssertionError(
                        ("norm", branch_label, parity, sample, residual)
                    )
                checked += 1
    print(f"norm audit: {checked} exact residuals are zero", flush=True)


def audit_masters(samples=2):
    began = time.perf_counter()
    checked = 0
    triples = tuple(
        itertools.product(
            stored.GRID_NS_LEVELS,
            stored.GRID_R_LEVELS,
            stored.GRID_R_LEVELS,
        )
    )
    for sample_number, sample in enumerate(stored.SAMPLES[:samples], start=1):
        for triple_number, labels in enumerate(triples, start=1):
            for epsilon2, eta in MASTERS:
                calculated = formula.raw_three_point(
                    *labels, epsilon2, 0, 0, eta, *sample
                )[1]
                direct = stored.enlarged_raw_three_point(
                    *labels, epsilon2, 0, 0, eta, *sample
                )[1]
                residual = exact_residual(calculated, direct)
                if residual != 0:
                    raise AssertionError(
                        ("master", sample_number, labels, epsilon2, eta, residual)
                    )
                checked += 1
            print(
                f"master audit sample={sample_number}/{samples} "
                f"triple={triple_number:02d}/27 labels={labels} "
                f"residuals=0 elapsed={time.perf_counter()-began:.1f}s",
                flush=True,
            )
    expected = 108 * samples
    if checked != expected:
        raise AssertionError((checked, expected))
    print(
        f"master audit: {checked} exact residuals are zero; "
        f"elapsed={time.perf_counter()-began:.1f}s",
        flush=True,
    )


def audit_every_stored_value(samples=2):
    """Check all 432 restrictions at each exact stored momentum sample."""

    began = time.perf_counter()
    checked = 0
    triples = tuple(
        itertools.product(
            stored.GRID_NS_LEVELS,
            stored.GRID_R_LEVELS,
            stored.GRID_R_LEVELS,
        )
    )
    for sample_number, sample in enumerate(stored.SAMPLES[:samples], start=1):
        for triple_number, labels in enumerate(triples, start=1):
            # Evaluate the four independent chi-path sums once.  The exact
            # zero-mode calculation below reconstructs every other choice.
            masters = {
                (epsilon2, eta): formula.raw_three_point(
                    *labels, epsilon2, 0, 0, eta, *sample
                )[1]
                for epsilon2, eta in MASTERS
            }
            for epsilon2, epsilon3, form_parity, eta in DISCRETE:
                calculated = (
                    masters[(epsilon2, eta)]
                    * formula.master_reduction_factor(
                        labels[1],
                        labels[2],
                        epsilon2,
                        epsilon3,
                        form_parity,
                        eta,
                    )
                )
                direct = stored.enlarged_raw_three_point(
                    *labels,
                    epsilon2,
                    epsilon3,
                    form_parity,
                    eta,
                    *sample,
                )[1]
                residual = exact_residual(calculated, direct)
                if residual != 0:
                    raise AssertionError(
                        (
                            "full-grid",
                            sample_number,
                            labels,
                            (epsilon2, epsilon3, form_parity, eta),
                            residual,
                        )
                    )
                checked += 1
            print(
                f"full audit sample={sample_number}/{samples} "
                f"triple={triple_number:02d}/27 labels={labels} "
                f"16/16 residuals=0 elapsed={time.perf_counter()-began:.1f}s",
                flush=True,
            )
    expected = 432 * samples
    if checked != expected:
        raise AssertionError((checked, expected))
    print(
        f"full stored-value audit: {checked} exact residuals are zero; "
        f"elapsed={time.perf_counter()-began:.1f}s",
        flush=True,
    )


def audit_high_ns_sample():
    labels = (sp.Rational(3, 2), sp.Rational(3, 4), sp.Rational(3, 4))
    sample = stored.SAMPLES[0]
    checked = 0
    masters = {
        (epsilon2, eta): formula.raw_three_point(
            *labels, epsilon2, 0, 0, eta, *sample
        )[1]
        for epsilon2, eta in MASTERS
    }
    for epsilon2, epsilon3, form_parity, eta in DISCRETE:
        calculated = masters[(epsilon2, eta)] * formula.master_reduction_factor(
            labels[1],
            labels[2],
            epsilon2,
            epsilon3,
            form_parity,
            eta,
        )
        direct = stored.enlarged_raw_three_point(
            *labels,
            epsilon2,
            epsilon3,
            form_parity,
            eta,
            *sample,
        )[1]
        residual = exact_residual(calculated, direct)
        if residual != 0:
            raise AssertionError(("high", labels, epsilon2, epsilon3, form_parity, eta, residual))
        checked += 1
    print(f"high-NS audit: {checked} exact residuals are zero", flush=True)


def audit_signed_ramond_sheets():
    """Check the direct +/- Ramond chain definition beyond the stored grid."""

    sample = stored.SAMPLES[0]
    signed_levels = (
        sp.Rational(1, 4),
        -sp.Rational(1, 4),
        sp.Rational(3, 4),
        -sp.Rational(3, 4),
    )
    checked = 0
    for n2, n3 in itertools.product(signed_levels, repeat=2):
        for epsilon2, eta in MASTERS:
            calculated = formula.raw_three_point(
                0, n2, n3, epsilon2, 0, 0, eta, *sample
            )[1]
            direct = stored.enlarged_raw_three_point(
                0, n2, n3, epsilon2, 0, 0, eta, *sample
            )[1]
            residual = exact_residual(calculated, direct)
            if residual != 0:
                raise AssertionError(
                    ("signed sheets", n2, n3, epsilon2, eta, residual)
                )
            checked += 1
    print(f"signed-sheet audit: {checked} exact residuals are zero", flush=True)


def quick_audit():
    audit_norms()
    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    for sample in stored.SAMPLES:
        b_value, p1, p2, p3 = sample
        for epsilon2, eta in MASTERS:
            calculated = formula.raw_three_point(
                *labels, epsilon2, 0, 0, eta, *sample
            )[1]
            closed = formula.hard_crossed_master(
                epsilon2, eta, b_value, p1, p2, p3
            )
            direct = stored.enlarged_raw_three_point(
                *labels, epsilon2, 0, 0, eta, *sample
            )[1]
            residual = exact_residual(calculated, direct)
            if residual != 0:
                raise AssertionError((labels, sample, epsilon2, eta, residual))
            closed_residual = exact_residual(closed, direct)
            if closed_residual != 0:
                raise AssertionError(
                    ("closed hard", labels, sample, epsilon2, eta, closed_residual)
                )
    print(
        "quick hard-case audit: 8 path and 8 closed-form exact residuals are zero",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masters", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--high", action="store_true")
    parser.add_argument("--samples", type=int, choices=(1, 2), default=2)
    arguments = parser.parse_args()
    if arguments.full:
        audit_norms()
        audit_every_stored_value(arguments.samples)
        audit_high_ns_sample()
        audit_signed_ramond_sheets()
    elif arguments.masters:
        audit_norms()
        audit_masters(arguments.samples)
        if arguments.high:
            audit_high_ns_sample()
    else:
        quick_audit()


if __name__ == "__main__":
    main()
