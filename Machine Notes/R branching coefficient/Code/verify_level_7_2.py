#!/usr/bin/env python3
"""Exact PBW/Ward comparison of the literal path function through onset 7/2.

The formula side is imported from ``screening_kernel.py`` and uses the finite
external-colour path function.  The comparison side instead assembles the simultaneous
Vir x Vir primary directly.  Both sides call the same NS--R--R Ward evaluator;
its zero-mode and parity signs are audited by a separate script.  Thus this is
an exact normalization/path check, not a proof of the Selberg average or its
generic-momentum interpolation.
For an odd NS branch primary we apply the generalized-Ward replacement
eta_eff=(-1)^p_phi eta explicitly.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import time

import sympy as sp


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRID_DIR = REPO / "python 2" / "ramond_three_point_grid"
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_grid as direct  # noqa: E402
import screening_kernel as screening  # noqa: E402


NS_LABELS = (sp.Integer(0), sp.Rational(1, 2), sp.Integer(1))
R_LABELS = (
    sp.Rational(1, 4),
    sp.Rational(3, 4),
    sp.Rational(5, 4),
)
MASTER_CHOICES = tuple(itertools.product((0, 1), (1, -1)))


def direct_master_square(labels, epsilon2, eta, sample):
    """Direct PBW/Ward master, with the intrinsic-primary parity restored."""

    eta_eff = screening.effective_eta(labels, eta)
    b_value, p1, p2, p3 = sample
    _, raw = direct.enlarged_raw_three_point(
        *labels,
        int(epsilon2),
        0,
        0,
        eta_eff,
        b_value,
        p1,
        p2,
        p3,
    )
    norms = direct.raw_norms(
        *labels, int(epsilon2), 0, b_value, p1, p2, p3
    )
    return sp.factor(sp.cancel(raw**2 / sp.prod(norms)))


def audit(sample_count=2):
    began = time.perf_counter()
    master_checks = 0
    discrete_checks = 0
    triples = tuple(itertools.product(NS_LABELS, R_LABELS, R_LABELS))
    for sample_index, sample in enumerate(direct.SAMPLES[:sample_count], start=1):
        for triple_index, labels in enumerate(triples, start=1):
            for epsilon2, eta in MASTER_CHOICES:
                predicted = screening.normalized_master_square(
                    labels, epsilon2, eta, sample
                )
                calculated = direct_master_square(labels, epsilon2, eta, sample)
                residual = sp.factor(sp.cancel(predicted - calculated))
                if residual != 0:
                    raise AssertionError(
                        f"master mismatch sample={sample_index}, labels={labels}, "
                        f"epsilon2={epsilon2}, eta={eta}: {residual}"
                    )
                master_checks += 1

                for epsilon3, form_parity in itertools.product((0, 1), repeat=2):
                    full = screening.normalized_square(
                        labels,
                        epsilon2,
                        epsilon3,
                        form_parity,
                        eta,
                        sample,
                    )
                    expected = (
                        (-1) ** epsilon3
                        * (-sp.I) ** form_parity
                        * calculated
                    )
                    residual = sp.factor(sp.cancel(full - expected))
                    if residual != 0:
                        raise AssertionError(
                            f"discrete mismatch sample={sample_index}, "
                            f"labels={labels}, epsilon2={epsilon2}, "
                            f"epsilon3={epsilon3}, f={form_parity}, eta={eta}: "
                            f"{residual}"
                        )
                    discrete_checks += 1
            print(
                f"sample={sample_index}/{sample_count} "
                f"triple={triple_index:02d}/27 labels={labels} residuals=0",
                flush=True,
            )

    print(
        f"PASS: {master_checks} exact path/PBW master comparisons; "
        f"{discrete_checks} exact discrete reductions; "
        f"elapsed={time.perf_counter()-began:.1f}s"
    )


if __name__ == "__main__":
    audit()
