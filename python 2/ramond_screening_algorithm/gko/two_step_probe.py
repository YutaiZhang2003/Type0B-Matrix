#!/usr/bin/env python3
"""Probe the two-consecutive-GKO realization of F x NSR -> Vir x Vir.

This is exploratory code.  The candidate side uses only Theorem 4.5 of
arXiv:2404.14350 twice and the exact diagonal-coset weight dictionary.  Ward
data are imported only by the optional diagnostics at the bottom.

Only ``physical_path_labels`` and the functions whose names start with
``physical_`` or ``reflected_physical_`` implement the Virasoro-weight-correct
Ramond dictionary.  The older orientation search is retained below solely as
a record of a falsified exploratory ansatz; it must not be interpreted as a
second physical path at fixed Ramond branch label.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID = HERE.parents[1] / "ramond_three_point_grid"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GRID))

from audit_gko_relevance import gko_ratio, triangle  # noqa: E402


def minus_one_power(exponent):
    exponent = sp.Rational(exponent)
    if exponent.q != 1:
        raise ValueError(f"nonintegral sign exponent {exponent}")
    return sp.Integer(-1) ** int(exponent)


def signed_gko_ratio(m, n, ell, mu, nu, lam, kappa):
    exponent = (
        (ell - m + n) * (ell - m + n + 1) / 2
        + 4 * n * (m - n) * (m - n - sp.Rational(1, 2))
    )
    return minus_one_power(exponent) * gko_ratio(
        m, n, ell, mu, nu, lam, kappa
    )


def affine_kappa(super_b):
    """k+2 from b^2=-(k+2)/(k+4)."""

    return sp.cancel(-2 * super_b**2 / (1 + super_b**2))


def affine_weight(super_b, momentum, sector, delta=1):
    """Affine weight whose diagonal coset has the requested SCA weight.

    For NS, the level-2 weight and diagonal shift are zero.  For R, the
    level-2 weight is one and the diagonal affine weight differs by
    ``delta`` = +/-1.  The chosen momentum sheet is fixed by the sign in
    x=2*b*P/(1+b^2).
    """

    kappa = affine_kappa(super_b)
    x = sp.cancel(2 * super_b * momentum / (1 + super_b**2))
    if sector == "NS":
        return sp.cancel(-1 + x)
    if sector == "R":
        return sp.cancel(-1 + delta * kappa / 2 + x)
    raise ValueError(sector)


def path_labels(branch, sector, orientation=0, delta=1):
    """Deprecated orientation ansatz (not a physical fixed-n dictionary).

    The two returned orientations differ by a physical branch shift of one
    half.  Use :func:`physical_path_labels` for all actual comparisons.
    """

    branch = sp.Rational(branch)
    if sector == "NS":
        return branch, -branch
    if delta == 1:
        if orientation == 0:
            return branch - sp.Rational(1, 4), sp.Rational(3, 4) - branch
        return branch + sp.Rational(1, 4), sp.Rational(1, 4) - branch
    # Reflection of all affine arrows gives the delta=-1 paths.
    if orientation == 0:
        return -branch + sp.Rational(1, 4), branch - sp.Rational(3, 4)
    return -branch - sp.Rational(1, 4), branch - sp.Rational(1, 4)


def physical_path_labels(branch, sector, delta=1):
    """Branch labels fixed by the two individual Virasoro momenta.

    In a Ramond module the intermediate level-two affine weight is one.
    If ``delta`` is the shift of the final diagonal affine weight, matching
    the first Virasoro momentum gives ``r=n+delta/4``.  Since the level-two
    affine channel imposes ``r+s=delta/2``, the second label is then
    ``s=delta/4-n``.  Thus the two Ramond copies are the two affine arrows,
    not the two ``orientation`` values used in the exploratory function
    above.
    """

    branch = sp.Rational(branch)
    if sector == "NS":
        return branch, -branch
    return branch + sp.Rational(delta, 4), sp.Rational(delta, 4) - branch


def physical_two_step_amplitude(labels, sample, delta2=1, delta3=1):
    """Two consecutive GKO factors with the Virasoro-weight-correct paths."""

    n1, n2, n3 = map(sp.Rational, labels)
    b, p1, p2, p3 = sample
    kappa = affine_kappa(b)
    weights = (
        affine_weight(b, p1, "NS"),
        affine_weight(b, p2, "R", delta2),
        affine_weight(b, p3, "R", delta3),
    )
    r1, s1 = physical_path_labels(n1, "NS")
    r2, s2 = physical_path_labels(n2, "R", delta2)
    r3, s3 = physical_path_labels(n3, "R", delta3)
    if (r1 + r2 + r3).q != 1 or (s1 + s2 + s3).q != 1:
        return sp.Integer(0)
    first = signed_gko_ratio(
        r1, r2, r3, weights[0], weights[1], weights[2], kappa
    )
    shifted = tuple(weight + 2 * r for weight, r in zip(weights, (r1, r2, r3)))
    second = signed_gko_ratio(
        s1, s2, s3, shifted[0], shifted[1], shifted[2], kappa + 1
    )
    return sp.factor(sp.cancel(first * second))


def physical_arrow_pair(labels, sample):
    """Return the two nonzero final-arrow amplitudes for an NS--R--R vertex."""

    answer = []
    for delta2 in (1, -1):
        for delta3 in (1, -1):
            value = physical_two_step_amplitude(labels, sample, delta2, delta3)
            if value != 0:
                answer.append(((delta2, delta3), value))
    return tuple(answer)


def reflected_physical_two_step_amplitude(
    labels, sample, sheets=(1, 1, 1), delta2=1, delta3=1
):
    """Physical GKO path on independently reflected ``(n,P)`` sheets."""

    reflected_labels = tuple(
        sp.Rational(sheet) * sp.Rational(label)
        for sheet, label in zip(sheets, labels)
    )
    b, p1, p2, p3 = sample
    reflected_sample = (
        b,
        sp.Rational(sheets[0]) * p1,
        sp.Rational(sheets[1]) * p2,
        sp.Rational(sheets[2]) * p3,
    )
    return physical_two_step_amplitude(
        reflected_labels, reflected_sample, delta2, delta3
    )


def valid_orientations(n1, n2, n3, delta2=1, delta3=1):
    answer = []
    r1, s1 = path_labels(n1, "NS")
    for orientation2 in (0, 1):
        r2, s2 = path_labels(n2, "R", orientation2, delta2)
        for orientation3 in (0, 1):
            r3, s3 = path_labels(n3, "R", orientation3, delta3)
            if (r1 + r2 + r3).q == 1 and (s1 + s2 + s3).q == 1:
                answer.append((orientation2, orientation3))
    return tuple(answer)


def two_step_amplitude(labels, sample, orientations, delta2=1, delta3=1):
    n1, n2, n3 = map(sp.Rational, labels)
    b, p1, p2, p3 = sample
    kappa = affine_kappa(b)
    weights = (
        affine_weight(b, p1, "NS"),
        affine_weight(b, p2, "R", delta2),
        affine_weight(b, p3, "R", delta3),
    )
    r1, s1 = path_labels(n1, "NS")
    r2, s2 = path_labels(n2, "R", orientations[0], delta2)
    r3, s3 = path_labels(n3, "R", orientations[1], delta3)

    first = signed_gko_ratio(
        r1, r2, r3, weights[0], weights[1], weights[2], kappa
    )
    shifted = tuple(weight + 2 * r for weight, r in zip(weights, (r1, r2, r3)))
    second = signed_gko_ratio(
        s1, s2, s3, shifted[0], shifted[1], shifted[2], kappa + 1
    )
    return sp.factor(sp.cancel(first * second))


def channel_pair(labels, sample, delta2=1, delta3=1):
    orientations = valid_orientations(*labels, delta2, delta3)
    return tuple(
        two_step_amplitude(labels, sample, choice, delta2, delta3)
        for choice in orientations
    )


def diagnostic():
    import compute_grid as grid

    samples = grid.SAMPLES
    cases = (
        (sp.Integer(0), sp.Rational(1, 4), sp.Rational(1, 4)),
        (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4)),
        (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(3, 4)),
    )
    for labels in cases:
        print("labels", labels)
        for sample in samples:
            print(" sample", sample, "physical arrows", physical_arrow_pair(labels, sample))
            masters = []
            for epsilon2 in (0, 1):
                for eta in (1, -1):
                    masters.append(
                        grid.enlarged_raw_three_point(
                            *labels, epsilon2, 0, 0, eta, *sample
                        )[1]
                    )
            print(" masters", tuple(map(sp.factor, masters)))


def feature_vector(labels, sample):
    """All two-step channels obtained from the two affine Ramond arrows."""

    names = []
    values = []
    for delta2 in (1, -1):
        for delta3 in (1, -1):
            orientations = valid_orientations(*labels, delta2, delta3)
            for orientation in orientations:
                names.append((delta2, delta3, orientation))
                values.append(
                    two_step_amplitude(
                        labels, sample, orientation, delta2, delta3
                    )
                )
    return tuple(names), tuple(values)


def hard_span_diagnostic():
    """Numerically test whether the first crossed masters lie in this span."""

    import numpy as np
    import compute_grid as grid
    import fit_signed_sectors as fit

    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    samples = fit.FIT_SAMPLES[:12]
    rows = []
    targets = [[] for _ in range(4)]
    names = None
    for sample in samples:
        current_names, values = feature_vector(labels, sample)
        if names is None:
            names = current_names
        assert names == current_names
        rows.append([complex(sp.N(value, 40)) for value in values])
        for target, (epsilon2, eta) in zip(
            targets, ((0, 1), (0, -1), (1, 1), (1, -1))
        ):
            raw = grid.enlarged_raw_three_point(
                *labels, epsilon2, 0, 0, eta, *sample
            )[1]
            target.append(complex(sp.N(raw, 40)))
    matrix = np.array(rows, dtype=complex)
    print("feature names", names)
    print("feature rank", np.linalg.matrix_rank(matrix, tol=1e-11))
    for key, target in zip(
        ((0, 1), (0, -1), (1, 1), (1, -1)), targets
    ):
        target = np.array(target, dtype=complex)
        coefficients, _, _, _ = np.linalg.lstsq(matrix, target, rcond=1e-12)
        residual = np.max(np.abs(matrix @ coefficients - target))
        sparse = [
            (names[index], coefficient)
            for index, coefficient in enumerate(coefficients)
            if abs(coefficient) > 1e-8
        ]
        print("target", key, "residual", residual, "coefficients", sparse)


def hard_single_sheet_search():
    """Search reflected momentum sheets for a single exact GKO channel."""

    import fit_signed_sectors as fit
    import compute_grid as grid

    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    samples = fit.FIT_SAMPLES[:4]
    targets = ((0, -1),)
    for target_key in targets:
        hits = []
        for sheet2 in (1, -1):
            for sheet3 in (1, -1):
                for delta2 in (1, -1):
                    for delta3 in (1, -1):
                        for orientation in valid_orientations(
                            *labels, delta2, delta3
                        ):
                            ratios = []
                            valid = True
                            for sample in samples:
                                b, p1, p2, p3 = sample
                                reflected = (b, p1, sheet2 * p2, sheet3 * p3)
                                candidate = two_step_amplitude(
                                    labels,
                                    reflected,
                                    orientation,
                                    delta2,
                                    delta3,
                                )
                                raw = grid.enlarged_raw_three_point(
                                    *labels, target_key[0], 0, 0, target_key[1], *sample
                                )[1]
                                if candidate == 0:
                                    valid = False
                                    break
                                ratios.append(sp.factor(sp.cancel(raw / candidate)))
                            if valid and all(
                                sp.simplify(value - ratios[0]) == 0
                                for value in ratios[1:]
                            ):
                                hits.append(
                                    (
                                        sheet2,
                                        sheet3,
                                        delta2,
                                        delta3,
                                        orientation,
                                        ratios[0],
                                    )
                                )
        print("single-sheet hits", target_key, hits)


def hard_reflected_sparse_search():
    """Search one/two constant combinations of all reflected GKO paths."""

    import itertools
    import numpy as np
    import fit_signed_sectors as fit
    import compute_grid as grid

    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    samples = fit.FIT_SAMPLES
    names = []
    specifications = []
    for sheet2 in (1, -1):
        for sheet3 in (1, -1):
            for delta2 in (1, -1):
                for delta3 in (1, -1):
                    for orientation in valid_orientations(
                        *labels, delta2, delta3
                    ):
                        names.append(
                            (sheet2, sheet3, delta2, delta3, orientation)
                        )
                        specifications.append(
                            (sheet2, sheet3, delta2, delta3, orientation)
                        )
    rows = []
    target = []
    for sample in samples:
        b, p1, p2, p3 = sample
        row = []
        for sheet2, sheet3, delta2, delta3, orientation in specifications:
            reflected = (b, p1, sheet2 * p2, sheet3 * p3)
            row.append(
                complex(
                    sp.N(
                        two_step_amplitude(
                            labels,
                            reflected,
                            orientation,
                            delta2,
                            delta3,
                        ),
                        40,
                    )
                )
            )
        rows.append(row)
        raw = grid.enlarged_raw_three_point(
            *labels, 0, 0, 0, -1, *sample
        )[1]
        target.append(complex(sp.N(raw, 40)))
    matrix = np.asarray(rows, dtype=complex)
    target = np.asarray(target, dtype=complex)
    scale = max(1.0, float(np.max(np.abs(target))))
    tolerance = 1e-10 * scale
    hits = []
    for column in range(matrix.shape[1]):
        vector = matrix[:, column]
        coefficient = np.vdot(vector, target) / np.vdot(vector, vector)
        residual = np.max(np.abs(coefficient * vector - target))
        if residual < tolerance:
            hits.append(((names[column],), (coefficient,), residual))
    for first, second in itertools.combinations(range(matrix.shape[1]), 2):
        pair = matrix[:, (first, second)]
        coefficients, _, rank, _ = np.linalg.lstsq(pair, target, rcond=1e-12)
        if rank != 2:
            continue
        residual = np.max(np.abs(pair @ coefficients - target))
        if residual < tolerance:
            hits.append(
                (
                    (names[first], names[second]),
                    tuple(coefficients),
                    residual,
                )
            )
    print("reflected one/two-channel hits", hits[:20], "count", len(hits))


def hard_determinant_search():
    """Test the antisymmetric two-arrow determinant suggested by 2505.23122."""

    import itertools
    import fit_signed_sectors as fit
    import compute_grid as grid

    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    samples = fit.FIT_SAMPLES[:8]
    # For the hard labels both valid choices are (0,0) and (1,1).
    choices = ((0, 0), (1, 1))
    hits = []
    for sheet2, sheet3 in itertools.product((1, -1), repeat=2):
        for selected in itertools.product(choices, repeat=4):
            ratios = []
            for sample in samples:
                b, p1, p2, p3 = sample
                reflected = (b, p1, sheet2 * p2, sheet3 * p3)
                app = two_step_amplitude(labels, reflected, selected[0], 1, 1)
                apm = two_step_amplitude(labels, reflected, selected[1], 1, -1)
                amp = two_step_amplitude(labels, reflected, selected[2], -1, 1)
                amm = two_step_amplitude(labels, reflected, selected[3], -1, -1)
                determinant = sp.factor(sp.cancel(app * amm - apm * amp))
                raw = grid.enlarged_raw_three_point(
                    *labels, 0, 0, 0, -1, *sample
                )[1]
                if determinant == 0:
                    ratios = []
                    break
                ratios.append(sp.factor(sp.cancel(raw / determinant)))
            if ratios and all(
                sp.simplify(value - ratios[0]) == 0 for value in ratios[1:]
            ):
                hits.append((sheet2, sheet3, selected, ratios[0]))
    print("two-arrow determinant hits", hits)


if __name__ == "__main__":
    if "--span" in sys.argv:
        raise SystemExit("deprecated: this used the falsified orientation ansatz")
    elif "--single" in sys.argv:
        raise SystemExit("deprecated: this used the falsified orientation ansatz")
    elif "--sparse" in sys.argv:
        raise SystemExit("deprecated: this used the falsified orientation ansatz")
    elif "--det" in sys.argv:
        raise SystemExit("deprecated: this used the falsified orientation ansatz")
    else:
        diagnostic()
