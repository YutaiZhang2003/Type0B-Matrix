#!/usr/bin/env python3
r"""Crossing-covariant contact reconstruction for the four-R evaluator.

The evaluator ordering is

``Psi(R4,p4) -> Psitilde(R1,p1) Psitilde(R2,p2) Psitilde(R3,p3)``,

with ``(R1,R2,R3,R4)=(0,z,1,infinity)`` and ``p4=p1+p2+p3``.  Numerical
rank coefficients are first passed through
:func:`subtract_four_ramond_ns_pair_poles`.  The resulting regular vectors
are then fitted to an *exactly* outgoing-fermion-crossing-covariant basis.

The nested basis does not assume that the regular part is Cayley-only.
Degree zero is the unique constant covariant, ``(0,3,1,0)``.  At every
higher total momentum degree, all vector-valued polynomials are projected
with the exact rational ``S_3`` Reynolds operator and exact row reduction.
Consequently degree ``d`` contains degree ``d-1`` literally as its first
columns.  Leave-one-complete-kinematic-sample-out diagnostics are mandatory
parts of every fit result.

Only the declared off-divisor pole lift is scheme dependent.  Changing that
lift shifts the fitted regular functions but does not affect the certified
on-divisor residues or any crossing statement in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import math
from typing import Mapping, Sequence

import numpy as np

from so7e8_four_ramond_assembly import (
    SPIN7_BILINEAR_EXCHANGE_SIGNS,
    SPIN7_FIERZ_02_TO_01,
    spin7_cayley_contact_direction,
)
from so7e8_tree_factorization import subtract_four_ramond_ns_pair_poles


RANK_LABELS = ("T0", "T1", "T2", "T3")
CAYLEY_DIRECTION = (0.0, 3.0, 1.0, 0.0)

Exponent = tuple[int, int, int]
Permutation = tuple[int, int, int]
FractionMatrix = tuple[tuple[Fraction, ...], ...]
SparsePolynomialVector = dict[tuple[int, Exponent], Fraction]


def _fraction_identity() -> FractionMatrix:
    return tuple(
        tuple(Fraction(int(i == j)) for j in range(4)) for i in range(4)
    )


def _fraction_matrix_multiply(
    left: FractionMatrix,
    right: FractionMatrix,
) -> FractionMatrix:
    return tuple(
        tuple(
            sum((left[i][k] * right[k][j] for k in range(4)), Fraction(0))
            for j in range(4)
        )
        for i in range(4)
    )


@lru_cache(maxsize=None)
def _fraction_matrix_inverse(matrix: FractionMatrix) -> FractionMatrix:
    identity = _fraction_identity()
    augmented = [list(matrix[row]) + list(identity[row]) for row in range(4)]
    for column in range(4):
        pivot = next(
            (row for row in range(column, 4) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise AssertionError("a crossing matrix was singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        normalization = augmented[column][column]
        augmented[column] = [value / normalization for value in augmented[column]]
        for row in range(4):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return tuple(tuple(row[4:]) for row in augmented)


_D_EXACT: FractionMatrix = tuple(
    tuple(
        Fraction(int(SPIN7_BILINEAR_EXCHANGE_SIGNS[i]))
        if i == j
        else Fraction(0)
        for j in range(4)
    )
    for i in range(4)
)
_F_EXACT: FractionMatrix = tuple(
    tuple(Fraction(value, 8) for value in row)
    for row in (
        (1, -7, 21, -35),
        (-1, -5, -9, -5),
        (1, -3, 1, 5),
        (-1, -1, 3, 3),
    )
)
_R12_EXACT = tuple(tuple(-value for value in row) for row in _D_EXACT)
_R23_EXACT = tuple(tuple(-value for value in row) for row in _F_EXACT)

if not np.allclose(
    np.asarray(_F_EXACT, dtype=float),
    SPIN7_FIERZ_02_TO_01,
):
    raise AssertionError("the exact contact Fierz matrix disagrees with assembly")
if not np.array_equal(
    np.asarray(CAYLEY_DIRECTION),
    spin7_cayley_contact_direction(),
):
    raise AssertionError("the contact Cayley convention disagrees with assembly")


def _crossing_group() -> tuple[tuple[Permutation, FractionMatrix], ...]:
    r"""Build the exact anti-representation for ``p -> p[permutation]``."""

    identity_permutation = (0, 1, 2)
    matrices: dict[Permutation, FractionMatrix] = {
        identity_permutation: _fraction_identity()
    }
    queue = [identity_permutation]
    generators = (
        ((1, 0, 2), _R12_EXACT),
        ((0, 2, 1), _R23_EXACT),
    )
    while queue:
        permutation = queue.pop(0)
        matrix = matrices[permutation]
        for swap, swap_matrix in generators:
            # Applying ``swap`` after the current permutation sends
            # p -> p[permutation][swap] and C -> R_swap R_permutation C.
            new_permutation = tuple(permutation[swap[index]] for index in range(3))
            new_matrix = _fraction_matrix_multiply(swap_matrix, matrix)
            if new_permutation in matrices:
                if matrices[new_permutation] != new_matrix:
                    raise AssertionError("the exact S3 braid relation failed")
                continue
            matrices[new_permutation] = new_matrix
            queue.append(new_permutation)
    if len(matrices) != 6:
        raise AssertionError("the outgoing crossing group is not S3")
    return tuple(matrices.items())


_OUTGOING_CROSSING_GROUP = _crossing_group()
_OUTGOING_CROSSING_BY_PERMUTATION = dict(_OUTGOING_CROSSING_GROUP)


def outgoing_fermion_crossing_matrix(
    permutation: Sequence[int],
) -> np.ndarray:
    r"""Return ``R_sigma`` in ``C(p[sigma])=R_sigma C(p)``.

    The permutation acts only on outgoing evaluator legs ``R1,R2,R3``.
    In particular, ``(1,0,2)`` returns ``-D`` and ``(0,2,1)`` returns
    ``-F``.  Both common fermion-exchange signs are therefore included.
    """

    raw = tuple(permutation)
    if len(raw) != 3 or any(not isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError("permutation must be one of the six permutations of (0,1,2)")
    key = tuple(map(int, raw))
    if key not in _OUTGOING_CROSSING_BY_PERMUTATION:
        raise ValueError("permutation must be one of the six permutations of (0,1,2)")
    return np.asarray(_OUTGOING_CROSSING_BY_PERMUTATION[key], dtype=float).copy()


def _permuted_exponent(exponent: Exponent, permutation: Permutation) -> Exponent:
    result = [0, 0, 0]
    for source_position, original_position in enumerate(permutation):
        result[original_position] = exponent[source_position]
    return tuple(result)  # type: ignore[return-value]


def _reynolds_covariant(exponent: Exponent, seed_rank: int) -> SparsePolynomialVector:
    r"""Project ``T_seed p^exponent`` to an exact crossing covariant."""

    result: SparsePolynomialVector = {}
    for permutation, matrix in _OUTGOING_CROSSING_GROUP:
        inverse = _fraction_matrix_inverse(matrix)
        transformed_exponent = _permuted_exponent(exponent, permutation)
        for rank in range(4):
            coefficient = inverse[rank][seed_rank]
            if coefficient:
                key = (rank, transformed_exponent)
                result[key] = result.get(key, Fraction(0)) + coefficient
    return {key: value for key, value in result.items() if value}


def _reduce_exact_covariant(
    candidate: SparsePolynomialVector,
    pivots: Mapping[tuple[int, Exponent], SparsePolynomialVector],
) -> SparsePolynomialVector:
    reduced = dict(candidate)
    for pivot, pivot_vector in pivots.items():
        if pivot not in reduced:
            continue
        factor = reduced[pivot]
        for key, coefficient in pivot_vector.items():
            value = reduced.get(key, Fraction(0)) - factor * coefficient
            if value:
                reduced[key] = value
            else:
                reduced.pop(key, None)
    if not reduced:
        return {}
    pivot = max(reduced)
    normalization = reduced[pivot]
    return {key: value / normalization for key, value in reduced.items()}


def _monomial_label(exponent: Exponent) -> str:
    factors = []
    for variable, power in zip(("p1", "p2", "p3"), exponent, strict=True):
        if power == 1:
            factors.append(variable)
        elif power > 1:
            factors.append(f"{variable}^{power}")
    return "*".join(factors) if factors else "1"


@dataclass(frozen=True)
class FourRamondContactFeature:
    """One exact rational vector-valued momentum polynomial."""

    label: str
    degree: int
    component_polynomials: tuple[
        tuple[tuple[Exponent, Fraction], ...],
        tuple[tuple[Exponent, Fraction], ...],
        tuple[tuple[Exponent, Fraction], ...],
        tuple[tuple[Exponent, Fraction], ...],
    ]

    def value(self, p1: complex, p2: complex, p3: complex) -> np.ndarray:
        momenta = tuple(map(complex, (p1, p2, p3)))
        values = np.zeros(4, dtype=np.complex128)
        for rank, polynomial in enumerate(self.component_polynomials):
            values[rank] = sum(
                (
                    complex(coefficient)
                    * np.prod(
                        [
                            momentum**power
                            for momentum, power in zip(
                                momenta, exponent, strict=True
                            )
                        ]
                    )
                    for exponent, coefficient in polynomial
                ),
                0.0j,
            )
        return values


def _feature_from_sparse(
    label: str,
    degree: int,
    sparse: SparsePolynomialVector,
) -> FourRamondContactFeature:
    components = tuple(
        tuple(
            sorted(
                (
                    (exponent, coefficient)
                    for (component, exponent), coefficient in sparse.items()
                    if component == rank
                ),
                key=lambda item: item[0],
            )
        )
        for rank in range(4)
    )
    return FourRamondContactFeature(
        label=label,
        degree=degree,
        component_polynomials=components,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FourRamondContactBasisSpec:
    """A complete nested crossing-covariant polynomial basis."""

    name: str
    maximum_polynomial_degree: int
    features: tuple[FourRamondContactFeature, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(feature.label for feature in self.features)

    @property
    def column_count(self) -> int:
        return len(self.features)


@lru_cache(maxsize=None)
def four_ramond_crossing_polynomial_basis(
    maximum_degree: int,
) -> FourRamondContactBasisSpec:
    r"""Return all exact covariants through a total momentum degree.

    The cumulative dimensions at degrees zero, one, two, and three are
    respectively ``1,3,7,14``.  The first column at every degree is exactly
    the Cayley direction ``(0,3,1,0)`` rather than an arbitrarily normalized
    group projection.
    """

    if not isinstance(maximum_degree, int) or maximum_degree < 0:
        raise ValueError("maximum_degree must be a nonnegative integer")

    pivots: dict[tuple[int, Exponent], SparsePolynomialVector] = {}
    features: list[FourRamondContactFeature] = []

    def try_add(
        candidate: SparsePolynomialVector,
        *,
        label: str,
        degree: int,
    ) -> bool:
        reduced = _reduce_exact_covariant(candidate, pivots)
        if not reduced:
            return False
        pivot = max(reduced)
        pivots[pivot] = reduced
        features.append(_feature_from_sparse(label, degree, reduced))
        return True

    cayley: SparsePolynomialVector = {
        (1, (0, 0, 0)): Fraction(3),
        (2, (0, 0, 0)): Fraction(1),
    }
    if not try_add(cayley, label="degree0:Cayley(3*T1+T2)", degree=0):
        raise AssertionError("the Cayley direction unexpectedly vanished")

    # Certify rather than assume that no second constant covariant exists.
    for seed_rank in range(4):
        if try_add(
            _reynolds_covariant((0, 0, 0), seed_rank),
            label=f"degree0:Reynolds[T{seed_rank}]",
            degree=0,
        ):
            raise AssertionError("the constant crossing-invariant space is not Cayley-only")

    for degree in range(1, maximum_degree + 1):
        exponents = sorted(
            (
                exponent
                for exponent in product(range(degree + 1), repeat=3)
                if sum(exponent) == degree
            ),
            key=lambda exponent: tuple(-power for power in exponent),
        )
        for exponent in exponents:
            for seed_rank in range(4):
                try_add(
                    _reynolds_covariant(exponent, seed_rank),
                    label=(
                        f"degree{degree}:Reynolds["
                        f"{_monomial_label(exponent)}*T{seed_rank}]"
                    ),
                    degree=degree,
                )

    return FourRamondContactBasisSpec(
        name=f"four_ramond_crossing_degree_{maximum_degree}",
        maximum_polynomial_degree=maximum_degree,
        features=tuple(features),
    )


CAYLEY_ONLY_BASIS = four_ramond_crossing_polynomial_basis(0)


def four_ramond_contact_basis_block(
    p1: complex,
    p2: complex,
    p3: complex,
    *,
    basis: FourRamondContactBasisSpec = CAYLEY_ONLY_BASIS,
) -> np.ndarray:
    """Return the ``4 x n`` contact-design block at one kinematic point."""

    return np.column_stack(
        tuple(feature.value(p1, p2, p3) for feature in basis.features)
    ).astype(np.complex128, copy=False)


@dataclass(frozen=True)
class FourRamondContactSample:
    """One full numerical four-R coefficient vector before pole subtraction."""

    p1: complex
    p2: complex
    p3: complex
    p4: complex
    coefficients: tuple[complex, complex, complex, complex]

    def momenta(self) -> tuple[complex, complex, complex, complex]:
        return tuple(map(complex, (self.p1, self.p2, self.p3, self.p4)))

    def values(self) -> np.ndarray:
        values = np.asarray(self.coefficients, dtype=np.complex128)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("sample coefficients must be four finite rank values")
        return values

    @classmethod
    def from_mapping(cls, sample: Mapping[str, object]) -> "FourRamondContactSample":
        """Build a sample from ``coefficients`` or ``A0,...,A3`` keys."""

        momenta = tuple(complex(sample[f"p{index}"]) for index in range(1, 5))
        if "coefficients" in sample:
            raw_coefficients = tuple(sample["coefficients"])  # type: ignore[arg-type]
        else:
            alternatives = (
                ("A0", "rank0", "T0"),
                ("A1", "rank1", "T1"),
                ("A2", "rank2", "T2"),
                ("A3", "rank3", "T3"),
            )
            def coefficient(keys: tuple[str, str, str]) -> object:
                for key in keys:
                    if key in sample:
                        return sample[key]
                raise KeyError(keys[0])

            raw_coefficients = tuple(coefficient(keys) for keys in alternatives)
        coefficients = tuple(map(complex, raw_coefficients))
        if len(coefficients) != 4:
            raise ValueError("coefficients must contain ranks 0,1,2,3")
        return cls(*momenta, coefficients)  # type: ignore[arg-type]

    @classmethod
    def from_evaluation(
        cls,
        external_momenta: Sequence[complex],
        evaluation: object,
    ) -> "FourRamondContactSample":
        """Adapt a four-R evaluator result while keeping momenta explicit."""

        momenta = tuple(map(complex, external_momenta))
        if len(momenta) != 4:
            raise ValueError("external_momenta must contain p1,p2,p3,p4")
        try:
            coefficients = tuple(map(complex, evaluation.coefficients))
        except AttributeError as error:
            raise TypeError("evaluation must expose a coefficients attribute") from error
        if len(coefficients) != 4:
            raise ValueError("evaluation coefficients must have length four")
        return cls(*momenta, coefficients)  # type: ignore[arg-type]


SampleLike = FourRamondContactSample | Mapping[str, object]


def _coerce_sample(sample: SampleLike) -> FourRamondContactSample:
    if isinstance(sample, FourRamondContactSample):
        return sample
    return FourRamondContactSample.from_mapping(sample)


def _checked_sample_momenta(
    sample: FourRamondContactSample,
    *,
    check_energy_conservation: bool,
    tolerance: float,
) -> tuple[complex, complex, complex, complex]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    momenta = sample.momenta()
    if any(
        not (math.isfinite(momentum.real) and math.isfinite(momentum.imag))
        for momentum in momenta
    ):
        raise ValueError("sample momenta must be finite")
    if check_energy_conservation:
        expected = sum(momenta[:3], 0.0j)
        scale = max(1.0, abs(expected), abs(momenta[3]))
        if abs(momenta[3] - expected) > tolerance * scale:
            raise ValueError("four-R samples require p4=p1+p2+p3")
    return momenta


def outgoing_fermion_crossed_sample(
    sample: SampleLike,
    permutation: Sequence[int],
    *,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> FourRamondContactSample:
    r"""Generate one exact outgoing-crossing image of a numerical sample.

    This is useful for checking or presenting a crossing orbit.  Such an
    algebraically generated orbit must not be counted as six independent
    kinematic evaluator samples in model selection.
    """

    coerced = _coerce_sample(sample)
    p1, p2, p3, p4 = _checked_sample_momenta(
        coerced,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    key = tuple(map(int, permutation))
    matrix = outgoing_fermion_crossing_matrix(key)
    finite = (p1, p2, p3)
    crossed_momenta = tuple(finite[index] for index in key)
    crossed_values = matrix @ coerced.values()
    return FourRamondContactSample(
        *crossed_momenta,
        p4,
        tuple(complex(value) for value in crossed_values),  # type: ignore[arg-type]
    )


def outgoing_fermion_crossing_orbit(
    sample: SampleLike,
) -> tuple[FourRamondContactSample, ...]:
    """Return all six exactly related outgoing permutations of one sample."""

    return tuple(
        outgoing_fermion_crossed_sample(sample, permutation)
        for permutation, _ in _OUTGOING_CROSSING_GROUP
    )


def four_ramond_contact_design_matrix(
    samples: Sequence[SampleLike],
    *,
    basis: FourRamondContactBasisSpec = CAYLEY_ONLY_BASIS,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Stack one four-rank crossing-covariant block per kinematic sample."""

    blocks = []
    for raw_sample in samples:
        sample = _coerce_sample(raw_sample)
        p1, p2, p3, _ = _checked_sample_momenta(
            sample,
            check_energy_conservation=check_energy_conservation,
            tolerance=tolerance,
        )
        blocks.append(four_ramond_contact_basis_block(p1, p2, p3, basis=basis))
    if not blocks:
        return np.empty((0, basis.column_count), dtype=np.complex128)
    return np.vstack(blocks)


def four_ramond_contact_target(
    samples: Sequence[SampleLike],
    *,
    subtract_certified_pair_poles: bool = True,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    r"""Stack regular rank vectors after the certified three-pole subtraction."""

    targets = []
    for raw_sample in samples:
        sample = _coerce_sample(raw_sample)
        momenta = _checked_sample_momenta(
            sample,
            check_energy_conservation=check_energy_conservation,
            tolerance=tolerance,
        )
        values = sample.values()
        if subtract_certified_pair_poles:
            subtraction = subtract_four_ramond_ns_pair_poles(
                values,
                momenta,
                check_energy_conservation=False,
            )
            values = np.asarray(subtraction.contact_remainder, dtype=np.complex128)
        targets.append(values)
    if not targets:
        return np.empty(0, dtype=np.complex128)
    return np.concatenate(targets)


def _relative_norm(residual: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.linalg.norm(residual))
    denominator = float(np.linalg.norm(target))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _rank_relative_norms(
    residual: np.ndarray,
    target: np.ndarray,
    original_rows: np.ndarray,
) -> tuple[float, float, float, float]:
    return tuple(
        _relative_norm(residual[original_rows % 4 == rank], target[original_rows % 4 == rank])
        for rank in range(4)
    )  # type: ignore[return-value]


def _scaled_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    rcond: float | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    scales = np.sqrt(np.mean(np.abs(matrix) ** 2, axis=0))
    scales[scales == 0] = 1.0
    scaled = matrix / scales
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        scaled,
        target,
        rcond=rcond,
    )
    return scaled_coefficients / scales, int(rank), singular_values


@dataclass(frozen=True)
class FourRamondContactFitDiagnostics:
    """Fit result with complete-kinematic-sample leave-one-out checks."""

    basis_name: str
    basis_labels: tuple[str, ...]
    coefficients: np.ndarray
    sample_count: int
    full_design_rank: int
    singular_values: np.ndarray
    scaled_condition_number: float
    training_residual_norm: float
    training_relative_residual: float
    training_rank_relative_residuals: tuple[float, float, float, float]
    leave_one_out_residual_norm: float
    leave_one_out_relative_residual: float
    leave_one_out_rank_relative_residuals: tuple[float, float, float, float]
    leave_one_out_sample_relative_residuals: tuple[float, ...]
    leave_one_out_training_ranks: tuple[int, ...]

    @property
    def identifiable(self) -> bool:
        column_count = len(self.basis_labels)
        return (
            self.full_design_rank == column_count
            and bool(self.leave_one_out_training_ranks)
            and min(self.leave_one_out_training_ranks) == column_count
        )

    def coefficient_map(self) -> dict[str, complex]:
        return dict(zip(self.basis_labels, self.coefficients, strict=True))


@dataclass(frozen=True)
class FourRamondContactModelComparison:
    """Nested fits; the reported preference is a diagnostic, not a proof."""

    fits: tuple[FourRamondContactFitDiagnostics, ...]
    best_identifiable_leave_one_out_basis_name: str | None

    def best_fit(self) -> FourRamondContactFitDiagnostics | None:
        return next(
            (
                fit
                for fit in self.fits
                if fit.basis_name == self.best_identifiable_leave_one_out_basis_name
            ),
            None,
        )


def fit_four_ramond_contact_basis(
    samples: Sequence[SampleLike],
    *,
    basis: FourRamondContactBasisSpec = CAYLEY_ONLY_BASIS,
    subtract_certified_pair_poles: bool = True,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
    rcond: float | None = None,
) -> FourRamondContactFitDiagnostics:
    r"""Fit a nested contact basis and leave each kinematic point out once."""

    coerced = tuple(_coerce_sample(sample) for sample in samples)
    if not coerced:
        raise ValueError("at least one sample is required")
    matrix = four_ramond_contact_design_matrix(
        coerced,
        basis=basis,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    target = four_ramond_contact_target(
        coerced,
        subtract_certified_pair_poles=subtract_certified_pair_poles,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    coefficients, rank, singular_values = _scaled_least_squares(
        matrix,
        target,
        rcond=rcond,
    )
    residual = matrix @ coefficients - target
    all_rows = np.arange(len(target), dtype=int)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == basis.column_count and len(singular_values) == basis.column_count
        else float("inf")
    )

    loo_residuals = []
    loo_targets = []
    loo_rows = []
    loo_sample_relative = []
    loo_ranks = []
    for held_out in range(len(coerced)):
        held_rows = np.arange(4 * held_out, 4 * held_out + 4, dtype=int)
        training_rows = np.asarray(
            [row for row in all_rows if row // 4 != held_out],
            dtype=int,
        )
        if len(training_rows) == 0:
            loo_sample_relative.append(float("inf"))
            loo_ranks.append(0)
            continue
        loo_coefficients, loo_rank, _ = _scaled_least_squares(
            matrix[training_rows],
            target[training_rows],
            rcond=rcond,
        )
        held_target = target[held_rows]
        held_residual = matrix[held_rows] @ loo_coefficients - held_target
        loo_residuals.append(held_residual)
        loo_targets.append(held_target)
        loo_rows.append(held_rows)
        loo_sample_relative.append(_relative_norm(held_residual, held_target))
        loo_ranks.append(loo_rank)

    if loo_residuals:
        combined_loo_residual = np.concatenate(loo_residuals)
        combined_loo_target = np.concatenate(loo_targets)
        combined_loo_rows = np.concatenate(loo_rows)
        loo_norm = float(np.linalg.norm(combined_loo_residual))
        loo_relative = _relative_norm(combined_loo_residual, combined_loo_target)
        loo_rank_relative = _rank_relative_norms(
            combined_loo_residual,
            combined_loo_target,
            combined_loo_rows,
        )
    else:
        loo_norm = float("nan")
        loo_relative = float("nan")
        loo_rank_relative = (float("nan"),) * 4

    return FourRamondContactFitDiagnostics(
        basis_name=basis.name,
        basis_labels=basis.labels,
        coefficients=coefficients,
        sample_count=len(coerced),
        full_design_rank=rank,
        singular_values=singular_values,
        scaled_condition_number=condition,
        training_residual_norm=float(np.linalg.norm(residual)),
        training_relative_residual=_relative_norm(residual, target),
        training_rank_relative_residuals=_rank_relative_norms(
            residual,
            target,
            all_rows,
        ),
        leave_one_out_residual_norm=loo_norm,
        leave_one_out_relative_residual=loo_relative,
        leave_one_out_rank_relative_residuals=loo_rank_relative,
        leave_one_out_sample_relative_residuals=tuple(loo_sample_relative),
        leave_one_out_training_ranks=tuple(loo_ranks),
    )


def compare_four_ramond_contact_bases(
    samples: Sequence[SampleLike],
    *,
    maximum_degrees: Sequence[int] = (0, 1, 2),
    subtract_certified_pair_poles: bool = True,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
    rcond: float | None = None,
) -> FourRamondContactModelComparison:
    r"""Compare nested models by leave-one-kinematic-point-out residual.

    An underidentified model is never preferred.  A preferred model is only
    the best member of the supplied finite list under this diagnostic; it is
    not evidence that omitted higher-degree covariants vanish.
    """

    degrees = tuple(dict.fromkeys(map(int, maximum_degrees)))
    if not degrees:
        raise ValueError("maximum_degrees must not be empty")
    fits = tuple(
        fit_four_ramond_contact_basis(
            samples,
            basis=four_ramond_crossing_polynomial_basis(degree),
            subtract_certified_pair_poles=subtract_certified_pair_poles,
            check_energy_conservation=check_energy_conservation,
            tolerance=tolerance,
            rcond=rcond,
        )
        for degree in degrees
    )
    identifiable = [
        fit
        for fit in fits
        if fit.identifiable and np.isfinite(fit.leave_one_out_relative_residual)
    ]
    best = min(
        identifiable,
        key=lambda fit: fit.leave_one_out_relative_residual,
        default=None,
    )
    return FourRamondContactModelComparison(
        fits=fits,
        best_identifiable_leave_one_out_basis_name=(
            None if best is None else best.basis_name
        ),
    )


def contact_vector_from_coefficients(
    p1: complex,
    p2: complex,
    p3: complex,
    coefficients: Sequence[complex],
    *,
    basis: FourRamondContactBasisSpec = CAYLEY_ONLY_BASIS,
) -> np.ndarray:
    """Evaluate one fitted regular vector."""

    values = np.asarray(tuple(coefficients), dtype=np.complex128)
    if values.shape != (basis.column_count,):
        raise ValueError(
            f"coefficients must have shape ({basis.column_count},)"
        )
    return four_ramond_contact_basis_block(p1, p2, p3, basis=basis) @ values


@dataclass(frozen=True)
class FourRamondMomentumDesign:
    """Deterministic amplitude-free safe points for the smallest models."""

    maximum_polynomial_degree: int
    basis_column_count: int
    momenta: tuple[tuple[complex, complex, complex, complex], ...]
    full_design_rank: int
    leave_one_out_min_design_rank: int
    scaled_condition_number: float
    worst_leave_one_out_scaled_condition_number: float
    minimum_chamber_margin: float
    minimum_collision_margin: float
    minimum_divisor_distance: float


def _matrix_rank_and_condition(matrix: np.ndarray) -> tuple[int, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if len(singular_values) == 0:
        return 0, float("inf")
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == matrix.shape[1]
        else float("inf")
    )
    return rank, condition


def _design_combinations(
    candidate_count: int,
    requested_count: int,
    *,
    seed: int,
    maximum_trials: int = 12000,
) -> tuple[tuple[int, ...], ...]:
    total = math.comb(candidate_count, requested_count)
    if total <= maximum_trials:
        return tuple(combinations(range(candidate_count), requested_count))
    rng = np.random.default_rng(seed + 104729)
    choices = {tuple(range(requested_count))}
    while len(choices) < maximum_trials:
        choices.add(
            tuple(
                sorted(
                    map(
                        int,
                        rng.choice(
                            candidate_count,
                            size=requested_count,
                            replace=False,
                        ),
                    )
                )
            )
        )
    return tuple(sorted(choices))


def design_four_ramond_contact_momenta(
    *,
    maximum_polynomial_degree: int = 1,
    count: int | None = None,
    seed: int = 91457,
    candidate_count: int = 40,
    minimum_chamber_margin: float = 0.08,
    minimum_collision_margin: float = 0.025,
    minimum_divisor_distance: float = 0.30,
) -> FourRamondMomentumDesign:
    r"""Select safe, well-conditioned points for degree-zero through two.

    Candidates obey the repository's sufficient undeformed-real-``P``
    chamber, ordinary convergence in all three collision channels, and a
    finite distance from each first NS-pair divisor.  No amplitude is
    evaluated.  The default sample count is the smallest integer for which
    every leave-one-point-out design can have full column rank.
    """

    if maximum_polynomial_degree not in (0, 1, 2):
        raise ValueError(
            "the deterministic safe design is certified only for degrees 0,1,2"
        )
    basis = four_ramond_crossing_polynomial_basis(maximum_polynomial_degree)
    # A single kinematic block has algebraic rank 1,2,3 at degrees 0,1,2,
    # respectively; counting its four tensor rows would underestimate the
    # number of independent kinematic points.  Exact generic stacked ranks
    # saturate after 1,2,4 points, hence one additional point is needed for
    # leave-one-sample-out identification.
    minimum_count = {0: 2, 1: 3, 2: 5}[maximum_polynomial_degree]
    requested_count = minimum_count if count is None else int(count)
    if requested_count < minimum_count:
        raise ValueError(
            "leave-one-kinematic-out identifiability requires at least "
            f"{minimum_count} samples for this basis"
        )
    pool_target = max(int(candidate_count), 8 * requested_count)
    if pool_target < requested_count:
        raise ValueError("candidate_count is smaller than the requested count")

    rng = np.random.default_rng(seed)
    candidates: list[tuple[complex, complex, complex, complex]] = []
    diagnostics: list[tuple[float, float, float]] = []
    attempts = 0
    while len(candidates) < pool_target and attempts < 200 * pool_target:
        attempts += 1
        real_parts = rng.uniform(-0.075, 0.075, size=3)
        imaginary_parts = rng.uniform(0.105, 0.215, size=3)
        finite = tuple(complex(value) for value in real_parts + 1.0j * imaginary_parts)
        p4 = sum(finite, 0.0j)
        chamber_margin = float(
            1.0 - (sum(imaginary_parts) + max(imaginary_parts))
        )
        pair_sums = (
            finite[0] + finite[1],
            finite[1] + finite[2],
            finite[2] + finite[0],
        )
        collision_margin = min(
            float(-(pair_sum * pair_sum).real) for pair_sum in pair_sums
        )
        divisor_distance = min(abs(1.0 + 1.0j * pair_sum) for pair_sum in pair_sums)
        if chamber_margin <= minimum_chamber_margin:
            continue
        if collision_margin <= minimum_collision_margin:
            continue
        if divisor_distance <= minimum_divisor_distance:
            continue
        if min(
            abs(finite[i] - finite[j])
            for i in range(3)
            for j in range(i + 1, 3)
        ) <= 0.018:
            continue
        candidates.append((*finite, p4))
        diagnostics.append((chamber_margin, collision_margin, divisor_distance))
    if len(candidates) < pool_target:
        raise RuntimeError(
            f"only generated {len(candidates)} safe candidates after {attempts} attempts"
        )

    # Forty candidates already give 658,008 five-point subsets for the
    # degree-two minimum design.  _design_combinations deterministically
    # subsamples large finite searches, so a caller may safely enlarge the
    # candidate pool without a combinatorial run-time explosion.
    search_count = len(candidates)
    blocks = np.asarray(
        [
            four_ramond_contact_basis_block(*momenta[:3], basis=basis)
            for momenta in candidates[:search_count]
        ]
    )
    scales = np.sqrt(np.mean(np.abs(blocks) ** 2, axis=(0, 1)))
    scales[scales == 0] = 1.0
    scaled_blocks = blocks / scales

    best_indices: tuple[int, ...] | None = None
    best_score = (float("inf"), float("inf"))
    best_full_rank = 0
    best_loo_rank = 0
    for indices in _design_combinations(
        search_count,
        requested_count,
        seed=seed,
    ):
        full = np.vstack([scaled_blocks[index] for index in indices])
        full_rank, full_condition = _matrix_rank_and_condition(full)
        if full_rank < basis.column_count:
            continue
        loo_diagnostics = [
            _matrix_rank_and_condition(
                np.vstack(
                    [
                        scaled_blocks[index]
                        for position, index in enumerate(indices)
                        if position != held_out
                    ]
                )
            )
            for held_out in range(requested_count)
        ]
        loo_rank = min(rank for rank, _ in loo_diagnostics)
        if loo_rank < basis.column_count:
            continue
        worst_loo_condition = max(condition for _, condition in loo_diagnostics)
        score = (worst_loo_condition, full_condition)
        if score < best_score:
            best_score = score
            best_indices = indices
            best_full_rank = full_rank
            best_loo_rank = loo_rank
    if best_indices is None:
        raise RuntimeError(
            "the safe candidate pool did not contain a leave-one-out identifiable design"
        )

    selected_diagnostics = [diagnostics[index] for index in best_indices]
    return FourRamondMomentumDesign(
        maximum_polynomial_degree=maximum_polynomial_degree,
        basis_column_count=basis.column_count,
        momenta=tuple(candidates[index] for index in best_indices),
        full_design_rank=best_full_rank,
        leave_one_out_min_design_rank=best_loo_rank,
        scaled_condition_number=best_score[1],
        worst_leave_one_out_scaled_condition_number=best_score[0],
        minimum_chamber_margin=min(value[0] for value in selected_diagnostics),
        minimum_collision_margin=min(value[1] for value in selected_diagnostics),
        minimum_divisor_distance=min(value[2] for value in selected_diagnostics),
    )


@dataclass(frozen=True)
class SinglePointCayleyDiagnostic:
    """A projection at one point, explicitly ineligible as a kinematic fit."""

    external_momenta: tuple[complex, complex, complex, complex] | None
    contact_remainder: tuple[complex, complex, complex, complex]
    cayley_coefficient: complex
    projected_contact: tuple[complex, complex, complex, complex]
    residual: tuple[complex, complex, complex, complex]
    residual_norm: float
    relative_residual: float
    provenance: str
    eligible_as_contact_fit: bool = False


def single_point_cayley_projection_diagnostic(
    contact_remainder: Sequence[complex],
    *,
    external_momenta: Sequence[complex] | None = None,
    provenance: str = "user-supplied single contact remainder",
) -> SinglePointCayleyDiagnostic:
    """Project one remainder onto Cayley without treating it as a fit."""

    contact = np.asarray(tuple(contact_remainder), dtype=np.complex128)
    if contact.shape != (4,) or not np.all(np.isfinite(contact)):
        raise ValueError("contact_remainder must contain four finite ranks")
    if external_momenta is None:
        momenta = None
    else:
        momenta = tuple(map(complex, external_momenta))
        if len(momenta) != 4:
            raise ValueError("external_momenta must contain p1,p2,p3,p4")
    cayley = np.asarray(CAYLEY_DIRECTION, dtype=np.complex128)
    coefficient = np.vdot(cayley, contact) / np.vdot(cayley, cayley)
    projected = coefficient * cayley
    residual = contact - projected
    return SinglePointCayleyDiagnostic(
        external_momenta=momenta,  # type: ignore[arg-type]
        contact_remainder=tuple(complex(value) for value in contact),  # type: ignore[arg-type]
        cayley_coefficient=complex(coefficient),
        projected_contact=tuple(complex(value) for value in projected),  # type: ignore[arg-type]
        residual=tuple(complex(value) for value in residual),  # type: ignore[arg-type]
        residual_norm=float(np.linalg.norm(residual)),
        relative_residual=_relative_norm(residual, contact),
        provenance=str(provenance),
    )


# Rounded transcript datum only.  It is deliberately stored as an already
# subtracted remainder, not silently mixed into any least-squares sample set.
DEEP_POINT_ORDER15_MEDIUM_CONTACT_REMAINDER = (
    0.00772 - 0.01784j,
    -0.37644 + 0.03509j,
    -0.12824 + 0.01773j,
    0.00110 - 0.00214j,
)
DEEP_POINT_ORDER15_MEDIUM_MOMENTA = (
    0.05 + 0.12j,
    0.06 + 0.13j,
    0.07 + 0.14j,
    0.18 + 0.39j,
)


def deep_point_order15_medium_cayley_diagnostic() -> SinglePointCayleyDiagnostic:
    r"""Return the rounded conversation datum as a projection, never a fit."""

    return single_point_cayley_projection_diagnostic(
        DEEP_POINT_ORDER15_MEDIUM_CONTACT_REMAINDER,
        external_momenta=DEEP_POINT_ORDER15_MEDIUM_MOMENTA,
        provenance=(
            "rounded order-15 medium-grid contact remainder reported in the "
            "four-R convergence conversation; diagnostic only, not fit input"
        ),
    )


__all__ = [
    "CAYLEY_DIRECTION",
    "CAYLEY_ONLY_BASIS",
    "DEEP_POINT_ORDER15_MEDIUM_CONTACT_REMAINDER",
    "DEEP_POINT_ORDER15_MEDIUM_MOMENTA",
    "FourRamondContactBasisSpec",
    "FourRamondContactFeature",
    "FourRamondContactFitDiagnostics",
    "FourRamondContactModelComparison",
    "FourRamondContactSample",
    "FourRamondMomentumDesign",
    "RANK_LABELS",
    "SinglePointCayleyDiagnostic",
    "compare_four_ramond_contact_bases",
    "contact_vector_from_coefficients",
    "deep_point_order15_medium_cayley_diagnostic",
    "design_four_ramond_contact_momenta",
    "fit_four_ramond_contact_basis",
    "four_ramond_contact_basis_block",
    "four_ramond_contact_design_matrix",
    "four_ramond_contact_target",
    "four_ramond_crossing_polynomial_basis",
    "outgoing_fermion_crossed_sample",
    "outgoing_fermion_crossing_matrix",
    "outgoing_fermion_crossing_orbit",
    "single_point_cayley_projection_diagnostic",
]
