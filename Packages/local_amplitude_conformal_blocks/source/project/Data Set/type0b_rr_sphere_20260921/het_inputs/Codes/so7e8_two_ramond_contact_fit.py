#!/usr/bin/env python3
r"""Contact-term reconstruction for the two-Ramond numerical evaluator.

This module uses only the ordering of
``evaluate_two_ramond_liouville_convergent``::

    R4(p4) -> R1(p1) + N2(p2) + N3(p3),   p4=p1+p2+p3,
    (R1,N2,N3,R4)=(0,z,1,infinity).

The fitted target is the numerical amplitude minus the *certified* ``z=1``
NS-pair pole.  In particular, no declared ``a_S,a_8,a_48`` mixed-channel
data and no boson-incoming contact ansatz enter this module.

The historical first hypothesis is a small leading-homogeneous basis with
the following assumptions:

* both NS soft limits supply ``u=p2*p3``;
* replacing an NS vector by the raw singlet descendant adds one power of
  momentum, as in the tested RRNS three-point amplitudes;
* ``N2 <-> N3`` sends ``F_SV <-> F_VS``, leaves ``F_SS,F_0`` invariant,
  and negates ``F_2``;
* the strong full-Liouville evidence for zero first mixed residue is used as
  a regularity hypothesis, so no ``D12`` or ``D13`` denominator is present.

Writing ``r=p1``, ``s=p2+p3``, and ``d=p2-p3``, the nine columns are

``F_SS = u * (c_r2*r**2 + c_rs*r*s + c_s2*s**2 + c_u*u)``,
``F_SV = u * (c_mr*r + c_ms*s + c_md*d)``,
``F_VS = u * (c_mr*r + c_ms*s - c_md*d)``,
``F_0  = u * c_0``, and ``F_2 = u*d*c_2``.

This is a reconstruction ansatz, not a claimed amplitude, and current generic
samples reject it.  :func:`crossing_polynomial_basis` therefore supplies
nested complete extensions in the even invariant ring ``(r,s,d^2,u)``;
the odd sectors carry one extra ``d``.  :func:`compare_evaluator_contact_bases`
compares them by leave-one-kinematic-point-out cross-validation and refuses
to select an underdetermined extension.  A nonzero residual calls for more
contact terms (or, more seriously, a reassessment of mixed-channel
regularity), not the silent reuse of the old boson-incoming formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from typing import Mapping, Sequence

import numpy as np

from so7e8_tree_factorization import (
    TwoRamondScalarFunctions,
    ns_pair_divisor,
    ramond_incoming_kernel_ns_pair_residues,
)


TENSOR_LABELS = ("F_SS", "F_SV", "F_VS", "F_0", "F_2")
CONTACT_BASIS_LABELS = (
    "SS:r^2",
    "SS:r*s",
    "SS:s^2",
    "SS:u",
    "SV/VS:r",
    "SV/VS:s",
    "SV/VS:+/-d",
    "F0:1",
    "F2:d",
)


@dataclass(frozen=True)
class ContactBasisSpec:
    """A crossing-covariant contact basis understood by this module."""

    name: str
    labels: tuple[str, ...]
    polynomial_degree: int | None = None


LEADING_HOMOGENEOUS_BASIS = ContactBasisSpec(
    name="leading_homogeneous",
    labels=CONTACT_BASIS_LABELS,
)


def _polynomial_exponents(degree: int) -> tuple[tuple[int, int, int, int], ...]:
    if degree < 0:
        raise ValueError("polynomial degree must be nonnegative")
    exponents = [
        exponent
        for exponent in product(range(degree + 1), repeat=4)
        if sum(exponent) <= degree
    ]
    return tuple(sorted(exponents, key=lambda exponent: (sum(exponent), exponent)))


def _expanded_invariant_monomial(
    exponent: tuple[int, int, int, int],
) -> dict[tuple[int, int, int], Fraction]:
    r_power, s_power, d2_power, u_power = exponent
    # d^2=s^2-4u exactly.  Expand into the independent ring C[r,s,u].
    return {
        (r_power, s_power + 2 * s2_power, u_power + d2_power - s2_power):
        Fraction(math.comb(d2_power, s2_power) * (-4) ** (d2_power - s2_power))
        for s2_power in range(d2_power + 1)
    }


def _independent_polynomial_exponents(
    degree: int,
) -> tuple[tuple[int, int, int, int], ...]:
    r"""Choose an exact basis modulo ``d^2=s^2-4u``."""

    pivots: dict[
        tuple[int, int, int],
        dict[tuple[int, int, int], Fraction],
    ] = {}
    selected = []
    for exponent in _polynomial_exponents(degree):
        vector = _expanded_invariant_monomial(exponent)
        for pivot, pivot_vector in pivots.items():
            if pivot not in vector:
                continue
            factor = vector[pivot]
            for monomial, coefficient in pivot_vector.items():
                new_value = vector.get(monomial, Fraction(0)) - factor * coefficient
                if new_value:
                    vector[monomial] = new_value
                else:
                    vector.pop(monomial, None)
        if not vector:
            continue
        pivot = max(vector)
        normalization = vector[pivot]
        pivots[pivot] = {
            monomial: coefficient / normalization
            for monomial, coefficient in vector.items()
        }
        selected.append(exponent)
    return tuple(selected)


def _monomial_label(exponent: tuple[int, int, int, int]) -> str:
    factors = []
    for variable, power in zip(("r", "s", "d2", "u"), exponent, strict=True):
        if power == 1:
            factors.append(variable)
        elif power > 1:
            factors.append(f"{variable}^{power}")
    return "*".join(factors) if factors else "1"


def crossing_polynomial_basis(degree: int) -> ContactBasisSpec:
    r"""Return the complete degree-``degree`` crossing-polynomial basis.

    The even invariant ring is generated by ``(r,s,d^2,u)``.  Each of the
    five independent crossing sectors gets every monomial of total formal
    degree at most ``degree``.  The odd sectors carry one extra prefactor
    ``d``.  The identity ``d^2=s^2-4u`` is quotiented out exactly, so degree
    zero has 5 columns, degree one has 25, and degree two has 70 rather than
    a rank-deficient 75.
    """

    exponents = _independent_polynomial_exponents(degree)
    monomials = tuple(_monomial_label(exponent) for exponent in exponents)
    labels = tuple(
        f"{sector}:{monomial}"
        for sector in ("SS", "M+", "M-", "F0", "F2")
        for monomial in monomials
    )
    return ContactBasisSpec(
        name=f"crossing_polynomial_degree_{degree}",
        labels=labels,
        polynomial_degree=degree,
    )


@dataclass(frozen=True)
class TwoRamondContactSample:
    """One full-amplitude sample in the numerical evaluator ordering."""

    p1: complex
    p2: complex
    p3: complex
    p4: complex
    F_SS: complex | None
    F_SV: complex | None
    F_VS: complex | None
    F_0: complex | None
    F_2: complex | None

    def momenta(self) -> tuple[complex, complex, complex, complex]:
        return tuple(map(complex, (self.p1, self.p2, self.p3, self.p4)))

    def values(self) -> np.ndarray:
        def optional_complex(value: complex | None) -> complex:
            return complex(np.nan, np.nan) if value is None else complex(value)

        return np.asarray(
            tuple(
                optional_complex(value)
                for value in (self.F_SS, self.F_SV, self.F_VS, self.F_0, self.F_2)
            ),
            dtype=complex,
        )

    @classmethod
    def from_mapping(cls, sample: Mapping[str, complex]) -> "TwoRamondContactSample":
        """Build a sample from generic mapping data.

        Both ``F_0,F_2`` and the common CSV-style spellings ``F0,F2`` are
        accepted.  All other keys use the tensor labels displayed above.
        """

        def value(primary: str, alternate: str | None = None) -> complex:
            if primary in sample:
                return complex(sample[primary])
            if alternate is not None and alternate in sample:
                return complex(sample[alternate])
            raise KeyError(primary)

        def optional_value(
            primary: str,
            alternate: str | None = None,
        ) -> complex | None:
            if primary in sample:
                value_or_none = sample[primary]
            elif alternate is not None and alternate in sample:
                value_or_none = sample[alternate]
            else:
                return None
            return None if value_or_none is None else complex(value_or_none)

        return cls(
            p1=value("p1"),
            p2=value("p2"),
            p3=value("p3"),
            p4=value("p4"),
            F_SS=optional_value("F_SS"),
            F_SV=optional_value("F_SV"),
            F_VS=optional_value("F_VS"),
            F_0=optional_value("F_0", "F0"),
            F_2=optional_value("F_2", "F2"),
        )


@dataclass(frozen=True)
class ContactFitDiagnostics:
    """Least-squares coefficients and train/holdout diagnostics."""

    basis_name: str
    basis_labels: tuple[str, ...]
    coefficients: np.ndarray
    sample_count: int
    training_sample_indices: tuple[int, ...]
    validation_sample_indices: tuple[int, ...]
    full_design_rank: int
    training_design_rank: int
    active_column_indices: tuple[int, ...]
    unconstrained_basis_labels: tuple[str, ...]
    singular_values: np.ndarray
    training_residual_norm: float
    training_relative_residual: float
    training_tensor_relative_residuals: tuple[float, ...]
    validation_residual_norm: float
    validation_relative_residual: float
    leave_one_out_residual_norm: float
    leave_one_out_relative_residual: float
    leave_one_out_tensor_relative_residuals: tuple[float, ...]
    leave_one_out_sample_relative_residuals: tuple[float, ...]
    leave_one_out_min_training_rank: int

    def coefficient_map(self) -> dict[str, complex]:
        return dict(zip(self.basis_labels, self.coefficients, strict=True))

    @property
    def active_column_count(self) -> int:
        return len(self.active_column_indices)

    @property
    def identifiable(self) -> bool:
        return (
            self.training_design_rank == self.active_column_count
            and self.leave_one_out_min_training_rank == self.active_column_count
        )


@dataclass(frozen=True)
class ContactModelComparison:
    """Nested model diagnostics and the leave-one-sample-out selection."""

    fits: tuple[ContactFitDiagnostics, ...]
    selected_basis_name: str | None

    def selected_fit(self) -> ContactFitDiagnostics | None:
        return next(
            (fit for fit in self.fits if fit.basis_name == self.selected_basis_name),
            None,
        )


@dataclass(frozen=True)
class EvaluatorMomentumDesign:
    """A deterministic, amplitude-free generic-chamber momentum design."""

    polynomial_degree: int
    invariant_monomial_count: int
    momenta: tuple[tuple[complex, complex, complex, complex], ...]
    feature_rank: int
    leave_one_out_min_feature_rank: int
    scaled_feature_condition_number: float
    minimum_chamber_margin: float
    minimum_collision_margin: float
    minimum_divisor_distance: float


SampleLike = TwoRamondContactSample | Mapping[str, complex]


def _coerce_sample(sample: SampleLike) -> TwoRamondContactSample:
    if isinstance(sample, TwoRamondContactSample):
        return sample
    return TwoRamondContactSample.from_mapping(sample)


def _checked_momenta(
    sample: TwoRamondContactSample,
    *,
    check_energy_conservation: bool,
    tolerance: float,
) -> tuple[complex, complex, complex, complex]:
    p1, p2, p3, p4 = sample.momenta()
    if check_energy_conservation:
        expected = p1 + p2 + p3
        scale = max(1.0, abs(expected), abs(p4))
        if abs(p4 - expected) > tolerance * scale:
            raise ValueError(
                "evaluator-order samples require p4=p1+p2+p3; "
                f"got residual {p4 - expected!r}"
            )
    return p1, p2, p3, p4


def _crossing_invariant_monomials(
    p1: complex,
    p2: complex,
    p3: complex,
    degree: int,
) -> np.ndarray:
    r, p2, p3 = map(complex, (p1, p2, p3))
    s = p2 + p3
    d2 = (p2 - p3) ** 2
    u = p2 * p3
    invariant_values = (r, s, d2, u)
    return np.asarray(
        [
            np.prod(
                [
                    value**power
                    for value, power in zip(
                        invariant_values,
                        exponent,
                        strict=True,
                    )
                ]
            )
            for exponent in _independent_polynomial_exponents(degree)
        ],
        dtype=complex,
    )


def evaluator_contact_basis_block(
    p1: complex,
    p2: complex,
    p3: complex,
    *,
    basis: ContactBasisSpec = LEADING_HOMOGENEOUS_BASIS,
) -> np.ndarray:
    r"""Return one evaluator-order contact design block.

    Rows have order ``(F_SS,F_SV,F_VS,F_0,F_2)``.  The default is the
    original nine-column leading-homogeneous hypothesis; pass a spec from
    :func:`crossing_polynomial_basis` for a complete nested basis.
    """

    r, p2, p3 = map(complex, (p1, p2, p3))
    s = p2 + p3
    d = p2 - p3
    u = p2 * p3
    if basis.name == LEADING_HOMOGENEOUS_BASIS.name:
        block = np.zeros((5, len(basis.labels)), dtype=complex)
        block[0, 0:4] = u * np.asarray((r * r, r * s, s * s, u))
        block[1, 4:7] = u * np.asarray((r, s, d))
        block[2, 4:7] = u * np.asarray((r, s, -d))
        block[3, 7] = u
        block[4, 8] = u * d
        return block

    if basis.polynomial_degree is None:
        raise ValueError(f"unsupported contact basis {basis.name!r}")
    monomials = _crossing_invariant_monomials(
        r,
        p2,
        p3,
        basis.polynomial_degree,
    )
    count = len(monomials)
    if len(basis.labels) != 5 * count:
        raise ValueError("crossing-polynomial basis labels are inconsistent")
    block = np.zeros((5, len(basis.labels)), dtype=complex)
    block[0, 0 * count : 1 * count] = u * monomials
    # F_SV/F_VS are the sum/difference crossing pair M+ +/- M-.
    block[1, 1 * count : 2 * count] = u * monomials
    block[2, 1 * count : 2 * count] = u * monomials
    block[1, 2 * count : 3 * count] = u * d * monomials
    block[2, 2 * count : 3 * count] = -u * d * monomials
    block[3, 3 * count : 4 * count] = u * monomials
    block[4, 4 * count : 5 * count] = u * d * monomials
    return block


def certified_z1_pole_lift(
    p1: complex,
    p2: complex,
    p3: complex,
    p4: complex,
) -> TwoRamondScalarFunctions:
    r"""Return the chosen off-divisor lift ``R_z1/d23``.

    Only the residue on ``d23=0`` is invariant.  This helper uses the
    polynomial continuation implemented by
    :func:`ramond_incoming_kernel_ns_pair_residues`; changing that
    continuation merely redefines the fitted regular contact coefficients.
    """

    divisor = ns_pair_divisor(p2, p3)
    if divisor == 0:
        raise ZeroDivisionError("the certified pole lift is undefined on d23=0")
    residue = ramond_incoming_kernel_ns_pair_residues(p1, p2, p3, p4)
    return TwoRamondScalarFunctions(
        *(entry / divisor for entry in residue.as_tuple())
    )


def evaluator_contact_design_matrix(
    samples: Sequence[SampleLike],
    *,
    basis: ContactBasisSpec = LEADING_HOMOGENEOUS_BASIS,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Stack one ``5 x 9`` contact block per kinematic sample."""

    blocks: list[np.ndarray] = []
    for raw_sample in samples:
        sample = _coerce_sample(raw_sample)
        p1, p2, p3, _ = _checked_momenta(
            sample,
            check_energy_conservation=check_energy_conservation,
            tolerance=tolerance,
        )
        blocks.append(evaluator_contact_basis_block(p1, p2, p3, basis=basis))
    if not blocks:
        return np.empty((0, len(basis.labels)), dtype=complex)
    return np.vstack(blocks)


def evaluator_contact_target(
    samples: Sequence[SampleLike],
    *,
    subtract_certified_z1_pole: bool = True,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Stack amplitudes, optionally subtracting only the certified pole."""

    rows: list[np.ndarray] = []
    for raw_sample in samples:
        sample = _coerce_sample(raw_sample)
        p1, p2, p3, p4 = _checked_momenta(
            sample,
            check_energy_conservation=check_energy_conservation,
            tolerance=tolerance,
        )
        values = sample.values()
        if subtract_certified_z1_pole:
            values = values - np.asarray(
                certified_z1_pole_lift(p1, p2, p3, p4).as_tuple(),
                dtype=complex,
            )
        rows.append(values)
    if not rows:
        return np.empty(0, dtype=complex)
    return np.concatenate(rows)


def _sample_rows(indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [5 * sample + tensor for sample in indices for tensor in range(5)],
        dtype=int,
    )


def _relative_norm(residual: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.linalg.norm(residual))
    denominator = float(np.linalg.norm(target))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _tensor_relative_norms(
    residual: np.ndarray,
    target: np.ndarray,
    original_rows: np.ndarray,
) -> tuple[float, ...]:
    values = []
    for tensor in range(5):
        mask = original_rows % 5 == tensor
        values.append(
            _relative_norm(residual[mask], target[mask])
            if np.any(mask)
            else float("nan")
        )
    return tuple(values)


def _finite_complex(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values.real) & np.isfinite(values.imag)


def _solve_scaled_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    active_columns: Sequence[int],
    *,
    total_column_count: int,
    rcond: float | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve after RMS column scaling and return unscaled coefficients."""

    active = np.asarray(active_columns, dtype=int)
    coefficients = np.zeros(total_column_count, dtype=complex)
    if len(active) == 0:
        return coefficients, 0, np.empty(0)
    reduced = matrix[:, active]
    scales = np.sqrt(np.mean(np.abs(reduced) ** 2, axis=0))
    scales[scales == 0] = 1.0
    scaled = reduced / scales
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        scaled,
        target,
        rcond=rcond,
    )
    coefficients[active] = scaled_coefficients / scales
    return coefficients, int(rank), singular_values


def fit_evaluator_contact_basis(
    samples: Sequence[SampleLike],
    *,
    basis: ContactBasisSpec = LEADING_HOMOGENEOUS_BASIS,
    validation_sample_indices: Sequence[int] = (),
    subtract_certified_z1_pole: bool = True,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
    rcond: float | None = None,
) -> ContactFitDiagnostics:
    r"""Fit one evaluator-order contact basis.

    Cross-validation is split by complete kinematic samples rather than by
    tensor rows.  Pass (for example) ``validation_sample_indices=(1,5)`` to
    keep those samples entirely out of the least-squares solve.  Empty
    validation data are reported by ``nan`` validation norms.
    """

    coerced = tuple(_coerce_sample(sample) for sample in samples)
    if not coerced:
        raise ValueError("at least one sample is required")
    sample_count = len(coerced)
    validation = tuple(sorted(set(map(int, validation_sample_indices))))
    if any(index < 0 or index >= sample_count for index in validation):
        raise IndexError("validation_sample_indices contains an invalid sample index")
    validation_set = set(validation)
    training = tuple(index for index in range(sample_count) if index not in validation_set)
    if not training:
        raise ValueError("at least one training sample is required")

    matrix = evaluator_contact_design_matrix(
        coerced,
        basis=basis,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    target = evaluator_contact_target(
        coerced,
        subtract_certified_z1_pole=subtract_certified_z1_pole,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    observed = _finite_complex(target)
    if not np.any(observed):
        raise ValueError("the samples contain no finite amplitude values")
    observed_matrix = matrix[observed]
    active_columns = tuple(
        int(index)
        for index in np.flatnonzero(np.any(np.abs(observed_matrix) > 0, axis=0))
    )
    unconstrained_labels = tuple(
        label
        for index, label in enumerate(basis.labels)
        if index not in set(active_columns)
    )

    training_rows_all = _sample_rows(training)
    training_rows = training_rows_all[observed[training_rows_all]]
    if len(training_rows) == 0:
        raise ValueError("the training samples contain no finite amplitude values")
    training_matrix = matrix[training_rows]
    training_target = target[training_rows]
    coefficients, training_rank, singular_values = _solve_scaled_least_squares(
        training_matrix,
        training_target,
        active_columns,
        total_column_count=len(basis.labels),
        rcond=rcond,
    )
    training_residual = training_matrix @ coefficients - training_target

    if validation:
        validation_rows_all = _sample_rows(validation)
        validation_rows = validation_rows_all[observed[validation_rows_all]]
        validation_target = target[validation_rows]
        validation_residual = matrix[validation_rows] @ coefficients - validation_target
        validation_norm = float(np.linalg.norm(validation_residual))
        validation_relative = _relative_norm(validation_residual, validation_target)
    else:
        validation_norm = float("nan")
        validation_relative = float("nan")

    loo_residuals: list[np.ndarray] = []
    loo_targets: list[np.ndarray] = []
    loo_original_rows: list[np.ndarray] = []
    loo_sample_relative: list[float] = []
    loo_ranks: list[int] = []
    all_indices = tuple(range(sample_count))
    for held_out in all_indices:
        held_rows_all = _sample_rows((held_out,))
        held_rows = held_rows_all[observed[held_rows_all]]
        if len(held_rows) == 0:
            loo_sample_relative.append(float("nan"))
            continue
        loo_training_indices = tuple(
            index for index in all_indices if index != held_out
        )
        loo_training_rows_all = _sample_rows(loo_training_indices)
        loo_training_rows = loo_training_rows_all[observed[loo_training_rows_all]]
        if len(loo_training_rows) == 0:
            loo_sample_relative.append(float("inf"))
            loo_ranks.append(0)
            continue
        loo_coefficients, loo_rank, _ = _solve_scaled_least_squares(
            matrix[loo_training_rows],
            target[loo_training_rows],
            active_columns,
            total_column_count=len(basis.labels),
            rcond=rcond,
        )
        held_target = target[held_rows]
        held_residual = matrix[held_rows] @ loo_coefficients - held_target
        loo_residuals.append(held_residual)
        loo_targets.append(held_target)
        loo_original_rows.append(held_rows)
        loo_sample_relative.append(_relative_norm(held_residual, held_target))
        loo_ranks.append(loo_rank)

    if loo_residuals:
        combined_loo_residual = np.concatenate(loo_residuals)
        combined_loo_target = np.concatenate(loo_targets)
        combined_loo_rows = np.concatenate(loo_original_rows)
        loo_norm = float(np.linalg.norm(combined_loo_residual))
        loo_relative = _relative_norm(combined_loo_residual, combined_loo_target)
        loo_tensor_relative = _tensor_relative_norms(
            combined_loo_residual,
            combined_loo_target,
            combined_loo_rows,
        )
    else:
        loo_norm = float("nan")
        loo_relative = float("nan")
        loo_tensor_relative = (float("nan"),) * 5
    loo_min_rank = min(loo_ranks) if loo_ranks else 0

    return ContactFitDiagnostics(
        basis_name=basis.name,
        basis_labels=basis.labels,
        coefficients=coefficients,
        sample_count=sample_count,
        training_sample_indices=training,
        validation_sample_indices=validation,
        full_design_rank=int(np.linalg.matrix_rank(observed_matrix)),
        training_design_rank=int(training_rank),
        active_column_indices=active_columns,
        unconstrained_basis_labels=unconstrained_labels,
        singular_values=singular_values,
        training_residual_norm=float(np.linalg.norm(training_residual)),
        training_relative_residual=_relative_norm(
            training_residual,
            training_target,
        ),
        training_tensor_relative_residuals=_tensor_relative_norms(
            training_residual,
            training_target,
            training_rows,
        ),
        validation_residual_norm=validation_norm,
        validation_relative_residual=validation_relative,
        leave_one_out_residual_norm=loo_norm,
        leave_one_out_relative_residual=loo_relative,
        leave_one_out_tensor_relative_residuals=loo_tensor_relative,
        leave_one_out_sample_relative_residuals=tuple(loo_sample_relative),
        leave_one_out_min_training_rank=loo_min_rank,
    )


def compare_evaluator_contact_bases(
    samples: Sequence[SampleLike],
    *,
    polynomial_degrees: Sequence[int] = (0, 1, 2),
    include_leading_homogeneous: bool = True,
    subtract_certified_z1_pole: bool = True,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
    rcond: float | None = None,
) -> ContactModelComparison:
    r"""Compare nested crossing-polynomial bases by kinematic LOOCV.

    A model is selectable only when every leave-one-sample-out training
    design identifies all columns touched by the available tensor data.
    This prevents a high-degree interpolant from winning with too few
    kinematic points.  Missing tensor sectors are listed as unconstrained
    rather than making the observed sectors unusable.
    """

    specs: list[ContactBasisSpec] = []
    if include_leading_homogeneous:
        specs.append(LEADING_HOMOGENEOUS_BASIS)
    specs.extend(crossing_polynomial_basis(degree) for degree in polynomial_degrees)
    fits = tuple(
        fit_evaluator_contact_basis(
            samples,
            basis=spec,
            subtract_certified_z1_pole=subtract_certified_z1_pole,
            check_energy_conservation=check_energy_conservation,
            tolerance=tolerance,
            rcond=rcond,
        )
        for spec in specs
    )
    identifiable = [
        fit
        for fit in fits
        if fit.identifiable and np.isfinite(fit.leave_one_out_relative_residual)
    ]
    selected = min(
        identifiable,
        key=lambda fit: fit.leave_one_out_relative_residual,
        default=None,
    )
    return ContactModelComparison(
        fits=fits,
        selected_basis_name=None if selected is None else selected.basis_name,
    )


def design_evaluator_contact_momenta(
    *,
    polynomial_degree: int = 1,
    count: int | None = None,
    seed: int = 71823,
    candidate_count: int = 1200,
    minimum_chamber_margin: float = 0.06,
    minimum_collision_margin: float = 0.025,
    minimum_divisor_distance: float = 0.06,
) -> EvaluatorMomentumDesign:
    r"""Select a well-conditioned generic-chamber momentum design.

    No amplitude or conformal block is evaluated.  A deterministic candidate
    pool is filtered by the repository's sufficient real-``P`` chamber,
    ordinary collision convergence, and distance from the three first
    divisors.  Greedy row-space/D-optimal selection then conditions the
    invariant monomial matrix.

    The default count is the exact minimum for leave-one-kinematic-point-out
    identifiability: one more than the number of independent invariant
    monomials.  It is six points at degree one and fifteen at degree two.
    """

    basis = crossing_polynomial_basis(polynomial_degree)
    monomial_count = len(basis.labels) // 5
    requested_count = monomial_count + 1 if count is None else int(count)
    if requested_count < monomial_count + 1:
        raise ValueError(
            "leave-one-out identifiability requires at least "
            f"{monomial_count + 1} points at degree {polynomial_degree}"
        )
    pool_target = max(int(candidate_count), 50 * requested_count)
    if pool_target < requested_count:
        raise ValueError("candidate_count is smaller than the requested design")

    rng = np.random.default_rng(seed)
    candidates: list[tuple[complex, complex, complex, complex]] = []
    diagnostics: list[tuple[float, float, float]] = []
    attempts = 0
    maximum_attempts = 200 * pool_target
    while len(candidates) < pool_target and attempts < maximum_attempts:
        attempts += 1
        real_parts = rng.uniform(-0.065, 0.075, size=3)
        imaginary_parts = rng.uniform(0.13, 0.27, size=3)
        p1, p2, p3 = tuple(real_parts + 1.0j * imaginary_parts)
        p4 = p1 + p2 + p3
        chamber_margin = float(
            1.0 - (sum(imaginary_parts) + max(imaginary_parts))
        )
        pair_sums = (p1 + p2, p2 + p3, p1 + p3)
        collision_margin = min(float(-(pair_sum * pair_sum).real) for pair_sum in pair_sums)
        divisor_distance = min(
            abs(0.5 + 1.0j * (p1 + p2)),
            abs(1.0 + 1.0j * (p2 + p3)),
            abs(0.5 + 1.0j * (p1 + p3)),
        )
        if chamber_margin <= minimum_chamber_margin:
            continue
        if collision_margin <= minimum_collision_margin:
            continue
        if divisor_distance <= minimum_divisor_distance:
            continue
        if abs(p2 * p3) <= 0.015 or abs(p2 - p3) <= 0.025:
            continue
        candidates.append((p1, p2, p3, p4))
        diagnostics.append((chamber_margin, collision_margin, divisor_distance))
    if len(candidates) < pool_target:
        raise RuntimeError(
            f"only generated {len(candidates)} safe candidates after {attempts} attempts"
        )

    features = np.vstack(
        [
            _crossing_invariant_monomials(p1, p2, p3, polynomial_degree)
            for p1, p2, p3, _ in candidates
        ]
    )
    feature_scales = np.sqrt(np.mean(np.abs(features) ** 2, axis=0))
    feature_scales[feature_scales == 0] = 1.0
    scaled_features = features / feature_scales

    selected: list[int] = []
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(requested_count):
        if selected:
            selected_matrix = scaled_features[selected]
            _, singular_values, right_vectors = np.linalg.svd(
                selected_matrix,
                full_matrices=False,
            )
            tolerance = max(selected_matrix.shape) * np.finfo(float).eps * singular_values[0]
            rank = int(np.sum(singular_values > tolerance))
        else:
            selected_matrix = np.empty((0, monomial_count), dtype=complex)
            right_vectors = np.empty((0, monomial_count), dtype=complex)
            rank = 0

        if rank < monomial_count:
            row_basis = right_vectors[:rank]
            projection = (
                (scaled_features @ row_basis.conj().T) @ row_basis
                if rank
                else 0.0
            )
            residual = scaled_features - projection
            scores = np.sum(np.abs(residual) ** 2, axis=1)
        else:
            gram_inverse = np.linalg.inv(selected_matrix.conj().T @ selected_matrix)
            scores = np.real(
                np.einsum(
                    "ij,jk,ik->i",
                    scaled_features,
                    gram_inverse,
                    scaled_features.conj(),
                )
            )
        scores[~available] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        available[chosen] = False

    selected_features = scaled_features[selected]
    feature_rank = int(np.linalg.matrix_rank(selected_features))
    loo_ranks = tuple(
        int(np.linalg.matrix_rank(np.delete(selected_features, held_out, axis=0)))
        for held_out in range(requested_count)
    )
    if min(loo_ranks) < monomial_count:
        raise RuntimeError(
            "the deterministic candidate design failed leave-one-out rank; "
            "increase candidate_count or change seed"
        )
    selected_diagnostics = [diagnostics[index] for index in selected]
    return EvaluatorMomentumDesign(
        polynomial_degree=polynomial_degree,
        invariant_monomial_count=monomial_count,
        momenta=tuple(candidates[index] for index in selected),
        feature_rank=feature_rank,
        leave_one_out_min_feature_rank=min(loo_ranks),
        scaled_feature_condition_number=float(np.linalg.cond(selected_features)),
        minimum_chamber_margin=min(value[0] for value in selected_diagnostics),
        minimum_collision_margin=min(value[1] for value in selected_diagnostics),
        minimum_divisor_distance=min(value[2] for value in selected_diagnostics),
    )


def contact_functions_from_coefficients(
    p1: complex,
    p2: complex,
    p3: complex,
    coefficients: Sequence[complex],
    *,
    basis: ContactBasisSpec = LEADING_HOMOGENEOUS_BASIS,
) -> TwoRamondScalarFunctions:
    """Evaluate the contact basis for a coefficient vector."""

    coefficient_array = np.asarray(coefficients, dtype=complex)
    if coefficient_array.shape != (len(basis.labels),):
        raise ValueError(
            f"coefficients must have shape ({len(basis.labels)},)"
        )
    values = evaluator_contact_basis_block(p1, p2, p3, basis=basis) @ coefficient_array
    return TwoRamondScalarFunctions(*map(complex, values))


__all__ = [
    "CONTACT_BASIS_LABELS",
    "ContactBasisSpec",
    "ContactFitDiagnostics",
    "ContactModelComparison",
    "EvaluatorMomentumDesign",
    "LEADING_HOMOGENEOUS_BASIS",
    "TENSOR_LABELS",
    "TwoRamondContactSample",
    "certified_z1_pole_lift",
    "compare_evaluator_contact_bases",
    "contact_functions_from_coefficients",
    "crossing_polynomial_basis",
    "design_evaluator_contact_momenta",
    "evaluator_contact_basis_block",
    "evaluator_contact_design_matrix",
    "evaluator_contact_target",
    "fit_evaluator_contact_basis",
]
