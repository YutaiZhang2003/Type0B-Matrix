"""Reuse descendant suffixes and sparse MP products in physical action solves.

The branch states, embedded Virasoro actions and span columns are those of
``compute_target``. Only construction reuse and the arithmetic work needed
to solve the same columns change. No conformal block or saved coefficient
is used here.
"""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np
from scipy import linalg as scipy_linalg

_BRANCH_DIR = Path(__file__).resolve().parents[1]/"ramond_branching_recursion"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))
import compute_target as br


class CachedActionModule(br.FreeFieldModule):
    """Action-local descendant suffix cache at one fixed MP precision.

    Full requested span columns belong to their caller and are not inserted
    into this cache. Recursive suffixes are retained only until the current
    action solve exits, including on failure. Expressions are read-only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._descendant_cache = None
        self._embedded_cache = None
        self._cache_primaries = None
        self._cache_stats = None
        self._action_precision = br.MP_DPS, mp.mp.dps

    @contextmanager
    def descendant_cache(self):
        if self._descendant_cache is not None:
            raise RuntimeError("A descendant action cache is already active")
        precision = br.MP_DPS, mp.mp.dps
        if precision != self._action_precision:
            raise RuntimeError("A physical action module cannot be reused at a different precision")
        self._descendant_cache, self._cache_primaries, self._embedded_cache = {}, {}, {}
        stats = {"requests": 0, "suffix_requests": 0, "cache_hits": 0,
                 "embedded_image_requests": 0, "embedded_image_hits": 0}
        self._cache_stats = stats
        try:
            yield stats
        finally:
            stats["retained_suffixes"] = len(self._descendant_cache)
            stats["retained_suffix_coefficients"] = sum(
                len(expression) for expression in self._descendant_cache.values())
            stats["retained_embedded_images"] = len(self._embedded_cache)
            stats["retained_embedded_image_coefficients"] = sum(
                len(expression) for expression in self._embedded_cache.values())
            self._descendant_cache = self._cache_primaries = self._cache_stats = None
            self._embedded_cache = None
            if (br.MP_DPS, mp.mp.dps) != precision:
                raise RuntimeError("Arithmetic precision changed inside a descendant action cache")

    def apply_embedded(self, copy, mode, expression):
        if self._embedded_cache is None:
            return super().apply_embedded(copy, mode, expression)

        def basis_image(state):
            self._cache_stats["embedded_image_requests"] += 1
            key = copy, mode, state
            if key in self._embedded_cache:
                self._cache_stats["embedded_image_hits"] += 1
                return self._embedded_cache[key]
            # Use precisely the inherited embedded-generator formula on one
            # oscillator basis vector, then extend by linearity. Cached
            # images are read-only and live only for the current action.
            image = super(CachedActionModule, self).apply_embedded(
                copy, mode, {state: br.complex_number(1)})
            self._embedded_cache[key] = image
            return image

        return br.apply_expression(expression, basis_image)

    def descendant(self, primary, first_partition, second_partition):
        if self._descendant_cache is None:
            return super().descendant(primary, first_partition, second_partition)
        # Retain a strong reference: an id cannot be recycled while cached.
        primary_id = id(primary)
        self._cache_primaries[primary_id] = primary
        self._cache_stats["requests"] += 1
        return self._descendant_suffix(primary_id, tuple(first_partition),
                                       tuple(second_partition), retain=False)

    def _descendant_suffix(self, primary_id, first, second, *, retain=True):
        self._cache_stats["suffix_requests"] += 1
        key = primary_id, first, second
        if key in self._descendant_cache:
            self._cache_stats["cache_hits"] += 1
            return self._descendant_cache[key]
        if first:
            suffix = self._descendant_suffix(primary_id, first[1:], second)
            answer = self.apply_embedded(1, -first[0], suffix)
        elif second:
            suffix = self._descendant_suffix(primary_id, (), second[1:])
            answer = self.apply_embedded(2, -second[0], suffix)
        else:
            return self._cache_primaries[primary_id]
        if retain:
            self._descendant_cache[key] = answer
        return answer


def action_span_fit(target, columns):
    """Original span equations with once-normalized sparse refinement rows.

    Double precision chooses a full-rank restriction and supplies only the
    LU preconditioner. Refinement retains compute_target's MP precision and
    selected-row tolerance. The original-column full-row residual is also
    evaluated and explicitly required to satisfy the outer Ward tolerance.
    """
    if not br.MP_DPS:
        return br.span_fit(target, columns)
    started = time.perf_counter()
    column_count = len(columns)
    keys = sorted(set(target).union(*(set(column) for column in columns)), key=repr)
    index = {key: row for row, key in enumerate(keys)}
    norms = [mp.sqrt(mp.re(br.sparse_inner(column, column))) for column in columns]
    if any(norm == 0 for norm in norms):
        raise AssertionError("A descendant column vanished at the sample point.")
    normalized_rows = [[] for _ in keys]
    shadow = np.zeros((len(keys), column_count), dtype=np.complex128)
    for column_index, (column, norm) in enumerate(zip(columns, norms)):
        inverse_norm = 1/norm
        for key, value in column.items():
            normalized = value*inverse_norm
            row = index[key]
            normalized_rows[row].append((column_index, normalized))
            shadow[row, column_index] = complex(normalized)
    assembly_seconds = time.perf_counter()-started
    factor_started = time.perf_counter()
    _, _, pivots = scipy_linalg.qr(
        shadow.T, mode="economic", pivoting=True, check_finite=False)
    selected_indices = tuple(int(index) for index in pivots[:column_count])
    selected = shadow[list(selected_indices), :]
    singular_values = scipy_linalg.svdvals(selected, check_finite=False)
    rank = int(np.count_nonzero(singular_values > br.RANK_TOLERANCE))
    if rank != column_count:
        raise np.linalg.LinAlgError(
            f"Pivoted oscillator restriction has rank {rank}/{column_count}.")
    selected_keys = tuple(keys[index] for index in selected_indices)
    selected_vector = np.asarray([complex(target.get(key, 0)) for key in selected_keys])
    lu = scipy_linalg.lu_factor(selected, check_finite=False)
    initial = scipy_linalg.lu_solve(lu, selected_vector, check_finite=False)
    values = [br.complex_number(value) for value in initial]
    factorization_seconds = time.perf_counter()-factor_started
    refine_started = time.perf_counter()
    selected_scale = max(mp.mpf(1), mp.sqrt(mp.fsum(
        abs(target.get(key, 0))**2 for key in selected_keys)))
    tolerance = mp.power(10, -max(20, br.MP_DPS-15))
    selected_relative_residual = mp.inf
    refinement_iterations = 0
    for iteration in range(1, 31):
        residual = [target.get(keys[row], 0)-mp.fsum(
            entry*values[column] for column, entry in normalized_rows[row])
            for row in selected_indices]
        selected_relative_residual = mp.sqrt(mp.fsum(
            abs(value)**2 for value in residual))/selected_scale
        if selected_relative_residual <= tolerance:
            break
        correction = scipy_linalg.lu_solve(
            lu, np.asarray([complex(value) for value in residual]), check_finite=False)
        values = [value+br.complex_number(delta)
                  for value, delta in zip(values, correction)]
        refinement_iterations = iteration
    else:
        raise FloatingPointError(
            "Mixed-precision descendant refinement did not reach the requested tolerance.")
    coefficients = [value/norm for value, norm in zip(values, norms)]
    # Certify the original, unnormalized columns on every oscillator row.
    # This also detects a loss of accuracy in the normalized-row arithmetic.
    all_row_residual = {key: -target.get(key, 0) for key in keys}
    for coefficient, column in zip(coefficients, columns):
        for state, value in column.items():
            all_row_residual[state] += coefficient*value
    absolute = mp.sqrt(mp.fsum(abs(value)**2 for value in all_row_residual.values()))
    target_norm = mp.sqrt(mp.re(br.sparse_inner(target, target)))
    relative = absolute/target_norm if target_norm else absolute
    if relative > mp.power(10, -max(15, br.MP_DPS-20)):
        raise ArithmeticError(f"Physical action span is inconsistent: {relative}")
    return {
        "coefficients": coefficients, "rows": len(keys), "columns": column_count,
        "rank": rank, "absolute_residual": float(absolute),
        "relative_residual": float(relative),
        "smallest_singular_value": float(singular_values[-1]),
        "scaled_condition_number": float(singular_values[0]/singular_values[-1]),
        "selected_rows": column_count,
        "refinement_iterations": refinement_iterations,
        "selected_relative_residual": float(selected_relative_residual),
        "solver": f"sparse-action-pivoted-refinement-{br.MP_DPS}dps",
        "normalized_matrix_nonzeros": sum(map(len, normalized_rows)),
        "span_assembly_seconds": assembly_seconds,
        "span_factorization_seconds": factorization_seconds,
        "span_refinement_seconds": time.perf_counter()-refine_started,
    }


def solve_ns_l1(module, label):
    """The existing NS L1 action, with reflection and cached span assembly."""
    label = Fraction(label)
    if label <= -1 and (2*label).denominator == 1:
        reflected = CachedActionModule("NS", module.b, -module.momentum)
        terms, fit = solve_ns_l1(reflected, -label)
        return [br.ActionTerm(-term.label, term.first, term.second, term.coefficient)
                for term in terms], fit
    if label < 1 or (2*label).denominator != 1:
        raise ValueError("The NS L1 reduction requires n >= 1 in Z/2.")
    with module.descendant_cache() as stats:
        high, low = module.ns_branch(label), module.ns_branch(label-1)
        pairs = br.partition_pairs(int(4*label-3))
        columns = [module.descendant(low, first, second) for first, second in pairs]
        fit = action_span_fit(module.apply_l(1, high), columns)
        terms = [br.ActionTerm(label-1, first, second, coefficient)
                 for (first, second), coefficient in zip(pairs, fit["coefficients"])]
    fit["descendant_cache"] = stats
    return terms, fit


def solve_ramond_lplus(module, label, parity):
    """Positive-chart Ramond L1 coefficients used by the middle recurrence."""
    with module.descendant_cache() as stats:
        high, low = module.r_branch(label, parity), module.r_branch(label-1, parity)
        pairs = br.partition_pairs(int(4*label-3))
        columns = [module.descendant(low, first, second) for first, second in pairs]
        fit = action_span_fit(module.apply_l(1, high), columns)
        terms = [br.ActionTerm(label-1, first, second, coefficient)
                 for (first, second), coefficient in zip(pairs, fit["coefficients"])]
    fit["descendant_cache"] = stats
    return terms, fit


def solve_ramond_lminus(module, label, parity):
    """The existing two same-primary columns plus neighboring-primary span."""
    with module.descendant_cache() as stats:
        high = module.r_branch(label, parity)
        neighbor_label, degree = br.ramond_lminus_structure(label)
        neighbor = module.r_branch(neighbor_label, parity)
        same = [module.descendant(high, (1,), ()), module.descendant(high, (), (1,))]
        pairs = br.partition_pairs(degree)
        columns = [module.descendant(neighbor, first, second) for first, second in pairs]
        target = module.apply_l(-1, high)
        fit = action_span_fit(target, same+columns)
        coefficients = fit["coefficients"]
        terms = [br.ActionTerm(label, (1,), (), coefficients[0]),
                 br.ActionTerm(label, (), (1,), coefficients[1])]
        terms.extend(br.ActionTerm(neighbor_label, first, second, coefficient)
                     for (first, second), coefficient in zip(pairs, coefficients[2:]))
        identity = br.combine((1, target), (-1, same[0]), (-1, same[1]),
                              (1, module.apply_lf(-1, high)))
        fit["inverse_identity_max_residual"] = br.max_abs(identity)
    fit["descendant_cache"] = stats
    return terms, fit
