"""Reuse commuting-copy descendants and local oscillator data in action solves.

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
from scipy.sparse import csc_matrix

_BRANCH_DIR = Path(__file__).resolve().parents[1]/"ramond_branching_recursion"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))
import compute_target as br


class CachedActionModule(br.FreeFieldModule):
    """Shared construction cache at fixed module parameters and precision.

    Full requested span columns belong to their caller and are not inserted
    into this cache. Recursive suffixes are retained only until the current
    action solve (or requested minus/plus pair) exits, including on failure.
    The default order peels the larger of the two leading modes; the copies
    commute, and the order within each copy is preserved. Expressions are
    read-only. Auxiliary images are reused across all physical spectators.
    """

    def __init__(self, *args, descendant_order="largest_mode", indexed_descendants=True,
                 real_descendants=True, sparse_native=True, **kwargs):
        super().__init__(*args, **kwargs)
        if descendant_order not in ("by_copy", "largest_mode"):
            raise ValueError("unknown descendant construction order")
        self.descendant_order = descendant_order
        self.indexed_descendants = bool(indexed_descendants)
        self.real_descendants = bool(real_descendants) and self.momentum.imag == 0
        self.sparse_native = bool(sparse_native) and not br.MP_DPS
        inverse_b = 1/self.b
        denominator = inverse_b-self.b
        self._embedded_prefactors = {
            1: (inverse_b/denominator, -(inverse_b+2*self.b)/denominator, 1/denominator),
            2: (-self.b/denominator, (self.b+2*inverse_b)/denominator, -1/denominator)}
        self._descendant_cache = None
        self._embedded_cache = None
        self._auxiliary_l_cache = None
        self._mixed_auxiliary_cache = None
        self._cache_primaries = None
        self._cache_stats = None
        self._action_precision = br.MP_DPS, mp.mp.prec
        self._state_ids = self._states = self._indexed_primaries = None
        self._state_phases = self._real_primaries = None
        self._generator_matrices = None

    @contextmanager
    def descendant_cache(self):
        if self._descendant_cache is not None:
            raise RuntimeError("A descendant action cache is already active")
        precision = br.MP_DPS, mp.mp.prec
        if precision != self._action_precision:
            raise RuntimeError("A physical action module cannot be reused at a different precision")
        self._descendant_cache, self._cache_primaries, self._embedded_cache = {}, {}, {}
        self._auxiliary_l_cache, self._mixed_auxiliary_cache = {}, {}
        self._state_ids, self._states, self._indexed_primaries = {}, [], {}
        self._state_phases, self._real_primaries = [], set()
        self._generator_matrices = {}
        self._action_zero = br.complex_number(0)
        self._real_zero = br.real_number(0)
        self._action_i = br.complex_number(1j)
        self._action_tolerance = br.arithmetic_tolerance()
        stats = {"requests": 0, "suffix_requests": 0, "cache_hits": 0,
                 "embedded_image_requests": 0, "embedded_image_hits": 0,
                 "auxiliary_l_requests": 0, "auxiliary_l_hits": 0,
                 "mixed_auxiliary_requests": 0, "mixed_auxiliary_hits": 0,
                 "descendant_order": self.descendant_order,
                 "indexed_descendants": self.indexed_descendants,
                 "real_descendant_primaries": 0,
                 "sparse_generator_applications": 0, "sparse_matrix_rebuilds": 0,
                 "sparse_native": self.sparse_native}
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
            stats["retained_auxiliary_l_images"] = len(self._auxiliary_l_cache)
            stats["retained_mixed_auxiliary_transitions"] = len(self._mixed_auxiliary_cache)
            stats["indexed_oscillator_states"] = len(self._states)
            stats["sparse_generator_matrices"] = len(self._generator_matrices)
            stats["sparse_generator_entries"] = sum(len(cell["data"]) for cell in self._generator_matrices.values())
            self._descendant_cache = self._cache_primaries = self._cache_stats = None
            self._embedded_cache = None
            self._auxiliary_l_cache = self._mixed_auxiliary_cache = None
            self._state_ids = self._states = self._indexed_primaries = None
            self._state_phases = self._real_primaries = None
            self._generator_matrices = None
            if (br.MP_DPS, mp.mp.prec) != precision:
                raise RuntimeError("Arithmetic precision changed inside a descendant action cache")

    def apply_lf(self, mode, expression):
        if self._auxiliary_l_cache is None:
            return super().apply_lf(mode, expression)

        def action(state):
            auxiliary, physical = self.split_state(state)
            key = mode, auxiliary
            self._cache_stats["auxiliary_l_requests"] += 1
            if key not in self._auxiliary_l_cache:
                # L_F acts only on the auxiliary state. Its image can be
                # reused with every physical spectator and both Virasoros.
                ground = ((), ()) if self.sector == "NS" else ((), (), 0)
                unit = {self.join_state(auxiliary, ground): br.complex_number(1)}
                image = super(CachedActionModule, self).apply_lf(mode, unit)
                self._auxiliary_l_cache[key] = tuple(
                    (self.split_state(final)[0], value) for final, value in image.items())
            else:
                self._cache_stats["auxiliary_l_hits"] += 1
            return {self.join_state(final, physical): value
                    for final, value in self._auxiliary_l_cache[key]}
        return br.apply_expression(expression, action)

    def apply_u(self, mode, expression):
        if self._mixed_auxiliary_cache is None:
            return super().apply_u(mode, expression)

        def action(state):
            auxiliary, physical = self.split_state(state)
            level = self.physical_level_units(physical)
            key = mode, auxiliary, level
            self._cache_stats["mixed_auxiliary_requests"] += 1
            if key not in self._mixed_auxiliary_cache:
                if self.sector == "NS":
                    lower = 2*mode-self.auxiliary_level_units(auxiliary)
                    upper = level
                    if lower % 2 == 0:
                        lower += 1
                    if upper % 2 == 0:
                        upper -= 1
                    modes = range(lower, upper+1, 2)
                    complement = lambda r: 2*mode-r
                else:
                    modes = range(mode-self.auxiliary_level_units(auxiliary), level+1)
                    complement = lambda r: mode-r
                sign = (-1)**self.auxiliary_parity(auxiliary)
                transitions = []
                for r in modes:
                    final, coefficient = self.apply_auxiliary(complement(r), auxiliary)
                    if coefficient:
                        transitions.append((r, final, sign*coefficient))
                self._mixed_auxiliary_cache[key] = tuple(transitions)
            else:
                self._cache_stats["mixed_auxiliary_hits"] += 1
            answer = {}
            for r, auxiliary_final, coefficient in self._mixed_auxiliary_cache[key]:
                for physical_final, physical_coefficient in self.physical_g_on_state(r, physical):
                    br.add_term(answer, self.join_state(auxiliary_final, physical_final),
                                coefficient*physical_coefficient)
            return answer
        return br.apply_expression(expression, action)

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
            image = self._embedded_basis_image(copy, mode, state)
            self._embedded_cache[key] = image
            return image

        return br.apply_expression(expression, basis_image)

    def _embedded_basis_image(self, copy, mode, state):
        unit = {state: br.complex_number(1)}
        pieces = (self.apply_l(mode, unit), self.apply_lf(mode, unit), self.apply_u(mode, unit))
        return br.combine(*zip(self._embedded_prefactors[copy], pieces))

    def _state_id(self, state):
        index = self._state_ids.get(state)
        if index is None:
            index = len(self._states)
            self._states.append(state)
            self._state_ids[state] = index
            physical = self.split_state(state)[1]
            # Multiplying each physical oscillator state by i to the number
            # of bosons plus fermions (including the Ramond ground bit)
            # makes every embedded generator real for real b and P.
            self._state_phases.append((len(physical[0])+len(physical[1])+
                                       (physical[2] if self.sector == "R" else 0)) % 4)
        return index

    @staticmethod
    def _real_at_phase(value, phase):
        if phase % 2:
            if value.real != 0:
                return None
            return value.imag if phase == 1 else -value.imag
        if value.imag != 0:
            return None
        return value.real if phase == 0 else -value.real

    def _apply_embedded_indexed(self, copy, mode, expression, *, real=False):
        if self.sparse_native:
            return self._apply_embedded_sparse(copy, mode, expression, real=real)
        answer = {}
        zero = self._real_zero if real else self._action_zero
        tolerance = self._action_tolerance
        cache, stats = self._embedded_cache, self._cache_stats
        stats["embedded_image_requests"] += len(expression)
        for state_id, outer in expression.items():
            key = "indexed", copy, mode, state_id, real
            image = cache.get(key)
            if image is None:
                image = self._indexed_image(copy, mode, state_id, real)
                cache[key] = image
            else:
                stats["embedded_image_hits"] += 1
            # The same operation order and pruning as br.add_term, with
            # action-local constants and integer rather than oscillator keys.
            for final, inner in image:
                value = answer.get(final, zero)+outer*inner
                if abs(value) <= tolerance:
                    answer.pop(final, None)
                else:
                    answer[final] = value
        return answer

    def _indexed_image(self, copy, mode, state_id, real):
        image = tuple((self._state_id(final), value) for final, value in
                      self._embedded_basis_image(copy, mode, self._states[state_id]).items())
        if not real:
            return image
        transformed = []
        for final, value in image:
            value = self._real_at_phase(value,
                (self._state_phases[final]-self._state_phases[state_id]) % 4)
            if value is None:
                raise ArithmeticError("Embedded image violated the exact oscillator phase")
            transformed.append((final, value))
        return tuple(transformed)

    def _apply_embedded_sparse(self, copy, mode, expression, *, real):
        """Extend the cached operator only on new inputs, then act on a sparse vector.

        The CSC product visits only occupied input columns. It does not scan
        every stored oscillator state or construct unrequested basis images.
        """
        if not expression:
            return {}
        key = copy, mode, real
        cell = self._generator_matrices.setdefault(key,
            dict(columns={}, data=[], indices=[], indptr=[0], matrix=None))
        changed = False
        stats = self._cache_stats
        stats["embedded_image_requests"] += len(expression)
        for state_id in expression:
            if state_id in cell["columns"]:
                stats["embedded_image_hits"] += 1
                continue
            cell["columns"][state_id] = len(cell["columns"])
            for final, value in self._indexed_image(copy, mode, state_id, real):
                cell["indices"].append(final)
                cell["data"].append(value)
            cell["indptr"].append(len(cell["data"]))
            changed = True
        dtype = np.float64 if real else np.complex128
        if changed:
            cell["matrix"] = csc_matrix((np.asarray(cell["data"], dtype=dtype),
                np.asarray(cell["indices"], dtype=np.int32),
                np.asarray(cell["indptr"], dtype=np.int32)),
                shape=(len(self._states), len(cell["columns"])))
            stats["sparse_matrix_rebuilds"] += 1
        vector = csc_matrix((np.asarray(tuple(expression.values()), dtype=dtype),
            np.asarray([cell["columns"][state_id] for state_id in expression], dtype=np.int32),
            np.asarray([0, len(expression)], dtype=np.int32)), shape=(len(cell["columns"]), 1))
        result = cell["matrix"] @ vector
        stats["sparse_generator_applications"] += 1
        # Matrix multiplication accumulates the complete action before the
        # existing numerical-zero tolerance is applied. No level is dropped.
        return {int(index): value.item() for index, value in zip(result.indices, result.data)
                if abs(value) > self._action_tolerance}

    def descendant(self, primary, first_partition, second_partition):
        if self._descendant_cache is None:
            return super().descendant(primary, first_partition, second_partition)
        # Retain a strong reference: an id cannot be recycled while cached.
        primary_id = id(primary)
        self._cache_primaries[primary_id] = primary
        self._cache_stats["requests"] += 1
        if self.indexed_descendants and primary_id not in self._indexed_primaries:
            indexed = {self._state_id(state): value for state, value in primary.items()}
            if self.real_descendants:
                real = {index: self._real_at_phase(value, self._state_phases[index])
                        for index, value in indexed.items()}
                if all(value is not None for value in real.values()):
                    indexed = real
                    self._real_primaries.add(primary_id)
                    self._cache_stats["real_descendant_primaries"] += 1
            self._indexed_primaries[primary_id] = indexed
        answer = self._descendant_suffix(primary_id, tuple(first_partition),
                                         tuple(second_partition), retain=False)
        if self.indexed_descendants:
            if primary_id in self._real_primaries:
                def restore(index, value):
                    phase = self._state_phases[index]
                    if phase % 2:
                        return self._action_i*(value if phase == 1 else -value)
                    return br.complex_number(value if phase == 0 else -value)
                return {self._states[index]: restore(index, value) for index, value in answer.items()}
            return {self._states[index]: value for index, value in answer.items()}
        return answer

    def _descendant_suffix(self, primary_id, first, second, *, retain=True):
        self._cache_stats["suffix_requests"] += 1
        key = primary_id, first, second
        if key in self._descendant_cache:
            self._cache_stats["cache_hits"] += 1
            return self._descendant_cache[key]
        if first and (not second or self.descendant_order == "by_copy"
                      or first[0] >= second[0]):
            suffix = self._descendant_suffix(primary_id, first[1:], second)
            answer = (self._apply_embedded_indexed(1, -first[0], suffix,
                                                 real=primary_id in self._real_primaries) if self.indexed_descendants
                      else self.apply_embedded(1, -first[0], suffix))
        elif second:
            # Copies commute. Peeling the larger available mode minimizes
            # the level of the input to this last action, without exchanging
            # any two modes within one Virasoro copy.
            suffix = self._descendant_suffix(primary_id, first, second[1:])
            answer = (self._apply_embedded_indexed(2, -second[0], suffix,
                                                 real=primary_id in self._real_primaries) if self.indexed_descendants
                      else self.apply_embedded(2, -second[0], suffix))
        else:
            return (self._indexed_primaries if self.indexed_descendants
                    else self._cache_primaries)[primary_id]
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
        started = time.perf_counter()
        fit = br.span_fit(target, columns)
        if fit["rank"] != len(columns):
            raise np.linalg.LinAlgError("Machine descendant span lost column rank")
        if not np.isfinite(fit["relative_residual"]) or fit["relative_residual"] > 1e-8:
            raise ArithmeticError(f"Machine physical action span is inconsistent: {fit['relative_residual']}")
        fit.update(span_assembly_seconds=0.0,
                   span_factorization_seconds=time.perf_counter()-started,
                   span_refinement_seconds=0.0,
                   timing_scope="native span assembly and least-squares solve combined")
        return fit
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
        terms, fit = _ramond_lplus_in_context(module, label, high, low)
    fit["descendant_cache"] = stats
    return terms, fit


def _ramond_lplus_in_context(module, label, high, low):
    pairs = br.partition_pairs(int(4*label-3))
    columns = [module.descendant(low, first, second) for first, second in pairs]
    fit = action_span_fit(module.apply_l(1, high), columns)
    terms = [br.ActionTerm(label-1, first, second, coefficient)
             for (first, second), coefficient in zip(pairs, fit["coefficients"])]
    return terms, fit


def solve_ramond_lminus(module, label, parity, *, plus_result=None):
    """Fit the full minus span and optionally reuse its cache for a needed plus.

    A caller-supplied list receives the plus terms and their fit. It is only
    requested on the second physical edge when the middle recurrence needs
    that label; the large temporary expressions are then released together.
    """
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
        if plus_result is not None and label > Fraction(1, 4):
            del columns, same, target, identity
            started = time.perf_counter()
            before = stats.copy()
            # Both actions descend from the same neighboring primary.
            # Keep its object identity and suffix/image caches alive while
            # constructing the lower-degree L1 columns for the middle stage.
            plus_terms, plus_fit = _ramond_lplus_in_context(module, label, high, neighbor)
            plus_fit["reused_minus_descendants"] = True
            plus_fit["construction_and_solve_seconds"] = time.perf_counter()-started
            plus_fit["shared_cache_increment"] = {
                key: value-before[key] for key, value in stats.items()
                if type(value) is int}
            plus_result.append((plus_terms, plus_fit))
    fit["descendant_cache"] = stats
    if plus_result:
        plus_result[-1][1]["descendant_cache"] = stats
    return terms, fit
