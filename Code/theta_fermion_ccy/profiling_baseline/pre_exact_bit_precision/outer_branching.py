"""Outer three-point coefficients from the repository's L1 Ward system.

The multiprecision path preserves rational input parameters, computes the
existing low-primary anchors at the same precision, and returns multiprecision
Ward solutions without casting them back to binary64.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import mpmath as mp
import numpy as np
from pathlib import Path
from scipy import linalg as scipy_linalg
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for directory in (
    CODE / "bosonic_c1_one_to_n_reference" / "reference_implementation",
    CODE / "full_ramond_block_runtime",
    CODE / "ramond_branching_recursion",
):
    sys.path.insert(0, str(directory))

from compute_full_block import BranchingGrid, ns_labels, ramond_labels
import compute_target as br
import direct_state_check as direct
from action_optimization import CachedActionModule, solve_ns_l1


def _mp(value):
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator)/value.denominator
    return mp.mpc(value)


class _MPPBWModule(direct.PBWModule):
    """The existing low-anchor PBW basis, with MP change-of-basis solves."""

    @lru_cache(None)
    def basis(self, level_units):
        if self.sector == "NS":
            rows, matrix = self.module._ns_level_transition(
                self.module.realization, level_units)
            metadata = tuple((ls, gs)
                for level in range(level_units//2+1)
                for ls in br.partitions(level)
                for gs in br.strict_odd_partitions(level_units-2*level))
        else:
            rows, matrix = self.module._level_transition(
                self.module.realization, level_units)
            metadata = tuple((ls, gs, ground)
                for level in range(level_units+1)
                for ls in br.partitions(level)
                for gs in br.strict_partitions(level_units-level)
                for ground in (0,1))
        if matrix.cols != len(metadata):
            raise AssertionError("MP low-anchor PBW metadata disagrees with basis")
        return rows, matrix, metadata

    def from_fock(self, expression):
        if not expression:
            return {}
        level = direct.expression_level(self.module, expression)
        rows, matrix, metadata = self.basis(level)
        vector = mp.matrix([expression.get(row, 0) for row in rows])
        values = mp.lu_solve(matrix, vector)
        error = mp.norm(matrix*values-vector)/max(1, mp.norm(vector))
        if error > mp.power(10, -max(15, br.MP_DPS-15)):
            raise ArithmeticError("MP low-anchor change of basis lost accuracy")
        return {state:value for state,value in zip(metadata,values)
                if abs(value) > br.arithmetic_tolerance()}


class _MPLowAnchors(direct.DirectBranchingCoefficient):
    """Only NS 0,+/-1/2 and R +/-1/4,+/-3/4 anchor states are allowed."""

    def __init__(self, b, momenta, primary_parity, dps):
        self.b, self.momenta = b, momenta
        self.primary_parity, self.dps = int(primary_parity), int(dps)
        self.free_modules = (br.FreeFieldModule("NS",b,momenta[0]),
                             br.FreeFieldModule("R",b,momenta[1]),
                             br.FreeFieldModule("R",b,momenta[2]))
        self.pbw_modules = tuple(_MPPBWModule(module) for module in self.free_modules)
        self.auxiliary_form = direct.AuxiliaryThreePoint(self.free_modules)
        self.auxiliary_form.base_value = lambda states: (
            mp.mpc(0) if states[1][1] != states[2][1]
            else mp.mpc(1) if states[1][1] == 0 else mp.j)
        self._reflected_ns_module = self._reflected_ns_pbw = None
        self._branch_cache, self._physical_forms = {}, {}
        for parity in (0,1):
            for eta in (-1,1):
                form = direct.PhysicalThreePoint(self.pbw_modules,parity,eta,
                                                 primary_parity=primary_parity)
                # Replace exp(3*pi*i/4) evaluated by binary64 cmath in the
                # legacy constructor by its algebraic MP value.
                form.ramond_odd_phase = (-1+mp.j)/mp.sqrt(2)
                self._physical_forms[parity,eta] = form

    def branch(self, slot, label, parity=0):
        label = Fraction(label)
        if (slot == 0 and abs(label)>Fraction(1,2)) or (slot and abs(label)>Fraction(3,4)):
            raise ValueError("MP direct evaluation is restricted to low anchors")
        key = slot,label,parity
        if key in self._branch_cache:
            return self._branch_cache[key]
        module, pbw = self.free_modules[slot], self.pbw_modules[slot]
        if slot == 0 and label < 0:
            if self._reflected_ns_module is None:
                self._reflected_ns_module = br.FreeFieldModule("NS",self.b,-self.momenta[0])
                self._reflected_ns_pbw = _MPPBWModule(self._reflected_ns_module)
            module, pbw = self._reflected_ns_module, self._reflected_ns_pbw
            state = module.ns_branch(-label)
        else:
            state = module.ns_branch(label) if slot == 0 else module.r_branch(label,parity)
        value = direct.branch_in_pbw(module,pbw,state)
        self._branch_cache[key] = value
        return value

    def raw(self, labels, alpha2, alpha3, eta):
        # BranchingGrid.solve intentionally suspends MP for its legacy oracle.
        # This adapter restores MP for the entire low-anchor calculation.
        previous = br.MP_DPS
        br.set_multiprecision(self.dps)
        try:
            return super().raw(labels,alpha2,alpha3,eta)
        finally:
            br.set_multiprecision(previous)


class _SparseWardFactorization:
    """Reusable mixed-precision solve, retaining the full Ward residual.

    This is the same column scaling, pivoted row selection and double LU
    preconditioner used by ``multiprecision_span_fit``. Only its MP matrix
    products change: a Ward row visits its nonzero entries, not every column.
    The matrix is independent of eta, so its factorization serves both RHSs.
    """

    def __init__(self, rows, column_count, dps):
        started = time.perf_counter()
        self.dps, self.column_count = int(dps), int(column_count)
        context = mp if dps else mp.fp
        squared_norms = [context.mpf(0) for _ in range(column_count)]
        for row in rows:
            for column, value in row.items():
                squared_norms[column] += abs(value)**2
        self.norms = tuple(context.sqrt(value) for value in squared_norms)
        if any(norm == 0 for norm in self.norms):
            raise AssertionError("An outer Ward-system column vanished.")
        self.rows = tuple(tuple((column, value/self.norms[column])
                                for column, value in row.items())
                          for row in rows)
        shadow = np.zeros((len(rows), column_count), dtype=np.complex128)
        for row_index, row in enumerate(self.rows):
            for column, value in row:
                shadow[row_index, column] = complex(value)
        _, _, pivots = scipy_linalg.qr(
            shadow.T, mode="economic", pivoting=True, check_finite=False)
        self.selected_indices = tuple(int(index) for index in pivots[:column_count])
        selected = shadow[list(self.selected_indices), :]
        singular_values = scipy_linalg.svdvals(selected, check_finite=False)
        self.rank = int(np.count_nonzero(singular_values > br.RANK_TOLERANCE))
        if self.rank != column_count:
            raise np.linalg.LinAlgError(
                f"Pivoted outer Ward restriction has rank {self.rank}/{column_count}.")
        self.smallest_singular_value = float(singular_values[-1])
        self.condition_number = float(singular_values[0]/singular_values[-1])
        self.lu = scipy_linalg.lu_factor(selected, check_finite=False)
        self.factorization_seconds = time.perf_counter()-started
        self.solve_count = 0

    def solve(self, rhs):
        if not self.dps:
            return self._solve_native(rhs)
        started = time.perf_counter()
        selected_target = tuple(rhs[index] for index in self.selected_indices)
        initial = scipy_linalg.lu_solve(
            self.lu, np.asarray([complex(value) for value in selected_target]),
            check_finite=False)
        values = [br.complex_number(value) for value in initial]
        selected_scale = max(mp.mpf(1), mp.sqrt(mp.fsum(
            abs(value)**2 for value in selected_target)))
        tolerance = mp.power(10, -max(20, self.dps-15))
        selected_residual = mp.inf
        iterations = 0
        for iteration in range(1, 31):
            residual = [rhs[index]-mp.fsum(value*values[column]
                        for column, value in self.rows[index])
                        for index in self.selected_indices]
            selected_residual = mp.sqrt(mp.fsum(
                abs(value)**2 for value in residual))/selected_scale
            if selected_residual <= tolerance:
                break
            correction = scipy_linalg.lu_solve(
                self.lu, np.asarray([complex(value) for value in residual]),
                check_finite=False)
            values = [value+br.complex_number(delta)
                      for value, delta in zip(values, correction)]
            iterations = iteration
        else:
            raise FloatingPointError(
                "Sparse outer Ward refinement did not reach the requested tolerance.")
        # No equations are projected out. The selected rows determine a
        # candidate, and every original row must accept that same candidate.
        all_residuals = [mp.fsum(value*values[column] for column, value in row)-target
                         for row, target in zip(self.rows, rhs)]
        absolute = mp.sqrt(mp.fsum(abs(value)**2 for value in all_residuals))
        target_norm = mp.sqrt(mp.fsum(abs(value)**2 for value in rhs))
        relative = absolute/target_norm if target_norm else absolute
        if relative > mp.power(10, -max(15, self.dps-20)):
            raise ArithmeticError(f"MP outer Ward system inconsistent: {relative}")
        coefficients = tuple(value/norm for value, norm in zip(values, self.norms))
        diagnostic = {
            "rows": len(self.rows), "columns": self.column_count,
            "rank": self.rank, "absolute_residual": float(absolute),
            "relative_residual": float(relative),
            "smallest_singular_value": self.smallest_singular_value,
            "scaled_condition_number": self.condition_number,
            "selected_rows": self.column_count,
            "refinement_iterations": iterations,
            "selected_relative_residual": float(selected_residual),
            "solver": f"sparse-mixed-precision-pivoted-refinement-{self.dps}dps",
            "matrix_nonzeros": sum(map(len, self.rows)),
            "factorization_reused": bool(self.solve_count),
            "factorization_seconds": 0.0 if self.solve_count else self.factorization_seconds,
            "solve_seconds": time.perf_counter()-started,
        }
        self.solve_count += 1
        return coefficients, diagnostic

    def _solve_native(self, rhs):
        """Reuse the same selected square system and retain every Ward row."""
        started = time.perf_counter()
        selected_target = np.asarray([rhs[index] for index in self.selected_indices],
                                     dtype=np.complex128)
        values = scipy_linalg.lu_solve(self.lu, selected_target, check_finite=False)
        residual = np.asarray([sum(value*values[column] for column,value in row)-target
                               for row,target in zip(self.rows,rhs)], dtype=np.complex128)
        absolute = float(np.linalg.norm(residual))
        target_norm = float(np.linalg.norm(np.asarray(rhs, dtype=np.complex128)))
        relative = absolute/target_norm if target_norm else absolute
        if not np.isfinite(relative) or relative > 1e-8:
            raise ArithmeticError(f"Machine outer Ward system inconsistent: {relative}")
        selected_residual = float(np.linalg.norm(residual[list(self.selected_indices)]))
        selected_scale = max(1.0, float(np.linalg.norm(selected_target)))
        diagnostic = {
            "rows": len(self.rows), "columns": self.column_count, "rank": self.rank,
            "absolute_residual": absolute, "relative_residual": relative,
            "smallest_singular_value": self.smallest_singular_value,
            "scaled_condition_number": self.condition_number,
            "selected_rows": self.column_count, "refinement_iterations": 0,
            "selected_relative_residual": selected_residual/selected_scale,
            "solver": "native-complex128-pivoted-lu",
            "matrix_nonzeros": sum(map(len,self.rows)),
            "factorization_reused": bool(self.solve_count),
            "factorization_seconds": 0.0 if self.solve_count else self.factorization_seconds,
            "solve_seconds": time.perf_counter()-started,
            "full_row_tolerance": 1e-8,
        }
        self.solve_count += 1
        return tuple(complex(value/norm) for value,norm in zip(values,self.norms)), diagnostic


class _MPBranchingGrid(BranchingGrid):
    def __init__(self, b, momenta, cutoff, primary_parity=0, mp_dps=60):
        self.mp_dps = int(mp_dps)
        if self.mp_dps and self.mp_dps < 30:
            raise ValueError("Ward precision must be zero (binary64) or at least 30 digits")
        br.set_multiprecision(self.mp_dps)
        self.arithmetic = mp if self.mp_dps else mp.fp
        self.b = br.real_number(b)
        self.momenta = tuple(_mp(value) if self.mp_dps else complex(value) for value in momenta)
        self.cutoff, self.primary_parity = int(cutoff), int(primary_parity)
        self.weights = br.BranchWeights(self.b,self.momenta)
        self.modules = (CachedActionModule("NS",self.b,self.momenta[0]),
                        br.FreeFieldModule("R",self.b,self.momenta[1]),
                        br.FreeFieldModule("R",self.b,self.momenta[2]))
        self.ns, self.r = ns_labels(cutoff), ramond_labels(cutoff)
        self.ns_actions, self.r_actions, self.action_diagnostics = {}, {}, []
        if self.mp_dps:
            self.direct = _MPLowAnchors(self.b,self.momenta,primary_parity,self.mp_dps)
        else:
            self.direct = direct.DirectBranchingCoefficient(self.b,self.momenta,primary_parity)
            self.direct.auxiliary_form.base_value = lambda states: (
                0j if states[1][1] != states[2][1] else 1+0j if states[1][1] == 0 else 1j)
            for parity in (0,1):
                for eta in (-1,1):
                    form = direct.PhysicalThreePoint(self.direct.pbw_modules,parity,eta,
                                                     primary_parity=primary_parity)
                    form.ramond_odd_phase = (-1+1j)/mp.fp.sqrt(2)
                    self.direct._physical_forms[parity,eta] = form
        self.required = None
        self._ward_systems, self._ordinary_forms = {}, {}

    def build_actions(self):
        """Use the abstract reflection before solving a negative-label action.

        The common SCA intertwiner sends (n,P) to (-n,-P), commutes with
        L and the two Virasoro algebras, and preserves the chi-string
        normalization. Its coefficients therefore need only a label flip.
        This avoids constructing a high-level Fock-to-PBW reflection matrix.
        The same cached action provider is used by the middle recurrence.
        """
        from middle_branching import RamondActions
        for label in self.ns:
            if abs(label) < 1:
                self.ns_actions[label] = ()
                continue
            terms, fit = solve_ns_l1(self.modules[0], label)
            self.ns_actions[label] = tuple(terms)
            self.action_diagnostics.append({
                "sector": "NS", "slot": 1, "label": str(label), "parity": 0,
                "relative_residual": float(fit["relative_residual"]),
                "descendant_cache": fit["descendant_cache"],
                "span_assembly_seconds": fit["span_assembly_seconds"],
                "span_factorization_seconds": fit["span_factorization_seconds"],
                "span_refinement_seconds": fit["span_refinement_seconds"],
            })
        self.r_action_providers = {
            slot: RamondActions(self.b, self.momenta[slot]) for slot in (1, 2)
        }
        for slot, provider in self.r_action_providers.items():
            for label in self.r:
                solved = provider.minus(label, 0)
                fit = provider.diagnostics["minus", label, 0]
                self.r_actions[slot, 0, label] = solved
                # Theta v_n^0 = -2^((-1)^M/2) v_n^1, M=2|n|-1/2.
                # Theta commutes with physical L and both embedded Virasoros:
                # V(n -> m;1)=V(n -> m;0)*t(m)/t(n). Reflection preserves
                # |n|, hence the same raw normalization in both signed charts.
                source_power = (-1)**int(2*abs(label)-Fraction(1, 2))
                transported = []
                for term in solved:
                    target_power = (-1)**int(2*abs(term.label)-Fraction(1, 2))
                    ratio = br.scalar_power_of_two(Fraction(target_power-source_power, 2))
                    transported.append(br.ActionTerm(
                        term.label, term.first, term.second, term.coefficient*ratio))
                self.r_actions[slot, 1, label] = tuple(transported)
                for parity in (0, 1):
                    self.action_diagnostics.append({
                        "sector": "R", "slot": slot+1, "label": str(label),
                        "parity": parity,
                        "relative_residual": float(fit["relative_residual"]),
                        "reflection": "positive n at -P" if label < 0 else "native positive n",
                        "method": "descendant-span solve" if parity == 0 else "exact Theta transport from parity 0",
                        "residual_source": "parity-0 all-row residual; relative norm preserved by Theta" if parity else "computed on all oscillator rows",
                        "descendant_cache": fit["descendant_cache"] if parity == 0 else None,
                        "span_assembly_seconds": fit["span_assembly_seconds"] if parity == 0 else 0.0,
                        "span_factorization_seconds": fit["span_factorization_seconds"] if parity == 0 else 0.0,
                        "span_refinement_seconds": fit["span_refinement_seconds"] if parity == 0 else 0.0,
                    })

    def close_requested(self, requested):
        """Close the requested triples under the actual action-label support."""
        if self._ward_systems:
            raise RuntimeError("Ward support cannot change after a factorization is cached")
        closed = set(requested)
        pending = list(closed)
        while pending:
            labels = pending.pop()
            action_sets = ((0, self.ns_actions[labels[0]]),)
            action_sets += tuple((slot, self.r_actions[slot, parity, labels[slot]])
                                 for slot in (1, 2) for parity in (0, 1))
            for slot, actions in action_sets:
                for term in actions:
                    changed = list(labels)
                    changed[slot] = term.label
                    changed = tuple(changed)
                    if changed not in closed:
                        closed.add(changed)
                        pending.append(changed)
        self.required = tuple(sorted(closed))

    def _ordinary_factor(self, labels, slot, term):
        changed = list(labels)
        changed[slot] = term.label
        changed = tuple(changed)
        answer = term.coefficient
        for copy in (0, 1):
            key = changed, copy
            if key not in self._ordinary_forms:
                self._ordinary_forms[key] = br.VirasoroThreePoint(
                    self.weights.triple(changed, copy), self.weights.central_charges[copy])
            words = [(), (), ()]
            words[slot] = term.first if copy == 0 else term.second
            answer *= self._ordinary_forms[key].value(*words)
        return changed, answer

    def _prepare_system(self, alpha2, alpha3, form_parity):
        started = time.perf_counter()
        context = self.arithmetic
        n1_parity = (form_parity-alpha2-alpha3) % 2
        support = self.required
        if support is None:
            support = tuple((first, second, third) for first in self.ns
                            for second in self.r for third in self.r)
        unknowns = tuple(labels for labels in support if int(2*labels[0]) % 2 == n1_parity)
        index = {labels: position for position, labels in enumerate(unknowns)}
        rows = []
        for labels in unknowns:
            equation = {}
            for sign, slot, actions in (
                    (1, 0, self.ns_actions[labels[0]]),
                    (-1, 1, self.r_actions[1, alpha2, labels[1]]),
                    (-1, 2, self.r_actions[2, alpha3, labels[2]])):
                for term in actions:
                    changed, coefficient = self._ordinary_factor(labels, slot, term)
                    if changed not in index:
                        raise AssertionError(f"Ward action left its closed support: {labels} -> {changed}")
                    column = index[changed]
                    equation[column] = equation.get(column, 0)+sign*coefficient
            norm = context.sqrt(context.fsum(abs(value)**2 for value in equation.values()))
            cutoff = (mp.power(10, -max(20, self.mp_dps-20)) if self.mp_dps
                      else br.arithmetic_tolerance())
            if norm > cutoff:
                rows.append({column: value/norm for column, value in equation.items() if value})
        ward_count = len(rows)
        anchor_first = (Fraction(0),) if n1_parity == 0 else (Fraction(-1, 2), Fraction(1, 2))
        anchor_r = tuple(Fraction(n, 4) for n in (-3, -1, 1, 3))
        anchors = tuple((first, second, third) for first in anchor_first
                        for second in anchor_r for third in anchor_r
                        if (first, second, third) in index)
        rows.extend({index[labels]: context.mpc(1)} for labels in anchors)
        assembly_seconds = time.perf_counter()-started
        factorization = _SparseWardFactorization(rows, len(unknowns), self.mp_dps)
        return unknowns, anchors, ward_count, factorization, assembly_seconds

    def solve(self, alpha2, alpha3, *, eta=1, form_parity=0):
        br.set_multiprecision(self.mp_dps)
        alpha2, alpha3, eta, form_parity = map(int, (alpha2, alpha3, eta, form_parity))
        if alpha2 not in (0, 1) or alpha3 not in (0, 1):
            raise ValueError("Ramond primary parity must be zero or one")
        if eta not in (-1, 1) or form_parity not in (0, 1):
            raise ValueError("eta must be +/-1 and form_parity zero or one")
        key = alpha2, alpha3, form_parity
        reused = key in self._ward_systems
        if not reused:
            self._ward_systems[key] = self._prepare_system(*key)
        unknowns, anchors, ward_count, factorization, assembly_seconds = self._ward_systems[key]
        rhs = [self.arithmetic.mpc(0)]*ward_count
        rhs.extend(self.direct.raw(labels, alpha2, alpha3, eta=eta) for labels in anchors)
        coefficients, diagnostic = factorization.solve(rhs)
        diagnostic.update(
            alpha_2=alpha2, alpha_3=alpha3, eta=eta, form_parity=form_parity,
            unknowns=len(unknowns), equations=len(factorization.rows), anchors=len(anchors),
            full_column_rank=factorization.rank == len(unknowns),
            system_assembly_seconds=0.0 if reused else assembly_seconds,
            boundary_anchor_backend=(f"low-anchor-pbw-mpmath-{self.mp_dps}dps" if self.mp_dps
                                     else "low-anchor-pbw-complex128"),
            input_parameters=("exact rational converted directly to MP" if self.mp_dps
                              else "rational inputs rounded to binary64"),
            support="action-closed requested triples" if self.required is not None else "Cartesian")
        return dict(zip(unknowns, coefficients)), diagnostic


def ramond_level(n):
    value = 2*n*n - Fraction(1, 8)
    assert value.denominator == 1
    return int(value)


def middle_pairs(level):
    bound = 1
    while (bound*bound-1)/8 <= 2*level + 2:
        bound += 2
    labels = tuple(Fraction(k, 4) for k in range(-bound, bound+1, 2))
    return tuple((n, changed) for n in labels
                 for changed in (n-Fraction(1, 2), n+Fraction(1, 2))
                 if ramond_level(n) + ramond_level(changed) <= 2*level)


class _SharedRamondActions:
    """Use the already prepared edge-two actions at the same precision."""

    def __init__(self, grid):
        self.grid = grid
        self.provider = grid.r_action_providers[1]
        self.diagnostics = self.provider.diagnostics

    def minus(self, label, parity):
        key = (1, int(parity), Fraction(label))
        if key not in self.grid.r_actions:
            self.grid.r_actions[key] = self.provider.minus(Fraction(label),int(parity))
        return self.grid.r_actions[key]

    def plus(self, label, parity):
        return self.provider.plus(Fraction(label),int(parity))


class OuterBranching:
    def __init__(self, b, momenta, level, *, f=0, p=0, dps=0, machine=False):
        started = time.perf_counter()
        self.level, self.f, self.p = int(level), int(f), int(p)
        pairs = middle_pairs(level)
        # The L_-1 ground Ward identities involve the +/-3/4 anchors even
        # when the requested physical block has only level-zero primaries.
        limit = max(Fraction(3, 4), max(abs(n) for pair in pairs for n in pair))
        grid_level = max(level, ramond_level(limit))
        self.grid = (_MPBranchingGrid(b,momenta,grid_level,primary_parity=p,mp_dps=dps)
                     if dps or machine else BranchingGrid(float(b),tuple(complex(x) for x in momenta),
                                               grid_level,primary_parity=p,mp_dps=0))
        self.grid.ns = ns_labels(level)
        self.grid.r = tuple(Fraction(k, 4)
                            for k in range(-int(4*limit), int(4*limit)+1, 2))
        self.grid.build_actions()
        action_seconds = time.perf_counter()-started
        requested = set()
        for n1 in self.grid.ns:
            for incoming, outgoing in pairs:
                for n3 in self.grid.r:
                    if (4*n1*n1+ramond_level(incoming)+ramond_level(outgoing)
                            +2*ramond_level(n3) <= 2*level):
                        requested.update(((n1, incoming, n3), (n1, outgoing, n3)))
        if dps or machine:
            self.grid.close_requested(requested)
        self.tables = {}
        self.diagnostics = {
            "action_seconds": action_seconds,
            "ward_solve_dps": dps, "anchor_arithmetic": (f"mpmath-{dps}dps" if dps else "binary64"),
            "requested_triples": len(requested),
            "closed_triples": len(self.grid.required) if dps or machine else None,
            "actions": self.grid.action_diagnostics, "ward_systems": [],
        }
        self.started = started

    def prepare(self, etas):
        for eta in dict.fromkeys(etas):
            for alpha in (0, 1):
                for gamma in (0, 1):
                    key = (alpha, gamma, eta)
                    if key in self.tables:
                        continue
                    values, diagnostic = self.grid.solve(
                        alpha, gamma, eta=(-1)**self.f*eta, form_parity=self.f)
                    # The archived odd-form oracle labels its two Ramond
                    # structures oppositely to the physical human-note eta.
                    diagnostic["physical_eta"] = eta
                    diagnostic["native_eta"] = (-1)**self.f*eta
                    self.tables[key] = values
                    self.diagnostics["ward_systems"].append(diagnostic)
        self.diagnostics["total_seconds"] = time.perf_counter()-self.started

    def raw(self, n1, n2, n3, alpha, gamma, eta):
        labels = tuple(map(Fraction, (n1, n2, n3)))
        return self.tables[alpha, gamma, eta][labels]

    def middle_actions(self, dps):
        if self.grid.mp_dps != dps or not hasattr(self.grid,"r_action_providers"):
            return None
        return _SharedRamondActions(self.grid)
