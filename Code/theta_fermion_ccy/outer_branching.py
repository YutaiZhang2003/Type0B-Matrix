"""Outer three-point coefficients from the repository's L1 Ward system.

The multiprecision path preserves rational input parameters, computes the
existing low-primary anchors at the same precision, and returns multiprecision
Ward solutions without casting them back to binary64.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import mpmath as mp
from pathlib import Path
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


class _MPBranchingGrid(BranchingGrid):
    def __init__(self, b, momenta, cutoff, primary_parity=0, mp_dps=60):
        self.mp_dps = int(mp_dps)
        if self.mp_dps < 30:
            raise ValueError("MP Ward precision must be at least 30 digits")
        br.set_multiprecision(self.mp_dps)
        self.b = br.real_number(b)
        self.momenta = tuple(_mp(value) for value in momenta)
        self.cutoff, self.primary_parity = int(cutoff), int(primary_parity)
        self.weights = br.BranchWeights(self.b,self.momenta)
        self.modules = (br.FreeFieldModule("NS",self.b,self.momenta[0]),
                        br.FreeFieldModule("R",self.b,self.momenta[1]),
                        br.FreeFieldModule("R",self.b,self.momenta[2]))
        self.ns, self.r = ns_labels(cutoff), ramond_labels(cutoff)
        self.ns_actions, self.r_actions, self.action_diagnostics = {}, {}, []
        self.direct = _MPLowAnchors(self.b,self.momenta,primary_parity,self.mp_dps)

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
            terms, fit = br.solve_ns_l1(self.modules[0], label)
            self.ns_actions[label] = tuple(terms)
            self.action_diagnostics.append({
                "sector": "NS", "slot": 1, "label": str(label), "parity": 0,
                "relative_residual": float(fit["relative_residual"]),
            })
        self.r_action_providers = {
            slot: RamondActions(self.b, self.momenta[slot]) for slot in (1, 2)
        }
        for slot, provider in self.r_action_providers.items():
            for parity in (0, 1):
                for label in self.r:
                    self.r_actions[slot, parity, label] = provider.minus(label, parity)
                    fit = provider.diagnostics["minus", label, parity]
                    self.action_diagnostics.append({
                        "sector": "R", "slot": slot+1, "label": str(label),
                        "parity": parity,
                        "relative_residual": float(fit["relative_residual"]),
                        "reflection": "positive n at -P" if label < 0 else "native positive n",
                    })

    def _solve_multiprecision(self,alpha2,alpha3,unknowns,rows,rhs,anchor_count):
        # The reusable action solver already implements MP iterative refinement
        # on a pivoted full-rank row subset, followed by an all-row residual.
        # This is a linear Ward-system solve, not a direct conformal-block sum.
        columns = [{} for _ in unknowns]
        for row_index,row in enumerate(rows):
            for column,value in row.items():
                columns[column][row_index] = value
        target = {row_index:value for row_index,value in enumerate(rhs) if value}
        fit = br.multiprecision_span_fit(target,columns)
        tolerance = 10.0**(-max(15,self.mp_dps-20))
        if fit["relative_residual"] > tolerance:
            raise ArithmeticError(f"MP outer Ward system inconsistent: {fit['relative_residual']}")
        solution = dict(zip(unknowns,fit.pop("coefficients")))
        diagnostic = dict(fit,alpha_2=alpha2,alpha_3=alpha3,
                          unknowns=len(unknowns),equations=len(rows),anchors=anchor_count,
                          full_column_rank=fit["rank"]==len(unknowns),
                          boundary_anchor_backend=f"low-anchor-pbw-mpmath-{self.mp_dps}dps",
                          input_parameters="exact rational converted directly to MP")
        return solution,diagnostic


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


class OuterBranching:
    def __init__(self, b, momenta, level, *, f=0, p=0, dps=0):
        started = time.perf_counter()
        self.level, self.f, self.p = int(level), int(f), int(p)
        pairs = middle_pairs(level)
        limit = max(abs(n) for pair in pairs for n in pair)
        grid_level = max(level, ramond_level(limit))
        self.grid = (_MPBranchingGrid(b,momenta,grid_level,primary_parity=p,mp_dps=dps)
                     if dps else BranchingGrid(float(b),tuple(complex(x) for x in momenta),
                                               grid_level,primary_parity=p,mp_dps=0))
        self.grid.ns = ns_labels(level)
        self.grid.r = tuple(Fraction(k, 4)
                            for k in range(-int(4*limit), int(4*limit)+1, 2))
        self.grid.build_actions()
        self.tables = {}
        self.diagnostics = {
            "action_seconds": time.perf_counter()-started,
            "ward_solve_dps": dps, "anchor_arithmetic": (f"mpmath-{dps}dps" if dps else "binary64"),
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
