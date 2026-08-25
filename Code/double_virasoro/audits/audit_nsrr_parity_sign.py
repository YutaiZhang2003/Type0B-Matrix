#!/usr/bin/env python3
"""Machine-suggestion audit of the NS--R--R three-point factorization.

This file does not edit or define the Human Note. It compares, term by term,
the hatted tensor-product three-point form on embedded Virasoro descendants
with the product of the two ordinary Virasoro three-point forms. It audits
both candidate tensor signs

    A A_F + (B+|alpha|+p_1)(C_F+c),

and the same exponent with the additional term ``f(C_F+c)``. Every physical
Ramond ground component is converted to the Human Note's ``w^+``/``w^-``
normalization before the three-point value is taken.

The test matrix contains both primary parities, both form parities, both eta
values, reflected and unreflected branch labels, and 34 descendant patterns:
the primary, L_-1/L_-2/L_-1^2 in each of six Virasoro slots, and every pair
of L_-1 insertions in two distinct slots.
"""

from __future__ import annotations

import cmath
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for directory in (
    CODE / "double_virasoro" / "nsrr",
    CODE / "ramond_branching_recursion",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compute_target import (  # noqa: E402
    BranchWeights,
    FreeFieldModule,
    VirasoroThreePoint,
)
from direct_state_check import (  # noqa: E402
    PBWModule,
    PhysicalThreePoint,
    branch_in_pbw,
)
from nsrr_genus2_block import HumanAuxiliaryThreePoint  # noqa: E402


B = 7 / 5
MOMENTA = (11 / 23, 13 / 29, 17 / 31)
LABEL_TRIPLES = (
    (Fraction(0), Fraction(1, 4), Fraction(1, 4)),
    (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)),
    (Fraction(-1, 2), Fraction(-1, 4), Fraction(1, 4)),
    (Fraction(0), Fraction(3, 4), Fraction(1, 4)),
    (Fraction(1), Fraction(3, 4), Fraction(3, 4)),
    (Fraction(-1), Fraction(-3, 4), Fraction(-3, 4)),
)
ABSOLUTE_TOLERANCE = 2.0e-8
NONZERO_TOLERANCE = 1.0e-8


class HumanWPhysicalThreePoint(PhysicalThreePoint):
    """Allow both eta values while retaining the Human-w ground table."""

    def __init__(self, modules, form_parity: int, eta: int, primary_parity: int):
        self.modules = tuple(modules)
        self.form_parity = int(form_parity)
        self.primary_parity = int(primary_parity)
        self.eta = int(eta)
        if self.form_parity not in (0, 1):
            raise ValueError("form_parity must be 0 or 1")
        if self.primary_parity not in (0, 1):
            raise ValueError("primary_parity must be 0 or 1")
        if self.eta not in (-1, 1):
            raise ValueError("eta must be +1 or -1")
        self.infinity_phase = -1j
        self.zero_phase = 1j
        self._cache = {}
        self._active = set()
        self.ramond_odd_phase = cmath.exp(3j * math.pi / 4)


def descendant_configurations():
    """Return the 34 low-order descendant patterns used in the audit."""

    positions = tuple((slot, copy) for slot in range(3) for copy in range(2))
    configurations = [(((), ()), ((), ()), ((), ()))]
    for slot, copy in positions:
        for word in ((1,), (2,), (1, 1)):
            configuration = [[(), ()] for _ in range(3)]
            configuration[slot][copy] = word
            configurations.append(tuple(tuple(words) for words in configuration))
    for first, second in combinations(positions, 2):
        configuration = [[(), ()] for _ in range(3)]
        configuration[first[0]][first[1]] = (1,)
        configuration[second[0]][second[1]] = (1,)
        configurations.append(tuple(tuple(words) for words in configuration))
    if len(configurations) != 34:
        raise AssertionError("the descendant test matrix has the wrong size")
    return tuple(configurations)


def branch_data(labels, alpha2: int, alpha3: int):
    """Build branch primaries, including the Human NS reflection chart."""

    modules = []
    pbw_modules = []
    primaries = []
    for slot, (label, momentum, parity, sector) in enumerate(
        zip(labels, MOMENTA, (0, alpha2, alpha3), ("NS", "R", "R"))
    ):
        effective_momentum = -momentum if slot == 0 and label < 0 else momentum
        module = FreeFieldModule(sector, B, effective_momentum)
        pbw_module = PBWModule(module)
        if slot == 0:
            primary = module.ns_branch(-label if label < 0 else label)
        else:
            primary = module.r_branch(label, parity)
        modules.append(module)
        pbw_modules.append(pbw_module)
        primaries.append(primary)
    return tuple(modules), tuple(pbw_modules), tuple(primaries)


def hatted_value(
    expressions,
    modules,
    pbw_modules,
    physical_form,
    auxiliary_form,
    *,
    primary_parity: int,
    form_parity: int,
    include_f_term: bool,
):
    """Evaluate one candidate hatted tensor-product form."""

    answer = 0.0j
    for (auxiliary1, physical1), coefficient1 in expressions[0].items():
        physical_parity1 = pbw_modules[0].parity(physical1)
        auxiliary_parity1 = modules[0].auxiliary_parity(auxiliary1)
        for (auxiliary2, physical2), coefficient2 in expressions[1].items():
            physical_parity2 = pbw_modules[1].parity(physical2)
            for (auxiliary3, physical3), coefficient3 in expressions[2].items():
                auxiliary_value = auxiliary_form.value(
                    (auxiliary1, auxiliary2, auxiliary3)
                )
                if abs(auxiliary_value) < 1.0e-13:
                    continue
                physical_value = physical_form.value(
                    (physical1, physical2, physical3)
                )
                if abs(physical_value) < 1.0e-13:
                    continue
                auxiliary_parity3 = modules[2].auxiliary_parity(auxiliary3)
                exponent = (
                    physical_parity1 * auxiliary_parity1
                    + (
                        physical_parity2
                        + primary_parity
                        + (form_parity if include_f_term else 0)
                    )
                    * auxiliary_parity3
                )
                answer += (
                    coefficient1
                    * coefficient2
                    * coefficient3
                    * (-1) ** exponent
                    * auxiliary_value
                    * physical_value
                )
    return answer


@dataclass
class Statistics:
    count: int = 0
    nonzero_count: int = 0
    failures: int = 0
    maximum_absolute_error: float = 0.0
    maximum_nonzero_relative_error: float = 0.0
    worst_case: object = None

    def record(self, *, left, right, case) -> None:
        error = abs(left - right)
        scale = max(abs(left), abs(right))
        self.count += 1
        if scale > NONZERO_TOLERANCE:
            self.nonzero_count += 1
            self.maximum_nonzero_relative_error = max(
                self.maximum_nonzero_relative_error, error / scale
            )
        if error > ABSOLUTE_TOLERANCE * max(scale, 1.0):
            self.failures += 1
        if error > self.maximum_absolute_error:
            self.maximum_absolute_error = error
            self.worst_case = (case, left, right)


def audit():
    """Return results categorized by sign candidate, p1, f, and eta."""

    configurations = descendant_configurations()
    branch_weights = BranchWeights(B, MOMENTA)
    statistics = {
        candidate: defaultdict(Statistics)
        for candidate in ("no_f", "with_f")
    }
    for labels in LABEL_TRIPLES:
        for alpha2, alpha3, primary_parity, eta in product(
            (0, 1), (0, 1), (0, 1), (-1, 1)
        ):
            form_parity = (int(2 * labels[0]) + alpha2 + alpha3) % 2
            modules, pbw_modules, primaries = branch_data(
                labels, alpha2, alpha3
            )
            auxiliary_form = HumanAuxiliaryThreePoint(modules)
            physical_form = HumanWPhysicalThreePoint(
                pbw_modules,
                form_parity,
                eta,
                primary_parity,
            )
            primary_expressions = tuple(
                branch_in_pbw(module, pbw_module, primary)
                for module, pbw_module, primary in zip(
                    modules, pbw_modules, primaries
                )
            )
            primary_values = {
                include_f: hatted_value(
                    primary_expressions,
                    modules,
                    pbw_modules,
                    physical_form,
                    auxiliary_form,
                    primary_parity=primary_parity,
                    form_parity=form_parity,
                    include_f_term=include_f,
                )
                for include_f in (False, True)
            }
            virasoro_forms = tuple(
                VirasoroThreePoint(
                    branch_weights.triple(labels, copy),
                    branch_weights.central_charges[copy],
                )
                for copy in (0, 1)
            )
            for configuration_index, configuration in enumerate(configurations):
                descendant_expressions = tuple(
                    branch_in_pbw(
                        module,
                        pbw_module,
                        module.descendant(primary, first_word, second_word),
                    )
                    for module, pbw_module, primary, (first_word, second_word)
                    in zip(modules, pbw_modules, primaries, configuration)
                )
                ordinary_product = 1.0 + 0.0j
                for copy in (0, 1):
                    ordinary_product *= virasoro_forms[copy].value(
                        *(configuration[slot][copy] for slot in range(3))
                    )
                case = (
                    labels,
                    alpha2,
                    alpha3,
                    primary_parity,
                    form_parity,
                    eta,
                    configuration_index,
                )
                for candidate, include_f in (("no_f", False), ("with_f", True)):
                    left = hatted_value(
                        descendant_expressions,
                        modules,
                        pbw_modules,
                        physical_form,
                        auxiliary_form,
                        primary_parity=primary_parity,
                        form_parity=form_parity,
                        include_f_term=include_f,
                    )
                    right = primary_values[include_f] * ordinary_product
                    statistics[candidate][
                        (primary_parity, form_parity, eta)
                    ].record(left=left, right=right, case=case)
    return statistics


def main() -> None:
    statistics = audit()
    for candidate in ("no_f", "with_f"):
        print(candidate)
        for (primary_parity, form_parity, eta), row in sorted(
            statistics[candidate].items()
        ):
            print(
                f"  p1={primary_parity}, f={form_parity}, eta={eta:+d}: "
                f"tests={row.count}, nonzero={row.nonzero_count}, "
                f"failures={row.failures}, "
                f"max_abs={row.maximum_absolute_error:.3e}, "
                f"max_nonzero_rel={row.maximum_nonzero_relative_error:.3e}"
            )
            if row.failures:
                print(f"    worst={row.worst_case}")


if __name__ == "__main__":
    main()
