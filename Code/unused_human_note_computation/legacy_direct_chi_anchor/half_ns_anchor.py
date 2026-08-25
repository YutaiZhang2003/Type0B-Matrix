#!/usr/bin/env python3
"""Direct half-integral NS anchors for NS--R--R branching.

The configurable recursion in :mod:`compute_target` advances the NS branch
label by one and therefore needs one independent anchor for each congruence
class.  This module supplies the missing ``n_NS in Z+1/2`` boundary directly:

1. expand the free-field chi strings on all three legs;
2. convert every physical endpoint to the NS/R super-Virasoro PBW basis;
3. evaluate the auxiliary Majorana form and the literal NS--R--R Ward form;
4. apply the hatted-form sign defined in Section 8; and
5. divide by the reflected Human-Note branch norms.

Every Ramond endpoint is retained in the Human Note's ``w^+,w^-`` basis.
No external Ramond convention or post-hoc frame conversion is used.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import sys
from typing import Sequence

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
REPOSITORY = CODE_DIR.parent
for directory in (
    CODE_DIR / "PBW_c_recursion_double_virasoro crosscheck",
    REPOSITORY / "python 2" / "nsrr_chi_branching",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ramond_pbw_generalized_ward import GeneralizedNRRWard  # noqa: E402


def _load_chi_branching():
    source = REPOSITORY / "python 2" / "nsrr_chi_branching" / "nsrr_chi_formula.py"
    specification = importlib.util.spec_from_file_location(
        "_type0b_half_ns_chi_formula", source
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load NSRR chi paths from {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CHI = _load_chi_branching()
SQRT2 = sp.sqrt(2)


def _rational(value) -> sp.Rational:
    if isinstance(value, float):
        return sp.Rational(str(value))
    return sp.Rational(value)


class DirectHalfNSAnchor:
    """Exact free-field/PBW evaluator for one NS--R--R branch trinion."""

    def __init__(self, *, b, momenta: Sequence[object]) -> None:
        if len(momenta) != 3:
            raise ValueError("momenta must contain P_NS, P_R1, and P_R2")
        self.b = _rational(b)
        self.momenta = tuple(_rational(value) for value in momenta)
        self.q = sp.cancel(self.b + 1 / self.b)
        self.central_charge = sp.cancel(
            sp.Rational(3, 2) + 3 * self.q**2
        )
        self.h_ns = sp.cancel(
            (self.q**2 / 4 - self.momenta[0] ** 2) / 2
        )
        self.h_ramond = tuple(
            sp.cancel(
                sp.Rational(1, 16)
                + self.q**2 / 8
                - momentum**2 / 2
            )
            for momentum in self.momenta[1:]
        )

    @staticmethod
    def _validate_labels(n_ns, n_r1, n_r2):
        n_ns, n_r1, n_r2 = map(_rational, (n_ns, n_r1, n_r2))
        if (2 * n_ns).q != 1:
            raise ValueError("n_NS must lie in Z/2")
        for name, label in (("n_R1", n_r1), ("n_R2", n_r2)):
            four_label = 4 * label
            if four_label.q != 1 or int(four_label) % 2 == 0:
                raise ValueError(f"{name} must lie in Z/2+1/4")
        return n_ns, n_r1, n_r2

    @lru_cache(maxsize=None)
    def numerator(
        self,
        n_ns,
        n_r1,
        n_r2,
        parity_r1: int,
        parity_r2: int,
        form_parity: int,
        eta: int,
    ) -> sp.Expr:
        """Return ``rhohat_f^eta(v_n,W_n^a,W_n^b)`` exactly."""

        n_ns, n_r1, n_r2 = self._validate_labels(n_ns, n_r1, n_r2)
        parity_r1, parity_r2 = int(parity_r1), int(parity_r2)
        form_parity, eta = int(form_parity), int(eta)
        if parity_r1 not in (0, 1) or parity_r2 not in (0, 1):
            raise ValueError("Ramond copy parities must be 0 or 1")
        if form_parity not in (0, 1):
            raise ValueError("form_parity must be 0 or 1")
        if eta not in (-1, 1):
            raise ValueError("eta must be +1 or -1")

        physical_form = GeneralizedNRRWard(
            p_phi=0,
            form_parity=form_parity,
            eta=eta,
            h_ns=self.h_ns,
            h_second=self.h_ramond[0],
            h_third=self.h_ramond[1],
            beta_second=self.momenta[1] / SQRT2,
            beta_third=self.momenta[2] / SQRT2,
            central_charge=self.central_charge,
        )
        first = CHI.ns_path_components(
            n_ns, self.q, self.momenta[0]
        )
        second = CHI.ramond_path_components(
            n_r1, parity_r1, self.q, self.momenta[1]
        )
        third = CHI.ramond_path_components(
            n_r2, parity_r2, self.q, self.momenta[2]
        )
        auxiliary_form_parity = (
            int(2 * n_ns) + parity_r1 + parity_r2 - form_parity
        ) % 2

        answer = sp.S.Zero
        for auxiliary_1, word_1, coefficient_1 in first:
            physical_parity_1 = CHI.stored.state_parity(word_1)
            for (
                auxiliary_2,
                auxiliary_ground_2,
                word_2,
                physical_ground_2,
                coefficient_2,
            ) in second:
                physical_parity_2 = CHI.stored.state_parity(
                    word_2, physical_ground_2
                )
                auxiliary_parity_2 = (
                    len(auxiliary_2) + auxiliary_ground_2
                ) % 2
                for (
                    auxiliary_3,
                    auxiliary_ground_3,
                    word_3,
                    physical_ground_3,
                    coefficient_3,
                ) in third:
                    auxiliary_parity_3 = (
                        len(auxiliary_3) + auxiliary_ground_3
                    ) % 2
                    auxiliary_value = CHI.stored.fermion_value(
                        auxiliary_form_parity,
                        auxiliary_1,
                        auxiliary_2,
                        auxiliary_ground_2,
                        auxiliary_3,
                        auxiliary_ground_3,
                    )
                    if auxiliary_value == 0:
                        continue
                    physical_value = physical_form.value(
                        word_1,
                        word_2,
                        physical_ground_2,
                        word_3,
                        physical_ground_3,
                    )
                    if physical_value == 0:
                        continue
                    hatted_form_sign = (-1) ** (
                        physical_parity_1 * (len(auxiliary_1) % 2)
                        + physical_parity_2 * auxiliary_parity_3
                        + form_parity * auxiliary_parity_3
                    )
                    answer += (
                        hatted_form_sign
                        * coefficient_1
                        * coefficient_2
                        * coefficient_3
                        * auxiliary_value
                        * physical_value
                    )
        return sp.factor(
            sp.cancel(
                CHI.ns_v_scale(n_ns, self.b, self.momenta[0]) * answer
            )
        )

    @lru_cache(maxsize=None)
    def norm_product(
        self,
        n_ns,
        n_r1,
        n_r2,
        parity_r1: int,
        parity_r2: int,
    ) -> sp.Expr:
        """Return the reflected Human-Note product of three branch norms."""

        n_ns, n_r1, n_r2 = self._validate_labels(n_ns, n_r1, n_r2)
        return sp.factor(
            sp.cancel(
                CHI.ns_v_norm(n_ns, self.b, self.momenta[0])
                * CHI.ramond_norm(
                    n_r1, parity_r1, self.b, self.momenta[1]
                )
                * CHI.ramond_norm(
                    n_r2, parity_r2, self.b, self.momenta[2]
                )
            )
        )

    def branching_product(
        self,
        *,
        labels,
        parities,
        form_parity: int,
        etas,
    ) -> sp.Expr:
        """Return the root-independent product ``B_f^eta B_f^eta'``."""

        if len(labels) != 3 or len(parities) != 2 or len(etas) != 2:
            raise ValueError("expected three labels, two parities, and two etas")
        left = self.numerator(
            *labels, *parities, int(form_parity), int(etas[0])
        )
        right = self.numerator(
            *labels, *parities, int(form_parity), int(etas[1])
        )
        denominator = self.norm_product(*labels, *parities)
        if denominator == 0:
            raise ZeroDivisionError("a direct half-NS branch norm vanished")
        return sp.factor(sp.cancel(left * right / denominator))


__all__ = ["DirectHalfNSAnchor"]
