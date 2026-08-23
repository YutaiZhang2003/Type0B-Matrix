#!/usr/bin/env python3
"""Literal finite Ramond branching function in the machine-note convention.

The note also gives a screening/Jack--Selberg representation at neutrality,
but its generic-momentum continuation requires a separate degree lemma.  This
module does not assume that lemma: it evaluates the literal finite
external-colour path function directly.  It is kept separate
from the direct simultaneous-primary assembly used by
``verify_level_7_2.py``; both representations call the same Ward functional,
whose signs are audited separately.

The public function ``raw_master`` is the literal finite sum A in the note;
``master_polynomial`` returns its pole-cleared abbreviation R, and
``branching_square`` implements the boxed branching formula with an explicit
argument list.  None of these functions uses a screening number or momentum
interpolation.  The current repository supplies branch-state
transport matrices through the requested onset level 7/2; the mathematical
formula itself is all-level at every fixed set of finite chi strings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRID_DIR = REPO / "python 2" / "ramond_three_point_grid"
FORMULA_PATH = GRID_DIR / "symbolic_formula_search" / "closed_path_formula.py"
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))


def _load_formula_module():
    specification = importlib.util.spec_from_file_location(
        "r_screening_closed_path", FORMULA_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {FORMULA_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


closed = _load_formula_module()
grid = closed.grid
I = sp.I


def intrinsic_ns_parity(n1) -> int:
    """Parity of the NS branch highest vector in the human-note basis."""

    return int(2 * sp.Rational(n1)) % 2


def effective_eta(labels, eta: int) -> int:
    """Convert the human eta to the fixed-even Ward representative."""

    eta = int(eta)
    if eta not in (-1, 1):
        raise ValueError("eta must be +1 or -1")
    return (-1) ** intrinsic_ns_parity(labels[0]) * eta


def master_polynomial(labels, epsilon2: int, eta: int, sample):
    """Return (R, Delta_1 Delta_2 Delta_3) from the finite contour paths."""

    eta_eff = effective_eta(labels, eta)
    numerator, denominator, _ = closed.cleared_ground_path_sum(
        tuple(map(sp.Rational, labels)), int(epsilon2), eta_eff, sample
    )
    return sp.factor(numerator), sp.factor(denominator)


def raw_master(labels, epsilon2: int, eta: int, sample):
    """Literal raw amplitude A from the finite external-colour path sum."""

    numerator, denominator = master_polynomial(labels, epsilon2, eta, sample)
    return sp.factor(sp.cancel(numerator / denominator))


def normalized_master_square(labels, epsilon2: int, eta: int, sample):
    """The epsilon3=f=0 normalized branching square."""

    labels = tuple(map(sp.Rational, labels))
    b_value, p1, p2, p3 = sample
    raw = raw_master(labels, epsilon2, eta, sample)
    norms = grid.raw_norms(
        *labels, int(epsilon2), 0, b_value, p1, p2, p3
    )
    return sp.factor(sp.cancel(raw**2 / sp.prod(norms)))


def normalized_square(
    labels,
    epsilon2: int,
    epsilon3: int,
    form_parity: int,
    eta: int,
    sample,
):
    """Full root-independent coefficient including epsilon3 and f."""

    master = normalized_master_square(labels, epsilon2, eta, sample)
    return sp.factor(
        (-1) ** int(epsilon3) * (-I) ** int(form_parity) * master
    )


def branching_square(
    b,
    p1,
    p2,
    p3,
    n1,
    n2,
    n3,
    epsilon2: int,
    epsilon3: int,
    form_parity: int,
    eta: int,
):
    """Unambiguous B^2(b,P_i,n_i,epsilon2,epsilon3,f,eta).

    The checked-in transition data cover n1 in {0,1/2,1} and
    n2,n3 in {1/4,3/4,5/4}; the finite recurrence in the note defines the
    same function beyond that range once the corresponding transition
    matrices are generated.
    """

    return normalized_square(
        (n1, n2, n3),
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        (b, p1, p2, p3),
    )


__all__ = [
    "branching_square",
    "effective_eta",
    "master_polynomial",
    "normalized_master_square",
    "normalized_square",
    "raw_master",
]
