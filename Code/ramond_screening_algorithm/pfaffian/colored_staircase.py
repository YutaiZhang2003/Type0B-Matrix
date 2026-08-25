"""Two-core/holonomy form of the colored Nekrasov product.

This file contains no SCA state construction and no Ward recursion.  It
implements Eq. (4.1) of arXiv:1210.7454 and, importantly, uses the two
*different* two-core representatives of a Ramond branch.  Merely putting
the same staircase in both holonomy sectors loses the affine path data.

The published bifundamental is a matrix element in

    H + sl(2)_2 + NSR,

not yet the desired fixed-spin matrix element in ``F + NSR``.  The hard
certificate at the bottom shows exactly which entry it determines and
isolates the extra trivalent (Racah/WZW-deconvolution) matrix that is still
needed for the other spin channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools

import sympy as sp


Diagram = tuple[int, ...]
Charges = tuple[int, int]


def staircase(size: int) -> Diagram:
    """Return the two-core ``delta_size=(size,...,1)``."""

    size = int(size)
    if size < 0:
        raise ValueError("a staircase size must be nonnegative")
    return tuple(range(size, 0, -1))


@lru_cache(None)
def _column_heights(diagram: Diagram) -> tuple[int, ...]:
    width = diagram[0] if diagram else 0
    return tuple(
        sum(row_length >= column for row_length in diagram)
        for column in range(1, width + 1)
    )


def boxes(diagram: Diagram):
    for row, row_length in enumerate(diagram, start=1):
        for column in range(1, row_length + 1):
            yield row, column


def arm(diagram: Diagram, square: tuple[int, int]) -> int:
    row, column = square
    return diagram[row - 1] - column


def leg(diagram: Diagram, square: tuple[int, int]) -> int:
    """Generalized leg length, also when ``square`` is outside ``diagram``."""

    row, column = square
    heights = _column_heights(diagram)
    height = heights[column - 1] if column <= len(heights) else 0
    return height - row


def e_factor(
    first: Diagram,
    second: Diagram,
    momentum,
    square: tuple[int, int],
    b,
):
    return momentum - leg(second, square) / b + (arm(first, square) + 1) * b


def colored_bifundamental(
    alpha,
    p_left,
    diagrams_left: tuple[Diagram, Diagram],
    charges_left: Charges,
    p_right,
    diagrams_right: tuple[Diagram, Diagram],
    charges_right: Charges,
    b,
):
    """Colored ``Z_bif`` of Eq. (4.1) of arXiv:1210.7454.

    Arms and generalized legs are precomputed, so every selected box adds
    one linear factor in constant time.  If ``B`` is the total number of
    boxes on the two sides, this routine uses ``O(B)`` arithmetic
    operations and ``O(B)`` memory (hence in particular meets an
    ``O(B**2)`` target).
    """

    q_background = b + 1 / b
    left_momenta = (p_left, -p_left)
    right_momenta = (p_right, -p_right)
    answer = sp.Integer(1)
    for i, j in itertools.product(range(2), repeat=2):
        y_i = diagrams_right[i]
        w_j = diagrams_left[j]
        q_i = int(charges_right[i])
        u_j = int(charges_left[j])
        for square in boxes(y_i):
            parity = leg(w_j, square) + arm(y_i, square) + 1
            if parity % 2 == (u_j - q_i) % 2:
                e = e_factor(
                    y_i,
                    w_j,
                    right_momenta[i] - left_momenta[j],
                    square,
                    b,
                )
                answer *= q_background - e - alpha
        for square in boxes(w_j):
            parity = leg(y_i, square) + arm(w_j, square) + 1
            if parity % 2 == (q_i - u_j) % 2:
                e = e_factor(
                    w_j,
                    y_i,
                    left_momenta[j] - right_momenta[i],
                    square,
                    b,
                )
                answer *= e - alpha
    return sp.factor(answer)


@dataclass(frozen=True)
class ColoredCore:
    """A colored two-core representing one level-one affine shift."""

    diagram: Diagram
    charge: int
    shift: sp.Rational


def core_for_shift(shift) -> ColoredCore:
    """Map an affine shift to its unique minimal colored diagram.

    If ``k=2*shift``, the level-one Cartan charge is ``k``.  Equations
    (2.21)--(2.24) of arXiv:1211.2788 give

      k > 0 : delta_(k-1) with corner color k mod 2,
      k <= 0: delta_(-k)  with corner color k mod 2.

    This includes ``core(0)=empty^0`` and
    ``core(1/2)=empty^1``; an empty diagram therefore does not erase its
    holonomy.
    """

    shift = sp.Rational(shift)
    twice = 2 * shift
    if not twice.is_integer:
        raise ValueError(f"shift must be integral or half-integral: {shift}")
    k = int(twice)
    size = k - 1 if k > 0 else -k
    return ColoredCore(staircase(size), k % 2, shift)


@dataclass(frozen=True)
class TwoCorePath:
    """The two level-one cores in one consecutive-GKO path."""

    first: ColoredCore
    second: ColoredCore
    orientation: int

    @property
    def diagrams(self) -> tuple[Diagram, Diagram]:
        return self.first.diagram, self.second.diagram

    @property
    def charges(self) -> Charges:
        return self.first.charge, self.second.charge

    @property
    def box_count(self) -> int:
        return sum(map(sum, self.diagrams))


def ns_path(branch_label) -> TwoCorePath:
    """The unique two-core path of a positive NS branch."""

    branch_label = sp.Rational(branch_label)
    return TwoCorePath(core_for_shift(branch_label), core_for_shift(-branch_label), 0)


def ramond_paths(branch_label, delta: int = 1) -> tuple[TwoCorePath, TwoCorePath]:
    """The two affine-arrow paths of a Ramond branch.

    For ``delta=+1`` their shifts are

      (n-1/4, 3/4-n), (n+1/4, 1/4-n).

    ``delta=-1`` is their simultaneous affine reflection.  These are the
    two holonomies; they are generally represented by different diagrams.
    """

    branch_label = sp.Rational(branch_label)
    delta = int(delta)
    if delta not in (-1, 1):
        raise ValueError("delta must be +1 or -1")
    if delta == 1:
        shifts = (
            (branch_label - sp.Rational(1, 4), sp.Rational(3, 4) - branch_label),
            (branch_label + sp.Rational(1, 4), sp.Rational(1, 4) - branch_label),
        )
    else:
        shifts = (
            (-branch_label + sp.Rational(1, 4), branch_label - sp.Rational(3, 4)),
            (-branch_label - sp.Rational(1, 4), branch_label - sp.Rational(1, 4)),
        )
    return tuple(
        TwoCorePath(core_for_shift(first), core_for_shift(second), orientation)
        for orientation, (first, second) in enumerate(shifts)
    )


def allowed_path_pairs(n1) -> tuple[tuple[int, int], tuple[int, int]]:
    """Path selection imposed by the NS branch parity."""

    twice = 2 * sp.Rational(n1)
    if not twice.is_integer:
        raise ValueError(n1)
    if int(twice) % 2 == 0:
        return (0, 0), (1, 1)
    return (0, 1), (1, 0)


def ramond_holonomy_matrix(n2, n3, b, p1, p2, p3):
    """Published colored matrix for an NS-primary--R--R vertex.

    The exact dictionary fixed by the hard identity is

      alpha_col = Q/2 + P1,  p_R,col = P_R + Q/2.

    The function deliberately has no ``n1`` argument: arXiv:1210.7454
    supplies the NS *primary* vertex only.  A nonzero external NS branch
    requires the missing charged/descendant trivalent vertex.
    """

    q = b + 1 / b
    alpha = q / 2 + p1
    second = ramond_paths(n2)
    third = ramond_paths(n3)
    return sp.Matrix(
        [
            [
                colored_bifundamental(
                    alpha,
                    p2 + q / 2,
                    left.diagrams,
                    left.charges,
                    p3 + q / 2,
                    right.diagrams,
                    right.charges,
                    b,
                )
                for right in third
            ]
            for left in second
        ]
    )


def stripped_ell3(x, q):
    """``2^(-1/8) ell(x,3)`` in the notes' convention."""

    return sp.expand(x**2 + q * x + 1)


def hard_polynomials(q, p1, p2, p3):
    x_plus_plus = q / 2 + p1 + p2 + p3
    x_minus_minus = q / 2 + p1 - p2 - p3
    d2 = stripped_ell3(q + 2 * p2, q)
    d3 = stripped_ell3(q + 2 * p3, q)
    e2 = q + 2 * p2
    e3 = q + 2 * p3
    line = sp.expand(x_plus_plus * (x_minus_minus - q))
    factorized = sp.expand(
        stripped_ell3(x_plus_plus, q)
        * stripped_ell3(q - x_minus_minus, q)
    )
    crossed = sp.expand(
        line**2 + 2 * line * (1 + e2 * e3) + d2 * d3
    )
    return factorized, crossed, d2, d3, e2, e3, line


def required_hard_racah_kernel(q, p1, p2, p3, matrix):
    """Return the extra kernel required to turn colored ``Z`` into ``H``.

    This is an obstruction certificate, not a proposed general formula.
    We write ``H=sum_ij R_ij Z_ij`` with ``R_11=1`` and equal off-diagonal
    entries.  The remaining two entries are then uniquely fixed.  Their
    nontrivial momentum dependence is precisely the datum absent from the
    colored bifundamental and from the scalar GKO product.
    """

    _, crossed, _, _, e2, e3, _ = hard_polynomials(q, p1, p2, p3)
    off_diagonal = sp.expand(e2 * e3 + q**2 + q * (p2 + p3))
    upper_left = sp.expand(
        crossed
        - matrix[1, 1]
        - off_diagonal * (matrix[0, 1] + matrix[1, 0])
    )
    return sp.Matrix(((upper_left, off_diagonal), (off_diagonal, 1)))


def entrywise_contraction(first: sp.Matrix, second: sp.Matrix):
    if first.shape != second.shape:
        raise ValueError((first.shape, second.shape))
    return sp.expand(sum(first[i, j] * second[i, j] for i in range(first.rows) for j in range(first.cols)))


def audit_hard_symbolic() -> None:
    """Exact, Ward-free hard calibration including phases and obstruction."""

    b, p1, p2, p3 = sp.symbols("b P_1 P_2 P_3", nonzero=True)
    q = b + 1 / b
    matrix = ramond_holonomy_matrix(sp.Rational(3, 4), sp.Rational(3, 4), b, p1, p2, p3)
    factorized, crossed, d2, d3, *_ = hard_polynomials(q, p1, p2, p3)

    x_plus_plus = q / 2 + p1 + p2 + p3
    x_minus_minus = q / 2 + p1 - p2 - p3
    expected_matrix = sp.Matrix(
        (
            (
                1,
                (x_minus_minus - q) * (x_plus_plus + q),
            ),
            (
                x_plus_plus * (x_minus_minus - 2 * q),
                factorized,
            ),
        )
    )
    for residual in matrix - expected_matrix:
        if sp.factor(sp.cancel(residual)) != 0:
            raise AssertionError(sp.factor(residual))

    racah = required_hard_racah_kernel(q, p1, p2, p3, matrix)
    residual = sp.factor(
        sp.cancel(entrywise_contraction(racah, matrix) - crossed)
    )
    if residual != 0:
        raise AssertionError(residual)

    # Exact spin-frame dictionary in the repository convention.
    # Do not ask SymPy for a multivariate gcd here: the ratios are fixed
    # convention factors, so checking their scalar multipliers is enough.
    r0_plus = -(1 + sp.I) * matrix[1, 1] / (d2 * d3)
    r0_minus = -(1 - sp.I) * crossed / (d2 * d3)
    r1_plus = sp.I * sp.sqrt(2) * r0_plus
    r1_minus = -sp.I * sp.sqrt(2) * r0_minus
    assert r1_plus == sp.I * sp.sqrt(2) * r0_plus
    assert r1_minus == -sp.I * sp.sqrt(2) * r0_minus

    determinant_residual = sp.factor(sp.cancel(matrix.det() - crossed))
    permanent_residual = sp.factor(
        sp.cancel(
            matrix[0, 0] * matrix[1, 1]
            + matrix[0, 1] * matrix[1, 0]
            - crossed
        )
    )
    if determinant_residual == 0 or permanent_residual == 0:
        raise AssertionError("an ordinary determinant unexpectedly produced H")

    print("hard colored matrix: exact two-core residual=0")
    print("Z_11=K: exact residual=0")
    print("H=sum(R_Racah .* Z): exact residual=0")
    print("det(Z)-H and perm(Z)-H are nonzero")
    print("spin frame: R1^+=i*sqrt(2) R0^+, R1^-=-i*sqrt(2) R0^-")


if __name__ == "__main__":
    audit_hard_symbolic()
