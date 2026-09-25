"""Canonical NS three-point convention used by ``Human Notes/SCblock.tex``.

The note uses a fixed-parity trilinear form.  Relative to component-ordered
Ward kernels, a nonzero parity-``a`` coefficient carries the third-slot sign

    rho_a^human(x1, x2, x3) = (-1)^(a |x3|) rho_a^component(x1, x2, x3).

For highest-weight states of intrinsic parities ``(p1,p2,p3)``, the graded
Ward ordering used in the note contributes the additional factor

    (-1)^(p1 |x1| + p2 |x3|).

Here ``|xi|`` denotes parity relative to the corresponding highest-weight
state.  Keeping this factor in the convention layer is equivalent to putting
the intrinsic primary parities into every odd contour crossing, and makes the
two normalizations ``rho_0(000)=1`` and ``rho_1(010)=1`` valid for every
choice of ``p_i``.

All public NS Ward and global-osp APIs use this convention.  Component-ordered
kernels, where needed to implement the recurrences, remain private.
"""

from __future__ import annotations

from collections.abc import Sequence


def normalize_parity_triple(
    parities: Sequence[int], *, name: str = "parities"
) -> tuple[int, int, int]:
    """Validate and return a three-bit parity tuple."""

    if len(parities) != 3:
        raise ValueError(f"{name} must contain exactly three entries")
    normalized = tuple(int(parity) for parity in parities)
    if any(parity not in (0, 1) for parity in normalized):
        raise ValueError(f"{name} must contain only zeroes and ones")
    return normalized  # type: ignore[return-value]


def primary_parity_ward_sign(
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    r"""Return ``(-1)^(p1*x1+p2*x3)`` for the graded Ward convention."""

    x1, _x2, x3 = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    p1, p2, _p3 = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    return -1 if (p1 * x1 + p2 * x3) % 2 else 1


def human_note_rho_sign(
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    """Return the full fixed-parity sign for one NS three-point tensor."""

    normalized = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    sector = sum(normalized) % 2
    fixed_parity_sign = -1 if sector * normalized[2] else 1
    return fixed_parity_sign * primary_parity_ward_sign(
        normalized, primary_parities
    )


def relative_three_form_label(descendant_parities: Sequence[int]) -> int:
    r"""Return the note's label ``a=sum_i A_i (mod 2)``.

    This is the label on ``rho_a`` and ``F_a``.  It does not include the
    intrinsic primary parities.
    """

    descendants = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    return sum(descendants) % 2


def absolute_three_form_parity(
    relative_label: int,
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    r"""Return ``a+p_1+p_2+p_3 (mod 2)``.

    The absolute parity is used to match holomorphic and antiholomorphic
    tensors in a local two-dimensional CFT.  It is not the label of the
    holomorphic block or of an NS fusion polynomial.
    """

    label = int(relative_label)
    if label not in (0, 1):
        raise ValueError("relative_label must be zero or one")
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    return (label + sum(primaries)) % 2


def relative_label_from_absolute(
    absolute_parity: int,
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    """Invert :func:`absolute_three_form_parity`."""

    parity = int(absolute_parity)
    if parity not in (0, 1):
        raise ValueError("absolute_parity must be zero or one")
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    return (parity + sum(primaries)) % 2


def theta_orientation_exponent(
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    r"""Return the theta sign exponent ``Q(A+p_1,C+p_2,E+p_3)``."""

    descendants = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    x1, x2, x3 = (
        descendant ^ primary
        for descendant, primary in zip(descendants, primaries)
    )
    return (x1 * x2 + x1 * x3 + x2 * x3) % 2


def theta_orientation_sign(
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    """Return the literal theta-channel sign in the current note."""

    return -1 if theta_orientation_exponent(
        descendant_parities, primary_parities
    ) else 1


def theta_polarization_exponent(
    left_parities: Sequence[int],
    right_parities: Sequence[int],
) -> int:
    r"""Return the polarization ``Q(l+r)-Q(l)-Q(r)`` modulo two."""

    left = normalize_parity_triple(left_parities, name="left_parities")
    right = normalize_parity_triple(right_parities, name="right_parities")
    return sum(
        left[i] * right[j] + right[i] * left[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ) % 2


def enlarged_ns_three_form_crossing_sign(
    descendant_parities: Sequence[int],
    auxiliary_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    r"""Return the ``SCA x F`` matrix-element crossing sign.

    Slots are ``(infinity,one,zero)``.  With the note's ungraded product BPZ
    convention, the three-point function is the corresponding matrix element
    and factorizes with

    ``(-1)^[(p_2+|x_2|)|u_3|]``.

    Thus the pairing convention is not an unrelated modification of the
    vertex: it fixes how the bra is evaluated and is inherited by the
    matrix-element definition of the three-point form.
    """

    _x1, x2, _x3 = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    _u1, _u2, u3 = normalize_parity_triple(
        auxiliary_parities, name="auxiliary_parities"
    )
    _p1, p2, _p3 = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    exponent = (p2 + x2) * u3
    return -1 if exponent % 2 else 1


def theta_primary_parity_rephasing(
    lifts: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> tuple[int, tuple[int, int, int]]:
    r"""Reduce a generic-``p_i`` theta block to the even-primary block.

    With literal note lifts ``eta_i``,

    ``F_a^(p)(eta) = prefactor * F_a^(0)(eta_effective)``,

    where ``prefactor=(-1)^Q(p) prod_i eta_i^p_i`` and
    ``eta_effective_i=eta_i*(-1)^sum_(j!=i) p_j``.
    """

    lift_tuple = tuple(int(value) for value in lifts)
    if len(lift_tuple) != 3 or any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("lifts must contain three signs")
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    prefactor = theta_orientation_sign((0, 0, 0), primaries)
    for lift, primary in zip(lift_tuple, primaries):
        prefactor *= lift**primary
    effective = tuple(
        lift_tuple[edge]
        * (-1) ** sum(
            primaries[other] for other in range(3) if other != edge
        )
        for edge in range(3)
    )
    return int(prefactor), effective  # type: ignore[return-value]


def glasses_primary_parity_rephasing(
    lifts: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> tuple[int, tuple[int, int, int]]:
    r"""Reduce the current glasses block with generic ``p_i`` to ``p_i=0``.

    Edge order is ``(left handle,right handle,bridge)``.  The three effective
    lift flips include both the glass orientation and the two graded Ward
    tensors.
    """

    lift_tuple = tuple(int(value) for value in lifts)
    if len(lift_tuple) != 3 or any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("lifts must contain three signs")
    p1, p2, p3 = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    prefactor = lift_tuple[0] ** p1 * lift_tuple[1] ** p2 * lift_tuple[2] ** p3
    effective = (
        lift_tuple[0] * (-1) ** (p1 + p3),
        lift_tuple[1] * (-1) ** (p2 + p3),
        lift_tuple[2] * (-1) ** (p1 + p2),
    )
    return int(prefactor), effective


def ns_null_factorization_sign(
    *,
    slot: int,
    null_parity: int,
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    r"""Return the one-null ``rho_a`` sign in the note's slot ordering.

    Slots are ``(infinity,one,zero)`` and the descendant parities are
    ``(A,C,E)``.  For ``delta=rs mod 2`` the three signs are

    ``(-1)^(delta*(p_1+A)), 1, (-1)^(delta*(1+p_2))``.
    """

    if slot not in (0, 1, 2):
        raise ValueError("slot must be zero, one, or two")
    delta = int(null_parity)
    if delta not in (0, 1):
        raise ValueError("null_parity must be zero or one")
    a_parity, _c_parity, _e_parity = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    p1, p2, _p3 = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    exponents = (delta * (p1 + a_parity), 0, delta * (1 + p2))
    return -1 if exponents[slot] % 2 else 1


def ns_double_null_factorization_sign(
    *,
    pair: Sequence[int],
    null_parity: int,
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    r"""Return the A.6 sign in its printed fusion-polynomial ordering.

    The allowed ordered pairs are ``(0,1)``, ``(0,2)``, and ``(1,2)``.
    The function composes the two verified one-null signs and shifts the
    intrinsic parity of the first extracted null module before the second
    extraction.
    """

    ordered_pair = tuple(int(value) for value in pair)
    if ordered_pair not in ((0, 1), (0, 2), (1, 2)):
        raise ValueError("pair must be (0,1), (0,2), or (1,2)")
    delta = int(null_parity)
    if delta not in (0, 1):
        raise ValueError("null_parity must be zero or one")
    descendants = normalize_parity_triple(
        descendant_parities, name="descendant_parities"
    )
    primaries = list(
        normalize_parity_triple(primary_parities, name="primary_parities")
    )
    first, second = ordered_pair
    sign = ns_null_factorization_sign(
        slot=first,
        null_parity=delta,
        descendant_parities=descendants,
        primary_parities=primaries,
    )
    primaries[first] ^= delta
    sign *= ns_null_factorization_sign(
        slot=second,
        null_parity=delta,
        descendant_parities=descendants,
        primary_parities=primaries,
    )
    return sign


__all__ = [
    "absolute_three_form_parity",
    "enlarged_ns_three_form_crossing_sign",
    "glasses_primary_parity_rephasing",
    "human_note_rho_sign",
    "ns_double_null_factorization_sign",
    "ns_null_factorization_sign",
    "normalize_parity_triple",
    "primary_parity_ward_sign",
    "relative_label_from_absolute",
    "relative_three_form_label",
    "theta_orientation_exponent",
    "theta_orientation_sign",
    "theta_polarization_exponent",
    "theta_primary_parity_rephasing",
]
