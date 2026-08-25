#!/usr/bin/env python3
"""Explicit Koszul assembly of the large-c NS regular block.

The large-c module separates into a non-global vacuum sector and a global
``osp(1|2)`` sector.  Their scalar graph coefficients cannot in general be
convolved without a sign, because each scalar coefficient has already been
put into the chosen plumbing half-edge order.  No new product is needed.

Fix the half-edge order obtained by concatenating the ordered slots of all
trinions, and a second order in which internal half-edges are paired for
contraction.  If ``Q(p)`` is the Koszul parity of this permutation for a
half-edge parity vector ``p``, the cross sign for vacuum parity ``a`` and
global parity ``b`` is the polarization

    (-1)**B(a,b),  B(a,b) = Q(a+b) + Q(a) + Q(b)  (mod 2).

This file implements that formula for an arbitrary ordered plumbing graph.
The optional linear BPZ/spin-frame ledger is included in ``Q``; as it must,
it cancels from the polarized cross sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

from ns_human_convention import ns_null_factorization_sign


Level = tuple[int, ...]


def _bits(values: Sequence[int], expected: int, name: str) -> tuple[int, ...]:
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}")
    return tuple(int(value) % 2 for value in values)


def _sign_bit(value: int, name: str) -> int:
    """Return zero for ``+1`` and one for ``-1``."""

    sign = int(value)
    if sign not in (-1, 1):
        raise ValueError(f"{name} entries must be +1 or -1")
    return int(sign == -1)


@dataclass(frozen=True)
class PlumbingOrientation:
    """One explicit half-edge and BPZ ordering convention.

    ``edge_half_edges[e]`` gives the two positions of internal edge ``e`` in
    vertex-slot order.  ``external_half_edges`` lists the remaining positions.
    ``contraction_order`` is a permutation of all half-edge positions.  The
    entries of ``edge_linear_bits`` and ``external_linear_bits`` encode fixed
    linear BPZ/spin-frame signs.
    """

    edge_half_edges: tuple[tuple[int, int], ...]
    external_half_edges: tuple[int, ...]
    contraction_order: tuple[int, ...]
    edge_linear_bits: tuple[int, ...] = ()
    external_linear_bits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        internal = tuple(index for pair in self.edge_half_edges for index in pair)
        positions = internal + self.external_half_edges
        if len(set(positions)) != len(positions):
            raise ValueError("each half-edge position must occur exactly once")
        if set(self.contraction_order) != set(positions):
            raise ValueError("contraction_order must permute all half-edges")
        if self.edge_linear_bits and len(self.edge_linear_bits) != len(
            self.edge_half_edges
        ):
            raise ValueError("edge_linear_bits has the wrong length")
        if self.external_linear_bits and len(self.external_linear_bits) != len(
            self.external_half_edges
        ):
            raise ValueError("external_linear_bits has the wrong length")

    @property
    def edge_count(self) -> int:
        return len(self.edge_half_edges)

    @property
    def external_count(self) -> int:
        return len(self.external_half_edges)

    def half_edge_parities(
        self,
        edge_parities: Sequence[int],
        external_parities: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """Expand edge and external parities into vertex-slot order."""

        edge_bits = _bits(edge_parities, self.edge_count, "edge_parities")
        external_bits = _bits(
            external_parities, self.external_count, "external_parities"
        )
        size = len(self.contraction_order)
        result = [0] * size
        for parity, (left, right) in zip(edge_bits, self.edge_half_edges):
            result[left] = parity
            result[right] = parity
        for parity, position in zip(external_bits, self.external_half_edges):
            result[position] = parity
        return tuple(result)

    def exponent(
        self,
        edge_parities: Sequence[int],
        external_parities: Sequence[int] = (),
    ) -> int:
        """Return the full orientation exponent ``Q`` modulo two."""

        edge_bits = _bits(edge_parities, self.edge_count, "edge_parities")
        external_bits = _bits(
            external_parities, self.external_count, "external_parities"
        )
        half_bits = self.half_edge_parities(edge_bits, external_bits)
        target_position = {
            half_edge: position
            for position, half_edge in enumerate(self.contraction_order)
        }
        inversions = sum(
            half_bits[left] * half_bits[right]
            for left in range(len(half_bits))
            for right in range(left + 1, len(half_bits))
            if target_position[left] > target_position[right]
        )
        edge_linear = self.edge_linear_bits or (0,) * self.edge_count
        external_linear = self.external_linear_bits or (0,) * self.external_count
        linear = sum(a * b for a, b in zip(edge_linear, edge_bits)) + sum(
            a * b for a, b in zip(external_linear, external_bits)
        )
        return int((inversions + linear) % 2)

    def sign(
        self,
        edge_parities: Sequence[int],
        external_parities: Sequence[int] = (),
    ) -> int:
        """Return ``(-1)^Q`` for one parity assignment."""

        return -1 if self.exponent(edge_parities, external_parities) else 1

    def polarized_exponent(
        self,
        vacuum_edge_parities: Sequence[int],
        global_edge_parities: Sequence[int],
        global_external_parities: Sequence[int] = (),
    ) -> int:
        """Return the vacuum/global cross exponent ``B`` modulo two.

        The vacuum factor has even external legs.  The global factor carries
        the prescribed external parities.  Linear BPZ terms cancel in this
        polarization, furnishing a useful convention check.
        """

        vacuum = _bits(
            vacuum_edge_parities, self.edge_count, "vacuum_edge_parities"
        )
        global_ = _bits(
            global_edge_parities, self.edge_count, "global_edge_parities"
        )
        external = _bits(
            global_external_parities,
            self.external_count,
            "global_external_parities",
        )
        combined = tuple(left ^ right for left, right in zip(vacuum, global_))
        zeros = (0,) * self.external_count
        return (
            self.exponent(combined, external)
            + self.exponent(vacuum, zeros)
            + self.exponent(global_, external)
        ) % 2

    def cross_sign(
        self,
        vacuum_edge_parities: Sequence[int],
        global_edge_parities: Sequence[int],
        global_external_parities: Sequence[int] = (),
    ) -> int:
        """Return the explicit Koszul sign multiplying one convolution term."""

        return (
            -1
            if self.polarized_exponent(
                vacuum_edge_parities,
                global_edge_parities,
                global_external_parities,
            )
            else 1
        )


@dataclass(frozen=True)
class PlumbingFrameLedger:
    """Construct the linear orientation bits from lifted plumbing frames.

    ``half_edge_frame_signs[h]`` is the literal sign relating the odd local
    coordinate at half-edge ``h`` to the canonical ``(infinity, one, zero)``
    trinion frame.  ``edge_transition_signs[e]`` is the remaining sign in the
    canonical-frame odd sewing map after the displayed spin lift has been
    factored out,

        theta_right_can = transition_sign * i * xi * sqrt(q)/z
                          * theta_left_can.

    Therefore an odd state on edge ``e=(h,hbar)`` acquires the product of the
    two frame signs and the transition sign.  This product, rather than a
    manually assigned ``beta_e``, defines the edge's linear orientation bit.
    """

    edge_half_edges: tuple[tuple[int, int], ...]
    external_half_edges: tuple[int, ...]
    contraction_order: tuple[int, ...]
    half_edge_frame_signs: tuple[int, ...]
    edge_transition_signs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        positions = tuple(
            position for pair in self.edge_half_edges for position in pair
        ) + self.external_half_edges
        if len(set(positions)) != len(positions):
            raise ValueError("each half-edge position must occur exactly once")
        if set(positions) != set(range(len(positions))):
            raise ValueError("half-edge positions must be numbered consecutively")
        if set(self.contraction_order) != set(positions):
            raise ValueError("contraction_order must permute all half-edges")
        if len(self.half_edge_frame_signs) != len(positions):
            raise ValueError("one frame sign is required for every half-edge")
        for sign in self.half_edge_frame_signs:
            _sign_bit(sign, "half_edge_frame_signs")
        transitions = self.edge_transition_signs or (1,) * len(
            self.edge_half_edges
        )
        if len(transitions) != len(self.edge_half_edges):
            raise ValueError("one transition sign is required for every edge")
        for sign in transitions:
            _sign_bit(sign, "edge_transition_signs")

    def orientation(self) -> PlumbingOrientation:
        """Return the orientation polynomial determined by these maps."""

        transitions = self.edge_transition_signs or (1,) * len(
            self.edge_half_edges
        )
        edge_bits = tuple(
            _sign_bit(
                self.half_edge_frame_signs[left]
                * self.half_edge_frame_signs[right]
                * transition,
                "derived edge frame sign",
            )
            for (left, right), transition in zip(
                self.edge_half_edges, transitions
            )
        )
        external_bits = tuple(
            _sign_bit(
                self.half_edge_frame_signs[position],
                "derived external frame sign",
            )
            for position in self.external_half_edges
        )
        return PlumbingOrientation(
            edge_half_edges=self.edge_half_edges,
            external_half_edges=self.external_half_edges,
            contraction_order=self.contraction_order,
            edge_linear_bits=edge_bits,
            external_linear_bits=external_bits,
        )


def canonical_null_endpoint_sign(
    *,
    r: int,
    s: int,
    null_slot: int,
    slot_parities: Sequence[int],
    canonical_order: Sequence[int],
) -> int:
    """Return the component-order sign for canonicalizing one null incidence.

    Slots are initially ordered as ``(infinity, one, zero)``.  The supplied
    ``canonical_order`` is a permutation that moves the null leg to either
    the first or third canonical slot.  Three independent effects are then
    included:

    * the Koszul sign of that slot permutation;
    * the NS reflection sign ``(-1)**(rs*b)`` when the null is third, where
      ``b`` is the parity in the canonical middle slot.

    This lower-level permutation/reflection sign is not the complete
    human-note ``rho_a`` factorization sign for generic intrinsic parities.
    That sign is :func:`ns_human_convention.ns_null_factorization_sign`.
    Local frame signs are deliberately excluded here: in the master recursion
    they occur once, through the linear part of ``Q_Gamma`` and its transport
    ratio.  Use :func:`null_incidence_sign` only in a formulation where that
    frame sign has not already been included in the graph orientation.
    """

    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS labels require positive r,s with r+s even")
    parities = _bits(slot_parities, 3, "slot_parities")
    order = tuple(int(value) for value in canonical_order)
    if set(order) != {0, 1, 2}:
        raise ValueError("canonical_order must permute the three slots")
    if null_slot not in (0, 1, 2):
        raise ValueError("null_slot must be 0, 1, or 2")
    null_parity = (r * s) % 2
    if parities[null_slot] != null_parity:
        raise ValueError("the null slot parity must equal rs modulo two")
    canonical_null_slot = order.index(null_slot)
    if canonical_null_slot not in (0, 2):
        raise ValueError("the canonical order must put the null first or third")

    target_position = {
        original_slot: position for position, original_slot in enumerate(order)
    }
    koszul = sum(
        parities[left] * parities[right]
        for left in range(3)
        for right in range(left + 1, 3)
        if target_position[left] > target_position[right]
    )
    middle_parity = parities[order[1]]
    reflection = (
        null_parity * middle_parity if canonical_null_slot == 2 else 0
    )
    return -1 if (koszul + reflection) % 2 else 1


def null_incidence_sign(
    *,
    r: int,
    s: int,
    null_slot: int,
    slot_parities: Sequence[int],
    canonical_order: Sequence[int],
    half_edge_frame_sign: int,
) -> int:
    """Return the canonical endpoint sign with its local frame included."""

    canonical = canonical_null_endpoint_sign(
        r=r,
        s=s,
        null_slot=null_slot,
        slot_parities=slot_parities,
        canonical_order=canonical_order,
    )
    frame = -1 if (
        (r * s) % 2
        and _sign_bit(half_edge_frame_sign, "half_edge_frame_sign")
    ) else 1
    return canonical * frame


# Vertex-slot order is
#
#   (0_L, 1_L, infinity_L, 0_R, 1_R, infinity_R),
#
# while contraction order pairs the two half-edges of each tube.  In the
# convention printed in ``Human Notes/SCblock.tex`` the fixed infinity-frame
# sign is absorbed into the lifted plumbing coordinate.  Consequently the
# theta orientation is the literal quadratic sign
#
#     (-1)^(p0 p1 + p0 p_infinity + p1 p_infinity)
#
# with no additional linear infinity-edge term.
THETA_FRAME_LEDGER = PlumbingFrameLedger(
    edge_half_edges=((0, 3), (1, 4), (2, 5)),
    external_half_edges=(),
    contraction_order=(0, 3, 1, 4, 2, 5),
    half_edge_frame_signs=(1, 1, 1, 1, 1, 1),
)
THETA_ORIENTATION = THETA_FRAME_LEDGER.orientation()


def regular_series(
    *,
    vacuum_coefficients: Mapping[Level, complex],
    global_coefficients: Mapping[Level, complex],
    orientation: PlumbingOrientation,
    global_edge_parity_offsets: Sequence[int] = (),
    global_external_parities: Sequence[int] = (),
) -> dict[Level, complex]:
    """Assemble the unrestricted all-level double sum on finite input series.

    The two mappings may contain every coefficient available at a chosen
    cutoff.  Unlike ``regular_coefficient``, this routine does not fix a
    target level: it sums every pair ``(vacuum_level, global_level)`` and
    returns all resulting regular-block coefficients at once.
    """

    result: dict[Level, complex] = {}
    offsets = (
        _bits(
            global_edge_parity_offsets,
            orientation.edge_count,
            "global_edge_parity_offsets",
        )
        if global_edge_parity_offsets
        else (0,) * orientation.edge_count
    )
    for vacuum_level, vacuum_value in vacuum_coefficients.items():
        if len(vacuum_level) != orientation.edge_count:
            raise ValueError("vacuum coefficient key has the wrong length")
        for global_level, global_value in global_coefficients.items():
            if len(global_level) != orientation.edge_count:
                raise ValueError("global coefficient key has the wrong length")
            level = tuple(
                int(vacuum_level[edge]) + int(global_level[edge])
                for edge in range(orientation.edge_count)
            )
            sign = orientation.cross_sign(
                tuple(value % 2 for value in vacuum_level),
                tuple(
                    value % 2 ^ offset
                    for value, offset in zip(global_level, offsets)
                ),
                global_external_parities,
            )
            result[level] = result.get(level, 0.0 + 0.0j) + (
                sign * complex(vacuum_value) * complex(global_value)
            )
    return result


def regular_series_parity_resummed(
    *,
    vacuum_coefficients: Mapping[Level, complex],
    global_coefficients: Mapping[Level, complex],
    orientation: PlumbingOrientation,
    global_edge_parity_offsets: Sequence[int] = (),
    global_external_parities: Sequence[int] = (),
) -> dict[Level, complex]:
    """Implement the finite parity-resummed function-level formula.

    For each global parity sector ``sigma``, bilinearity turns the cross sign
    into a sign flip of the vacuum plumbing variables.  This is the truncated
    series realization of Eq. (regular-parity-resummed) in the note.
    """

    edge_count = orientation.edge_count
    offsets = (
        _bits(
            global_edge_parity_offsets,
            edge_count,
            "global_edge_parity_offsets",
        )
        if global_edge_parity_offsets
        else (0,) * edge_count
    )
    result: dict[Level, complex] = {}
    for sigma in product((0, 1), repeat=edge_count):
        flip = tuple(
            orientation.polarized_exponent(
                tuple(int(index == edge) for index in range(edge_count)),
                sigma,
                global_external_parities,
            )
            for edge in range(edge_count)
        )
        for vacuum_level, vacuum_value in vacuum_coefficients.items():
            if len(vacuum_level) != edge_count:
                raise ValueError("vacuum coefficient key has the wrong length")
            vacuum_sign = -1 if sum(
                int(vacuum_level[edge]) * flip[edge]
                for edge in range(edge_count)
            ) % 2 else 1
            for global_level, global_value in global_coefficients.items():
                if len(global_level) != edge_count:
                    raise ValueError("global coefficient key has the wrong length")
                if tuple(
                    int(value) % 2 ^ offset
                    for value, offset in zip(global_level, offsets)
                ) != sigma:
                    continue
                level = tuple(
                    int(vacuum_level[edge]) + int(global_level[edge])
                    for edge in range(edge_count)
                )
                result[level] = result.get(level, 0.0 + 0.0j) + (
                    vacuum_sign * complex(vacuum_value) * complex(global_value)
                )
    return result


def regular_coefficient(
    *,
    level: Sequence[int],
    vacuum_coefficients: Mapping[Level, complex],
    global_coefficient: Callable[[Level], complex],
    orientation: PlumbingOrientation,
    global_edge_parity_offsets: Sequence[int] = (),
    global_external_parities: Sequence[int] = (),
) -> complex:
    """Evaluate one coefficient of the regular block without a named product.

    The supplied vacuum and global coefficients are scalar graph coefficients
    that already include their individual orientation signs.  The displayed
    polarized sign is therefore the only extra factor in their convolution.
    """

    target = tuple(int(value) for value in level)
    if len(target) != orientation.edge_count:
        raise ValueError("level vector has the wrong number of internal edges")
    offsets = (
        _bits(
            global_edge_parity_offsets,
            orientation.edge_count,
            "global_edge_parity_offsets",
        )
        if global_edge_parity_offsets
        else (0,) * orientation.edge_count
    )
    result = 0.0 + 0.0j
    for vacuum_level, vacuum_value in vacuum_coefficients.items():
        if len(vacuum_level) != orientation.edge_count:
            raise ValueError("vacuum coefficient key has the wrong length")
        global_level = tuple(
            target[edge] - int(vacuum_level[edge])
            for edge in range(orientation.edge_count)
        )
        if any(value < 0 for value in global_level):
            continue
        sign = orientation.cross_sign(
            tuple(value % 2 for value in vacuum_level),
            tuple(
                value % 2 ^ offset
                for value, offset in zip(global_level, offsets)
            ),
            global_external_parities,
        )
        result += sign * complex(vacuum_value) * complex(
            global_coefficient(global_level)
        )
    return result


def _self_check() -> None:
    if THETA_ORIENTATION.edge_linear_bits != (0, 0, 0):
        raise AssertionError("theta orientation no longer matches the human note")

    # A redefinition of one intermediate canonical odd coordinate changes
    # its local frame sign and the adjacent transition sign together.  The
    # compiled edge bits, hence every block coefficient, are invariant.
    reframed_theta = PlumbingFrameLedger(
        edge_half_edges=THETA_FRAME_LEDGER.edge_half_edges,
        external_half_edges=(),
        contraction_order=THETA_FRAME_LEDGER.contraction_order,
        half_edge_frame_signs=(-1, 1, 1, 1, 1, 1),
        edge_transition_signs=(-1, 1, 1),
    ).orientation()
    if reframed_theta.edge_linear_bits != THETA_ORIENTATION.edge_linear_bits:
        raise AssertionError("compiled edge bits changed under a frame gauge move")

    for e0 in (0, 1):
        for e1 in (0, 1):
            for einf in (0, 1):
                expected = (e0 * e1 + e0 * einf + e1 * einf) % 2
                if THETA_ORIENTATION.exponent((e0, e1, einf)) != expected:
                    raise AssertionError("theta orientation polynomial changed")

    # The lower-level component-order canonicalizations are sign-free in the
    # three minimal configurations.  They must not be confused with the full
    # fixed-parity rho_a factorization signs checked immediately below.
    component_endpoint_signs = (
        null_incidence_sign(
            r=3,
            s=1,
            null_slot=0,
            slot_parities=(1, 0, 0),
            canonical_order=(0, 1, 2),
            half_edge_frame_sign=1,
        ),
        null_incidence_sign(
            r=3,
            s=1,
            null_slot=1,
            slot_parities=(0, 1, 0),
            canonical_order=(1, 2, 0),
            half_edge_frame_sign=1,
        ),
        null_incidence_sign(
            r=3,
            s=1,
            null_slot=2,
            slot_parities=(0, 0, 1),
            canonical_order=(0, 1, 2),
            half_edge_frame_sign=1,
        ),
    )
    if component_endpoint_signs != (1, 1, 1):
        raise AssertionError("component-order endpoint signs changed")

    human_endpoint_signs = tuple(
        ns_null_factorization_sign(
            slot=slot,
            null_parity=1,
            descendant_parities=(0, 0, 0),
            primary_parities=(0, 0, 0),
        )
        for slot in range(3)
    )
    if human_endpoint_signs != (1, 1, -1):
        raise AssertionError("human-note odd-null rho signs changed")

    # Belavin--Geiko reflection: a third-slot odd null crossing an odd middle
    # component supplies (-1)^rs.  A first-slot null has no such factor.
    if canonical_null_endpoint_sign(
        r=3,
        s=1,
        null_slot=2,
        slot_parities=(0, 1, 1),
        canonical_order=(0, 1, 2),
    ) != -1:
        raise AssertionError("third-slot NS reflection sign changed")
    if canonical_null_endpoint_sign(
        r=3,
        s=1,
        null_slot=0,
        slot_parities=(1, 1, 0),
        canonical_order=(0, 1, 2),
    ) != 1:
        raise AssertionError("first-slot NS fusion sign changed")

    # The polarized sign is independent of every linear BPZ bit.
    changed = PlumbingOrientation(
        edge_half_edges=THETA_ORIENTATION.edge_half_edges,
        external_half_edges=(),
        contraction_order=THETA_ORIENTATION.contraction_order,
        edge_linear_bits=(1, 1, 0),
    )
    for left in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
        for right in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)):
            if THETA_ORIENTATION.cross_sign(left, right) != changed.cross_sign(
                left, right
            ):
                raise AssertionError("linear BPZ data survived polarization")

    vacuum = {
        (0, 0, 0): 1,
        (1, 0, 0): 2,
        (0, 1, 1): -3,
    }
    global_ = {
        (0, 0, 0): 5,
        (0, 1, 0): 7,
        (1, 0, 1): 11,
    }
    direct = regular_series(
        vacuum_coefficients=vacuum,
        global_coefficients=global_,
        orientation=THETA_ORIENTATION,
    )
    resummed = regular_series_parity_resummed(
        vacuum_coefficients=vacuum,
        global_coefficients=global_,
        orientation=THETA_ORIENTATION,
    )
    if direct != resummed:
        raise AssertionError("parity-resummed all-level series changed")


if __name__ == "__main__":
    _self_check()
    print("explicit NS regular-block orientation checks: PASS")
