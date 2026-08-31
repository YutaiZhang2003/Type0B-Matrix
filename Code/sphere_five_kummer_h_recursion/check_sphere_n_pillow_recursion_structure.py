#!/usr/bin/env python3
"""Algebraic checks for the sphere n-point pillow h-recursion proposal."""

from __future__ import annotations

import sympy as sp


def effective_plumbing(segment_count: int) -> tuple[tuple[sp.Symbol, ...], tuple[sp.Expr, ...]]:
    """Return raw and endpoint-normalized plumbing parameters."""

    raw = sp.symbols(f"p1:{segment_count + 1}")
    effective = tuple(
        4 ** (int(index == 0) + int(index == segment_count - 1)) * value
        for index, value in enumerate(raw)
    )
    return raw, effective


def aligned_phases(segment_count: int) -> tuple[int, ...]:
    """Collision-aligned phases for an open necklace with mobile insertions."""

    if segment_count < 2:
        return (1,)
    return (-1,) + (1,) * (segment_count - 2) + (-1,)


def check_endpoint_normalization(max_segments: int = 8) -> None:
    """Every open necklace must have total effective plumbing 16 q."""

    for segment_count in range(1, max_segments + 1):
        raw, effective = effective_plumbing(segment_count)
        q = sp.prod(raw)
        assert sp.simplify(sp.prod(effective) - 16 * q) == 0


def check_collision_alignment(max_segments: int = 8) -> None:
    """The nested collision limit fixes the endpoint signs and all 4 factors."""

    for segment_count in range(2, max_segments + 1):
        ope_coordinates = sp.symbols(f"x1:{segment_count + 1}")
        raw_leading = (
            (-ope_coordinates[0] / 4,)
            + tuple(ope_coordinates[1:-1])
            + (-ope_coordinates[-1] / 4,)
        )
        phases = aligned_phases(segment_count)
        assert sp.prod(phases) == 1
        aligned = tuple(
            phase * value for phase, value in zip(phases, raw_leading)
        )
        effective = tuple(
            4 ** (int(index == 0) + int(index == segment_count - 1)) * value
            for index, value in enumerate(aligned)
        )
        assert all(
            sp.simplify(value - expected) == 0
            for value, expected in zip(effective, ope_coordinates)
        )


def check_fixed_difference_shifts(max_segments: int = 8) -> None:
    """A residue shift changes only its singular physical internal weight."""

    pole, level = sp.symbols("h_pole ell")
    for segment_count in range(1, max_segments + 1):
        offsets = (sp.Integer(0),) + sp.symbols(f"a2:{segment_count + 1}")
        for singular_edge in range(segment_count):
            pole_h = pole - offsets[singular_edge]
            physical_at_pole = tuple(pole_h + offset for offset in offsets)

            if singular_edge == 0:
                shifted_h = pole + level
                shifted_offsets = (sp.Integer(0),) + tuple(
                    offset - level for offset in offsets[1:]
                )
            else:
                shifted_h = pole_h
                shifted_offsets = tuple(
                    offset + level if index == singular_edge else offset
                    for index, offset in enumerate(offsets)
                )

            physical_after_shift = tuple(
                shifted_h + offset for offset in shifted_offsets
            )
            for index in range(segment_count):
                expected = (
                    pole + level
                    if index == singular_edge
                    else physical_at_pole[index]
                )
                assert sp.simplify(physical_after_shift[index] - expected) == 0


def check_low_point_specializations() -> None:
    """The endpoint convention reproduces the four- and five-point variables."""

    raw_four, effective_four = effective_plumbing(1)
    assert effective_four == (16 * raw_four[0],)

    raw_five, effective_five = effective_plumbing(2)
    assert effective_five == (4 * raw_five[0], 4 * raw_five[1])

    raw_six, effective_six = effective_plumbing(3)
    assert effective_six == (4 * raw_six[0], raw_six[1], 4 * raw_six[2])


def main() -> None:
    check_endpoint_normalization()
    check_collision_alignment()
    check_fixed_difference_shifts()
    check_low_point_specializations()
    print("sphere n-point pillow recursion structural checks: PASS")
    print("checked 1 through 8 internal segments")
    print("effective plumbing product: 16 q")
    print("collision-aligned phases: (-1, +1, ..., +1, -1)")
    print("every fixed-difference residue shift changes only its pole edge")


if __name__ == "__main__":
    main()
