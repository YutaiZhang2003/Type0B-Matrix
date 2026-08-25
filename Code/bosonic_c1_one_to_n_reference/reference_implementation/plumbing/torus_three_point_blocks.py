#!/usr/bin/env python3
"""Exact descendant baseline for a primary torus three-point necklace block."""

from __future__ import annotations

from itertools import product

import numpy as np

try:
    from torus_descendant_blocks import gram_matrix, rho_primary_external
except ImportError:  # pragma: no cover
    from plumbing.torus_descendant_blocks import gram_matrix, rho_primary_external


def necklace_descendant_coefficients_three_point(
    c: complex,
    internal_weights: tuple[complex, complex, complex],
    external_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
) -> np.ndarray:
    """Evaluate the defining three-edge descendant contraction exactly.

    This finite-level sewing sum is the regular c=25 value of the block.  It
    is also the reference used to validate the regulated h-recursion, whose
    individual residues are resonant at b=1.
    """
    if len(internal_weights) != 3 or len(external_weights) != 3 or len(orders) != 3:
        raise ValueError("the torus three-point necklace requires three edges")
    if any(int(order) < 0 for order in orders):
        raise ValueError("block orders must be non-negative")

    edge_data: list[list[tuple[tuple[tuple[int, ...], ...], np.ndarray]]] = []
    for weight, maximum_level in zip(internal_weights, orders):
        levels = []
        for level in range(int(maximum_level) + 1):
            basis, gram = gram_matrix(weight, c, level)
            levels.append((basis, np.linalg.inv(gram)))
        edge_data.append(levels)

    shape = tuple(int(order) + 1 for order in orders)
    coefficients = np.zeros(shape, dtype=np.complex128)
    for levels in np.ndindex(shape):
        dimensions = tuple(len(edge_data[edge][levels[edge]][0]) for edge in range(3))
        value = 0.0 + 0.0j
        for ket_indices in product(*(range(dimension) for dimension in dimensions)):
            for bra_indices in product(*(range(dimension) for dimension in dimensions)):
                term = 1.0 + 0.0j
                for edge in range(3):
                    next_edge = (edge + 1) % 3
                    basis, inverse_gram = edge_data[edge][levels[edge]]
                    next_basis, _ = edge_data[next_edge][levels[next_edge]]
                    term *= inverse_gram[ket_indices[edge], bra_indices[edge]]
                    term *= rho_primary_external(
                        basis[bra_indices[edge]],
                        next_basis[ket_indices[next_edge]],
                        internal_weights[edge],
                        external_weights[edge],
                        internal_weights[next_edge],
                        c,
                    )
                value += term
        coefficients[levels] = value
    return coefficients

