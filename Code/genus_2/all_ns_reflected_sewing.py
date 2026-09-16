"""All-NS theta sewing in a declared radial reflection frame.

The supplied blocks remain the descendant-only Human-Note blocks, including
their quadratic parity sign K. This adapter acts on all four lift values;
it does not edit their Ward identities, recursion, or primary powers.

For real physical weights, the reflected NS pants norm uses the Hermitian
Gram and a conjugate vertex. In the Human-Note lift basis this removes K by
a finite Fourier transform. The derivation and independent checks live in
audit_ns_sewing_identity_limits.py. This local conversion does not establish
the global NSRR-to-NSNSNS spin transport.
"""
from __future__ import annotations

import cmath
from itertools import product
from typing import Mapping, Sequence


# Geometry order (0, 1, infinity); fix the redundant infinity lift to +1.
LIFTS = tuple((a, b, 1) for a, b in product((1, -1), repeat=2))


def quadratic_parity(parities: Sequence[int]) -> int:
    if len(parities) != 3 or any(p not in (0, 1) for p in parities):
        raise ValueError("Three parity bits are required.")
    a, b, c = parities
    return (a*b + a*c + b*c) % 2


def lift_conversion(sector: int, primary_parities=(0, 0, 0)):
    """Return the exact half-integer matrix H_a = U_a F_a.

    a is relative descendant parity. The matrix depends on the absolute
    parity a+sum(primary_parities), as does the Human-Note K sign.
    """
    if sector not in (0, 1):
        raise ValueError("The relative three-form sector must be 0 or 1.")
    quadratic_parity(primary_parities)  # Validate the full parity triple.
    absolute = (sector + sum(primary_parities)) % 2
    parities = tuple(p for p in product((0, 1), repeat=3)
                     if sum(p) % 2 == absolute)
    characters = tuple(tuple(
        a**p[0]*b**p[1] for p in parities) for a, b, _ in LIFTS)
    return tuple(tuple(
        sum(characters[i][k] * (-1)**quadratic_parity(p) * characters[j][k]
            for k, p in enumerate(parities)) / 4
        for j in range(4)) for i in range(4))


def reflected_blocks(descendant_blocks: Mapping, sector: int,
                     primary_parities=(0, 0, 0)):
    """Convert four complex lift values, retaining their relative phases."""
    if set(descendant_blocks) != set(LIFTS):
        raise ValueError("All four complex Human-Note lift values are required.")
    values = tuple(complex(descendant_blocks[lift]) for lift in LIFTS)
    conversion = lift_conversion(sector, primary_parities)
    return {lift: sum(coefficient*value for coefficient, value
                      in zip(conversion[i], values))
            for i, lift in enumerate(LIFTS)}


def coefficient_matrix(sector: int, left_constant: complex,
                       right_constant: complex, lift: Sequence[int]):
    """Matrix in the literal lift basis for even NS highest states.

    Constants are BRY (C for sector 0, Ctilde for sector 1), with independent
    oriented left/right products. The explicit Human-Note odd sewing minus
    and the two odd-vertex i factors have already been combined. No q^h is
    included. The convention is F^T M conjugate(F).
    """
    try:
        index = LIFTS.index(tuple(lift))
    except ValueError as exc:
        raise ValueError("Use a canonical geometry lift with infinity +1.") from exc
    row = lift_conversion(sector)[index]
    constant = complex(left_constant)*complex(right_constant)
    return tuple(tuple(constant*a*b for b in row) for a in row)


def primary_from_weights(q_values, weights):
    """Primary propagation, separately from the descendant lift transform."""
    if len(q_values) != 3 or len(weights) != 3:
        raise ValueError("Three plumbing parameters and three weights are required.")
    return cmath.exp(sum(complex(h)*cmath.log(complex(q))
                         for h, q in zip(weights, q_values)))


def contract_reflected_ns(blocks, left_constants, right_constants, lift, primary):
    """Evaluate both three-form sectors, with P outside their matrices."""
    if set(blocks) != {0, 1} or len(left_constants) != 2 or len(right_constants) != 2:
        raise ValueError("Both NS three-form sectors and both constants are required.")
    values = {a: reflected_blocks(blocks[a], a)[tuple(lift)] for a in (0, 1)}
    terms = {
        a: complex(left_constants[a])*complex(right_constants[a])
        * abs(complex(primary))**2 * abs(values[a])**2 for a in (0, 1)
    }
    return {"total": sum(terms.values()), "terms": terms,
            "reflected_descendant_blocks": values}
