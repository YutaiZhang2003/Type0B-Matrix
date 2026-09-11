"""Formal four-edge series with the manuscript's theta parity cocycle.

No physical product constraint is imposed on the quotient. In particular,
coefficients with different left and right Ramond exponents are retained.
Scalars can be ordinary complex numbers or mpmath complex numbers.
"""

from __future__ import annotations

from itertools import product
from functools import lru_cache

ZERO = (0, 0, 0, 0)


def indices(twice_cutoff):
    """Increasing balanced total level; q3 remains integer-powered."""
    for degree in range(twice_cutoff + 1):
        for first in range(degree + 1):
            for third in range(0, degree - first + 1, 2):
                remaining = degree - first - third
                for left in range(remaining + 1):
                    yield first, left, remaining - left, third


def diagonal_indices(twice_cutoff):
    """Physical coefficients: both split middle exponents are equal."""
    for degree in range(twice_cutoff+1):
        for first in range(degree+1):
            for third in range(0, degree-first+1, 2):
                remaining = degree-first-third
                if remaining % 2 == 0:
                    yield first, remaining//2, remaining//2, third


@lru_cache(None)
def diagonal_virasoro_indices(left_budget, right_budget):
    """Downward closure of shifted equal-power product targets."""
    return tuple(sorted(((a,b,c,d)
        for a in range(min(left_budget,right_budget)+1)
        for d in range(min(left_budget,right_budget)-a+1)
        for b in range(left_budget-a-d+1)
        for c in range(right_budget-a-d+1)), key=lambda n:(sum(n),n)))


def diagonal_virasoro_product(left, right, left_budget, right_budget):
    """Multiply ordinary four-level series only at the required targets.

    Inputs include all lower unequal-power coefficients. Returned exponents
    have the outer levels doubled, ready for the pipeline's branching shift.
    """
    result = {}
    for t in range(min(left_budget,right_budget)+1):
        b, c = left_budget-t, right_budget-t
        for a in range(t+1):
            for d in range(t-a+1):
                value = 0
                for aa in range(a+1):
                    for bb in range(b+1):
                        for cc in range(c+1):
                            for dd in range(d+1):
                                value += left[aa,bb,cc,dd]*right[a-aa,b-bb,c-cc,d-dd]
                result[2*a,b,c,2*d] = value
    return result


def theta_sign(index):
    a, b, c = ((index >> j) & 1 for j in range(3))
    return (-1) ** (a*b + a*c + b*c)


STAR_SIGN = tuple(tuple(theta_sign(a)*theta_sign(b)*theta_sign(a ^ b)
                       for b in range(8)) for a in range(8))


def star(left, right):
    answer = [0] * 8
    for a, x in enumerate(left):
        if x:
            for b, y in enumerate(right):
                if y:
                    answer[a ^ b] += STAR_SIGN[a][b] * x * y
    return tuple(answer)


def minus_projection(value):
    x_times = star((0, 0, 0, 0, 0, 0, 1, 0), value)
    return tuple((a-b)/2 for a, b in zip(value, x_times))


def scalar_product(left, right, twice_cutoff):
    # A carry-free integer encoding replaces tuple construction in the inner
    # loop. The degree bound guarantees each resulting coordinate is smaller
    # than the encoding base. Grouping by degree avoids forbidden pairs.
    width = max(1, twice_cutoff.bit_length())
    mask = (1 << width)-1
    def packed(key):
        return sum(value << (width*j) for j,value in enumerate(key))
    right_terms = sorted((sum(key), packed(key), value)
                         for key,value in right.items() if value and sum(key) <= twice_cutoff)
    answer = {}
    for a, x in left.items():
        if not x:
            continue
        remaining = twice_cutoff-sum(a)
        encoded_a = packed(a)
        for degree, encoded_b, y in right_terms:
            if degree > remaining:
                break
            key = encoded_a+encoded_b
            answer[key] = answer.get(key, 0)+x*y
    return {tuple((key >> (width*j)) & mask for j in range(4)):value
            for key,value in answer.items()}


def recover_minus(numerator, auxiliary, twice_cutoff, *, tolerance=1e-9,
                  sector_policy="error", diagnostics=None, diagonal_only=False):
    """Triangular division; the constant acts as sqrt(2)*Q on the minus ideal.

    The scalar is read from the supplied auxiliary constant, so this also
    handles a consistently rescaled external fermion. By default sector
    residuals exceeding tolerance raise an error. Explicit sector_policy=
    "record" continues with the unaltered coefficients and records residuals.
    Neither policy projects the coefficients into the sector.
    """
    if sector_policy not in ("error", "record"):
        raise ValueError("sector_policy must be error or record")
    if diagonal_only and any(key[1] != key[2] for series in (numerator,auxiliary) for key in series):
        raise ValueError('diagonal recovery requires equal-middle-power inputs')
    if diagnostics is not None:
        diagnostics.update(policy=sector_policy, tolerance=tolerance,
                           processed_vectors=0, vectors_above_tolerance=0,
                           maximum_absolute_residual=0.0,
                           maximum_scaled_residual=0.0,
                           first_exceedance=None, worst_scaled_exponents=None)
    zero_vector = (0,) * 8
    constant = auxiliary.get(ZERO, zero_vector)
    expected = (constant[0], 0, 0, 0, 0, 0, -constant[0], 0)
    if not constant[0] or max(abs(x-y) for x,y in zip(constant, expected)) > tolerance:
        raise ValueError("Auxiliary constant must be a nonzero multiple of 1-eta2*eta3")
    scalar = 2*constant[0]
    positive = sorted((sum(key), key, tuple((j,x) for j,x in enumerate(value) if x))
                      for key, value in auxiliary.items()
                      if any(key) and any(value) and sum(key) <= twice_cutoff)
    # Propagate each solved coefficient forward. This is the same triangular
    # division, but it never scans shifts incompatible with a target's degree
    # or constructs and rejects negative multi-indices.
    residual = {key:list(value) for key,value in numerator.items() if sum(key) <= twice_cutoff}
    answer = {}
    for key in (diagonal_indices(twice_cutoff) if diagonal_only else indices(twice_cutoff)):
        value = tuple(a/scalar for a in residual.pop(key, zero_vector))
        projected = minus_projection(value)
        error = max(abs(a-b) for a,b in zip(value, projected))
        scale = max(1, *(abs(a) for a in value))
        if any(a.real != a.real or a.imag != a.imag or abs(a) == float("inf")
               for a in value):
            raise ArithmeticError(f"Nonfinite recovered coefficient at {key}")
        if diagnostics is not None:
            diagnostics["processed_vectors"] += 1
            diagnostics["maximum_absolute_residual"] = max(
                diagnostics["maximum_absolute_residual"], float(error))
            scaled = float(error/scale)
            if scaled > diagnostics["maximum_scaled_residual"]:
                diagnostics["maximum_scaled_residual"] = scaled
                diagnostics["worst_scaled_exponents"] = key
        if error > tolerance*scale:
            if diagnostics is not None:
                diagnostics["vectors_above_tolerance"] += 1
                if diagnostics["first_exceedance"] is None:
                    diagnostics["first_exceedance"] = {
                        "exponents":key, "absolute_residual":float(error),
                        "scaled_residual":float(error/scale)}
            if sector_policy == "error":
                raise ArithmeticError(f"Numerator leaves the minus ideal at {key}: {error}")
        answer[key] = value
        nonzero = tuple((j,x) for j,x in enumerate(value) if x)
        if not nonzero:
            continue
        remaining = twice_cutoff-sum(key)
        for degree, shift, coefficient in positive:
            if degree > remaining:
                break
            future = (key[0]+shift[0],key[1]+shift[1],key[2]+shift[2],key[3]+shift[3])
            target = residual.setdefault(future, [0]*8)
            for a,x in coefficient:
                for b,y in nonzero:
                    target[a ^ b] -= STAR_SIGN[a][b]*x*y
    return answer


def evaluate(series, *, q1, q2_left, q2_right, q3, spin_signs=(1, 1, 1)):
    answer = 0
    for (first, left, right, third), vector in series.items():
        parity_value = sum(value * product_sign(index, spin_signs)
                           for index, value in enumerate(vector))
        answer += (parity_value * q1**(first/2) * q2_left**left
                   * q2_right**right * q3**(third/2))
    return answer


def product_sign(index, signs):
    value = 1
    for edge in range(3):
        if (index >> edge) & 1:
            value *= signs[edge]
    return value
