"""Formal four-edge series with the manuscript's theta parity cocycle.

No physical product constraint is imposed on the quotient. In particular,
coefficients with different left and right Ramond exponents are retained.
Scalars can be ordinary complex numbers or mpmath complex numbers.
"""

from __future__ import annotations

from itertools import product

ZERO = (0, 0, 0, 0)


def indices(twice_cutoff):
    """Increasing balanced total level; q3 remains integer-powered."""
    for degree in range(twice_cutoff + 1):
        for first in range(degree + 1):
            for third in range(0, degree - first + 1, 2):
                remaining = degree - first - third
                for left in range(remaining + 1):
                    yield first, left, remaining - left, third


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


def recover_minus(numerator, auxiliary, twice_cutoff, *, tolerance=1e-9):
    """Triangular division; the constant acts as sqrt(2)*Q on the minus ideal.

    The scalar is read from the supplied auxiliary constant, so this also
    handles a consistently rescaled external fermion. Invariant failures
    raise errors instead of projecting away evidence of an incorrect input.
    """
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
    for key in indices(twice_cutoff):
        value = tuple(a/scalar for a in residual.pop(key, zero_vector))
        projected = minus_projection(value)
        error = max(abs(a-b) for a,b in zip(value, projected))
        scale = max(1, *(abs(a) for a in value))
        if error > tolerance*scale:
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
