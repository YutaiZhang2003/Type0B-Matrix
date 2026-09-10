"""Exact finite-field arithmetic and blocked linear algebra for validation.

Matrices store canonical integer residues in float64. A dot product is exact
because every partial integer sum is below 2**53; this bound is checked.
No floating-point division or approximate rank decision is used.
"""

from fractions import Fraction
import numpy as np
import sympy as sp

PRIME = 65521


class F(int):
    def __new__(cls, value=0):
        if isinstance(value, Fraction):
            value = value.numerator * pow(value.denominator % PRIME, -1, PRIME)
        elif isinstance(value, complex):
            return F(value.real) + I * F(value.imag)
        elif isinstance(value, float):
            return F(Fraction(value))
        return int.__new__(cls, int(value) % PRIME)

    def __add__(self, other):
        return F(int(self) + int(F(other)))

    __radd__ = __add__

    def __sub__(self, other):
        return F(int(self) - int(F(other)))

    def __rsub__(self, other):
        return F(other) - self

    def __mul__(self, other):
        return F(int(self) * int(F(other)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return F(int(self) * pow(int(F(other)), -1, PRIME))

    def __rtruediv__(self, other):
        return F(other) / self

    def __neg__(self):
        return F(-int(self))

    def __pow__(self, exponent):
        exponent = Fraction(exponent)
        if exponent.denominator != 1:
            raise ValueError("nonintegral finite-field power")
        return F(pow(int(self), int(exponent), PRIME))

    def __abs__(self):
        return int(bool(self))


I = F(sp.sqrt_mod(-1, PRIME))
SQRT2 = F(sp.sqrt_mod(2, PRIME))


def divide(a, b):
    if isinstance(a, F) or isinstance(b, F):
        return F(a) / b
    return Fraction(a) / Fraction(b)


def array(values):
    return np.asarray(values, dtype=np.float64) % PRIME


def mm(a, b):
    assert a.shape[-1] * (PRIME - 1) ** 2 < 2**53
    return np.remainder(a @ b, PRIME)


def lu_factor(matrix, block=32):
    a = array(matrix).copy()
    n = len(a)
    assert a.shape == (n, n)
    permutation = np.arange(n)
    for start in range(0, n, block):
        stop = min(n, start + block)
        for j in range(start, stop):
            nonzero = np.flatnonzero(a[j:, j])
            if not len(nonzero):
                raise np.linalg.LinAlgError(f"singular modulo {PRIME}, pivot {j}/{n}")
            pivot = j + nonzero[0]
            if pivot != j:
                a[[j, pivot], :] = a[[pivot, j], :]
                permutation[[j, pivot]] = permutation[[pivot, j]]
            inverse = pow(int(a[j, j]), -1, PRIME)
            a[j + 1:, j] = np.remainder(a[j + 1:, j] * inverse, PRIME)
            if j + 1 < stop:
                a[j + 1:, j + 1:stop] = np.remainder(
                    a[j + 1:, j + 1:stop]
                    - a[j + 1:, j, None] * a[None, j, j + 1:stop], PRIME)
        for j in range(start, stop):
            if j > start:
                a[j, stop:] = np.remainder(a[j, stop:] - mm(a[j, start:j], a[start:j, stop:]), PRIME)
        if stop < n:
            a[stop:, stop:] = np.remainder(a[stop:, stop:] - mm(a[stop:, start:stop], a[start:stop, stop:]), PRIME)
    return a, permutation


def lu_solve(factor, rhs, block=32):
    lu, permutation = factor
    rhs = array(rhs)
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs[:, None]
    b = rhs[permutation, :].copy()
    n = len(lu)
    for start in range(0, n, block):
        stop = min(n, start + block)
        for j in range(start, stop):
            if j > start:
                b[j] = np.remainder(b[j] - mm(lu[j, start:j], b[start:j]), PRIME)
        if stop < n:
            b[stop:] = np.remainder(b[stop:] - mm(lu[stop:, start:stop], b[start:stop]), PRIME)
    for stop in range(n, 0, -block):
        start = max(0, stop - block)
        for j in range(stop - 1, start - 1, -1):
            if j + 1 < stop:
                b[j] = np.remainder(b[j] - mm(lu[j, j + 1:stop], b[j + 1:stop]), PRIME)
            b[j] = np.remainder(b[j] * pow(int(lu[j, j]), -1, PRIME), PRIME)
        if start:
            b[:start] = np.remainder(b[:start] - mm(lu[:start, start:stop], b[start:stop]), PRIME)
    return b[:, 0] if vector else b


def inverse(matrix):
    result = lu_solve(lu_factor(matrix), np.eye(len(matrix)))
    if not np.array_equal(mm(array(matrix), result), np.eye(len(matrix))):
        raise AssertionError("exact inverse residual is nonzero")
    return result
