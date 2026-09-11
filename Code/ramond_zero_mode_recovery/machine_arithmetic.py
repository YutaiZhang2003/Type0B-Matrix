"""Native binary64 arithmetic for the existing direct physical PBW rules.

Level labels and rational constants may remain exact until they enter a
numerical expression. Physical weights, Gram matrices and contractions use
Python complex / NumPy complex128. No small floating coefficient is clipped.
"""

from fractions import Fraction
import cmath

import numpy as np


F = complex
I = 1j
SQRT2 = cmath.sqrt(2)
PRIME = None
PRECISION_BITS = 53


def divide(a, b):
    if isinstance(a, (complex, np.complexfloating)) or isinstance(b, (complex, np.complexfloating)):
        return complex(a)/complex(b)
    return Fraction(a)/Fraction(b)


def array(values):
    return np.asarray(values, dtype=np.complex128)


def mm(a, b):
    # BPZ contractions are bilinear: no conjugation is introduced here.
    return array(a) @ array(b)


def inverse(a):
    return np.linalg.inv(array(a))


def lu_factor(a):
    # Preserve the legacy provider's interface (a cached inverse).
    return inverse(a)


def lu_solve(a, b):
    return mm(a, b)


class NumpyProxy:
    def __getattr__(self, name):
        return getattr(np, name)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.complex128)

    def zeros_like(self, a):
        return self.zeros(a.shape)

    def eye(self, n):
        return np.eye(n, dtype=np.complex128)

    def remainder(self, a, p):
        return a

