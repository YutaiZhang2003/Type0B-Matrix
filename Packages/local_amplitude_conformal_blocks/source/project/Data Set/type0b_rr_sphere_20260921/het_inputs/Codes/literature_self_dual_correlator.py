"""b=1 super-Liouville correlators in the literal literature normalization.

Only the common external Upsilon factor is stripped, as in the c=3
benchmark. These are fixed-coordinate correlators, not heterotic amplitudes.
The spectral measure is dP; no moduli integral occurs in this module.
"""
from functools import lru_cache

import mpmath as mp
import numpy as np

from literature_component_blocks import sewn_integrand
from literature_self_dual_blocks import (
    SelfDualLiteratureBlocks, ExtrapolatedLiteratureBlocks, analytic_antiholomorphic,
)
from literature_super_liouville import PaperConstants


class SelfDualConstants(PaperConstants):
    """Equations (10)--(13),(28) at b=1, evaluated with Barnes G."""
    def __init__(self, precision=70):
        self.b, self.Q, self.precision = 1., 2., precision

    @lru_cache(None)
    def log_u(self, x):
        with mp.workdps(self.precision):
            x = mp.mpc(x)
            return complex(mp.log(mp.barnesg(x)) + mp.log(mp.barnesg(2-x)))

    @lru_cache(None)
    def density(self, family, p, P, parity=0, left=1, right=1):
        a1, a2, a3, a4 = map(self.a, p)
        ap, am = self.a(P), self.a(-P)
        if family == "rrrr":
            logval = (self.numerator(P, "NS")
                      - self.log_rr_denominator(ap, a3, a4, left)
                      - self.log_rr_denominator(am, a2, a1, right))
            factor = 1
        elif family == "mixed_ns":
            logval = (self.numerator(P, "NS")
                      - self.log_ns_denominator((a4, a3, ap), parity)
                      - self.log_rr_denominator(am, a2, a1, right))
            factor = 2 if parity else 1
        elif family == "mixed_r":
            logval = (self.numerator(P, "R")
                      - self.log_rr_denominator(a4, a3, ap, left)
                      - self.log_rr_denominator(a1, a2, am, right))
            factor = 1
        else:
            raise ValueError(family)
        value = factor*np.exp(logval)
        if not np.isfinite(value):
            raise ArithmeticError("nonfinite literature spectral density")
        return complex(value)


class AnalyticSewing:
    """One fixed-P pair of holomorphic and coefficient-conjugate blocks."""
    def __init__(self, family, momenta, P, maximum_twice_level, *, options=None, constants=None,
                 backend="extrapolation"):
        if backend not in ("extrapolation", "cauchy"):
            raise ValueError("backend must be extrapolation or cauchy")
        builder = ExtrapolatedLiteratureBlocks if backend == "extrapolation" else SelfDualLiteratureBlocks
        self.blocks = builder(family, momenta, P, maximum_twice_level, options=options)
        pp = tuple(complex(p).conjugate() for p in momenta)
        # On the real slice the two coefficient tables are exactly the same.
        self.dual = (self.blocks if pp == self.blocks.p and complex(P).imag == 0 else
                     builder(family, pp, complex(P).conjugate(), maximum_twice_level, options=options))
        self.constants = constants or SelfDualConstants()
        self.P = P

    def value(self, z, external):
        return sewn_integrand(self.blocks, self.constants, self.P, z, external,
                              antiholomorphic_value=analytic_antiholomorphic(self.dual))

    def metadata(self):
        return dict(holomorphic=self.blocks.metadata(), analytic_dual=self.dual.metadata(),
                    physical_momenta_conjugated=False, absolute_amplitude_normalization=False)
