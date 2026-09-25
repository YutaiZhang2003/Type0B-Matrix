"""Analytic b=1 limit of the literature double-Virasoro/CCY blocks.

The Cauchy projection acts on the complete branch sum, not on its singular
individual Virasoro factors. Two different contours must agree coefficient
by coefficient. This check is independent of truncation convergence.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np

from literature_component_blocks import SECTORS, _bit
from literature_double_virasoro import AnalyticModule, LiteratureDoubleVirasoroBlocks
from spin23_genus1_recursion import dictionary_finite_part


@dataclass(frozen=True)
class ExtrapolationOptions:
    epsilon: float = .01

    def __post_init__(self):
        if (not math.isfinite(self.epsilon) or self.epsilon <= 0
                or 1+self.epsilon == 1):
            raise ValueError("positive resolvable epsilon required")


def duality_extrapolate(first, second, epsilon=.01):
    """Type0B endpoint estimator, linear in x=Q-2=(b-1)^2/b.

    Applied to assembled coefficients at fixed physical momenta and fixed
    external component/sign labels. Never extrapolate single branch terms.
    Formal O(epsilon^4) accuracy assumes analyticity and b-duality.
    """
    ExtrapolationOptions(epsilon)
    first, second = np.asarray(first, complex), np.asarray(second, complex)
    if first.shape != second.shape or not np.isfinite([first, second]).all():
        raise ValueError("matching finite assembled tables required")
    return (4*(1+epsilon)*first-(1+2*epsilon)*second)/(3+2*epsilon)


@dataclass(frozen=True)
class FinitePartOptions:
    radius: float = .10
    check_radius: float = .12
    samples: int = 32
    tolerance: float = 1e-8

    def __post_init__(self):
        if (not 0 < self.radius < self.check_radius or self.samples < 8
                or not isinstance(self.samples, int) or isinstance(self.samples, bool)
                or not math.isfinite(self.tolerance) or self.tolerance <= 0):
            raise ValueError("two ordered positive radii, >=8 samples and positive tolerance required")


class SelfDualLiteratureBlocks(LiteratureDoubleVirasoroBlocks):
    """Same public native sign frame, c=27/2 and signed complex momenta."""
    def __init__(self, family, momenta, P, maximum_twice_level=8, *, options=None):
        if family not in SECTORS or len(momenta) != 4:
            raise ValueError("unknown channel or non-four-point data")
        if (not isinstance(maximum_twice_level, int)
                or isinstance(maximum_twice_level, bool) or maximum_twice_level < 0):
            raise ValueError("a nonnegative integral truncation is required")
        self.b = 1.+0j
        self.family, self.p, self.order = family, tuple(map(complex, momenta)), maximum_twice_level
        self.modules = tuple(AnalyticModule(s, p, 1) for s, p in zip(SECTORS[family], self.p))
        self.internal = AnalyticModule("R" if family == "mixed_r" else "NS", P, 1)
        if any(m.sector == "R" and m.p == 0 for m in (*self.modules, self.internal)):
            raise ValueError("Ramond zero momentum requires a separate branch limit")
        self.options = options or FinitePartOptions()
        self.diagnostics = {}
        for name in ("_generic", "_native_components", "coefficients", "elliptic_transform",
                     "elliptic_coefficients", "value"):
            setattr(self, name, lru_cache(None)(getattr(self, name)))

    def bpz_coefficients(self, *args, **kwargs):
        raise TypeError("project the fully assembled native table through coefficients()")

    def _generic(self, b):
        return LiteratureDoubleVirasoroBlocks(self.family, self.p, self.internal.p,
                                             self.order, b=b)

    def _native_components(self, external, parity):
        external = tuple(map(_bit, external))
        parity = _bit(parity)
        count = 2 if self.family == "mixed_ns" else 4
        keys = tuple((j, n) for j in range(count) for n in range(self.order+1))

        def evaluate(b):
            table = self._generic(b)._native_components(external, parity)
            return {(j, n): table[j, n] for j, n in keys}

        values, diagnostics = dictionary_finite_part(
            evaluate, keys=keys, radius=self.options.radius,
            check_radius=self.options.check_radius, samples=self.options.samples)
        error = 0.
        for key, diagnostic in diagnostics.items():
            x, y = diagnostic.value, diagnostic.check_value
            scaled = abs(x-y)/max(1., abs(x), abs(y))
            if not np.isfinite([x, y]).all() or scaled > self.options.tolerance:
                raise ArithmeticError(
                    f"literature b=1 finite-part check failed: {self.family}, "
                    f"external={external}, parity={parity}, sign/level={key}, "
                    f"two-radius scaled discrepancy={scaled:.6g}")
            error = max(error, scaled)
        self.diagnostics[external, parity] = error
        return np.array([[values[j, n] for n in range(self.order+1)] for j in range(count)])

    def metadata(self):
        return dict(method="assembled two-radius Cauchy finite part in t=log(b)",
                    options=asdict(self.options),
                    maximum_scaled_discrepancy=max(self.diagnostics.values(), default=None),
                    component_tables_checked=len(self.diagnostics),
                    ordinary_virasoro_backend="CCY sphere c-recursion",
                    c=13.5, maximum_twice_level=self.order)


class ExtrapolatedLiteratureBlocks(SelfDualLiteratureBlocks):
    """Fast b=1 coefficient bank with the user's Type0B prescription.

    The exact b=1 primary/elliptic prefactors multiply the extrapolated
    descendant coefficients. The two original samples are retained in the
    generic block cache. This does not estimate its own regulator error;
    the preparation audit compares with independent b=1 reference tables.
    """
    def __init__(self, *args, options=None, **kwargs):
        super().__init__(*args, options=options or ExtrapolationOptions(), **kwargs)

    def _native_components(self, external, parity):
        external, parity = tuple(map(_bit, external)), _bit(parity)
        eps = self.options.epsilon
        first = self._generic(1+eps)._native_components(external, parity)
        second = self._generic(1+2*eps)._native_components(external, parity)
        result = duality_extrapolate(first, second, eps)
        self.diagnostics[external, parity] = float(np.max(abs(first-second)))
        return result

    def metadata(self):
        return dict(method="Type0B b-dual two-sample extrapolation of assembled coefficients",
                    formula="[4*(1+epsilon)*F(1+epsilon)-(1+2*epsilon)*F(1+2*epsilon)]/(3+2*epsilon)",
                    options=asdict(self.options), sample_b=[1+self.options.epsilon, 1+2*self.options.epsilon],
                    variable="x=Q-2=(b-1)^2/b", formal_error="O(epsilon^4), conditional on analyticity and b-duality",
                    regulator_error_estimate=None, primary_prefactors="exact b=1",
                    component_tables_checked=len(self.diagnostics),
                    ordinary_virasoro_backend="CCY sphere c-recursion",
                    c=13.5, maximum_twice_level=self.order)


def analytic_antiholomorphic(dual_blocks):
    """Return Fbar(p,zbar)=conj(F(conj(p),z)), including fixed frame phases.

    The caller must build dual_blocks at conjugate parameters. This is
    coefficientwise analytic continuation, not abs(F(p,z))**2.
    """
    def value(z, external, parity, left, right):
        return np.conj(dual_blocks.value(z, external, parity, left, right))
    return value
