"""Endpoint/bulk/infinite-tail quadrature adapted to P**beta exp(-a P**2+s P).

The weight is a positive numerical envelope, not a change to dP/pi or an
assertion about a physical asymptotic. Returned weights integrate the ORIGINAL
integrand against dP; the amplitude builders supply their usual factor 1/pi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import operator
from typing import Mapping

import numpy as np
from scipy.special import roots_jacobi, roots_laguerre, roots_legendre


@dataclass(frozen=True)
class ThresholdConfig:
    endpoint: float = 0.18
    tail: float = 3.2
    beta: float = 2.0
    a: float = 1.0
    s: float = 0.0
    bulk_breakpoints: tuple[float, ...] = (0.36, 0.56, 0.85, 1.30, 2.10)

    def __post_init__(self):
        for name in ("endpoint", "tail", "beta", "a", "s"):
            value = getattr(self, name)
            if isinstance(value, (bool, complex)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and real")
        if not 0 < self.endpoint < self.tail:
            raise ValueError("thresholds must satisfy 0 < endpoint < tail")
        if self.beta <= -1:
            raise ValueError("endpoint beta must be > -1")
        if self.a <= 0:
            raise ValueError("the Gaussian envelope needs a > 0")
        if 2 * self.a * self.tail - self.s <= 0:
            raise ValueError("tail must lie beyond the exponential saddle: 2*a*tail > s")
        points = tuple(float(x) for x in self.bulk_breakpoints)
        if any(not math.isfinite(x) or x <= 0 for x in points):
            raise ValueError("bulk breakpoints must be finite and positive")
        if any(y <= x for x, y in zip(points, points[1:])):
            raise ValueError("bulk breakpoints must be strictly increasing")
        object.__setattr__(self, "bulk_breakpoints", points)

    @classmethod
    def from_options(cls, options=None):
        if options is None:
            return cls()
        if isinstance(options, cls):
            return options
        if not isinstance(options, Mapping):
            raise ValueError("momentum_threshold_options must be a mapping or ThresholdConfig")
        unknown = set(options) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown threshold options: {sorted(unknown)}")
        return cls(**options)

    def log_weight(self, p):
        p = np.asarray(p, dtype=float)
        if np.any(p <= 0):
            raise ValueError("evaluate the envelope only at positive P")
        return self.beta * np.log(p) - self.a * p**2 + self.s * p


def _counts(p_nodes):
    """Integer means TOTAL nodes; a triple specifies endpoint, bulk, tail."""
    if isinstance(p_nodes, str):
        if p_nodes != "segmented":
            raise ValueError("threshold p_nodes string must be 'segmented'")
        return (12, 72, 24)
    if not isinstance(p_nodes, (bool, np.bool_)):
        try:
            total = operator.index(p_nodes)
        except TypeError:
            total = None
        if total is not None:
            if total < 3:
                raise ValueError("threshold quadrature requires at least three total nodes")
            endpoint, tail = max(1, total // 6), max(1, total // 4)
            return endpoint, total - endpoint - tail, tail
    try:
        values = tuple(p_nodes)
        if len(values) != 3 or any(isinstance(n, (bool, np.bool_)) for n in values):
            raise ValueError
        values = tuple(operator.index(n) for n in values)
        if any(n < 1 for n in values):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("threshold p_nodes must be a total integer or (endpoint, bulk, tail) counts") from None
    return values


@dataclass(frozen=True)
class ThresholdRule:
    momenta: np.ndarray
    weights: np.ndarray
    regions: np.ndarray
    config: ThresholdConfig
    counts: tuple[int, int, int]
    bulk_boundaries: tuple[float, ...]

    def metadata(self):
        return {
            "scheme": "threshold_weighted",
            "measure": "dP; amplitude builder applies 1/pi",
            "weight": "P**beta * exp(-a*P**2+s*P)",
            "parameters": asdict(self.config),
            "region_counts": dict(zip(("endpoint", "bulk", "tail"), self.counts)),
            "bulk_boundaries": list(self.bulk_boundaries),
            "node_count": len(self.momenta),
            "largest_sampled_momentum": float(self.momenta[-1]),
            "tail_domain": "[tail, infinity); largest node is NOT a cutoff",
            "accuracy_status": "requires_node_threshold_and_envelope_refinement",
        }

    def integrate(self, values, *, factored=False):
        """Values have momentum as axis 0; factored=True means values=f/w."""
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(self.momenta):
            raise ValueError("values must have momentum as their first axis")
        weights = self.weights
        if factored:
            weights = np.exp(np.log(weights) + self.config.log_weight(self.momenta))
        return np.tensordot(weights, values, axes=(0, 0))


def threshold_weighted_rule(p_nodes="segmented", options=None):
    """Build positive dP weights; never sample P=0 or truncate the tail.

    Endpoint: Gauss-Jacobi for P**beta, with the remaining envelope in the
    smooth residual. Bulk: panelled Gauss-Legendre. Tail: Gauss-Laguerre after
    t=a*(P**2-T**2)-s*(P-T), so dP/dt=1/(2*a*P-s).
    The tail P**beta factor stays in its smooth residual. This is NOT a claim
    to construct Gaussian polynomials for the full shifted weight globally.
    """
    config = ThresholdConfig.from_options(options)
    ne, nb, nt = _counts(p_nodes)
    e, t, beta, a, s = config.endpoint, config.tail, config.beta, config.a, config.s

    x, w = roots_jacobi(ne, 0.0, beta)
    pe = e * (x + 1) / 2
    we = np.exp((beta + 1) * np.log(e / 2) + np.log(w) - beta * np.log(pe))

    interior = [x for x in config.bulk_breakpoints if e < x < t]
    # Tiny smoke-test budgets merge panels, rather than silently adding nodes.
    if len(interior) >= nb:
        indices = np.linspace(0, len(interior)-1, nb-1, dtype=int)
        interior = [interior[i] for i in indices]
    bounds = (e, *interior, t)
    panel_count = len(bounds) - 1
    counts = [nb // panel_count + (j < nb % panel_count) for j in range(panel_count)]
    momenta, weights = [pe], [we]
    for lo, hi, count in zip(bounds[:-1], bounds[1:], counts):
        x, w = roots_legendre(count)
        momenta.append((lo + hi) / 2 + (hi - lo) * x / 2)
        weights.append((hi - lo) * w / 2)

    x, w = roots_laguerre(nt)
    if np.any(w <= 0) or not np.all(np.isfinite(x)):
        raise ArithmeticError("Laguerre weights underflowed; reduce tail order or change thresholds")
    slope = 2*a*t - s
    jacobian_denominator = np.sqrt(slope*slope + 4*a*x)
    pt = t + 2*x/(jacobian_denominator + slope)
    wt = np.exp(np.log(w) + x - np.log(jacobian_denominator))
    momenta.append(pt)
    weights.append(wt)
    momenta, weights = np.concatenate(momenta), np.concatenate(weights)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0) or np.any(np.diff(momenta) <= 0):
        raise ArithmeticError("invalid threshold nodes or weights")
    return ThresholdRule(momenta, weights, np.repeat([0, 1, 2], [ne, nb, nt]),
                         config, (ne, nb, nt), bounds)
