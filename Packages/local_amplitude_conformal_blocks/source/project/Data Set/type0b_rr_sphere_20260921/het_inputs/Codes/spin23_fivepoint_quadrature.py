"""Whole-half-line Gaussian quadrature with an explicit TOTAL node budget.

This is the generalized-Laguerre transformation used by the c=1 reference,
not a sum of endpoint/bulk/tail node budgets. The positive envelope
P**beta exp(-a P**2) is removed from the returned dP weights. It changes
convergence, never the integrand or measure; callers still supply 1/pi.
Fixed envelope scales permit coefficient-bank reuse across moduli. They are
not asserted to be the exact heterotic propagator or its asymptotic decay.
"""
from dataclasses import asdict, dataclass
import math
import operator

import numpy as np
from scipy.special import roots_genlaguerre


@dataclass(frozen=True)
class GaussianEnvelope:
    a: float = 1.0
    beta: float = 2.0
    diagnostic_endpoint: float = .18
    diagnostic_tail: float = 3.2

    def __post_init__(self):
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('finite real Gaussian envelope parameters required')
        if self.a <= 0 or self.beta <= -1:
            raise ValueError('Gaussian envelope requires a>0 and beta>-1')
        if not 0 < self.diagnostic_endpoint < self.diagnostic_tail:
            raise ValueError('ordered positive diagnostic region boundaries required')

    def log_weight(self, p):
        p = np.asarray(p, dtype=float)
        if np.any(p <= 0):
            raise ValueError('positive momentum required')
        return self.beta*np.log(p)-self.a*p**2


@dataclass(frozen=True)
class GaussianMomentumRule:
    momenta: np.ndarray
    weights: np.ndarray
    regions: np.ndarray
    config: GaussianEnvelope

    def metadata(self):
        return dict(scheme='threshold_gaussian', node_count=len(self.momenta),
            node_budget='TOTAL over [0,infinity), not per region',
            measure='dP; amplitude builder applies 1/pi',
            weight='P**beta * exp(-a*P**2)', parameters=asdict(self.config),
            laguerre_alpha=(self.config.beta-1)/2,
            momentum_domain='[0,infinity)', p_max_used=False,
            largest_sampled_momentum=float(self.momenta[-1]),
            diagnostic_region_counts=np.bincount(self.regions, minlength=3).tolist(),
            region_labels_are_not_separate_quadrature_panels=True,
            accuracy_status='baseline only; requires order and envelope refinements')

    def integrate(self, values, *, factored=False):
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(self.momenta):
            raise ValueError('one value per momentum node required')
        weights = self.weights
        if factored:
            weights = np.exp(np.log(weights)+self.config.log_weight(self.momenta))
        return np.tensordot(weights, values, axes=(0, 0))


def gaussian_weighted_rule(node_count, options=None):
    """N Laguerre roots, u=a*P^2, alpha=(beta-1)/2, on [0,infinity).

Returned weights are w*exp(u)*u**(-beta/2)/(2*sqrt(a)), so the
complete original integrand is evaluated without multiplying it by an
extra Gaussian or threshold factor. No endpoint/tail nodes are appended.
"""
    if isinstance(node_count, (bool, np.bool_)):
        raise ValueError('positive integer TOTAL node count required')
    try:
        node_count = operator.index(node_count)
    except TypeError as exc:
        raise ValueError('positive integer TOTAL node count required') from exc
    if node_count < 1:
        raise ValueError('positive integer TOTAL node count required')
    config = options if isinstance(options, GaussianEnvelope) else GaussianEnvelope(**(options or {}))
    u, w = roots_genlaguerre(node_count, (config.beta-1)/2)
    p = np.sqrt(u/config.a)
    weights = np.exp(np.log(w)+u-(config.beta/2)*np.log(u)-math.log(2)-.5*math.log(config.a))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0) or np.any(np.diff(p) <= 0):
        raise ArithmeticError('invalid whole-half-line Gaussian nodes or weights')
    regions = np.where(p < config.diagnostic_endpoint, 0,
        np.where(p < config.diagnostic_tail, 1, 2))
    return GaussianMomentumRule(p, weights, regions, config)


def gaussian_spectral_rules(counts=(8, 8), envelopes=None):
    if len(counts) != 2:
        raise ValueError('two TOTAL momentum node budgets required')
    if envelopes is None:
        envelopes = ({'a': 1., 'beta': 2.}, {'a': 1.017, 'beta': 2.})
    if len(envelopes) != 2:
        raise ValueError('two Gaussian envelopes required')
    rules = tuple(gaussian_weighted_rule(n, e) for n, e in zip(counts, envelopes))
    gap = np.abs(rules[0].momenta[:, None]-rules[1].momenta[None, :])
    if np.any(gap < 1e-12*np.maximum(1, rules[0].momenta[:, None])):
        raise ValueError('quadrature grids contain coincident recursion poles; no nodes may be dropped')
    return rules
