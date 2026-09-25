"""Remove the known collision power before truncating recursive VV blocks.

For either loop sector the external fields fuse into an NS continuum with
h >= 1/2. A chiral pair of weight h_E+w/2 therefore has threshold power
(1-q)**(1/2-2*h_E-w). We expand the remaining function, preserving EVERY
input Taylor coefficient. The remaining continuum logarithms are not
approximated by an extra pole or fitted to a modular/MQM answer.

This is a numerical series reorganization, not a complete collision chart.
It does not improve the momentum rule or establish convergence near q=1.
The source bank is never mutated or saved under a different interpretation.
"""
import copy
import numpy as np

from spin23_genus1_vv_bank import WORDS


def binomial_coefficients(exponent, order):
    """Taylor coefficients of (1-q)**exponent, also at integer exponents."""
    out = np.ones(order+1, complex)
    for n in range(1, order+1):
        out[n] = out[n-1]*(n-1-exponent)/n
    return out


def factor_coefficients(coefficients, levels, exponent):
    """Multiply both edges by (1-q)**exponent in a triangular level table."""
    levels = np.asarray(levels)
    b = binomial_coefficients(exponent, int(levels.max())//2)
    delta = levels[:, None, :]-levels[None, :, :]
    valid = np.all(delta >= 0, axis=-1) & np.all(delta % 2 == 0, axis=-1)
    indices = np.maximum(delta//2, 0)
    matrix = valid*b[indices[..., 0]]*b[indices[..., 1]]
    return np.asarray(coefficients) @ matrix.T


class FactoredVVBank:
    """Evaluation-only view; all expensive recursion remains in preparation."""
    def __init__(self, source):
        source.validate()
        self.metadata = dict(source.metadata, evaluation='NS-threshold-factored-v1')
        self.momenta, self.weights = source.momenta, source.weights
        self._bank = copy.copy(source)
        omega = complex(*source.metadata['energy'])
        self.exponents = np.array([.5+omega**2+word[0] for word in WORDS])
        for name, levels in (('ns', source.ns_levels), ('r', source.r_levels)):
            raw = getattr(source, name)
            value = np.empty_like(raw)
            for wi, exponent in enumerate(self.exponents):
                value[:, wi] = factor_coefficients(raw[:, wi], levels, exponent)
            setattr(self._bank, name, value)

    def evaluate(self, plumbing, lifts, cutoffs=(6, 8), *, momentum_chunk=128):
        result = self._bank.evaluate(plumbing, lifts, cutoffs,
                                     momentum_chunk=momentum_chunk)
        logs = np.log1p(-np.asarray(plumbing, complex)).sum(axis=-1)
        # External energies are analytically continued without conjugating
        # their weights in the antiholomorphic block.
        prefactor = np.exp(-logs[:, None]*self.exponents[None, :]
                          -logs.conj()[:, None]*self.exponents[1])
        return result*prefactor[:, None, None, :]

    def short_factors(self, z, cutoff):
        """The SAME truncated leading cusp, for an exact control subtraction."""
        from spin23_genus1_vv_tail import _short_factors
        q=np.exp(2j*np.pi*z)
        # The removed long-edge factors start at twice-level two, so they
        # do not alter the leading long-edge zero/half-level coefficients.
        prefactor=np.exp(-np.log1p(-q)*self.exponents
                         -np.log1p(-q.conjugate())*self.exponents[1])
        return _short_factors(self._bank,z,cutoff)*prefactor
