"""Overall correlator prefactors fixed by overlap and modularity.

This layer acts AFTER the original primary/descendant sewing is complete.
Primary norms, inverse Grams, vertices, and block coefficients are not
changed. The remaining common string-amplitude constant is separate.
"""
from dataclasses import dataclass

import numpy as np

from local_density import LocalRROPE
from radial_trace import RadialTrace


@dataclass(frozen=True)
class CorrelatorNormalization:
    """Partition/correlation-function normalization, not a state metric.

    The factors compare the existing radial and OPE trace conventions.
    They must not be interpreted as a multiplicity per internal R edge.
    """

    @property
    def radial_factor(self):
        return 2.

    def ope_factor(self, delta):
        if delta not in (1, 2, 3, 4):
            raise ValueError('theta characteristic must be 1,2,3,4')
        return 2. if delta < 3 else 1.

    def metadata(self):
        return dict(id='rr-correlator-prefactor-v1',
            layer='overall partition/correlation function, after descendant summation',
            convention='OPE NS-handle correlator sets the relative reference',
            radial_factor=self.radial_factor, ope_theta_factors=[self.ope_factor(d) for d in (1, 2, 3, 4)],
            determined_by='channel overlap and modular covariance',
            geometry_or_energy_dependent=False, descendant_dependent=False,
            primary_norms_changed=False, inverse_Grams_changed=False,
            trinion_couplings_changed=False, descendant_coefficients_changed=False,
            factor_per_internal_edge=False,
            GSO_weights_changed=False, external_vertices_rescaled=False,
            common_amplitude_constant=None,
            common_constant_prescription='factorization onto independently normalized lower-genus string amplitudes',
            matrix_amplitude_used=False)


CORRELATOR = CorrelatorNormalization()


class CorrelatorLocalRROPE(LocalRROPE):
    """Multiply completed OPE correlators, leaving their block engine intact.

    Inherited components/evaluate/collision_polynomials consume the
    already multiplied correlators or their Taylor coefficients. They do
    not apply another prefactor. Ground-handle limits use the same layer.
    """
    def liouville(self, tau, z, delta, **kwargs):
        return CORRELATOR.ope_factor(delta)*super().liouville(tau, z, delta, **kwargs)

    def component_polynomials(self, tau, delta, **kwargs):
        # These are Taylor coefficients of the fully contracted correlator,
        # not the normalized chiral descendant coefficients in holo/anti.
        raw = super().component_polynomials(tau, delta, **kwargs)
        factor = CORRELATOR.ope_factor(delta)
        return [{key:factor*value for key,value in family.items()} for family in raw]


class CorrelatorRadialTrace(RadialTrace):
    """Only the assembled correlator is multiplied; state sums are inherited."""
    def __init__(self, *args, **kwargs):
        if kwargs.get('c', 13.5) != 13.5:
            raise ValueError('the correlator convention is fixed for the interacting c=27/2 theory')
        super().__init__(*args, **kwargs)

    def bare(self, tau, z, delta, **kwargs):
        return CORRELATOR.radial_factor*super().bare(tau, z, delta, **kwargs)

    def raised(self, tau, z, delta, **kwargs):
        return CORRELATOR.radial_factor*super().raised(tau, z, delta, **kwargs)


def normalized_spin_array(raw, *, theta_axis):
    """Convert a documented historical OPE array once, without mutating it."""
    raw = np.asarray(raw)
    if raw.shape[theta_axis] != 4:
        raise ValueError('the selected axis must have theta order 1,2,3,4')
    shape = [1]*raw.ndim
    shape[theta_axis] = 4
    return raw*np.array([CORRELATOR.ope_factor(d) for d in (1, 2, 3, 4)]).reshape(shape)
