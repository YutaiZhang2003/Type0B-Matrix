"""Corrected finite-level SVV integrand with an explicitly ordered trace.

This is a new entry point; the historical Codes assembler is left intact.
The singlet is vertex zero, followed by the two vectors in necklace order.
This evaluates fixed punctures, not the integrated genus-one amplitude.
"""
import cmath
from dataclasses import replace
import math

from spin23_genus1_amplitude import (
    antiholomorphic_liouville_words, evaluate_genus_one_integrand,
    liouville_annulus_to_additive_factor, ordered_necklace_coordinates,even_pco_components,
)
from spin23_genus1_spectral import (SpectralIntegralDiagnostics, gauss_laguerre_spectral_integral,
                                  gauss_laguerre_spectral_integral_batch)
from spin23_genus1_threepoint_integral import states_for_energy, _KERNEL_KEYS
from spin23_genus1_ns_point_sewing import paired_momentum_integrand as ns_pair
from spin23_genus1_ramond_point_sewing import paired_momentum_integrand as r_pair
from spin23_genus1_svv_operator_order import grouping_sign
from audit_spin23_genus1_threepoint_modularity import trace_frame_factor


def ns_edge_lifts(coordinates, temporal_lift_sign):
    """Lift half-level powers to the additive chart; put parity at closure."""
    if temporal_lift_sign not in (-1, 1):
        raise ValueError('temporal lift must be +1 or -1')
    z=coordinates.additive_points
    increments=tuple(z[i+1]-z[i] for i in range(len(z)-1))+(coordinates.tau+z[0]-z[-1],)
    windings=tuple(round((2*math.pi*d.real-cmath.phase(q))/(2*math.pi))
                   for d,q in zip(increments,coordinates.plumbing_parameters))
    signs=[(-1)**w for w in windings]
    signs[-1]*=temporal_lift_sign
    return tuple(signs)


def evaluate_svv_integrand(p,q,*,tau,points,maximum_twice_levels=6,
        quadrature_order=11,coarse_order=None,workers=1,precision=30,
        spin_labels=('NS','NS_tilde','R'),internal_momenta=None,spectral_weight=1.,
        block_backend='recursion',maximum_total_twice_level=None):
    """Return the finite-level density, including GSO and the string phase.

    With internal_momenta=None integrate all three internal momenta on the
    half line using Gaussian Laguerre quadrature. Alternatively evaluate a
    single Gaussian-stripped importance node, multiplying spectral_weight.
    Its caller must supply the Gaussian and sampling Jacobian in that weight.
    A coarse_order estimates quadrature error only, never block truncation.
    The default block backend is recursion in both sectors. direct_fast is
    retained as an explicit low-level comparison. A total-level cutoff is
    optional and independent of the per-edge cutoffs.
    """
    states=states_for_energy(p,q)
    coordinates=ordered_necklace_coordinates(tau,points)
    if block_backend=='recursion':
        from spin23_genus1_recursive_sewing import ns_pair as ns_evaluator,r_pair as r_evaluator
    elif block_backend=='direct_fast':
        if maximum_total_twice_level is not None:
            raise ValueError('The finite-matrix audit backend uses a rectangular cutoff')
        ns_evaluator,r_evaluator=ns_pair,r_pair
    else:raise ValueError('block_backend must be recursion or direct_fast')
    component_words=tuple(c.holomorphic_liouville_words for c in even_pco_components(states))
    spectral_batches={}

    def callback(**options):
        kw={key:options[key] for key in _KERNEL_KEYS}
        kw['strip_internal_gaussian']=True
        if block_backend=='recursion':kw['maximum_total_twice_level']=maximum_total_twice_level
        if options['sector']=='NS':
            kw['edge_lift_signs']=ns_edge_lifts(coordinates,options['temporal_lift_sign'])
            pair=ns_evaluator
        else:
            pair=r_evaluator
        if internal_momenta is None:
            if block_backend=='recursion':
                key=(options['sector'],options['temporal_lift_sign'])
                if key not in spectral_batches:
                    def all_components(internal):
                        return tuple(pair(internal,**dict(kw,holomorphic_words=words)) for words in component_words)
                    spectral_batches[key]=gauss_laguerre_spectral_integral_batch(all_components,
                        output_size=len(component_words),
                        gaussian_scales=tuple(-math.log(abs(x)) for x in kw['plumbing_parameters']),
                        quadrature_order=quadrature_order,coarse_order=coarse_order,workers=workers)
                return spectral_batches[key][component_words.index(kw['holomorphic_words'])]
            return gauss_laguerre_spectral_integral(lambda internal:pair(internal,**kw),
                gaussian_scales=tuple(-math.log(abs(x)) for x in kw['plumbing_parameters']),
                quadrature_order=quadrature_order,coarse_order=coarse_order,workers=workers)
        return SpectralIntegralDiagnostics(value=pair(internal_momenta,**kw)*spectral_weight,
            refined_value=None,extended_value=None,quadrature_absolute_error=None,
            tail_absolute_error=None,estimated_absolute_error=None,dimension=3,
            p_max=None,quadrature_order=0,refined_order=None,extended_p_max=None,
            function_evaluations=1,quadrature_method='corrected_importance_node')

    raw=evaluate_genus_one_integrand(states,tau,points,
        maximum_twice_levels=maximum_twice_levels,p_max=None,
        quadrature_order=quadrature_order,structure_precision=precision,
        block_digits=precision,free_field_precision=precision,condition_limit=1e11,
        block_backend=block_backend,include_string_phase=True,
        spin_labels=spin_labels,liouville_evaluator=callback)
    external=tuple(state.liouville_momentum for state in states)
    anti=antiholomorphic_liouville_words(states)
    fixed={}
    for spin,evaluation in raw.fixed_spin.items():
        components=[];errors=[]
        for item in evaluation.components:
            holo=item.component.holomorphic_liouville_words
            factor=grouping_sign(item.component.time_fermion_indices)*trace_frame_factor(
                external,holo,anti)/liouville_annulus_to_additive_factor(
                    coordinates,external,holo,anti)
            corrected=replace(item,value=item.value*factor,multiplier=item.multiplier*factor)
            components.append(corrected)
            if item.liouville.estimated_absolute_error is not None:
                errors.append(abs(corrected.multiplier)*item.liouville.estimated_absolute_error)
        fixed[spin]=replace(evaluation,components=tuple(components),
            value=sum(item.value for item in components),
            estimated_absolute_error=sum(errors) if errors else None)
    errors=[.5*x.estimated_absolute_error for x in fixed.values()
            if x.estimated_absolute_error is not None]
    return replace(raw,fixed_spin=fixed,value=-.5j*sum(x.value for x in fixed.values()),
                   estimated_absolute_error=sum(errors) if errors else None)
