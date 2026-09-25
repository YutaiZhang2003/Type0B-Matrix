#!/usr/bin/env python3
"""Reviewable positive-NS-metric wrappers; existing kernels stay untouched.

Only the dedicated audit JSON is written when this file is run as a script.
Importing it exposes the two wrappers without running the audit.
"""
import cmath
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Codes'))
from so7e8_four_ramond_nonchiral import build_nonchiral_fixed_p_four_ramond_kernel
from so7e8_four_ramond_assembly import general_four_ramond_sld_chiral_block
from spin23_super_liouville_data import rr_ns_structure_constants, rr_ns_chiral_structure_constant
from so7e8_ramond_threepoint import (
    hjs_minus_singlet_ward_factor, hjs_plus_singlet_ward_factor,
    strip_hjs_singlet_phase, strip_hjs_plus_singlet_phase,
)
from ns_algebra.ns_sca import G, descendant_inner_product


def pair(v):
    return [complex(v).real,complex(v).imag]


def positive_endpoint_kernel(internal_momentum, **kwargs):
    """Use +P at both RRNS endpoints, preserving every other kernel datum.

    This independent re-evaluation of the right structure constant is
    intended for audit.  It is not a production implementation.
    """
    kernel = build_nonchiral_fixed_p_four_ramond_kernel(internal_momentum,**kwargs)
    m = tuple(kwargs['external_liouville_momenta'])
    digits = kwargs.get('digits',40)
    ratios = {}
    for sr in (-1,1):
        old = rr_ns_chiral_structure_constant(m[1],m[0],-complex(internal_momentum),
            structure_sign=sr,precision=digits)
        new = rr_ns_chiral_structure_constant(m[1],m[0],complex(internal_momentum),
            structure_sign=sr,precision=digits)
        if old == 0:
            raise ValueError('This audit wrapper requires a generic nonzero structure constant.')
        ratios[sr] = new/old
    return replace(kernel,terms=tuple(replace(t,coefficient=t.coefficient*
        ratios[t.right_structure_sign]) for t in kernel.terms))


def reflected_metric_kernel(internal_momentum, **kwargs):
    """Keep P,-P endpoints and include inverse reflection metric -1."""
    kernel = build_nonchiral_fixed_p_four_ramond_kernel(internal_momentum,**kwargs)
    return replace(kernel,terms=tuple(replace(t,coefficient=-t.coefficient)
                                     for t in kernel.terms))


def run_checks():
    P = sp.symbols('P',real=True)
    norm = descendant_inner_product((G(-sp.Rational(1,2)),),
        (G(-sp.Rational(1,2)),),h=(1+P**2)/2,c=sp.Rational(27,2))
    assert sp.simplify(norm-(1+P**2))==0
    checks=[]
    for x,y,z,Pv in [(.2,.3,.4,.7),(.13+.2j,.19+.37j,-.07+.18j,.42),
                     (.21+.08j,.16+.12j,.09+.13j,.31+.06j)]:
        E=x+y+z; L=(E+x)*(y-z); N=1+Pv*Pv
        le,lo=rr_ns_structure_constants(E,x,Pv,precision=60)
        re,ro=rr_ns_structure_constants(z,y,Pv,precision=60)
        rem,rom=rr_ns_structure_constants(z,y,-Pv,precision=60)
        reflection_errors=[abs(rem/re+1),abs(rom/ro+1)]
        assert max(reflection_errors)<1e-13
        # Both compact cubic orientations, using the existing phase helpers.
        wL=hjs_minus_singlet_ward_factor(E,x)
        wR=hjs_plus_singlet_ward_factor(y,z)
        cL=strip_hjs_singlet_phase(.5*wL*lo)
        cR=strip_hjs_plus_singlet_phase(.5*wR*re)
        B=general_four_ramond_sld_chiral_block(Pv,external_momenta=(y,z,x,E),
            external_ground_parities=(0,0,0,0),maximum_twice_level=1,
            component='odd',left_structure_sign=-1,right_structure_sign=1,
            chirality='antiholomorphic',digits=60)
        odd=B.coefficients[1]
        assert abs(odd/(L/(2*N))-1)<1e-12
        geometry_phase=odd*N/(wL*wR)
        assert abs(geometry_phase+1j)<1e-12
        # Correctly reflected coefficient includes a minus; equal to +P,+P.
        old_scalar=.25*lo*rem*odd
        scalar_sameP=.25*lo*re*odd
        expected_scalar=cL*cR/N
        scalar_error=abs(scalar_sameP/expected_scalar-1)
        assert scalar_error<1e-12
        assert abs(old_scalar/expected_scalar+1)<1e-12
        vector_sameP=lo*re/8
        cubic_vector=(lo/(2*math.sqrt(2)))*(re/(2*math.sqrt(2)))
        assert abs(vector_sameP/cubic_vector-1)<1e-12
        kwargs=dict(external_liouville_momenta=(y,z,x,E),
            time_momenta=(y,z,x,-E),families=('Psi_tilde',)*3+('Psi',),
            maximum_twice_level=13,spin7_maximum_order=25,
            sld_block_backend='elliptic_recursion',digits=90)
        old=build_nonchiral_fixed_p_four_ramond_kernel(Pv,**kwargs)
        same=positive_endpoint_kernel(Pv,**kwargs)
        reflected=reflected_metric_kernel(Pv,**kwargs)
        term_error=max(abs(a.coefficient-b.coefficient) for a,b in zip(same.terms,reflected.terms))
        assert term_error<1e-13
        tensor_checks=[]
        for u in [.2+.1j,.37+.07j,.45-.12j]:
            av,bv,cv=old.evaluate(u),same.evaluate(u),reflected.evaluate(u)
            error=float(np.max(np.abs(bv-cv)))
            sign_error=float(np.max(np.abs(bv+av)))
            assert max(error,sign_error)<1e-12
            tensor_checks.append(dict(modulus=pair(u),old=[pair(v) for v in av],
                same_positive_P=[pair(v) for v in bv],endpoint_metric_error=error,
                common_sign_error=sign_error))
        checks.append(dict(energies=[pair(v) for v in (x,y,z,E)],P=pair(Pv),
            reflection_errors=reflection_errors,scalar_cubic_error=scalar_error,
            effective_HJS_orientation=pair(geometry_phase),
            old_scalar_to_cubic=pair(old_scalar/expected_scalar),
            term_comparison_error=term_error,tensor_checks=tensor_checks))

    gamma,kappa,ZF2,L = sp.symbols('gamma kappa ZF2 L',nonzero=True)
    C3=-sp.I*sp.sqrt(2)*gamma
    C4=sp.I*gamma**2/sp.pi
    norm_R_sq=1/(2*kappa)
    raw_residue=-sp.pi*kappa**2*L
    mapped=sp.simplify(C4*norm_R_sq**2*raw_residue)
    assert sp.simplify(mapped+sp.I*gamma**2*L/4)==0
    result=dict(primary_metric='pi delta(P-Pprime), P>=0',
        descendant_metric=str(norm),reflected_metric='-(1+P^2) times pi delta',
        original_kernel_global_sign=-1,corrected_GS_residue=str(mapped),
        C3=str(C3),C4=str(C4),C3_squared_over_C4=str(sp.simplify(C3**2/C4)),
        samples=checks,
        limitations=['Does not settle the separate microscopic spectator cocycle.',
                     'Does not independently derive external Ramond kinetic magnitude.',
                     'Does not establish the full nonresonant four-point amplitude.'])
    (ROOT/'data_exports/heterotic_long_string_trees_20260912'/'four_ramond_reflected_metric_results.json').write_text(
        json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run_checks()
