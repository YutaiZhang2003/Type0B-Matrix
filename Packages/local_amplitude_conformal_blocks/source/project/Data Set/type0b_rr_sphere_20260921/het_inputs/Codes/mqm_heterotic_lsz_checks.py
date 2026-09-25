#!/usr/bin/env python3
"""Exact free-kinetic and external-leg checks for the heterotic NS states."""
import json
from pathlib import Path
import sys
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Codes'))
from so7e8_ns_threepoint import v_to_vs_unit_descendants
from spin23_three_point_amplitudes import s_to_vv_unit_descendants


def main():
    P,E,Z=s.symbols('P E Z',positive=True,real=True)
    n2=1+P*P
    hsl=n2/2
    right=hsl-E*E/2+s.Rational(1,2)-1
    left_vector=hsl-E*E/2+s.Rational(1,2)-1
    left_singlet=hsl+s.Rational(1,2)-E*E/2-1
    assert s.simplify(right+left_vector-(P*P-E*E)) == 0
    assert s.simplify(right+left_singlet-(P*P-E*E)) == 0
    assert s.simplify(n2/s.sqrt(n2)**2) == 1
    kinetic=P*P-E*E
    positive_residue_factor=-s.diff(kinetic,E).subs(E,P)
    assert positive_residue_factor == 2*P
    mode_coefficient=s.sqrt(s.pi/(2*Z*P))
    assert s.simplify(2*P*mode_coefficient**2-s.pi/Z) == 0
    full_wave_coefficient=1/s.sqrt(2*s.pi*Z*P)
    assert s.simplify(full_wave_coefficient**2*2*s.pi*Z*P) == 1

    a,b,x=s.symbols('a b x',positive=True,real=True)
    q=a+b
    w0=x+q
    W=w0*x*a*b
    cft_product=(w0*x*q/s.sqrt(1+q*q))*(q*a*b/s.sqrt(1+q*q))
    canonical_product=s.simplify(cft_product/s.sqrt(w0*x*q*q*a*b))
    assert s.simplify(canonical_product-s.sqrt(W)*q/(1+q*q)) == 0

    points=[]
    for aa,bb,xx in [(0.23,0.41,0.31),(0.6,1.1,0.7)]:
        qq=aa+bb; ww0=xx+qq; WW=ww0*xx*aa*bb
        first=v_to_vs_unit_descendants(xx,qq)/(ww0*xx*qq)**0.5
        second=s_to_vv_unit_descendants(aa,bb)/(qq*aa*bb)**0.5
        target=WW**0.5*qq/(1+qq*qq)
        assert abs(first*second-target)<1.e-13
        points.append({'q':qq,'canonical_cubic_product':str(first*second),
                       'quartic_imaginary_energy_magnitude':target})

    eps,H=s.symbols('epsilon H',positive=True,real=True)
    soft_ratio=eps*s.sqrt(H/((H+eps)*eps))
    assert s.limit(soft_ratio/s.sqrt(eps),eps,0,dir='+') == 1
    result={'kinetic_both_species':str(kinetic),
            'singlet_norm_after_given_n_division':'pi*delta(P-Pprime)',
            'positive_frequency_kinetic_residue':str(positive_residue_factor),
            'unit_delta_external_wave_factor':str(full_wave_coefficient),
            'energy_dependence_per_external_leg':'omega**(-1/2)',
            'canonical_cubic_cut':str(canonical_product),
            'numerical_crossed_checks':points,
            'soft_F_epsilon_to_canonical_sqrt_epsilon':'pass',
            'previous_magnitude_obstruction':'superseded by derived normalization'}
    output=ROOT/'data_exports/mqm/mqm_heterotic_lsz_results.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
