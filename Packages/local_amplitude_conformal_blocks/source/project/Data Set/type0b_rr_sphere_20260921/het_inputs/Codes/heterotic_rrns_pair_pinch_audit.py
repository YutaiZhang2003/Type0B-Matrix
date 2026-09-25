#!/usr/bin/env python3
"""Read-only audit of the actual RRNS endpoint pinch, September 12, 2026.

Only the dedicated JSON result is written. Existing calculations and the
manuscript are not modified.
The full-amplitude analytic-continuation convention remains a qualification.
"""
import cmath
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Codes'))
from spin23_super_liouville_data import (
    _upsilon_r_mp, _n_r_mp, rr_ns_structure_constants,
)
from so7e8_four_ramond_nonchiral import nonchiral_heterotic_sewing_weight
from so7e8_four_ramond_assembly import (
    normalized_time_ising_scalar_block, spin7_spectator_block_series,
)
from so7e8_four_ramond_elliptic_recursion import (
    general_four_ramond_sld_elliptic_h_series,
)


def pair(value):
    c = complex(value)
    return [c.real, c.imag]


mp.mp.dps = 60
x = mp.mpc('.13', '.20')
y = mp.mpc('.19', '.37')
z = 1j-y
E = x+1j
den = (E+x)*(y-z)
ratio_r = _n_r_mp(y)*_n_r_mp(z)/(
    _upsilon_r_mp(1+1j*(y-z))*_upsilon_r_mp(1-1j*(y-z)))
ratio_l = _n_r_mp(E)*_n_r_mp(x)/_upsilon_r_mp(1+1j*(E+x))**2
ratio_errors = [abs(ratio_r/((y-z)**2/4)-1),
                abs(ratio_l/((E+x)**2/4)-1)]
assert max(ratio_errors) < mp.mpf('1e-50')
limit = -den**2/4
pinch_samples = []
for eps in [1e-2, 1e-3, 1e-4]:
    zp = 1j*(1-eps)-complex(y)
    Ep = complex(x)+complex(y)+zp
    left = rr_ns_structure_constants(Ep, complex(x), eps, precision=60)
    right = rr_ns_structure_constants(zp, complex(y), -eps, precision=60)
    value = 4*eps**2*left[0]*right[1]
    regular = left[1]*right[0]
    pinch_samples.append(dict(eps=eps, singular_scaled=pair(value),
                              relative_error=float(abs(value/complex(limit)-1)),
                              regular_product=pair(regular)))
assert pinch_samples[-1]['relative_error'] < .001

# Derive the four-row factorization from the actual current sewing code.
families = ('Psi_tilde',)*3+('Psi',)
rows = []
expected = {(0,1,0,0):-.5, (0,1,1,1):-.25,
            (1,0,0,0):1, (1,0,1,1):.5}
for qh in range(2):
    for qt in range(2):
        for qa in range(2):
            for q7 in range(2):
                q = (qh,qt,qa,q7)
                w = nonchiral_heterotic_sewing_weight(families,*q,1,-1)
                assert abs(w-expected.get(q,0)) < 1e-13
                if w:
                    rows.append(dict(parities=q, weight=pair(w)))

# Exact algebraic verification of the Spin7 x Ising identity, using KZ.
# Set t=sqrt(1-u); d/du=-(2t)^(-1)d/dt.  For U=I0 K0 and
# V=-I1 K1/4, U'=(M+a)U and V'=(M+b)V.
t = sp.symbols('t', positive=True)
u = 1-t*t
deriv = lambda f: sp.diff(f,t)/(-2*t)
A0 = sp.diag(-sp.Rational(7,8),-sp.Rational(3,8),
             -sp.Rational(1,24),sp.Rational(1,8))
A1 = sp.Matrix([[0,0,sp.Rational(7,8),0],
                [0,-sp.Rational(1,4),0,sp.Rational(5,8)],
                [sp.Rational(1,24),0,-sp.Rational(5,12),-sp.Rational(5,12)],
                [0,sp.Rational(1,8),-sp.Rational(1,4),-sp.Rational(1,2)]])
M = A0/u + A1/(u-1)
a = -1/(8*u)+1/(8*(1-u))-1/(4*t*(1+t))
b = -1/(8*u)+1/(8*(1-u))+1/(4*t*(1-t))
W = sp.Matrix([(1/u-sp.Rational(1,2))/t,-1/(4*t),-1/(4*t),0])
U = ((W.applyfunc(deriv)-M*W-b*W)/(a-b)).applyfunc(sp.factor)
V = (W-U).applyfunc(sp.factor)
ode_u = (U.applyfunc(deriv)-M*U-a*U).applyfunc(sp.factor)
ode_v = (V.applyfunc(deriv)-M*V-b*V).applyfunc(sp.factor)
assert ode_u == sp.zeros(4,1) and ode_v == sp.zeros(4,1)
spin = {c:spin7_spectator_block_series(c,maximum_order=35)
        for c in ('vacuum','vector')}
spin_checks = []
for uv in [.01,.2,.2+.1j]:
    actual = (normalized_time_ising_scalar_block('identity',uv)*spin['vacuum'].value(uv)
              -.25*normalized_time_ising_scalar_block('fermion',uv)*spin['vector'].value(uv))
    exact = np.asarray([1/uv-.5,-.25,-.25,0],complex)/cmath.sqrt(1-uv)
    err = float(np.max(np.abs(actual-exact)))
    assert err < 1e-11
    spin_checks.append(dict(modulus=pair(uv), max_absolute_error=err))

# Verify the zero-screening SLD blocks using the actual new elliptic
# R recursion, independently of the KZ simplification above.
uv = .2+.1j
momenta = tuple(map(complex,(y,z,x,E)))
f = cmath.exp(-(1+1j*complex(y))*(1+1j*complex(z))*cmath.log(uv)
              -(1+1j*complex(z))*(1-1j*complex(x))*cmath.log(1-uv))
block_checks = []
for order in [9,13,17]:
    for component in ('even','odd'):
        B = general_four_ramond_sld_elliptic_h_series(
            0,external_momenta=momenta,external_ground_parities=(0,0,0,0),
            maximum_twice_level=order,component=component,
            left_structure_sign=1,right_structure_sign=-1,digits=90)
        exact = f*normalized_time_ising_scalar_block(
            'identity' if component=='even' else 'fermion',uv)
        if component=='odd':
            exact *= -.5
        err = abs(B.value(uv)/exact-1)
        block_checks.append(dict(order=order,component=component,relative_error=err))
assert max(v['relative_error'] for v in block_checks if v['order']==17)<1e-12

# Integrating the remaining scalar term.  After u=1-zeta, angular
# integration of 1/(1-ubar) is 2pi for |u|<1 and zero for |u|>1.
exponent = -mp.mpf('1.5')+1j*(x-z)-2*x*z
assert -1 < exponent.real < -.5  # Absolute convergence for this term.
angular_integrated = 2*mp.pi*mp.quad(
    lambda r:r**(2*exponent+1), [0,1])
target_integral = 2*mp.pi/den
assert abs(angular_integrated/target_integral-1)<mp.mpf('1e-25')
# Constant tensor terms vanish in analytic regularization: the complex
# beta integral with |zeta|^(2a) regulator tends to zero as a->0.
regulator_checks = []
for eps in [1e-3,1e-4,1e-5]:
    bv = (mp.pi*mp.gamma(1+eps)*mp.gamma(1+exponent)*mp.gamma(-1-eps-exponent)
          /(mp.gamma(-eps)*mp.gamma(-exponent)*mp.gamma(2+eps+exponent)))
    regulator_checks.append(dict(eps=eps, integral=pair(bv)))

natural_regulator_checks = []
for d in [1e-3,1e-4,1e-5]:
    P = d
    a = -d+(P*P+d*d)/2
    jv = (mp.pi*mp.gamma(1+a)*mp.gamma(1+exponent)*mp.gamma(-1-a-exponent)
          /(mp.gamma(-a)*mp.gamma(-exponent)*mp.gamma(2+a+exponent)))
    # J(a,A) = pi*a/(A+1)^2 + O(a^2): a spectral 1/d produces
    # a finite term, while its putative vector residue tends to zero.
    natural_regulator_checks.append(dict(d=d,a=a,putative_residue=pair(jv),
        linear_limit_error=float(abs(jv/(mp.pi*a/(1+exponent)**2)-1))))
assert natural_regulator_checks[-1]['linear_limit_error'] < .0001

result = dict(
    sample=dict(x=pair(x),y=pair(y),z=pair(z),E=pair(E)),
    upsilon_ratio_errors=[float(v) for v in ratio_errors],
    pinch_coefficient=pair(limit), pinch_samples=pinch_samples,
    sewing_rows=rows, exact_kz_ode_checks=True,
    spin7_checks=spin_checks, elliptic_recursion_checks=block_checks,
    A=pair(exponent), scalar_modulus_integral=pair(angular_integrated),
    scalar_integral_error=float(abs(angular_integrated/target_integral-1)),
    constant_integral_regulator=regulator_checks,
    off_pole_regulator=natural_regulator_checks,
    residue_T0_current_kernel=pair(mp.pi*den/32),
    residue_vector_zero=True,
    qualification='Meromorphic continuation of the current nonchiral sewing; overall external phase/sign and contour completeness remain separate checks.')
output = ROOT/'data_exports/heterotic_long_string_trees_20260912'/'rrns_pair_pinch_results.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
