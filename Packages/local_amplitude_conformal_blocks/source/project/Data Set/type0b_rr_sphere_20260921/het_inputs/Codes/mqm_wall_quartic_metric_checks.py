#!/usr/bin/env python3
"""Quartic wall kernels: a kinematic exclusion and a possible metric term.

This tests a continuum interaction, not its derivation from a matrix sea.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as s


def main():
    a, b, c, t = s.symbols('a b c t', positive=True)
    w = [a+b+c, a, b, c]
    q, r, d = b+c, 2*a+b+c, b-c
    A = s.cos(w[0]*t)*s.cos(w[1]*t)
    B = s.cos(w[2]*t)*s.cos(w[3]*t)
    curvature = q*q*A*B-s.diff(A,t)*s.diff(B,t)
    coefficients = [q*q/4, (q*q-q*d)/8, (q*q+q*d)/8,
                    (q*q-r*q)/8, (q*q+r*q)/8,
                    (q*q-r*d)/8, (q*q+r*d)/8]
    frequencies = [2*q, q-d, q+d, r-q, r+q, r-d, r+d]
    expansion = sum(x*s.cos(y*t) for x,y in zip(coefficients,frequencies))
    assert s.simplify(s.expand(s.expand_power_exp((curvature-expansion).rewrite(s.exp)))) == 0
    assert all(s.simplify(x).is_positive for x in frequencies)
    eps = s.symbols('eps', positive=True)
    regulated = sum(x*eps/(eps*eps+y*y) for x,y in zip(coefficients,frequencies))
    assert s.limit(regulated, eps, 0, dir='+') == 0

    # Coefficient of epsilon0 epsilon1 epsilon2 epsilon3 for distinct
    # flavors (0,1) and (2,3), in L=1/2 h (sum V_t V_tau)^2.
    c0,c1,c2,c3=[s.cos(x*t) for x in w]
    u0,u1,u2,u3=[s.sin(x*t) for x in w]
    pair01 = s.I*(c0*u1-c1*u0)
    pair23 = -s.I*(c2*u3+c3*u2)
    assert s.simplify(s.expand(s.expand_power_exp((pair01*pair23+s.sin(q*t)**2).rewrite(s.exp)))) == 0

    mp.mp.dps=45
    values=[]
    for qval in ['0.2','0.7','1.3','2.4']:
        z=mp.mpf(qval)
        integral=mp.quad(lambda x:(mp.sin(z*x)/mp.sinh(x))**2,[0,1,5,mp.inf])
        expected=(mp.pi*z*mp.coth(mp.pi*z)-1)/2
        assert abs(integral-expected)<mp.mpf('1e-38')
        values.append({'q':qval,'convergent_overlap':mp.nstr(integral,30)})
    z=s.symbols('z')
    exchange=s.pi*z*s.coth(s.pi*z)+1/(1+s.I*z)
    metric_contact=-(s.pi*z*s.coth(s.pi*z)-1)
    boundary_contact=-s.Integer(1)
    assert s.simplify(exchange+metric_contact+boundary_contact-1/(1+s.I*z))==0
    result={
        'intrinsic_curvature_1to3_contact':'zero for positive energies under Abel regulation',
        'curvature_cosine_frequencies':[str(s.expand(x)) for x in frequencies],
        'mixed_derivative_channel':'-sin(q*tau)**2 for L=h*(V_t dot V_tau)**2/2',
        'convergent_wall_overlap':values,
        'metric_plus_boundary_fit':'I(q)-(pi*q*coth(pi*q)-1)-1=1/(1+i*q)',
        'scope':'Kinematic identities only. The added metric/boundary couplings have not been derived from MQM; other channels, state and normalization remain untested.'}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_quartic_metric_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
