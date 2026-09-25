#!/usr/bin/env python3
"""A one-screening SL residue, checked independently by convergent integrals."""
import json
from pathlib import Path
import mpmath as mp
import sympy as s


def gamma_ratio(z):
    return mp.gamma(z)/mp.gamma(1-z)


def main():
    mp.mp.dps=50
    cases=[]
    for b,q in ((mp.mpf(4)/5,mp.mpf(41)/50),
                (mp.mpf(9)/10,mp.mpf(9)/10)):
        # q=alpha*b. Both endpoints and infinity converge here.
        assert b*b<q<1
        a=1-b*b
        def integrand(v):
            z=mp.exp(-v)
            return (mp.exp(-(1-q)*v)+mp.exp(-(q-b*b)*v))*mp.hyp2f1(a,a,1,z)
        direct=mp.pi*mp.quad(integrand,[0,1,10,100,mp.inf])
        closed=mp.pi*gamma_ratio(1-q)*gamma_ratio(b*b)*gamma_ratio(q-b*b)
        relative=abs(direct/closed-1)
        assert relative<mp.mpf('1e-40')
        cases.append({'b':str(b),'alpha_b':str(q),
                      'radial_integral':mp.nstr(direct,32),
                      'gamma_expression':mp.nstr(closed,32),
                      'relative_error':mp.nstr(relative,5)})

    continued=[]
    for alpha in (mp.mpc(1,mp.mpf(2)/3),mp.mpc(mp.mpf(4)/5,mp.mpf(7)/9)):
        target=-2*mp.pi/(alpha-1)**2
        errors=[]
        for exponent in (4,6,8):
            b=1+mp.mpf(10)**(-exponent)
            value=(mp.pi*gamma_ratio(1-alpha*b)*gamma_ratio(b*b)
                   *gamma_ratio(alpha*b-b*b)/gamma_ratio((1+b*b)/2))
            errors.append(abs(value/target-1))
        assert errors[2]<errors[1]<errors[0]
        assert errors[-1]<mp.mpf('1e-6')
        continued.append({'alpha':str(alpha),'limit':str(target),
                          'relative_errors':list(map(str,errors))})
    p,k,eps=s.symbols('p k eps', nonzero=True)
    assert s.cancel((k*k/p**2).subs(p,k)-1)==0
    assert s.cancel((k*k/p**2).subs(p,-k)-1)==0
    result={'all_checks_passed':True,'convergent_integrals':cases,
            'continued_limits':continued,
            'on_shell_k2_over_p2':1,
            'scope':'Screening residue after a stated coupling renormalization; not a full heterotic BRST Ward coefficient.'}
    path=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_screened_adjoint_kernel_results.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
