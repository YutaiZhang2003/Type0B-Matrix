#!/usr/bin/env python3
"""A boundary-kinetic-split stress test, not a full MQM limit theorem."""
import json
from pathlib import Path
import mpmath as mp
import sympy as s


def symbolic_check():
    v,m,c,alpha=s.symbols('v m c alpha',positive=True)
    k=4*c*(1-alpha)/(alpha*m)
    r=v-k*v**3+3*k*k*v**5
    lag=m*(v-alpha*r)**2/(2*(1-alpha))+m*alpha*r*r/2+c*r**4
    series=s.series(lag,v,0,8).removeO()
    expected=m*v*v/2+c*v**4-8*c*c*(1-alpha)*v**6/(alpha*m)
    assert s.simplify(series-expected)==0
    r=s.symbols('r',real=True)
    positive_form=m*(v-alpha*r)**2/(2*(1-alpha))+m*alpha*r*r/2+c*r**4
    stationary=s.diff(positive_form,r)
    assert s.simplify(stationary.subs(v,r+k*r**3))==0
    compact=m*v*v/(2*(1-alpha))-m*alpha*r*r/(2*(1-alpha))-3*c*r**4
    assert s.simplify((positive_form-compact).subs(v,r+k*r**3))==0
    return {'small_velocity_series':str(expected),
            'stationary_root':'v=r+4*c*(1-alpha)*r**3/(alpha*m)',
            'infimal_convolution':'pass','status':'pass'}


def spectral_h(d,sval):
    p=mp.sqrt(2*sval)
    r=mp.sinh(mp.asinh(3*mp.sqrt(3*d)*p)/3)/mp.sqrt(3*d)
    return r*r/2+3*d*r**4


def split_L(m,c,alpha,v):
    k=4*c*(1-alpha)/(alpha*m)
    r=2/mp.sqrt(3*k)*mp.sinh(mp.asinh(3*mp.sqrt(3*k)*v/2)/3)
    value=m*(v-alpha*r)**2/(2*(1-alpha))+m*alpha*r*r/2+c*r**4
    p=m*(v-alpha*r)/(1-alpha)
    H=(1-alpha)*p*p/(2*m)+spectral_h(c/(alpha*alpha*m*m),alpha*p*p/(2*m))
    assert abs(p*v-H-value)<mp.mpf('1e-65')*(1+abs(value))
    return value,r


def tau_of_count(count,lam):
    guess=(3*count/(2*lam))**(mp.mpf(1)/3)
    return mp.findroot(lambda t:mp.sinh(2*t)/2-t-count/lam,guess)


def main():
    mp.mp.dps=80
    out={'symbolic':symbolic_check(),'fixed_velocity':[],
         'full_quartic_Legendre':[],'wall_inertia':[]}
    alpha=mp.mpf('.3');c=mp.mpf('.7');v=mp.mpf('.8')
    expected_ratio=v*v/(2*(1-alpha))
    for mass in ['1e-3','1e-6','1e-12','1e-24']:
        m=mp.mpf(mass);value,r=split_L(m,c,alpha,v)
        correction=-3*c*(alpha*m*v/(4*c*(1-alpha)))**(mp.mpf(4)/3)
        assert abs((value-m*expected_ratio)/correction-1)<20*m**(mp.mpf(1)/3)
        out['fixed_velocity'].append({'m':mass,'L_over_m':mp.nstr(value/m,30),
            'limiting_ratio':mp.nstr(expected_ratio,30),'L':mp.nstr(value,30)})
    p=mp.mpf('.9')
    limiting_H=3*abs(p)**(mp.mpf(4)/3)/(4**(mp.mpf(4)/3)*c**(mp.mpf(1)/3))
    for mass in ['1e-3','1e-6','1e-12']:
        m=mp.mpf(mass)
        full=spectral_h(c/(m*m),p*p/(2*m))
        velocity=mp.findroot(lambda vv:m*vv+4*c*vv**3-p,(p/(4*c))**(mp.mpf(1)/3))
        assert abs(full-(m*velocity**2/2+3*c*velocity**4))<mp.mpf('1e-65')
        out['full_quartic_Legendre'].append({'m':mass,'H':mp.nstr(full,30),'massless_H':mp.nstr(limiting_H,30)})
    # First dynamical site i=1, c*=1: its centered density uses labels1,3.
    gv=mp.mpf('.7');b=mp.mpf('2');cm=mp.mpf(1);cp=mp.mpf(3)
    predicted=(b*b/(gv*gv))*(mp.mpf(3)/2)**(mp.mpf(4)/3)*(cp**(mp.mpf(2)/3)-cm**(mp.mpf(2)/3))**2/8
    for lvalue in ['1e3','1e6','1e9']:
        lam=mp.mpf(lvalue)
        ym=mp.sqrt(2)*mp.cosh(tau_of_count(cm,lam))
        yp=mp.sqrt(2)*mp.cosh(tau_of_count(cp,lam))
        rhohat=2/(yp-ym)
        inertia=b*b*lam/(gv*gv*rhohat*rhohat)
        scaled=inertia*lam**(mp.mpf(1)/3)
        assert abs(scaled/predicted-1)<10*lam**(-mp.mpf(2)/3)
        out['wall_inertia'].append({'lambda':lvalue,'m1':mp.nstr(inertia,30),
            'm1_times_lambda_one_third':mp.nstr(scaled,30),'limit':mp.nstr(predicted,30)})
    out['status']='all checks pass; one-coordinate and aligned-reference diagnostics only'
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_boundary_inertia_limit_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
