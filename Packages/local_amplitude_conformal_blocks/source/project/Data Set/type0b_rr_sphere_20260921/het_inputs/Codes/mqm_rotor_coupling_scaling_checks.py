#!/usr/bin/env python3
"""Exact finite-regulator coupling rescaling; no genus identification."""
import json
from pathlib import Path
import sympy as s
import mpmath as mp


def symbolic_checks():
    g,lam,b,rho,dy,epsbar=s.symbols('g lambda b rhohat dy epsbar',positive=True)
    a=1/(g*g*lam*lam);mu=1/(g*g*lam);beta=b/g
    rhoi=g*s.sqrt(lam)*rho;gap=dy/(g*s.sqrt(lam))
    K=s.sqrt(a)/(2*mu);nu=1/(2*mu)
    assert s.simplify(mu/a-lam)==0
    assert s.simplify(s.sqrt(a)/mu-g)==0
    assert s.simplify(K-g/2)==0
    checks={
        'dx_metric':((1/(g*s.sqrt(lam)))**2,1/lam),
        'F_metric':(1/(g*g*a*rhoi*rhoi),lam/rho**2),
        'rotor_metric':(beta**2/(a*rhoi*rhoi),lam*b*b/rho**2),
        'nearest_gap_potential':(a*a/(6*gap*gap),1/(6*lam**3*dy**2)),
        'rotor_bond_potential':(a*beta*beta/(2*gap*gap),b*b/(2*lam*dy**2)),
        'inverse_bulk_metric':(nu*beta**4/2,lam*b**4/4),
        'boundary_velocity_quartic':(K*K*beta**4/8,b**4/32),
        # F,eta,Gvector,V each scale 1/g in the selective cross tensor.
        'selective_cross_metric':(K*K/g**4,s.Rational(1,4)),
        'selective_boundary_metric':(5*K*K*beta**2/(6*g*g),5*b*b/24),
    }
    for name,(actual,barred) in checks.items():
        assert s.simplify(g*g*actual-barred)==0,name
    ee=epsbar/(g*g);c=K*K*beta**4/8;d=c/(ee*ee)
    assert s.simplify(d-g*g*b**4/(32*epsbar**2))==0
    # Pullback metric scaling makes G0^-1 B independent of g, even if
    # the two tensors do not commute. Check a non-diagonal example.
    G0=s.Matrix([[2,1],[1,3]]);B=s.Matrix([[1,-2],[-2,0]])
    assert (G0/g**2).inv()*(B/g**2)==G0.inv()*B
    return {'coefficient_checks':{k:str(v[1]) for k,v in checks.items()},
            'boundary_d':'g**2*b**4/(32*epsbar**2)',
            'noncommuting_metric_endomorphism':'pass','status':'pass'}


def h(d,argument):
    if not argument:return mp.mpf(0)
    if not d:return argument
    p=mp.sqrt(2*argument)
    r=mp.sinh(mp.asinh(3*mp.sqrt(3*d)*p)/3)/mp.sqrt(3*d)
    assert abs((r+4*d*r**3)-p)<mp.mpf('1e-65')*(1+p)
    return r*r/2+3*d*r**4


def main():
    out={'symbolic':symbolic_checks(),'spectral_function_checks':[],
         'counting_function_checks':[]}
    mp.mp.dps=75
    for gv,ds,ss in [('.2','.7','1.3'),('.7','1.2','.09'),('2.4','.11','5.2')]:
        g,dhat,argument=map(mp.mpf,[gv,ds,ss])
        left=g*g*h(g*g*dhat,g*g*argument)
        right=h(dhat,g**4*argument)
        error=abs(left-right)
        assert error<mp.mpf('1e-65')*(1+abs(right))
        out['spectral_function_checks'].append({'g':gv,'dhat':ds,'s':ss,'error':mp.nstr(error,5)})
    for gv,lv,yv in [('.2','3','2.1'),('1.3','7','3.7')]:
        g,lam,y=map(mp.mpf,[gv,lv,yv]);a=1/(g*g*lam*lam);mu=1/(g*g*lam)
        xmin=mp.sqrt(2*mu);x=y/(g*mp.sqrt(lam))
        # Integrate in the physical x coordinate using a nonnegative
        # parametrization of the distance above the turning point.
        width=x-xmin
        integral=mp.quad(lambda t:mp.sqrt(width*t*(2*xmin+width*t))*width/a,[0,1])
        expected=lam*(y*mp.sqrt(y*y-2)/2-mp.acosh(y/mp.sqrt(2)))
        assert abs(integral-expected)<mp.mpf('1e-65')
        out['counting_function_checks'].append({'g':gv,'lambda':lv,'y':yv,'error':mp.nstr(abs(integral-expected),5)})
    # A connected graph with m external legs and loop number L has
    # sum_v(r_v-2)=m+2I-2V=m-2+2L.
    m,I,V=s.symbols('m I V',integer=True)
    assert s.expand((m+2*I-2*V)-(m-2+2*(I-V+1)))==0
    out['formal_connected_graph_count']='g**(m-2+2*L), fixed barred data and normalized fluctuations'
    out['status']='all checks pass; no statement about string genus or a continuum state'
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_rotor_coupling_scaling_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
