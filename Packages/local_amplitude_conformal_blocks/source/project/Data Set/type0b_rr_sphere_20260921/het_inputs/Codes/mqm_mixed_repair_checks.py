#!/usr/bin/env python3
"""Selective mixed-contact repair and independent all-S contact checks."""
from __future__ import annotations
import json
from pathlib import Path
import mpmath as mp
import sympy as s

from mqm_wall_mixed_quartic_audit import coefficient
from mqm_wall_mixed_quartic_checks import make_vertex


def profile_checks():
    t,z,q=s.symbols("t z q",positive=True)
    D=lambda f:s.diff(f,t)-2*z*s.diff(f,z)
    g=2*t*z/(1-z*z)
    h=8*z/(1-z)**2-16*z*z/(1-z)**4+2*D(g)-D(D(D(g)))/2
    compact=16*z**3*(12*t*(1+z*z)+z**4-8*z-9)/(1-z*z)**4
    assert s.factor(h-compact)==0
    assert s.series(h,z,0,5).removeO()==(192*t-144)*z**3-128*z**4
    actual=compact.subs(z,s.exp(-2*t))
    small=s.series(actual,t,0,2).removeO()
    assert s.simplify(small-(-t**-4+s.Rational(8,3)*t**-2
                            -s.Rational(41,45)-s.Rational(16,5)*t))==0
    # The double-pole coefficient from the leading tau exp(-6 tau)
    # tail of -int h sin²(q tau).
    eps=s.symbols("eps")
    growing_piece=48/(6+2*s.I*(3*s.I+eps))**2
    assert s.simplify(growing_piece*eps**2)==-12
    return compact, {"profile":str(compact),
                     "tail":"(192 tau-144)e^-6tau -128e^-8tau + ...",
                     "double_pole_at_q_3i":"-12/(q-3i)^2",
                     "turning_point_series":str(small)}


def direct_finite_parts(compact):
    mp.mp.dps=50
    t,z=s.symbols("t z",positive=True)
    hfun=s.lambdify((t,z),compact,"mpmath")
    rows=[]
    for value in [".2",".7","1.2","1.3","2.1"]:
        q=mp.mpf(value)
        Tg=mp.quad(lambda x:x*mp.sin(2*q*x)/mp.sinh(2*x),
                   [0,1,4,mp.inf])
        H=mp.pi*q*mp.coth(mp.pi*q)
        expected=-mp.mpf(1)/3-q*q/2-(4+q*q)*H/3+2*q*(1+q*q)*Tg
        trigammaJ=q/2+q/(1+1j*q)+(1+q*q)/(8j)*(
            mp.polygamma(1,(1+1j*q)/2)-mp.polygamma(1,(1-1j*q)/2))
        old=mp.mpf(2)/3-(4+q*q)*H/3-q*trigammaJ-1/(1+1j*q)-1j*q
        assert abs(expected-old)<mp.mpf("1e-42")
        eps=mp.mpf("0.0001")
        integral=mp.quad(lambda x:hfun(x,mp.exp(-2*x))*mp.sin(q*x)**2,
                         [eps,mp.mpf(".1"),1,4,12,mp.inf])
        # h sin²(qt)=−q²/t²+c0+c2t²+c3t³+O(t⁴).
        c0=(q**4+8*q*q)/3
        c2=-2*q**6/45-8*q**4/9-41*q*q/45
        c3=-16*q*q/5
        finite=integral+q*q/eps+c0*eps+c2*eps**3/3+c3*eps**4/4
        repair=-finite-mp.mpf(5)/3
        assert abs(repair-expected)<mp.mpf("1e-17")
        rows.append({"q":value,"direct_repair":mp.nstr(repair,30),
                     "Delta":mp.nstr(expected,30),
                     "absolute_error":mp.nstr(abs(repair-expected),5)})
    return rows


def selective_metric():
    u,v,du,dv,eps=s.symbols("u v du dv eps",real=True)
    L=s.cosh(eps**2*u*v)*eps**2*(du*du+dv*dv)/2\
        +s.sinh(eps**2*u*v)*eps**2*du*dv
    expansion=s.series(L,eps,0,7).removeO()
    assert expansion.coeff(eps,4)==u*v*du*dv
    assert s.simplify(expansion.coeff(eps,6)-u*u*v*v*(du*du+dv*dv)/4)==0
    assert (u*v*du*dv).subs({u:0,du:0})==0
    assert (u*v*du*dv).subs({v:0,dv:0})==0
    return {"positive_metric_eigenvalues":"exp(+u v), exp(-u v)",
            "quartic":"u v du dv",
            "first_extra_diagonal_interaction_order":6,
            "pure_scalar_and_pure_vector_quartics":"zero"}


def all_s_contact_crosscheck():
    e=s.symbols("e0:4")
    a,b,c,t=s.symbols("a b c t",positive=True)
    w=[a+b+c,a,b,c]
    eta=sum(e[i]*s.sin(w[i]*t) for i in range(4))
    et=sum(e[i]*s.I*(-1 if i==0 else 1)*w[i]*s.sin(w[i]*t)
           for i in range(4))
    ex=s.diff(eta,t); wt=s.diff(et,t); wx=s.diff(ex,t)
    W=s.prod(w); B=1+sum(v*v for v in w)/2; qs=[w[0]-v for v in w[1:]]
    den=coefficient(ex**2*et**2/2,e)/W
    spin=coefficient(ex**2*wt**2/2+2*ex*et*wt*wx+et**2*wx**2/2,e)/W
    want_den=s.Rational(3,2)-sum(s.cos(2*q*t) for q in qs)/2
    want_spin=3*(B-1)/2+sum((q*q-(B-1)/2)*s.cos(2*q*t) for q in qs)
    for expr,want in [(den,want_den),(spin,want_spin)]:
        assert s.simplify(s.expand((expr-want).rewrite(s.exp)))==0
    bulk=coefficient((wt*wx)**2,e)
    want_bulk=-2*W**2*sum(s.sin(q*t)**2 for q in qs)
    assert s.simplify(s.expand((bulk-want_bulk).rewrite(s.exp)))==0
    boundary=coefficient(wt**4/8,e).subs(t,0)
    assert s.simplify(boundary+3*W**2)==0
    C,D=s.Rational(3,5),s.Rational(7,10);q=C+D
    vertex,_=make_vertex(["S","S","S"],[C,D],[C,D,-q])
    for pp in [".2",".8","1.7","2.3"]:
        p=mp.mpf(pp);qq=mp.mpf(str(q))
        denominator=mp.cosh(mp.pi*p)-mp.cosh(mp.pi*qq)
        j=(qq*qq-p*p)*mp.pi/2*mp.sinh(mp.pi*p)/denominator
        k=(qq*qq-p*p)*mp.pi/2*mp.sinh(mp.pi*qq)/denominator
        want=-mp.mpf(str(C*D))*((1-mp.mpf(str(C*D)))*j+qq*p*k)
        assert abs(vertex(p)-want)<mp.mpf("1e-40")
    return {"density_contact":"3/2 - (1/2) sum cos(2q tau)",
            "spin_contact":"3(B0-1)/2 + sum [q²-(B0-1)/2]cos(2q tau)",
            "all_W_bulk":"-2 Wprod² sum sin²(q tau)",
            "all_W_boundary":"-3 Wprod²",
            "two_outgoing_source":"-CD[(1-CD)j+qp k]: pass",
            "selective_mixed_repair":"does not change this channel"}


def main():
    compact,profile=profile_checks()
    result={"scope":"Inverse classical quartic repair; no quantum endpoint or full duality claim.",
            "profile":profile,
            "direct_contact_checks":direct_finite_parts(compact),
            "positive_metric_completion":selective_metric(),
            "independent_all_S_audit":all_s_contact_crosscheck()}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_mixed_repair_results.json').write_text(
        json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
