#!/usr/bin/env python3
"""SSVV tree test of the explicitly prescribed, inverse-designed wall fluid.

This tests continuum finite parts, not a finite-matrix origin or a duality.
The external n(omega) and oscillator factors are stripped uniformly.
"""
import itertools
import json
from pathlib import Path
import mpmath as mp
import sympy as s


def vertex_terms(types, kappas, energies):
    terms = []
    for signs in itertools.product((-1, 1), repeat=3):
        spatial = [s.I*signs[i]*kappas[i] for i in range(3)]
        temporal = [s.I*energies[i] for i in range(3)]
        eta = [signs[i]/(2*s.I) if types[i] == 'S' else 0 for i in range(3)]
        spin = [kappas[i]/2 if types[i] == 'S' else s.Rational(1,2) for i in range(3)]
        flavor = ['W' if t == 'S' else t for t in types]
        coefficient = s.prod(spatial[i]*eta[i] for i in range(3))
        for i in range(3):
            j,k = [r for r in range(3) if r != i]
            coefficient += spatial[i]*eta[i]*temporal[j]*eta[j]*temporal[k]*eta[k]
            if flavor[j] == flavor[k]:
                coefficient += eta[i]*spin[j]*spin[k]*(
                    temporal[i]*(temporal[j]*spatial[k]+temporal[k]*spatial[j])
                    +spatial[i]*(temporal[j]*temporal[k]+spatial[j]*spatial[k]))
        terms.append((signs, s.factor(-coefficient)))
    return terms


def f2(k):
    if abs(k) < mp.mpf('1e-30'):
        return -1
    return -mp.pi*k/2*mp.coth(mp.pi*k/2)


def f4(k):
    if abs(k) < mp.mpf('1e-30'):
        return mp.mpf(2)/3
    return mp.pi*k*(k*k+4)/12*mp.coth(mp.pi*k/2)


def make_vertex(types, external_k, energies):
    p = s.symbols('p', real=True)
    kappas = [s.sympify(external_k[0]), s.sympify(external_k[1]), p]
    energies = list(map(s.sympify, energies))
    terms = vertex_terms(types, kappas, energies)
    prepared=[]
    asymptotic=0
    for signs, coeff in terms:
        # F2 is even; combine the signs so its argument is p+shift.
        shift = signs[2]*(signs[0]*kappas[0]+signs[1]*kappas[1])
        prepared.append((mp.mpf(str(shift)), s.lambdify(p, coeff, 'mpmath')))
        asymptotic += coeff*(-s.pi/2)*(p+shift)
    asymptotic=s.factor(asymptotic)
    def evaluate(value):
        return mp.fsum(c(value)*f2(value+shift) for shift,c in prepared)
    return evaluate, asymptotic


def channel(left, right, energy, constrained):
    p=s.symbols('p', real=True)
    lv,lp=left; rv,rp=right
    e=mp.mpf(str(energy)); es=s.sympify(energy)
    def numerator(z):
        return lv(z)*rv(z)/(1+z*z if constrained else 1)
    pole=numerator(e)
    # asymptotic rational function before subtraction of the on-shell pole.
    raw=s.factor(lp*rp/((1+p*p) if constrained else 1)/(es*es-p*p))
    polynomial,remainder=s.div(s.together(raw).as_numer_denom()[0],s.together(raw).as_numer_denom()[1],p)
    poly=s.lambdify(p,polynomial,'mpmath')
    # Symbolically combine the tail before quadrature; separately evaluating
    # large terms at infinity would destroy the cancellation of the UV part.
    pole_sym=s.Float(str(pole),mp.mp.dps)
    tail_expr=s.factor(raw-polynomial-pole_sym/(es*es-p*p))
    tail=s.lambdify(p,tail_expr,'mpmath')
    def smooth(z):
        if abs(z-e)<mp.mpf('1e-18'):
            return -mp.diff(numerator,e)/(2*e)-poly(e)
        return (numerator(z)-pole)/(e*e-z*z)-poly(z)
    cutoff=mp.mpf(60)
    breaks=sorted(set([mp.mpf(0),e/2,e,e+1,e+8,cutoff]))
    real=-(2/mp.pi)*(mp.quad(smooth,breaks)+mp.quad(tail,[cutoff,mp.inf]))
    imaginary=pole/e
    return real+1j*imaginary, {'energy':str(energy),'left_UV':str(lp),
        'right_UV':str(rp),'integrand_UV_subtraction':str(polynomial),
        'value':mp.nstr(real+1j*imaginary,35),'pole_numerator':mp.nstr(pole,35)}


def ssvv(energies):
    w1,w2,w3=map(s.Rational,energies); w0=w1+w2+w3
    q=w2+w3; product=w0*w1*w2*w3
    scalar_left=make_vertex(['S','S','S'],[w0,w1],[-w0,w1,q])
    scalar_right=make_vertex(['a','a','S'],[w2,w3],[w2,w3,-q])
    exs, report=channel(scalar_left,scalar_right,q,True)
    exchange=exs
    reports=[{'kind':'singlet',**report}]
    for j,k in [(w2,w3),(w3,w2)]:
        e=w0-j
        vl=make_vertex(['S','a','a'],[w0,j],[-w0,j,e])
        vr=make_vertex(['S','a','a'],[w1,k],[w1,k,-e])
        value,report=channel(vl,vr,e,False)
        exchange+=value
        reports.append({'kind':'vector',**report})
    W=mp.mpf(str(product)); qq=mp.mpf(str(q)); r=mp.mpf(str(w0+w1)); d=mp.mpf(str(w2-w3))
    parent=W*(-mp.mpf('1.5')*f4(0)+mp.mpf('.5')*f4(2*qq)-mp.mpf('.5')*f4(r+d)-mp.mpf('.5')*f4(r-d))
    deformation=-W*mp.mpf(str(w0*w1))*mp.pi*qq*mp.coth(mp.pi*qq)
    # The same vector phase i that matches the cubic multiplies SSVV by -1.
    result=-(exchange+parent+deformation)/W
    target=1+2j*mp.mpf(str(w0))-mp.mpf(str(w0*w1))/(1+1j*qq)
    # Compare this direct three-channel integration with the separately
    # derived closed mixed kernel, including its real-energy cut.
    J=qq/2+qq/(1+1j*qq)+(1+qq*qq)/(8j)*(mp.polygamma(1,(1+1j*qq)/2)-mp.polygamma(1,(1-1j*qq)/2))
    expected_difference=mp.mpf(2)/3-(qq*qq+4)*mp.pi*qq*mp.coth(mp.pi*qq)/3-qq*mp.re(J)-1/(1+qq*qq)
    assert abs(result-target-expected_difference)<mp.mpf('1e-30')
    return {'outgoing':energies,'exchange_channels':reports,
        'parent_contact_over_W':mp.nstr(parent/W,35),
        'added_contact_over_W':mp.nstr(deformation/W,35),
        'result_after_vector_phase':mp.nstr(result,35),
        'target':mp.nstr(target,35),'difference':mp.nstr(result-target,35),
        'closed_difference':mp.nstr(expected_difference,35)}


def elementary_checks():
    # Calibrate the signed action vertices against the independently derived
    # pure-vector source and all on-shell cubic flavor kernels.
    x,y=s.Rational(7,10),s.Rational(4,5); q=x+y
    v,_=make_vertex(['a','a','S'],[x,y],[x,y,-q])
    def j(p):
        if abs(p-mp.mpf(str(q)))<mp.mpf('1e-30'):return -mp.mpf(str(q))
        return (mp.mpf(str(q))**2-p*p)*mp.pi/2*mp.sinh(mp.pi*p)/(mp.cosh(mp.pi*p)-mp.cosh(mp.pi*mp.mpf(str(q))))
    for pp in ['.3','1.5','2.1']:
        p=mp.mpf(pp)
        assert abs(v(p)-mp.mpf(str(x*y))*j(p))<mp.mpf('1e-35')
    v,_=make_vertex(['S','S','S'],[x+y,x],[-x-y,x,y])
    expected=(x+y)*x*y*(1+x*x+y*y+x*y)
    assert abs(v(mp.mpf(str(y)))-mp.mpf(str(expected)))<mp.mpf('1e-35')
    return 'pass'


def main():
    mp.mp.dps=45
    results={'elementary_checks':elementary_checks(),'prescription':'individual channel Hadamard UV finite parts, common sine/cosine FP vertex transforms', 'samples':[]}
    for energies in [['0.4','0.6','0.7'],['0.3','0.4','0.8']]:
        result=ssvv(energies)
        results['samples'].append(result)
        print(json.dumps(result,indent=2),flush=True)
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_mixed_quartic_results.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
