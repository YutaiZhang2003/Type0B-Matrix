#!/usr/bin/env python3
"""Exact local all-singlet inverse repair with at most two time derivatives.

This checks continuum vertices and Hadamard transforms, not a microscopic
prediction or a quantum continuum limit.
"""
import itertools
import json
from pathlib import Path
import mpmath as mp
import sympy as s
from mqm_wall_mixed_kernel_checks import delta,thermal
from mqm_wall_all_singlet_quartic_checks import theta


def exact_vertices():
    b,c,d=s.symbols('b c d',positive=True)
    w=[b+c+d,b,c,d];e=[-w[0],b,c,d];W=s.prod(w)
    B0=1+sum(x*x for x in w)/2;qs=[w[0]-w[i] for i in range(1,4)]
    permutations=list(itertools.permutations(range(4)))
    def coefficient(f,g,h,k):
        return sum(f[i]*g[j]*h[l]*k[m] for i,j,l,m in permutations)
    OD={};OH={}
    for signs in itertools.product((-1,1),repeat=4):
        eta_t=[signs[i]*e[i]/2 for i in range(4)]
        chi=[x/2 for x in w]
        chi_t=[s.I*e[i]*w[i]/2 for i in range(4)]
        chi_x=[s.I*signs[i]*w[i]*w[i]/2 for i in range(4)]
        od=coefficient(chi,chi,eta_t,eta_t)/2
        oh=coefficient(eta_t,chi,chi_t,chi_x)+coefficient(chi,chi,eta_t,eta_t)
        freq=s.expand(sum(signs[i]*w[i] for i in range(4)))
        if freq!=0 and next(freq.coeff(x) for x in [b,c,d] if freq.coeff(x)!=0)<0:freq=-freq
        OD[freq]=s.expand(OD.get(freq,0)+od)
        OH[freq]=s.expand(OH.get(freq,0)+oh)
    expected_D={s.Integer(0):3*W/2}
    expected_H={s.Integer(0):W*sum(1+B0-q*q for q in qs)/2}
    for q in qs:
        expected_D[s.expand(2*q)]=-W/2
        expected_H[s.expand(2*q)]=-W*(1+B0-q*q)/2
    for actual,expected in [(OD,expected_D),(OH,expected_H)]:
        for freq in set(actual)|set(expected):
            assert s.simplify(actual.get(freq,0)-expected.get(freq,0))==0
    # Actual boundary monomials, evaluated on all four distinct external
    # waves, including signs of the incoming time derivative.
    chi=w;chi_t=[s.I*e[i]*w[i] for i in range(4)];chi_xx=[-x**3 for x in w]
    boundary=(s.Rational(7,24)*coefficient(chi,chi,chi,chi)
        +(s.Rational(5,12)-s.pi**2/8)*coefficient(chi,chi,chi_t,chi_t)
        -s.pi**2/8*coefficient(chi,chi,chi_xx,chi_xx)
        +s.pi**2/8*coefficient(chi,chi_xx,chi_t,chi_t))
    expected_boundary=W*(s.Rational(5,3)*B0+s.Rational(16,3)-s.pi**2/4*sum(q*q*(1+q*q) for q in qs))
    assert s.simplify(boundary-expected_boundary)==0
    return {'OD_kernel':'sum sin(q_i*tau)**2',
            'OH_kernel':'sum (1+B0-q_i**2)*sin(q_i*tau)**2',
            'boundary_over_W':'5*B0/3+16/3-pi**2*sum(q_i**2*(1+q_i**2))/4',
            'all_generic_fourier_coefficients':'pass'}


def make_profiles():
    t,q=s.symbols('t q',positive=True)
    u=t/s.sinh(2*t)
    h=2/s.sinh(t)**2-1/s.sinh(t)**4+2*s.diff(u,t)-s.diff(u,t,3)/2
    r=-4/s.sinh(t)**2-2/s.sinh(t)**4+2*s.diff(u,t,2)-s.diff(u,t,4)/2
    # Expand the regular endpoint integrands before numeric evaluation;
    # this avoids subtracting tau^-2 quantities at tiny quadrature nodes.
    hs=s.series(h,t,0,9).removeO()
    rs=s.series(r,t,0,9).removeO()
    sin2=s.series(s.sin(q*t)**2,t,0,14).removeO()
    hreg=s.series(hs*sin2+q*q/t**2,t,0,7).removeO().expand()
    rreg=s.series(rs*sin2+2*q*q/t**2,t,0,7).removeO().expand()
    return {'h':(s.lambdify(t,h,'mpmath'),s.lambdify((t,q),hreg,'mpmath'),1),
            'r':(s.lambdify(t,r,'mpmath'),s.lambdify((t,q),rreg,'mpmath'),2)}


def transform(profile,q):
    full,local,divergence=profile
    def regular(t):
        if t<mp.mpf('1e-7'):return local(t,q)
        return full(t)*mp.sin(q*t)**2+divergence*q*q/(t*t)
    # FP integral of -d*q²/t² on [0,1] is +d*q².
    return mp.quad(regular,[0,mp.mpf('.02'),mp.mpf('.3'),1])+mp.quad(lambda t:full(t)*mp.sin(q*t)**2,[1,3,8,20,40,60])+divergence*q*q


def main():
    mp.mp.dps=55
    out={'vertices':exact_vertices(),'profile_transforms':[],'full_repairs':[]}
    profiles=make_profiles();cache={}
    for q in map(mp.mpf,['.2','.7','1.0','1.1','1.2','1.3','2.1']):
        H=transform(profiles['h'],q);R=transform(profiles['r'],q)
        expected_H=-delta(q)-mp.mpf(5)/3
        expected_R=-theta(q)+1+mp.pi**2*q*q*(1+q*q)/4
        assert abs(H-expected_H)<mp.mpf('1e-36'),(q,'h',H-expected_H)
        assert abs(R-expected_R)<mp.mpf('1e-36'),(q,'r',R-expected_R)
        cache[mp.nstr(q,20)]=(H,R)
        out['profile_transforms'].append({'q':str(q),'h_integral':mp.nstr(H,38),
            'h_error':mp.nstr(H-expected_H,5),'r_integral':mp.nstr(R,38),'r_error':mp.nstr(R-expected_R,5)})
    for raw in [('.4','.6','.7'),('.3','.4','.8')]:
        w=list(map(mp.mpf,raw));A=sum(w);B0=1+(A*A+sum(x*x for x in w))/2;qs=[A-x for x in w]
        bulk=0
        for q in qs:
            H,R=cache[mp.nstr(q,20)]
            bulk+=(1+B0-q*q)*H+R
        boundary=mp.mpf(5)*B0/3+mp.mpf(16)/3-mp.pi**2*sum(q*q*(1+q*q) for q in qs)/4
        computed=bulk+boundary
        required=-sum((1+B0-q*q)*delta(q)+theta(q) for q in qs)
        assert abs(computed-required)<mp.mpf('1e-35')
        out['full_repairs'].append({'outgoing':raw,'bulk':mp.nstr(bulk,38),
            'boundary':mp.nstr(boundary,38),'computed_shift':mp.nstr(computed,38),
            'required_shift':mp.nstr(required,38),'error':mp.nstr(computed-required,5)})
    out['status']='all checks pass; pure-singlet inverse interaction at quartic order only'
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_all_singlet_repair_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
