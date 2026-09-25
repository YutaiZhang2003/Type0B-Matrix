#!/usr/bin/env python3
"""Full all-singlet test of the fixed holonomic wall-fluid completion.

Common endpoint Hadamard transforms are followed by the common spectral
finite part used in the earlier vector and mixed tests. No finite-MQM
regulator equivalence or string duality is assumed.
"""
import itertools
import json
from pathlib import Path
import mpmath as mp
import sympy as s
import mqm_wall_mixed_quartic_checks as direct
from mqm_wall_mixed_kernel_checks import jk, thermal, kernel_j, delta


def exact_contact_check():
    b,c,d=s.symbols('b c d',positive=True)
    w=[b+c+d,b,c,d]; e=[-w[0],b,c,d]
    W=s.prod(w); B0=1+sum(x*x for x in w)/2
    q=[w[0]-w[i] for i in range(1,4)]
    parent={}; bulk={}
    permutations=list(itertools.permutations(range(4)))
    def multilinear(f,g,h,k):
        return sum(f[i]*g[j]*h[l]*k[m] for i,j,l,m in permutations)
    for signs in itertools.product((-1,1),repeat=4):
        eta_x=[x/2 for x in w]
        eta_t=[signs[i]*e[i]/2 for i in range(4)]
        W_t=[s.I*e[i]*w[i]/2 for i in range(4)]
        W_x=[s.I*signs[i]*w[i]*w[i]/2 for i in range(4)]
        coeff=(multilinear(eta_x,eta_x,eta_t,eta_t)/2
               +multilinear(eta_x,eta_x,W_t,W_t)/2
               +2*multilinear(eta_x,eta_t,W_t,W_x)
               +multilinear(eta_t,eta_t,W_x,W_x)/2)
        added=multilinear(W_t,W_t,W_x,W_x)
        freq=s.expand(sum(signs[i]*w[i] for i in range(4)))
        # Energy conservation makes every nonzero coefficient in freq have
        # the same sign. Pair +/- frequencies into one cosine coefficient.
        if freq!=0 and next(freq.coeff(x) for x in [b,c,d] if freq.coeff(x)!=0)<0:
            freq=-freq
        parent[freq]=s.expand(parent.get(freq,0)+coeff)
        bulk[freq]=s.expand(bulk.get(freq,0)+added)
    expected_parent={s.Integer(0):3*B0*W/2}
    expected_bulk={s.Integer(0):-3*W*W}
    for energy in q:
        expected_parent[s.expand(2*energy)]=W*(energy*energy-B0/2)
        expected_bulk[s.expand(2*energy)]=W*W
    for key in set(parent)|set(expected_parent):
        assert s.simplify(parent.get(key,0)-expected_parent.get(key,0))==0
    for key in set(bulk)|set(expected_bulk):
        assert s.simplify(bulk.get(key,0)-expected_bulk.get(key,0))==0
    # Actual boundary coefficient of (W_t)^4/8 at tau=0.
    boundary=s.factor(s.factorial(4)*s.prod(s.I*e[i]*w[i] for i in range(4))/8)
    assert s.simplify(boundary+3*W*W)==0
    return {'parent_integrand_over_W':'3*B0/2 + sum((q_i**2-B0/2)*cos(2*q_i*tau))',
            'added_bulk_integrand_over_W2':'-3 + sum(cos(2*q_i*tau))',
            'boundary_over_W2':'-3','status':'pass'}


def G(q):return mp.pi**2*(1+q*q)*mp.tanh(mp.pi*q/2)**2/4


def exact_structural_check():
    qs=s.symbols('q1:4',positive=True)
    Ts=s.symbols('T1:4');Js=s.symbols('J1:4');Gs=s.symbols('G1:4')
    W=s.symbols('W');A=sum(qs)/2;B0=1+sum(q*q for q in qs)/2
    exchanges=[];deltas=[];thetas=[]
    for q,T,J,Gv in zip(qs,Ts,Js,Gs):
        R=1/(1+s.I*q);I=T+R
        L=(q*q+4)*T/3-s.Rational(1,3)-s.I*q-R-Gv
        exchanges.append(-((B0-q*q-W)*I+q*(1+B0-q*q)*J+q*q*L))
        deltas.append(s.Rational(2,3)-(q*q+4)*T/3-q*J-R-s.I*q)
        thetas.append((4-2*q*q)*T/3+q*q*Gv-s.Rational(1,3))
    parent=B0+sum((2*q*q-B0)*(1+q*q)*T/3 for q,T in zip(qs,Ts))
    target=W*sum(1/(1+s.I*q) for q in qs)+(1+2*s.I*A)*B0
    lhs=sum(exchanges)+parent-W*sum(Ts)-target
    rhs=sum((1+B0-q*q)*D+Th for q,D,Th in zip(qs,deltas,thetas))
    assert s.simplify(s.expand(lhs-rhs))==0
    b,c,q=s.symbols('b c q')
    assert s.expand((1+b+q*q)*(1-c+q*q)-(1+q*q)*(1+b-c+q*q)+b*c)==0
    return {'generic_structural_identity':'pass','on_shell_cubic_product':'pass'}

def kernel_L(q):
    return (q*q+4)*thermal(q)/3-mp.mpf(1)/3-1j*q-1/(1+1j*q)-G(q)

def theta(q):return (4-2*q*q)*thermal(q)/3+q*q*G(q)-mp.mpf(1)/3


def integral_L(q,cutoff):
    def numerator(p):return p*p*jk(p,q)[1]**2/(1+p*p)
    pole=numerator(q)
    def smooth(p):
        if p==q:return -mp.diff(numerator,q)/(2*q)
        return (numerator(p)-pole)/(q*q-p*p)
    tail=pole/(2*q)*mp.log((cutoff+q)/(cutoff-q))
    return 2/mp.pi*(mp.quad(smooth,[0,q/2,q,q+1,q+8,cutoff])+tail)-1j*pole/q


def all_singlet(raw):
    outgoing=list(map(s.Rational,raw));a=sum(outgoing)
    ws=[a]+outgoing; W_s=s.prod(ws);W=mp.mpf(str(W_s))
    B0=mp.mpf(str(1+sum(x*x for x in ws)/2))
    exchange=0;reports=[];qs=[]
    for i in range(3):
        b=outgoing[i];c,d=[outgoing[j] for j in range(3) if j!=i]
        q=c+d;qq=mp.mpf(str(q));qs.append(qq)
        left=direct.make_vertex(['S','S','S'],[a,b],[-a,b,q])
        right=direct.make_vertex(['S','S','S'],[c,d],[c,d,-q])
        # Off-shell right source is independently checked before integrating.
        cd=mp.mpf(str(c*d))
        for p in map(mp.mpf,['.17','.83','1.51','2.73']):
            j,k=jk(p,qq)
            expected=-cd*((1-cd)*j+qq*p*k)
            assert abs(right[0](p)-expected)<mp.mpf('1e-37')
        value,report=direct.channel(left,right,q,True)
        exchange+=value
        # Closed exchange includes all components of the constrained
        # singlet source, including the internal W components.
        I=thermal(qq)+1/(1+1j*qq)
        expected_E=-((B0-qq*qq-W)*I+qq*(1+B0-qq*qq)*kernel_j(qq)+qq*qq*kernel_L(qq))
        assert abs(value/W-expected_E)<mp.mpf('1e-29')
        report['closed_value_over_W']=mp.nstr(expected_E,35)
        reports.append(report)
    parent=B0+mp.fsum((2*q*q-B0)*(1+q*q)*thermal(q)/3 for q in qs)
    added=-W*mp.fsum(thermal(q) for q in qs)
    measured=exchange/W+parent+added
    target=W*mp.fsum(1/(1+1j*q) for q in qs)+B0*(1+2j*mp.mpf(str(a)))
    expected_difference=mp.fsum((1+B0-q*q)*delta(q)+theta(q) for q in qs)
    assert abs(measured-target-expected_difference)<mp.mpf('1e-28')
    assert abs(mp.im(measured-target))<mp.mpf('1e-38')
    return {'outgoing':raw,'B0':str(B0),'exchange_channels':reports,
            'parent_contact':mp.nstr(parent,35),'added_contacts':mp.nstr(added,35),
            'computed':mp.nstr(measured,35),'target':mp.nstr(target,35),
            'difference':mp.nstr(measured-target,35),
            'structural_difference':mp.nstr(expected_difference,35)}


def main():
    mp.mp.dps=45
    out={'exact_contacts':exact_contact_check(),'exact_structure':exact_structural_check(),'L_checks':[],'samples':[]}
    for q in map(mp.mpf,['.2','.7','1.3','2.1']):
        integral=integral_L(q,mp.mpf(60));closed=kernel_L(q)
        assert abs(integral-closed)<mp.mpf('1e-37')
        out['L_checks'].append({'q':str(q),'integral':mp.nstr(integral,37),'closed':mp.nstr(closed,37),'error':mp.nstr(abs(integral-closed),5)})
    for raw in [['.4','.6','.7'],['.3','.4','.8']]:
        sample=all_singlet(raw);out['samples'].append(sample)
        print(json.dumps(sample,indent=2),flush=True)
    with mp.workdps(105):
        eps=mp.mpf('1e-17');q=1j+eps
        assert abs(kernel_L(q))<10*eps**2
        assert abs(theta(q)+2)<10*eps
    out['pole_checks']={'L(i)':'0','Theta(i)':'-2','status':'pass'}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_all_singlet_quartic_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'all checks pass'},indent=2))

if __name__=='__main__':main()
