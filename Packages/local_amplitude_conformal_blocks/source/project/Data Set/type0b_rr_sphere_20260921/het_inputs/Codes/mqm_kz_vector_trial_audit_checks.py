"""Independent exact minimal-vector action/domain and scalar-sector checks."""
from collections import defaultdict
from itertools import combinations, permutations, product
from pathlib import Path
import json
import sympy as s


def sign(p):
    return (-1)**sum(p[i]>p[j] for i in range(len(p)) for j in range(i+1,len(p)))


def combine(terms):
    out = defaultdict(lambda:s.Integer(0))
    for factor,vec in terms:
        for labels,value in vec.items(): out[labels] += factor*value
    return {k:s.factor(v) for k,v in out.items() if s.factor(v)!=0}


def R(vec,site,a,b):
    out = {}
    for labels,value in vec.items():
        if labels[site] in (a,b):
            target = list(labels)
            target[site] = b if labels[site]==a else a
            out[tuple(target)] = value*(-1 if labels[site]==a else 1)
    return out


def omega(vec,i,j,n):
    out = defaultdict(lambda:s.Integer(0))
    for labels,value in vec.items():
        target = list(labels); target[i],target[j] = target[j],target[i]
        out[tuple(target)] += value
        if labels[i]==labels[j]:
            for a in range(n):
                target=list(labels);target[i]=target[j]=a
                out[tuple(target)] -= value
    return {k:s.factor(v) for k,v in out.items() if s.factor(v)!=0}


def C(vec,i,j,a,b,n,transpose=False):
    if not transpose:
        return combine([(s.Rational(2,3),R(vec,j,a,b)),
                        (-s.Rational(1,n-1),R(omega(vec,i,j,n),i,a,b)),
                        (s.Rational(1,3),omega(R(vec,i,a,b),i,j,n))])
    return combine([(-s.Rational(2,3),R(vec,j,a,b)),
                    (s.Rational(1,n-1),omega(R(vec,i,a,b),i,j,n)),
                    (-s.Rational(1,3),R(omega(vec,i,j,n),i,a,b))])


def pairings(sites):
    if not sites: return [(1,[])]
    i=sites[0];out=[]
    for k in range(1,len(sites)):
        j=sites[k]
        for sg,pairs in pairings(sites[1:k]+sites[k+1:]):
            out.append(((-1)**(k-1)*sg,[(i,j)]+pairs))
    return out


def majorana(n,x):
    out=[]
    for sg,pairs in pairings(list(range(len(x)))):
        vec={}
        for colors in product(range(n),repeat=len(pairs)):
            labels=[None]*len(x)
            for color,(i,j) in zip(colors,pairs):labels[i]=labels[j]=color
            vec[tuple(labels)]=s.Integer(1)
        out.append((sg/s.prod(x[i]-x[j] for i,j in pairs),vec))
    return combine(out)


def charged_case(n):
    N=n-1;ell=N//2;x=s.symbols('x0:'+str(N),real=True)
    eps={p:s.Integer(sign(p)) for p in permutations(range(1,n))}
    assert len(eps)==s.factorial(N)
    psi=majorana(n,x)
    terms=[]
    for p in permutations(range(1,n)):
        vec=psi
        for i in range(ell):vec=R(vec,i,p[2*i],p[2*i+1])
        terms.append((sign(p),vec))
    real_action=combine(terms)
    permanent=s.factor(sum((s.prod(1/(x[i]-x[ell+p[i]]) for i in range(ell))
                           for p in permutations(range(ell))),s.Integer(0)))
    assert combine([(1,real_action),(-2**ell*permanent,eps)])=={}
    delta=s.prod(x[j]-x[i] for i,j in combinations(range(N),2))
    polynomial=s.factor(delta*permanent)
    assert s.Poly(polynomial,*x).total_degree()==N*(N-2)//2
    for i,j in combinations(range(N),2):
        assert omega(eps,i,j,n)=={k:-v for k,v in eps.items()}
    u_a=s.Rational(2*(n*n-4),3*(n-1))
    pair_result=combine([(1,C(C(eps,0,1,a,b,n),0,1,a,b,n,True))
                         for a,b in combinations(range(n),2)])
    assert pair_result=={k:u_a*v/2 for k,v in eps.items()}
    triple_value=None
    if N>=3:
        triple=combine([(1,C(C(eps,0,2,a,b,n),0,1,a,b,n,True))
                         for a,b in combinations(range(n),2)])
        first=tuple(range(1,n));triple_value=triple.get(first,0)
        assert triple=={k:triple_value*v for k,v in eps.items() if triple_value*v}
        # Independent relabelled ordered triple has the identical scalar.
        other=combine([(1,C(C(eps,2,3,a,b,n),2,0,a,b,n,True))
                        for a,b in combinations(range(n),2)])
        assert other==triple
    rho=s.symbols('rho',positive=True)
    collision={x[0]:0,x[1]:rho} if N==2 else {x[0]:0,x[1]:1,x[2]:1+rho,x[3]:3}
    collision_limit=s.limit(polynomial.subs(collision),rho,0,dir='+')
    assert collision_limit!=0
    A=[sum((1/(x[i]-x[j]) for j in range(N) if i!=j),s.Integer(0)) for i in range(N)]
    sum_pair=sum((1/(x[i]-x[j])**2 for i,j in combinations(range(N),2)),s.Integer(0))
    assert s.factor(sum(a*a for a in A)-2*sum_pair)==0
    assert s.factor(sum(x[i]*A[i] for i in range(N))-N*(N-1)/2)==0
    return {'n':n,'N':N,'real_operator_action_coefficient':str(2**ell),
            'full_operator_phase_coefficient':str((2*s.I)**ell),
            'Cauchy_permanent':str(permanent),'flattened_polynomial':str(polynomial),
            'isolated_cross_pair_collision_limit':str(collision_limit),
            'null_ordered_pair_scalar':str(u_a/2),
            'null_distinct_triple_scalar':str(triple_value)}


def main():
    cases=[charged_case(3),charged_case(5)]
    N,gamma=s.symbols('N gamma',positive=True)
    beta=1-1/N
    u_a=2*((N+1)**2-4)/(3*N)
    b=s.Rational(1,2)+s.sqrt((beta-s.Rational(1,2))**2+gamma*u_a)
    assert s.simplify(b*(b-1)-beta*(beta-1)-gamma*u_a)==0
    ordinary_gap=N/2+(b-1)*N*(N-1)/2
    square_gap=N*(N-1)*(b-beta)/2
    assert s.simplify(ordinary_gap-square_gap-s.Rational(1,2))==0
    for size in [2,4,22]:
        assert s.limit(ordinary_gap.subs(N,size),gamma,0,dir='+')==s.Rational(1,2)
        assert s.limit(square_gap.subs(N,size),gamma,0,dir='+')==0
    output={'all_checks_passed':True,'small_n_exact_cases':cases,
            'ordinary_minus_positive_square_vector_gap_over_Omega':'1/2',
            'scope':'Exact minimal N=n−1 vector sector; no large-N or Liouville scattering conclusion.'}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_vector_trial_audit_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
