"""Exact unfitted test: can null squares cancel KZ three-site terms?"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import json
import sympy as s


def combine(terms):
    out = defaultdict(F)
    for factor, vector in terms:
        for labels, value in vector.items():
            out[labels] += factor*value
    return {labels:value for labels,value in out.items() if value}


def rotation(vector, site):
    # L^{01}=i R.  The common i cancels in N^dagger N.
    out = {}
    for labels,value in vector.items():
        if labels[site] in (0,1):
            target = list(labels)
            target[site] = 1-labels[site]
            out[tuple(target)] = value*(1 if labels[site] == 1 else -1)
    return out


def omega(vector,i,j,n):
    out = defaultdict(F)
    for labels,value in vector.items():
        target = list(labels)
        target[i],target[j] = target[j],target[i]
        out[tuple(target)] += value
        if labels[i] == labels[j]:
            for a in range(n):
                target = list(labels)
                target[i] = target[j] = a
                out[tuple(target)] -= value
    return {labels:value for labels,value in out.items() if value}


def ward_pair(vector,i,j,n):
    return combine([(F(2,3),rotation(vector,j)),
                    (-F(1,n-1),rotation(omega(vector,i,j,n),i)),
                    (F(1,3),omega(rotation(vector,i),i,j,n))])


def dot(v,w):
    return sum((value*w.get(labels,F(0)) for labels,value in v.items()),F(0))


def singlet_basis(n):
    out = []
    for pairs in [((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]:
        vec = {}
        for a,b in product(range(n),repeat=2):
            labels = [None]*4
            for pair,color in zip(pairs,[a,b]):
                for site in pair:
                    labels[site] = color
            vec[tuple(labels)] = F(1)
        out.append(vec)
    return out


def stringify(mat):
    return [[str(v) for v in row] for row in mat.tolist()]


def case(n,x):
    basis = singlet_basis(n)
    gram = s.Matrix(3,3,lambda a,b:dot(basis[a],basis[b]))
    assert gram == s.Matrix(3,3,lambda a,b:n*n if a==b else n)
    selectors = [(0,0,1,1),(0,1,0,1),(0,1,1,0)]
    om = {}
    for i,j in combinations(range(4),2):
        images = [omega(v,i,j,n) for v in basis]
        om[i,j] = s.Matrix(3,3,lambda a,b:images[b].get(selectors[a],F(0)))
        assert gram*om[i,j] == om[i,j].T*gram
    nullpair = {(i,j):[ward_pair(v,i,j,n) for v in basis]
                for i in range(4) for j in range(4) if i!=j}
    null_three_gram = s.zeros(3)
    null_pair_gram = s.zeros(3)
    null_total_gram = s.zeros(3)
    dim_adj = s.Rational(n*(n-1),2)
    for i in range(4):
        others = [j for j in range(4) if i!=j]
        full = [combine([(F(1,x[i]-x[j]),nullpair[i,j][a]) for j in others])
                for a in range(3)]
        null_total_gram += dim_adj*s.Matrix(3,3,lambda a,b:dot(full[a],full[b]))
        for j,k in product(others,repeat=2):
            term = dim_adj*s.Matrix(3,3,lambda a,b:dot(nullpair[i,j][a],nullpair[i,k][b]))
            term /= (x[i]-x[j])*(x[i]-x[k])
            if j==k:
                null_pair_gram += term
            else:
                null_three_gram += term
    assert null_total_gram == null_pair_gram+null_three_gram
    assert null_three_gram == null_three_gram.T
    # Independent N2 irrep formula verifies both the generator-sum factor
    # and the ordered-site counting in the directly computed pair piece.
    analytic_pair = s.zeros(3)
    c_st = s.Rational(2*n*(n-2),n-1)
    c_a = s.Rational(2*(n*n-4),3*(n-1))
    for (i,j),O in om.items():
        Q = (O*O-s.eye(3))/(n-2)
        P = O+Q
        analytic_pair += (c_st*((s.eye(3)+P)/2-Q/n)
                          +c_a*(s.eye(3)-P)/2)/(x[i]-x[j])**2
    assert null_pair_gram == gram*analytic_pair
    null_three = gram.inv()*null_three_gram
    alpha = s.Rational(1,n-1)
    kz_three = s.zeros(3)
    for i in range(4):
        for j,k in combinations([a for a in range(4) if a!=i],2):
            B = s.eye(3)+alpha*om[tuple(sorted((i,j)))]
            C = s.eye(3)+alpha*om[tuple(sorted((i,k)))]
            kz_three += (B*C+C*B)/(2*(x[i]-x[j])*(x[i]-x[k]))
    assert gram*kz_three == kz_three.T*gram
    ratios = {(a,b):s.factor(-kz_three[a,b]/null_three[a,b])
              for a,b in product(range(3),repeat=2) if null_three[a,b]}
    unique = sorted(set(ratios.values()),key=str)
    impossible_zero_entry = any(kz_three[a,b] and not null_three[a,b]
                                for a,b in product(range(3),repeat=2))
    gamma = unique[0] if len(unique)==1 and not impossible_zero_entry else None
    if gamma is not None:
        assert kz_three+gamma*null_three == s.zeros(3)
    pf = s.Matrix([s.Rational(1,(x[0]-x[1])*(x[2]-x[3])),
                   -s.Rational(1,(x[0]-x[2])*(x[1]-x[3])),
                   s.Rational(1,(x[0]-x[3])*(x[1]-x[2]))])
    assert null_total_gram*pf == s.zeros(3,1)
    return {"n":n,"x":x,"KZ_three":stringify(kz_three),
            "null_three":stringify(null_three),
            "entry_gamma_ratios":{str(k):str(v) for k,v in ratios.items()},
            "constant_gamma":str(gamma),"positive_gamma":bool(gamma is not None and gamma>0),
            "physical_gram":stringify(gram)}


def main():
    rows = [case(n,x) for n in [3,4,23,24]
            for x in [(0,1,3,6),(-4,-1,2,8)]]
    assert all(row['constant_gamma']=='None' for row in rows)
    witness = next(row for row in rows if row['n']==23 and row['x']==(0,1,3,6))
    assert witness['entry_gamma_ratios']['(0, 0)']=='69/2068'
    assert witness['entry_gamma_ratios']['(0, 1)']=='-3/110'
    result = {"cases":rows,"all_checks_passed":True,
              "scope":"N4 metric-contraction singlet subspace; at n23/24 this is the full global-singlet space. No amplitudes or fitted coefficients enter."}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_pairwise_selection_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:row[k] for k in ['n','x','constant_gamma','positive_gamma','entry_gamma_ratios']}
                      for row in rows],indent=2))


if __name__ == '__main__':
    main()
