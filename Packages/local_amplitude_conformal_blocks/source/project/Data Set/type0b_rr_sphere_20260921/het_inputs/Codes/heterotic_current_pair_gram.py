from functools import lru_cache
import sympy as s
k=s.symbols('k', positive=True)
r=s.sqrt(2)
metric={('H','H'):1,('+','-'):1,('-','+'):1}
bracket={('H','+'):(r,'+'),('+','H'):(-r,'+'),('H','-'):(-r,'-'),('-','H'):(r,'-'),('+','-'):(r,'H'),('-','+'):(-r,'H')}
@lru_cache(None)
def vev(word):
    if not word:return s.Integer(1)
    a,n=word[0]
    if n<=0:return s.Integer(0)
    out=0
    rest=word[1:]
    for j,(b,m) in enumerate(rest):
        pre,post=rest[:j],rest[j+1:]
        if n+m==0:out+=k*n*metric.get((a,b),0)*vev(pre+post)
        if (a,b) in bracket:
            c,v=bracket[a,b]
            out+=c*vev(pre+((v,n+m),)+post)
    return s.expand(out)
checks=0
for n in range(1,5):
    for m in range(1,5):
        p=n+m
        raw=vev((('+',m),('-',n),('+',-n),('-',-m)))
        overlap=vev((('H',p),('+',-n),('-',-m)))
        norm=s.expand(raw-overlap**2/(k*p))
        assert s.simplify(raw-(k*k*n*m+2*k*m))==0
        assert s.simplify(overlap-r*k*m)==0
        assert s.simplify(norm-k*n*m*(k+s.Rational(2,p)))==0
        sugawara_overlap=0
        for rmode in range(1,p):
            for a,b in [('H','H'),('+','-'),('-','+')]:
                sugawara_overlap+=vev((('+',m),('-',n),(a,-rmode),(b,-p+rmode)))
        sugawara_overlap=s.simplify(sugawara_overlap/(2*(k+2)))
        assert s.simplify(sugawara_overlap-k*n*m)==0
        checks+=4
print('Passed',checks,'independent finite-mode Gram checks')
for n,m in [(1,1),(1,2),(2,3)]:
    print(n,m,'raw norm=',vev((('+',m),('-',n),('+',-n),('-',-m))).subs(k,1),
          'projected norm=',s.Rational(n*m)*(1+s.Rational(2,n+m)),
          'normalized stress overlap=',s.sqrt(s.Rational(n*m)/(1+s.Rational(2,n+m))))

for p in range(2,9):
    dimension=p-1
    gram=s.zeros(dimension)
    d=s.diag(*[n*(p-n) for n in range(1,p)])
    bridge=s.zeros(dimension)
    for n in range(1,p):
        for m in range(1,p):
            raw=vev((('+',p-m),('-',m),('+',-n),('-',-p+n)))
            projected=s.expand(raw-2*k*s.Rational((p-n)*(p-m),p))
            target=k*k*n*(p-n)*int(n==m)+2*k*s.Rational(min(n,m)*(p-max(n,m)),p)
            assert s.simplify(projected-target)==0
            gram[m-1,n-1]=projected
            bridge[m-1,n-1]=s.Rational(min(n,m)*(p-max(n,m)),p)
            checks+=1
    # Same eigenvalues as the symmetric matrix D^-1/2 G D^-1/2 / k^2.
    normalized=s.eye(dimension)+2*d.inv()*bridge # set k=1
    actual=normalized.eigenvals()
    expected={1+s.Rational(2,ell*(ell+1)):1 for ell in range(1,p)}
    assert actual==expected,(p,actual,expected)
    # Stress vector is sqrt(D) in symmetric normalization, or 1 in this similarity basis.
    assert normalized*s.ones(dimension,1)==2*s.ones(dimension,1)
    print('p=',p,'Gram eigenvalues at k=1:',list(expected))
    checks+=2
print('TOTAL',checks,'checks passed')

import json
from pathlib import Path
output = Path(__file__).resolve().parents[1]/'data_exports/heterotic_long_string_trees_20260912'/'current_pair_gram_results.json'
result = {'status':'passed','checks':checks,'scope':'Affine root-pair Gram matrix and normalized spectrum; current-created states are not independent Fock particles','projected_gram':'k^2*n*(p-n)*delta_nm + (2*k/p)*min(n,m)*(p-max(n,m))','normalized_eigenvalues':'1+2/[k*l*(l+1)], l=1,...,p-1','level_one_stress_identity':'sum_n projected_root_pair(n,p-n)=2 L_root(-p)|0>'}
output.write_text(json.dumps(result,indent=2)+'\n')
