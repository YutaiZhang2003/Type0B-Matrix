"""Full finite-level BRST ghost audit of the O22 chiral representative.

The SCA actions at this level are displayed explicitly; all bc and beta-gamma
mode contractions are performed by an independent canonical-ordering routine.
No repository algebra package is imported.
"""
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import json
import sympy as s
R=s.Rational
half=R(1,2)
checks=[]

def check(name,v):
    if isinstance(v,dict):
        good=all(s.simplify(x)==0 for x in v.values())
    elif isinstance(v,s.MatrixBase):
        good=all(s.simplify(x)==0 for x in v)
    else:
        good=s.simplify(v)==0
    assert good,(name,v)
    checks.append(name)

def ann(o):
    t,r=o
    return bool(r>=(-1 if t=="b" else 2 if t=="c" else half))

def key(o):
    t,r=o
    return (ann(o),{"c":0,"b":1,"B":2,"Y":3}[t],-r)

def ferm(o):
    return o[0] in ("b","c")

def bracket(a,b):
    t,r=a;u,q=b
    if r+q!=0: return 0
    if {t,u}=={"b","c"}:return 1
    if (t,u)==("Y","B"):return 1
    if (t,u)==("B","Y"):return -1
    return 0

@lru_cache(None)
def norm(word):
    """Fermionic anticommutators, bosonic commutators, picture -1 vacuum."""
    for j in range(len(word)-1):
        a,b=word[j:j+2]
        if a==b and ferm(a):return ()
        if key(a)>key(b):
            answer=defaultdict(lambda:s.S.Zero)
            sign=-1 if ferm(a) and ferm(b) else 1
            swapped=word[:j]+(b,a)+word[j+2:]
            for w,c in norm(swapped):answer[w]+=sign*c
            k=bracket(a,b)
            if k:
                for w,c in norm(word[:j]+word[j+2:]):answer[w]+=k*c
            return tuple((w,s.expand(c)) for w,c in answer.items() if c)
    if any(ann(o) for o in word):return ()
    return ((word,s.S.One),)

def add(out,coef,ghost,matter):
    for w,c in norm(tuple(ghost)):
        out[(w,matter)]+=coef*c

def clean(out):
    return {k:s.simplify(v) for k,v in out.items() if s.simplify(v)!=0}

C1=(("c",s.Integer(1)),)
Bm=(("B",-half),)
Ym=(("Y",-half),)
# Matter labels:
# A=psi_-1/2 g, W=psi_-1/2 N, U=a_-1 N,
# B=a_-1 g, C=psi_-1/2 L_-1 N, D=psi_-3/2 N.
# Five independent pure-c output labels:
# t1=psi_-3/2 g, t2=a_-1 L_-1 N, t3=psi_-1/2 L_-1 g,
# t4=a_-2 N, t5=psi_-1/2 G_-3/2 N.
weights={"A":-half,"W":-1,"B":0,"C":0,"D":0}
Lminus={"A":{"t1":1,"t3":1},"W":{"LW":1},
        "B":{"LB":1},"C":{"LC":1},"D":{"LD":1}}
Lplus={"A":{},"W":{},"B":{},"C":{"W":-3},"D":{"W":1}}
Gplus={"A":{"W":3},"W":{},"B":{"A":1,"U":-3},
       "C":{"A":-1},"D":{"U":1}}
Gminus={
 ("B",-half):{"t1":1,"t2":1},
 ("C",-half):{"t2":1,"t3":-1},
 ("D",-half):{"t1":-1,"t4":1},
 ("W",-3*half):{"t4":1,"t5":-1}
}

def qpure(v):
    out=defaultdict(lambda:s.S.Zero)
    for a,k in Lminus[v].items():add(out,k,C1,a)
    add(out,-1,C1+Bm+Ym,v)
    add(out,weights[v]+half,(("c",0),),v)
    for a,k in Lplus[v].items():add(out,k,(("c",-1),),a)
    for a,k in Gplus[v].items():add(out,k,Ym,a)
    return clean(out)

def q_cbeta(v,r,K=4):
    """Q(c1 beta_r v) via Qc1, [Q,beta_r], and Qv, with full ghosts."""
    out=defaultdict(lambda:s.S.Zero)
    B=(( "B",r),)
    # Qc1 = [c partial c - gamma^2] coefficient z^0.
    for m in range(-K,K+1):
        n=1-m
        add(out,1-n,(("c",m),("c",n))+B,v)
    for j in range(-K,K):
        q=R(2*j+1,2)
        add(out,-1,(("Y",q),("Y",1-q))+B,v)
    # -c1 [Q,beta_r], matter piece +G_r.
    for a,k in Gminus[(v,r)].items():add(out,-k,C1,a)
    # c partial beta +3/2 partial c beta
    for m in range(-K,K+1):
        add(out,-(R(m,2)-r),C1+(("c",-m),("B",r+m)),v)
    # [Q,beta_r] includes -2 sum gamma_q b_{r-q}.
    for j in range(-K,K):
        q=R(2*j+1,2)
        add(out,2,C1+(("Y",q),("b",r-q)),v)
    # -c1 beta_r Qv.
    for (w,a),k in qpure(v).items():add(out,-k,C1+B+w,a)
    return clean(out)

# Vacuum convention checks; these are where the bosonization signs enter.
check("beta gamma annihilator commutator",
      dict(norm((("B",half),("Y",-half)))) .get((),0)+1)
check("gamma beta annihilator commutator",
      dict(norm((("Y",half),("B",-half)))) .get((),0)-1)
check("bc annihilator anticommutator",
      dict(norm((("b",-1),("c",1)))) .get((),0)-1)

# L_-1^beta-gamma = sum_q (q-1/2) beta_-1-q gamma_q.
ghostL=defaultdict(lambda:s.S.Zero)
for j in range(-4,4):
    q=R(2*j+1,2)
    for w,k in norm((("B",-1-q),("Y",q))):ghostL[w]+=(q-half)*k
ghostL[tuple(Bm+Ym)]+=1
check("picture-minus-one ghost translation is minus beta gamma",ghostL)

b,c,d,e=s.symbols("b c d e")
total=defaultdict(lambda:s.S.Zero)
for key_,v in qpure("A").items():total[key_]+=v
for coef,name,r in [(b,"B",-half),(c,"C",-half),
                    (d,"D",-half),(e,"W",-3*half)]:
    small=q_cbeta(name,r,3)
    big=q_cbeta(name,r,6)
    check(name+" finite ghost-mode support",{k:small.get(k,0)-big.get(k,0)
          for k in set(small)|set(big)})
    for key_,v in big.items():total[key_]+=coef*v
total=clean(total)

expected=defaultdict(lambda:s.S.Zero)
for matter,coef in [("t1",1-b+d),("t2",-b-c),("t3",1+c),
                    ("t4",-d-e),("t5",e)]:
    add(expected,coef,C1,matter)
add(expected,3-2*e,Ym,"W")
add(expected,-1-b+c,C1+Bm+Ym,"A")
add(expected,3*b-d,C1+Bm+Ym,"U")
add(expected,3*c-d-2*e,C1+(("c",-1),)+Bm,"W")
check("complete BRST variation has exactly the displayed sectors",
      {k:total.get(k,0)-expected.get(k,0) for k in set(total)|set(expected)})
values={b:-half,c:half,d:-3*half,e:3*half}
on_shell={k:s.simplify(v.subs(values)) for k,v in total.items()}
null=defaultdict(lambda:s.S.Zero)
add(null,3*half,C1,"t3")
add(null,3*half,C1,"t5")
check("full variation is 3/2 c1 psi times exact SL null",
      {k:on_shell.get(k,0)-null.get(k,0) for k in set(on_shell)|set(null)})

equations=[3-2*e,-1-b+c,3*b-d,3*c-d-2*e,
           1-b+d,-b-c,-d-e,1+c-e]
solutions=s.solve(equations,[b,c,d,e],dict=True)
assert solutions==[values],solutions
checks.append("unique BRST completion within leading-normalized five-term ansatz")
result={"passed":True,"exact_checks":len(checks),"checks":checks,
        "solution":{str(k):str(v) for k,v in values.items()},
        "full_variation":"(3/2) c1 psi_X,-1/2 (G_-3/2 N + L_-1 G_-1/2 N)",
        "ghost_sectors_before_substitution":[str(k) for k in total],
        "scope":"Full finite-level chiral BRST closure in the exact SL null quotient. No paired bulk/heterotic contour normalization."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_o22_full_brst_independent_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
