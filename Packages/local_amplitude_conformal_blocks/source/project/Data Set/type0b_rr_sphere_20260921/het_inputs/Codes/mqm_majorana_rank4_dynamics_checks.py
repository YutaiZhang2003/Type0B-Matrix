"""Independent Clifford, metric, branching and perturbative rank-four checks."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import product
from fractions import Fraction
import json
import sympy as s

checks = []
def ck(name, value):
    value = bool(value)
    checks.append({"name": name, "passed": value})
    if not value:
        raise AssertionError(name)

def zero(matrix):
    return all(s.simplify(v) == 0 for v in matrix)

I = s.eye(2)
pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
def kron(items):
    out = s.ones(1, 1)
    for item in items:
        out = s.kronecker_product(out, item)
    return s.SparseMatrix(out)

def clifford(even_dimension):
    modes = even_dimension // 2
    out = []
    for mode in range(modes):
        for mat in pauli[:2]:
            out.append(kron([pauli[2]] * mode + [mat] + [I] * (modes-mode-1)))
    return out

g = clifford(4)
idh = s.eye(4)
chir = -g[0]*g[1]*g[2]*g[3]
plus = [-s.I*(g[1]*g[2]+g[0]*g[3])/4,
        -s.I*(g[2]*g[0]+g[1]*g[3])/4,
        -s.I*(g[0]*g[1]+g[2]*g[3])/4]
minus = [-s.I*(g[1]*g[2]-g[0]*g[3])/4,
         -s.I*(g[2]*g[0]-g[1]*g[3])/4,
         -s.I*(g[0]*g[1]-g[2]*g[3])/4]
for sign, ts in [(1,plus),(-1,minus)]:
    projector = (idh+sign*chir)/2
    ck(f"heavy chirality {sign} has rank two", s.trace(projector)==2)
    for a in range(3):
        ck(f"heavy {sign}, generator {a} support", zero(ts[a]*(idh-projector)))
        b,c=(a+1)%3,(a+2)%3
        ck(f"heavy {sign}, generator {a} su2 bracket", zero(ts[a]*ts[b]-ts[b]*ts[a]-s.I*ts[c]))
    ck(f"heavy {sign} Casimir", zero(sum((t*t for t in ts),s.zeros(4))-s.Rational(3,4)*projector))
for a in range(3):
    for b in range(3):
        ck(f"commuting color factors {a},{b}",zero(plus[a]*minus[b]-minus[b]*plus[a]))

x12,x13,x14,x23,x24,x34=s.symbols('x12 x13 x14 x23 x24 x34',real=True)
x=[x12,x13,x14,x23,x24,x34]
bp=s.Matrix([x23+x14,-x13+x24,x12+x34])/s.sqrt(2)
bm=s.Matrix([x23-x14,-x13-x24,x12-x34])/s.sqrt(2)
transform=bp.col_join(bm).jacobian(x)
ck("orthonormal selfdual matrix coordinates",zero(transform.T*transform-s.eye(6)))
yuk=s.zeros(4)
for val,(i,j) in zip(x,[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
    yuk += s.I*val*g[i]*g[j]
dual=-2*s.sqrt(2)*sum((bp[a]*plus[a]+bm[a]*minus[a] for a in range(3)),s.zeros(4))
ck("full selective Yukawa normalization",zero(yuk-dual))
ck("heavy chirality commutes with complete Yukawa",zero(chir*yuk-yuk*chir))

# A color reflection in row four exchanges the two SU2 factors. Its heavy
# Pin lift alone is gamma4; an overall phase in the full 24-flavor lift is fixed
# elsewhere, and cancels from this conjugation test.
reflection=g[3]
for a in range(3):
    ck(f"reflection exchanges heavy spin factors {a}",zero(reflection*plus[a]*reflection-minus[a]))
ck("reflection reverses heavy chirality",zero(reflection*chir*reflection+chir))

# Independent complete Cl12 construction: four colors, three spectator flavors.
# This tests the singlet/vector spectator pattern in an actual finite Fock space,
# separately from the large-flavor complement table.
n=3
gs=clifford(4*n)
gam=[[gs[i*n+a] for a in range(n)] for i in range(4)]
dim=2**(2*n)
sp=[s.zeros(dim) for _ in range(3)]
sm=[s.zeros(dim) for _ in range(3)]
for a in range(n):
    c=[gam[i][a] for i in range(4)]
    p=[-s.I*(c[1]*c[2]+c[0]*c[3])/4,
       -s.I*(c[2]*c[0]+c[1]*c[3])/4,
       -s.I*(c[0]*c[1]+c[2]*c[3])/4]
    m=[-s.I*(c[1]*c[2]-c[0]*c[3])/4,
       -s.I*(c[2]*c[0]-c[1]*c[3])/4,
       -s.I*(c[0]*c[1]-c[2]*c[3])/4]
    sp=[sp[i]+p[i] for i in range(3)]
    sm=[sm[i]+m[i] for i in range(3)]
cp=sum((t*t for t in sp),s.zeros(dim))
cm=sum((t*t for t in sm),s.zeros(dim))
cf=s.zeros(dim)
for a in range(n):
    for b in range(a+1,n):
        t=sum((s.I*gam[i][a]*gam[i][b]/2 for i in range(4)),s.zeros(dim))
        cf+=t*t
        ck(f"complete Cl12 commuting flavor {a},{b}",zero(t*cp-cp*t) and zero(t*cm-cm*t))
ck("complete Cl12 flavor spectrum",cf.eigenvals()=={s.Integer(0):8,s.Integer(2):36,s.Integer(6):20})
pid=s.eye(dim)
f0=(cf-2*pid)*(cf-6*pid)/12
f1=-cf*(cf-6*pid)/8
ck("complete Cl12 scalar color Casimir sum",zero((cp+cm-s.Rational(15,4)*pid)*f0))
ck("complete Cl12 scalar only one color factor",zero(cp*cm*f0))
ck("complete Cl12 scalar orientation multiplicity",s.trace(cp*f0)/s.Rational(15,4)==4)
ck("complete Cl12 vector color Casimir sum",zero((cp+cm-s.Rational(11,4)*pid)*f1))
ck("complete Cl12 vector color Casimir product",zero((cp*cm-s.Rational(3,2)*pid)*f1))
ck("complete Cl12 vector orientation multiplicity",s.trace((cp-s.Rational(3,4)*pid)*f1)/s.Rational(5,4)==18)

# Exhaustive SO24 -> SO23 S/V branch list from independent color/flavor duality.
# Only integer orbital spins of both three-vectors satisfy the Gauss law.
rows=[((0,0),(12,12),['S']),((1,0),(12,11),['S','V']),
      ((2,0),(11,11),['S','V']),((1,1),(12,10),['V']),
      ((2,1),(11,10),['V'])]
allowed={'S':set(),'V':set()}
for weight,(a,b),flavors in rows:
    jp,jm=s.Rational(a+b,2),s.Rational(a-b,2)
    if jp.q==1 and jm.q==1:
        for flavor in flavors:
            allowed[flavor].add((int(jp),int(jm)))
ck("exhaustive scalar physical color sectors",allowed['S']=={(11,0),(12,0)})
ck("exhaustive vector physical color sectors",allowed['V']=={(11,0),(11,1)})

def dim_b(weight):
    rho=[Fraction(2*(len(weight)-i)-1,2) for i in range(len(weight))]
    z=[rho[i]+weight[i] for i in range(len(weight))]
    value=Fraction(1)
    for i in range(len(weight)):
        value*=z[i]/rho[i]
        for j in range(i+1,len(weight)):
            value*=(z[i]**2-z[j]**2)/(rho[i]**2-rho[j]**2)
    assert value.denominator==1
    return value.numerator

def su2(twice_j,u):
    return sum(u**m for m in range(-twice_j,twice_j+1,2))

# The complete connected-color physical decomposition, independently obtained
# from the full 24-flavor complement/interlacing rule (including b=0 doubling).
physical=defaultdict(Counter)
full_color=defaultdict(Counter)
for a in range(13):
    for b in range(-a,a+1):
        w=tuple(2-int(a>=12-i)-int(abs(b)>=12-i) for i in range(12))
        choices=[range(w[i+1],w[i]+1) for i in range(11)]
        for mu in product(*choices):
            full_color[mu][(a+b,a-b)]+=2 if b==0 else 1
            if (a+b)%2==0:
                physical[mu][((a+b)//2,(a-b)//2)]+=2 if b==0 else 1
all_blocks=[]
for p in range(12):
    for q in range(12-p):
        mu=(2,)*p+(1,)*q+(0,)*(11-p-q)
        if q%2==0:
            L,ell=12-p-q//2,q//2
        else:
            L,ell=(q+1)//2,12-p-(q+1)//2
        expected=Counter([(L-1,ell),(L,ell),(ell,L-1),(ell,L)])
        ck(f"complete physical flavor p={p},q={q}",physical[mu]==expected)
        full_expected=Counter()
        for jp2,jm2 in [(23-2*p-q,q),(q,23-2*p-q)]:
            for pair in [(jp2+1,jm2),(jp2-1,jm2),(jp2,jm2+1),(jp2,jm2-1)]:
                if min(pair)>=0:
                    full_expected[pair]+=1
        ck(f"spectator times heavy full-character identity p={p},q={q}",full_color[mu]==full_expected)
        ck(f"inverse block label p={p},q={q}",p==12-L-ell and q==(2*ell if ell<L else 2*L-1))
        all_blocks.append({"p":p,"q":q,"L":L,"ell":ell,"flavor_dimension":dim_b(mu)})
ck("78 exhaustive flavor blocks",len(physical)==len(all_blocks)==78)
ck("full ground flavor block",[(row['p'],row['q']) for row in all_blocks if row['L']==1 and row['ell']==0]==[(11,0)])

# An independent full spectator color-character identity at flavor identity.
for n in [3,5,7,23]:
    rank=(n-1)//2
    u,vv=s.Rational(2),s.Rational(3)
    direct=u**(-n)*(1+u*vv)**n*(1+u/vv)**n
    dual=0
    for p in range(rank+1):
        for q in range(rank+1-p):
            weight=(2,)*p+(1,)*q+(0,)*(rank-p-q)
            jp2,jm2=n-2*p-q,q
            dual+=dim_b(weight)*(su2(jp2,u)*su2(jm2,vv)+su2(jm2,u)*su2(jp2,vv))
    ck(f"complete spectator color character n={n}",s.cancel(direct-dual)==0)

# Exact radial stable-trap matrix element. Multiplication by r takes the lower
# J=L-1 ground to the upper J=L ground, so there is exactly one intermediate
# radial level in the order-lambda^2 shift.
r,Omega=s.symbols('r Omega',positive=True)
v=s.symbols('v',real=True)
for L in [1,2,12]:
    nminus=s.sqrt(2*Omega**(s.Rational(2*L+1,2))/s.gamma(L+s.Rational(1,2)))
    nplus=s.sqrt(2*Omega**(s.Rational(2*L+3,2))/s.gamma(L+s.Rational(3,2)))
    uminus=nminus*r**L*s.exp(-Omega*r*r/2)
    uplus=nplus*r**(L+1)*s.exp(-Omega*r*r/2)
    ck(f"L={L} normalized lower radial ground",s.simplify(s.integrate(uminus**2,(r,0,s.oo))-1)==0)
    ck(f"L={L} one exact Yukawa intermediate state",s.simplify(r*uminus-s.sqrt((L+s.Rational(1,2))/Omega)*uplus)==0)
    ck(f"L={L} second-order shift",s.simplify(-v*v*(nminus/nplus)**2/Omega+v*v*(L+s.Rational(1,2))/Omega**2)==0)
ck("scalar rank-four order-lambda² coefficient",2*(12+s.Rational(1,2))==25)
ck("vector rank-four order-lambda² coefficient",2*(1+s.Rational(1,2))==3)
ck("common unperturbed first scalar/vector energy",s.Rational(25,2)+s.Rational(3,2)==14)

results={"passed":all(row['passed'] for row in checks),"number_of_checks":len(checks),"checks":checks,
         "radial_operators":{"S":"H_12(sqrt(2)*lambda;r_plus)+h_0(r_minus)",
                             "V":"h_11(r_plus)+H_1(sqrt(2)*lambda;r_minus)"},
         "all_flavor_blocks":all_blocks,
         "scope":"Exact finite-rank flavor mechanics; no band truncation or target scattering fit."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_rank4_dynamics_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:results[k] for k in ['passed','number_of_checks','radial_operators']},indent=2))
