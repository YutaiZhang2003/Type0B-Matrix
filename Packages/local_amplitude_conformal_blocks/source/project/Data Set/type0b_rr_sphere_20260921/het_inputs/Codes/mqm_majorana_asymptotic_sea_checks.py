"""Independent root reduction / CAR / asymptotic checks for canonical SO MQM."""
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import json
import sympy as s

checks=[]
def eq(name,a,b=0):
    z=a-b
    vals=list(z) if isinstance(z,s.MatrixBase) else [z]
    assert all(s.simplify(v)==0 for v in vals),(name,z)
    checks.append(name)

# Weyl denominators of the adjoint B/D matrix orbit have no scalar potential
# after multiplication by Delta, because their flat Cartan Laplacian vanishes.
for rank in (1,2,3):
    x=s.symbols("x:"+str(rank))
    delta=s.prod(x[i]**2-x[j]**2 for i in range(rank) for j in range(i+1,rank))
    for kind,d in (("D",delta),("B",s.prod(x)*delta)):
        eq("harmonic Weyl denominator "+kind+str(rank),sum(s.diff(d,v,2) for v in x))

# Orthonormal real root planes in so(4), with norm -Tr(T²)/2.
def T(i,j,n=4):
    z=s.zeros(n); z[i,j]=1; z[j,i]=-1
    return z
def comm(a,b):return a*b-b*a
x1,x2=s.symbols("x1 x2")
X=x1*T(0,1)+x2*T(2,3)
roots={
 "minus":((T(0,2)+T(1,3))/s.sqrt(2),(T(0,3)-T(1,2))/s.sqrt(2),x1-x2),
 "plus":((T(0,2)-T(1,3))/s.sqrt(2),(T(0,3)+T(1,2))/s.sqrt(2),x1+x2)}
for name,(A,B,gap) in roots.items():
    eq(name+" root A normalization",-s.trace(A*A)/2,1)
    eq(name+" root B normalization",-s.trace(B*B)/2,1)
    eq(name+" root orthogonality",-s.trace(A*B)/2)
    eq(name+" orbit metric", -s.trace(comm(A,X)*comm(A,X))/2,gap**2)
    eq(name+" ad square",comm(X,comm(X,A)),-gap**2*A)

# Sparse exact fermion actions. The first site's p modes precede the second.
def clean(v): return {b:c for b,c in v.items() if c}
def add(*vs):
    out=defaultdict(int)
    for v in vs:
        for b,c in v.items():out[b]+=c
    return clean(out)
def scale(v,a):return clean({b:a*c for b,c in v.items()})
def f(v,j,create=False):
    out={}
    for bits,c in v.items():
        occupied=(bits>>j)&1
        if occupied==create:continue
        sign=(-1)**((bits&((1<<j)-1)).bit_count())
        out[bits^(1<<j)]=sign*c
    return out
def chain(v,ops):
    for j,cr in reversed(ops):v=f(v,j,cr)
    return v
def E(v,p,i,j):
    return add(*(chain(v,[(i*p+a,True),(j*p+a,False)]) for a in range(p)))
def D(v,p,dag=False):
    return add(*(chain(v,[(p+a,True),(a,True)] if dag else [(a,False),(p+a,False)])
                 for a in range(p)))
def Pspin(v,p):
    return add(*(chain(v,[(a,True),(b,False),(p+b,True),(p+a,False)])
                 for a in range(p) for b in range(p)))
def Qspin(v,p):
    return add(*(chain(v,[(a,True),(b,False),(p+a,True),(p+b,False)])
                 for a in range(p) for b in range(p)))
def same(a,b):return not add(a,scale(b,-1))
def inner(a,b):return sum(s.conjugate(v)*b.get(k,0) for k,v in a.items())

total_states=0
for p,k in ((2,1),(3,2),(4,2),(5,3),(6,3)):
    combos=list(combinations(range(p),k))
    for aa in combos:
        for bb in combos:
            bits=sum(1<<a for a in aa)+sum(1<<(p+b) for b in bb)
            v={bits:1}
            q=D(D(v,p),p,True)
            assert same(q,Qspin(v,p))
            dd=D(D(v,p,True),p)
            assert same(add(dd,scale(q,-1)),scale(v,p-2*k))
            ee=E(E(v,p,1,0),p,0,1)
            assert same(ee,add(scale(v,k),scale(Pspin(v,p),-1)))
            total_states+=1
    checks.append(f"CAR root identities all neutral states p={p},k={k}")
    # A simple aligned Slater vector has zero difference-root energy,
    # but a nontrivial image-root fluctuation.
    bits=sum(1<<a for a in range(k))+sum(1<<(p+a) for a in range(k))
    v={bits:1};q=D(D(v,p),p,True)
    eq(f"aligned Q mean p={p}",inner(v,q),k)
    eq(f"aligned Q variance p={p}",inner(q,q)-k*k,k*(p-k))
    assert not E(v,p,0,1) and not E(v,p,1,0)
    checks.append(f"aligned difference root kernel p={p}")

# For odd spectator p and k=(p+1)/2, the SO(p) singlet is expensive in the
# image-root interaction, whereas the Cartan-product highest state saturates
# both positive bounds P<=k and Q>=1.
for p in (3,5,7):
    k=(p+1)//2
    singlet={sum(1<<a for a in aa)+sum(1<<(p+a) for a in aa):1
             for aa in combinations(range(p),k)}
    assert same(Pspin(singlet,p),scale(singlet,k))
    assert same(Qspin(singlet,p),scale(singlet,k*k))
    checks.append(f"SO({p}) singlet P=k Q=k²")
    # A local highest wedge consists of isotropic vectors e_(2j)+i e_(2j+1)
    # and the final real direction. Overall normalization is unnecessary.
    highest={0:1}
    for site in (1,0):
        highest=f(highest,site*p+p-1,True)
        for j in reversed(range((p-1)//2)):
            highest=add(f(highest,site*p+2*j,True),
                        scale(f(highest,site*p+2*j+1,True),s.I))
    assert same(Pspin(highest,p),scale(highest,k))
    assert same(Qspin(highest,p),highest)
    checks.append(f"SO({p}) Cartan-product highest P=k Q=1")
    eq(f"SO({p}) maximal pair Casimir",
       2*k*(p-k)+2*(k-1),s.Rational((p-1)*(p+3),2))
blocks=s.symbols("r",integer=True,positive=True)
eq("SO23 all-block maximal Casimir",132*blocks+11*blocks*(blocks-1),11*blocks*(blocks+11))
eq("SO23 Cartan-product Casimir from highest weight",
   sum(blocks*(blocks+23-2*j) for j in range(1,12)),11*blocks*(blocks+11))

# Exact heavy-pair leakage, checked in the FULL neutral Fock space before
# compression. Odd p spectators plus one heavy flavor have total 2k modes.
# Different root pairs land in orthogonal heavy-occupation sectors.
def Dpair(v,stride,i,j,flavors,dag=False):
    return add(*(chain(v,[(j*stride+a,True),(i*stride+a,True)] if dag
                         else [(i*stride+a,False),(j*stride+a,False)])
                 for a in flavors))
def Epair(v,stride,i,j,flavors):
    return add(*(chain(v,[(i*stride+a,True),(j*stride+a,False)]) for a in flavors))
leakage_states=0
for p,r in ((3,2),(3,3),(5,2)):
    stride=p+1;k=stride//2
    pairs=list(combinations(range(r),2))
    from itertools import product
    basis=[{sum(1<<(site*stride+a) for site,aa in enumerate(occupations) for a in aa):1}
           for occupations in product(list(combinations(range(p),k)),repeat=r)]
    # A complex superposition additionally tests interference within each
    # spectator contraction, rather than only occupation-state diagonals.
    basis.append(add(*(scale(v,s.I**j) for j,v in enumerate(basis))))
    heavy_mask=sum(1<<(i*stride+p) for i in range(r))
    for v in basis:
        off={};expected={};rhs=0
        for weight,(i,j) in enumerate(pairs,2):
            d=Dpair(v,stride,i,j,range(stride))
            dd=Dpair(Dpair(v,stride,i,j,range(stride),True),stride,i,j,range(stride))
            q=Dpair(d,stride,i,j,range(stride),True)
            image=scale(add(q,dd),s.Rational(1,2))
            off=add(off,scale({b:c for b,c in image.items() if b&heavy_mask},weight))
            ds=Dpair(v,stride,i,j,range(p))
            expected=add(expected,scale(Dpair(ds,stride,i,j,[p],True),weight))
            rhs+=weight**2*inner(ds,ds)
            # Full difference-root energy has no heavy output.
            difference=add(Epair(Epair(v,stride,j,i,range(stride)),stride,i,j,range(stride)),
                           Epair(Epair(v,stride,i,j,range(stride)),stride,j,i,range(stride)))
            assert not {b:c for b,c in difference.items() if b&heavy_mask}
        assert same(off,expected)
        assert s.simplify(inner(off,off)-rhs)==0
        assert rhs >= sum(w*w for w in range(2,len(pairs)+2))*inner(v,v)
        leakage_states+=1
    checks.append(f"full heavy leakage identity and strict bound p={p},r={r}")

# Matrix-valued shifts cannot be simultaneous canonical translations:
# in the N=3 irreducible heavy factor T=-lambda*sigma.
lam,omega,E0,z=s.symbols("lambda omega E z",positive=True)
sx=s.Matrix([[0,1],[1,0]]);sy=s.Matrix([[0,-s.I],[s.I,0]]);sz=s.diag(1,-1)
eq("noncommuting heavy linear coefficients",comm(-lam*sx,-lam*sy),2*s.I*lam**2*sz)

# Both asymptotic branches have real large-x momentum; its Laurent series
# has a branch-dependent linear phase and a common shifted logarithm.
for sigma in (-1,1):
    p_trial=omega/z-sigma*lam/omega+(E0/omega-lam**2/(2*omega**3))*z
    exact_square=omega**2/z**2-2*sigma*lam/z+2*E0
    eq("WKB square through constant sigma="+str(sigma),
       s.series(p_trial**2-exact_square,z,0,1).removeO())
    eq("common branch energy shift sigma="+str(sigma),
       -omega**2*(x1-sigma*lam/omega**2)**2/2+lam**2/(2*omega**2),
       -omega**2*x1**2/2+sigma*lam*x1)

out={"passed":True,"checks":checks,"neutral_Fock_states_tested":total_states,
     "full_neutral_heavy_leakage_states_tested":leakage_states,
     "p23_k12_image_bound":"Q=Ddag D=1+D Ddag >=1",
     "p23_two_site_singlet_image_eigenvalue":144,
     "p23_two_site_Cartan_product_image_eigenvalue":1,
     "scope":"Exact finite root/CAR algebra and asymptotic expansion; no sea state, integrability, scattering dictionary or large-rank uniform estimate."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_asymptotic_sea_results.json').write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
