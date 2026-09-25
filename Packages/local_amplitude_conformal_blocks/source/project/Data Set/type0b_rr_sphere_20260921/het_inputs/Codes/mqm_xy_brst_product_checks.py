"""Direct free-block xy product and its full oscillator BRST quotient.

Local operator modes at k=0, alpha=-1, Q_Liouville=2. Mode indices are
doubled integers. This is an exact finite-level Fock calculation, not a
replacement of the interacting bulk OPE by its direct conformal block.
"""
import json
from collections import defaultdict
from pathlib import Path
import sympy as S

I=S.I
R=S.Rational
ORDER={k:n for n,k in enumerate(['c','b','be','ga','ax','ap','fx','fp'])}
ODD={'c','b','fx','fp'}
CUT=5
EV=list(range(-2*CUT,2*CUT+1,2))
OD=list(range(-2*CUT+1,2*CUT,2))

def clean(st):
    return {w:S.expand(v) for w,v in st.items() if S.expand(v)!=0}

def add(*states):
    out=defaultdict(lambda:S.S.Zero)
    for st in states:
        for w,v in st.items(): out[w]+=v
    return clean(out)

def scale(st,a): return clean({w:a*v for w,v in st.items()})

def key(op): return (ORDER[op[0]],-op[1])

def creator(op):
    k,n=op
    if k=='c': return n<=2
    if k=='b': return n<=-4
    return n<0

def contraction(a,b):
    ka,na=a; kb,nb=b
    if na+nb: return 0
    if ka==kb and ka in ('ax','ap'): return R(na,2)
    if ka==kb and ka in ('fx','fp'): return 1
    if (ka,kb) in [('c','b'),('b','c'),('ga','be')]: return 1
    if (ka,kb)==('be','ga'): return -1
    return 0

def one(op,word):
    k,n=op
    if k in ('ax','ap') and n==0:
        return {word:0 if k=='ax' else I}
    if creator(op):
        if k in ODD and op in word: return {}
        place=next((j for j,v in enumerate(word) if key(op)<key(v)),len(word))
        sign=(-1)**sum(v[0] in ODD for v in word[:place]) if k in ODD else 1
        return {word[:place]+(op,)+word[place:]:sign}
    out=defaultdict(lambda:S.S.Zero)
    for j,v in enumerate(word):
        cc=contraction(op,v)
        if cc:
            sign=(-1)**sum(u[0] in ODD for u in word[:j]) if k in ODD else 1
            out[word[:j]+word[j+1:]]+=sign*cc
    return clean(out)

def apply(ops,st):
    out=st
    for op in reversed(ops):
        nxt=defaultdict(lambda:S.S.Zero)
        for w,v in out.items():
            for ww,a in one(op,w).items(): nxt[ww]+=a*v
        out=clean(nxt)
    return out

def state(*ops): return apply(ops,{():S.S.One})

def gterms(rr):
    out=[]
    for n in EV:
        for ak,fk in [('ax','fx'),('ap','fp')]:
            out.append((1,((ak,n),(fk,rr-n))))
    out.append((I*(rr+1),(('fp',rr),)))
    return out

def qop(op):
    k,ss=op
    out=[]
    if k=='c':
        for m in EV:
            nn=ss-m
            out.append((R(m-nn,4),(('c',m),('c',nn))))
        for rr in OD: out.append((-1,(('ga',ss-rr),('ga',rr))))
    elif k=='be':
        out+=gterms(ss)
        for n in EV:
            out.append((-R(ss,2)-R(n,4),(('be',ss-n),('c',n))))
            out.append((-2,(('b',-n),('ga',n+ss))))
    elif k=='ga':
        for n in EV: out.append((R(3*n,4)-R(ss,2),(('c',n),('ga',ss-n))))
    elif k in ('ax','ap'):
        fk='fx' if k=='ax' else 'fp'
        for n in EV:
            out.append((-R(ss,2),(('c',-n),(k,n+ss))))
            if k=='ap' and n+ss==0:
                out.append((I*R(n*(n+2),4),(('c',-n),)))
        for rr in OD: out.append((-R(ss,2),(('ga',-rr),(fk,rr+ss))))
    elif k in ('fx','fp'):
        ak='ax' if k=='fx' else 'ap'
        for n in EV: out.append((-R(n,4)-R(ss,2),(('c',-n),(k,n+ss))))
        for rr in OD:
            out.append((1,(('ga',-rr),(ak,rr+ss))))
            if k=='fp' and rr+ss==0:
                out.append((I*(rr+1),(('ga',-rr),)))
    else:
        raise ValueError('Q on b not needed for these input states')
    return [(a,w) for a,w in out if a]

QVAC=add(scale(state(('c',2),('ap',-2)),I),
          scale(state(('c',2),('be',-1),('ga',-1)),-1),
          scale(state(('c',0)),-1))

def q(st):
    ans={}
    for word,coef in st.items():
        sign=1
        for j,op in enumerate(word):
            for a,ops in qop(op):
                ans=add(ans,scale(apply(word[:j]+ops+word[j+1:],{():1}),coef*sign*a))
            if op[0] in ODD: sign=-sign
        ans=add(ans,scale(apply(word,QVAC),coef*sign))
    return clean(ans)

checks=[]
def check(name,st):
    if clean(st): raise AssertionError((name,clean(st)))
    checks.append(name)

def scalar_check(name,expr):
    if isinstance(expr,S.MatrixBase):
        assert all(S.simplify(v)==0 for v in expr),(name,expr)
    else: assert S.simplify(expr)==0,(name,expr)
    checks.append(name)

# First test the individual converted Ramond representatives. Basis S+,S-.
px=S.Matrix([[0,1],[1,0]])/S.sqrt(2)
pp=S.Matrix([[0,-I],[I,0]])/S.sqrt(2)
for name,kval,spin,flipped,dc in [
    ('x',R(1,2),S.Matrix([0,1]),S.Matrix([1,0]),1/S.sqrt(2)),
    ('y',-R(1,2),S.Matrix([1,0]),S.Matrix([0,1]),-1/S.sqrt(2))]:
    g0=kval*px+3*I*pp/2
    scalar_check(name+' pure gamma0 BRST sector',g0*spin-2*dc*flipped)
    scalar_check(name+' c1 beta_-1 gamma0 BRST sector',-spin/2-dc*g0*flipped)
    # Explicit L_-1−(1/2)G_-1G0 free-field coefficients.
    scalar_check(name+' Ramond null aX coefficient',kval*spin-px*g0*spin/2)
    scalar_check(name+' Ramond null aphi coefficient',I*spin/2-pp*g0*spin/2)
    scalar_check(name+' Ramond null psiX_-1 coefficient',px*spin/2-kval*g0*spin/2)
    scalar_check(name+' Ramond null psiphi_-1 coefficient',pp*spin/2+I*g0*spin/4)

check('BRST square on base vacuum',q(QVAC))
N=state()
U=state(('ax',-2))
A=scale(state(('fx',-1),('fp',-1)),I)
B=scale(state(('ax',-2),('fp',-1)),I)
C=scale(state(('ap',-2),('fx',-1)),I)
D=state(('fx',-3))
W=state(('fx',-1))
O=add(A,apply((('c',2),('be',-1)),add(scale(B,-R(1,2)),scale(C,R(1,2)),scale(D,-R(3,2)))),
      scale(apply((('c',2),('be',-3)),W),R(3,2)))
check('sourced Ohat is BRST closed in free degenerate Fock module',q(O))

# In the explicit Ohat convention used here, x has H charge -1/2 at X
# charge +1/2 (the user notes' H is reversed). The singular coefficient
# fixes product phases (00,01,10,11)=(+,-,+,+). Its BRST preimage is c1 beta² N.
prepole=state(('c',2),('be',-1),('be',-1))
pole=add(N,scale(state(('c',2),('be',-1),('fp',-1)),I),
         scale(state(('c',2),('c',0),('be',-1),('be',-1)),-R(1,2)))
check('singular product coefficient is minus one half Q(prepole)',add(pole,scale(q(prepole),R(1,2))))
check('singular product coefficient is BRST closed',q(pole))
# Determine, rather than assume, the relative component product phases.
s01,s10,s11=S.symbols('s01 s10 s11')
pole_general=add(N,
    scale(state(('c',2),('be',-1),('fx',-1)),(s01+s10)/2),
    scale(state(('c',2),('be',-1),('fp',-1)),I*(-s01+s10)/2),
    scale(state(('c',2),('c',0),('be',-1),('be',-1)),-s11/2))
phase_solutions=S.solve(list(q(pole_general).values()),[s01,s10,s11],dict=True)
assert phase_solutions==[{s01:-1,s10:1,s11:1}],phase_solutions
checks.append('relative component phases fixed by singular BRST closure')

# Remove one half the derivative of the singular coefficient from the
# literal regular product. This change is BRST exact. Direct Taylor
# expansion gives the following symmetrized constant coefficient.
reg=add(scale(add(U,A),R(1,2)),
        scale(apply((('c',2),('be',-1)),B),R(1,2)),
        scale(apply((('c',0),('be',-1)),W),R(1,2)),
        scale(apply((('c',2),('be',-3)),W),R(1,2)),
        scale(apply((('c',2),('c',0),('be',-1),('be',-1)),add(U,scale(A,-1))),-R(1,4)))
check('symmetrized direct xy regular product is BRST closed',q(reg))

# A finite list of useful ghost-minus-one, weight-zero preimages. The short
# explicit preimage below proves the decomposition without a completeness
# assumption about exceptional ghost representatives.
pre=[]
for fk in ['fx','fp']:
    pre.append(state(('be',-1),(fk,-1)))
for mat in [U,state(('ap',-2)),state(('fx',-1),('fp',-1))]:
    pre.append(apply((('c',2),('be',-1),('be',-1)),mat))
pre.extend([state(('c',2),('be',-3),('be',-1)),
            state(('c',0),('be',-1),('be',-1)),
            state(('c',2),('be',-1),('be',-1),('be',-1),('ga',-1))])
for fk in ['fx','fp']:
    pre.append(state(('c',2),('c',0),('be',-1),('be',-1),('be',-1),(fk,-1)))
images=[q(v) for v in pre]
words=sorted(set().union(reg,O,*images),key=str)
cols=images+[O]
mat=S.Matrix([[col.get(w,0) for col in cols] for w in words])
rhs=S.Matrix([reg.get(w,0) for w in words])
sol,params=mat.gauss_jordan_solve(rhs)
zeta=S.simplify(sol[-1])
assert not zeta.free_symbols,('cohomology coefficient not unique',zeta)
reconstructed={}
for a,col in zip(sol,cols): reconstructed=add(reconstructed,scale(col,a))
check('complete direct-product cohomology decomposition',add(reg,scale(reconstructed,-1)))
explicit_pre=add(scale(state(('be',-1),('fx',-1)),-R(1,6)),
                 scale(state(('c',2),('be',-1),('be',-1),('ax',-2)),-R(1,3)),
                 scale(state(('c',2),('c',0),('be',-1),('be',-1),('be',-1),('fx',-1)),-R(1,12)))
check('short exact preimage for symmetrized product',add(reg,scale(O,-R(1,3)),scale(q(explicit_pre),-1)))
check('BRST square on the explicit preimage',q(q(explicit_pre)))
saved_EV,saved_OD=EV,OD
for radius in [3,7]:
    EV=list(range(-2*radius,2*radius+1,2))
    OD=list(range(-2*radius+1,2*radius,2))
    for v,old_image in zip(pre,images):
        assert not clean(add(q(v),scale(old_image,-1))),radius
    assert not q(O) and not q(reg),radius
    checks.append(f'BRST mode sums stable at cutoff {radius}')
EV,OD=saved_EV,saved_OD

out={'passed':True,'checks':checks,'preimage_count':len(pre),
     'matrix_rank':mat.rank(),'image_rank':mat[:,:-1].rank(),
     'zeta_Ohat':str(zeta),'preimage_coefficients':[str(v) for v in sol[:-1]],
     'input_convention':'H_note=-H with the fixed Ohat bosonization; x has (Hcharge,Xmomentum)=(-1/2,+1/2), y=(+1/2,-1/2). Individual BRST closure is checked before the product.',
     'subtracted_regular_product':'R_sub=[x(z)y(0)]_(z^0)-(1/2)partial K; no averaging of the two radial orderings.',
     'phase_scope':'Component phases selected by direct-block BRST consistency; a globally associative Ramond/Klein cocycle extension is not independently certified.',
     'scope':'Conditional direct free conformal block; no multiplication by a separately normalized paired FH coefficient.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_xy_brst_product_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
