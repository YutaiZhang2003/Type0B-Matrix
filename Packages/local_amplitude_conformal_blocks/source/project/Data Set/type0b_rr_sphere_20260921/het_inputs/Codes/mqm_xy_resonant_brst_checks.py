"""Transport the exact-b=1 xy quotient off its degenerate momentum.

The analytic section is explicitly fixed; it is not an off-shell definition
of the full heterotic charge or a calculation of a bulk contact term.
"""
import json
from pathlib import Path
import sympy as S

I = S.I
R = S.Rational
eps = S.symbols('epsilon')
a = -1 + eps
source = (Path(__file__).resolve().parents[1] / 'Codes/mqm_xy_brst_product_checks.py').read_text()
source = source.split('checks=[]')[0]
source = source.replace('CUT=5', 'CUT=3')
source = source.replace("return {word:0 if k=='ax' else I}",
                        "return {word:0 if k=='ax' else -I*ALPHA}")
start = source.index('QVAC=')
end = source.index('\ndef q(st):', start)
source = source[:start] + '''QVAC=add(
    scale(state(('c',2),('ap',-2)),-I*ALPHA),
    scale(state(('c',2),('be',-1),('ga',-1)),-1),
    scale(state(('c',0)),(ALPHA*(2-ALPHA)+1)/2))
''' + source[end:]
ns = {'ALPHA': a}
exec(source, ns)
state,add,scale,apply,q,clean = [ns[k] for k in ('state','add','scale','apply','q','clean')]

def sub(st, value):
    return clean({w: S.expand(v.subs(eps,value)) for w,v in st.items()})

def derivative(st):
    return clean({w: S.expand(S.diff(v,eps).subs(eps,0)) for w,v in st.items()})

checks=[]
def zero(name, st):
    assert not clean(st), (name,clean(st))
    checks.append(name)

def eq(name, x, y): zero(name,add(x,scale(y,-1)))

U=state(('ax',-2))
v=state(('fx',-1))
A=scale(state(('fx',-1),('fp',-1)),-I*a)
B=scale(state(('ax',-2),('fp',-1)),-I*a)
C=scale(state(('ap',-2),('fx',-1)),-I*a)
D=state(('fx',-3))
O=add(A,apply((('c',2),('be',-1)),add(scale(B,-R(1,2)),scale(C,R(1,2)),scale(D,-R(3,2)))),
      scale(apply((('c',2),('be',-3)),v),R(3,2)))
reg=add(scale(add(U,A),R(1,2)),
        scale(apply((('c',2),('be',-1)),B),R(1,2)),
        scale(apply((('c',0),('be',-1)),v),R(1,2)),
        scale(apply((('c',2),('be',-3)),v),R(1,2)),
        scale(apply((('c',2),('c',0),('be',-1),('be',-1)),add(U,scale(A,-1))),-R(1,4)))
chi=add(scale(state(('be',-1),('fx',-1)),-R(1,6)),
        scale(state(('c',2),('be',-1),('be',-1),('ax',-2)),-R(1,3)),
        scale(state(('c',2),('c',0),('be',-1),('be',-1),('be',-1),('fx',-1)),-R(1,12)))
defect=add(reg,scale(O,-R(1,3)),scale(q(chi),-1))
zero('quotient defect vanishes at epsilon zero',sub(defect,0))
zero('Ohat closure at epsilon zero',sub(q(O),0))
zero('regular-product closure at epsilon zero',sub(q(reg),0))
zero('generic-momentum BRST square on vacuum',q(ns['QVAC']))
zero('generic-momentum BRST square on short preimage',q(q(chi)))
first=derivative(defect)
assert first
checks.append('first-order quotient defect is nonzero')
weight=S.expand(a*(2-a)/2+R(3,2))
Z=add(scale(state(('c',0),('be',-1),('fx',-1)),R(1,6)),
      scale(state(('c',2),('c',0),('be',-1),('be',-1),('ax',-2)),-R(1,3)))
eq('exact defect equals output weight times the short c0 polynomial',defect,scale(Z,weight))
image1=add(U,scale(A,-1),apply((('c',2),('be',-1)),add(C,D)),
           scale(state(('c',2),('be',-1),('be',-1),('ga',-1),('fx',-1)),-1),
           scale(state(('c',0),('be',-1),('fx',-1)),weight))
image2=add(scale(U,-2),scale(apply((('c',2),('be',-1)),add(B,D)),-2),
           scale(state(('c',2),('be',-1),('be',-1),('ga',-1),('fx',-1)),-1),
           scale(state(('c',2),('c',0),('be',-1),('be',-1),('ax',-2)),-weight))
image3=add(scale(state(('c',0),('be',-1),('fx',-1)),-6),
           scale(state(('c',2),('be',-1),('be',-1),('ga',-1),('fx',-1)),6),
           scale(apply((('c',2),('c',0),('be',-1),('be',-1)),add(U,scale(A,-1))),3))
eq('first generic primary BRST image',q(state(('be',-1),('fx',-1))),image1)
eq('second generic primary BRST image',q(state(('c',2),('be',-1),('be',-1),('ax',-2))),image2)
eq('third generic primary BRST image',q(state(('c',2),('c',0),('be',-1),('be',-1),('be',-1),('fx',-1))),image3)
assert derivative(q(O))
checks.append('first-order Ohat closure defect is nonzero')

# The original quotient is absolute chiral. These identities do not by
# themselves obstruct its local descended contour equivalence.
b0chi=apply((('b',0),),chi)
b0reg=apply((('b',0),),reg)
eq('b0 chi',b0chi,scale(state(('c',2),('be',-1),('be',-1),('be',-1),('fx',-1)),R(1,12)))
expected_b0reg=add(scale(state(('be',-1),('fx',-1)),R(1,2)),
                   scale(apply((('c',2),('be',-1),('be',-1)),add(U,scale(A,-1))),R(1,4)))
eq('b0 regular product',b0reg,expected_b0reg)
zero('Ohat is relative along chosen section',apply((('b',0),),O))
zero('at epsilon zero b0 defect is Q exact',sub(add(b0reg,q(b0chi)),0))

def serial(st):
    return {str(w):str(S.factor(x)) for w,x in sorted(st.items(),key=lambda z:str(z[0]))}

out={'passed':True,'checks':checks,
     'section':'a=-1+epsilon; g_a=G_-1/2 N_a=-ia psi_phi N_a; same displayed oscillator-polynomial coefficients',
     'weight':str(weight),
     'first_order_quotient_defect':serial(first),
     'exact_quotient_defect':serial(defect),
     'first_order_Ohat_BRST_variation':serial(derivative(q(O))),
     'first_order_regular_BRST_variation':serial(derivative(q(reg))),
     'scope':'Tests noncommutation risk; does not calculate a paired bulk residue, its correction factor, or the actual analytically continued xy intertwiner.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_xy_resonant_brst_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
