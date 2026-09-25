"""Independent rational/Ward audit of the (2,2) even-floor calculation.
No import of the repository NS algebra. This is a check of chiral relative
coefficients, not a computation of a paired heterotic OPE/PCO coefficient.
"""
import json
from pathlib import Path
import sympy as s

R=s.Rational
H=-R(3,2)
c=R(27,2)
h_g=H+R(1,2)
checks=[]

def check(name, value):
    if isinstance(value, s.MatrixBase):
        ok=all(s.simplify(v)==0 for v in value)
    else:
        ok=s.simplify(value)==0
    if not ok:
        raise AssertionError((name,value))
    checks.append(name)

# Contravariant forms, derived directly by the SCA commutators.
normg=2*H
gram_null=s.Matrix([[2*H+2*c/3,4*H],
                    [4*H,2*h_g*normg]])
check("level-3/2 exact singular vector", gram_null*s.Matrix([1,1]))
check("level-3/2 quotient rank",gram_null.rank()-1)
gram=s.Matrix([
 [2*H+(c/3)*(R(25,4)-R(1,4)), R(7,2)*normg,6*normg],
 [R(7,2)*normg,(4*h_g+c/2)*normg,6*h_g*normg],
 [6*normg,6*h_g*normg,4*h_g*(2*h_g+1)*normg]
])
js=s.Matrix([1,2,R(3,2)])
check("independent jSL norm",(js.T*gram*js)[0]+24)
check("jSL L1 image is 3 chi",s.Matrix([[3,0,0],[0,3,-2]])*js-s.Matrix([3,3]))
check("jSL L2 image",s.Matrix([[R(7,2),R(11,4),-6]])*js)

# Primaries after the exact SL null, before the critical BRST quotient.
P=s.Matrix([[-3,3,-2,0,0],
            [R(7,2),R(11,4),-6,R(1,2),R(23,4)]])
vSL=s.Matrix([1,2,R(3,2),0,0])
vX=s.Matrix([0,R(2,25),R(3,25),1,0])
vF=s.Matrix([0,R(23,25),R(69,50),0,1])
vII=s.Matrix([0,1,R(3,2),1,1])
for name,v in [("jSL",vSL),("jX",vX),("jf",vF),("type II",vII)]:
    check(name+" is primary",P*v)
check("scalar primary dimension",len(P.nullspace())-3)
check("critical type-II direction",vX+vF-vII)
adjP=s.Matrix([[2,-2,2,0]])
adjremove=s.Matrix.hstack(s.Matrix([1,1,0,0]),s.Matrix([0,1,1,0]))
check("adjoint null and derivative are primary",adjP*adjremove)
check("adjoint quotient is one-dimensional",4-adjP.rank()-adjremove.rank()-1)
# Q g = c dg - dc g. The following independently normal-ordered actions
# use {Q,b_-2}=L_-2^tot and L_-2^gh|0>=-2 b_-2 c_0|0>-b_-3 c_1|0>.
# Coordinates: c1 Lm_-2 g, c1 Lm_-1^2 g, c_-1 g,
# b_-2 c1 c0 g. The cubic-ghost contribution cancels in Q(b_-2 c1 g).
Qbc=s.Matrix([1,0,3,2])-s.Matrix([0,0,0,2])
Qdg=s.Matrix([0,1,-2,0])
check("explicit type-II BRST preimage",Qbc+R(3,2)*Qdg-s.Matrix([1,R(3,2),0,0]))
check("critical L2 annihilation",4*h_g+26/2+R(3,2)*6*h_g)
check("weight-zero derivative preimage",h_g+1)

# Direct block using Ising global descendants and exponential Taylor series.
ising_a1=R(1,2)
ising_a2=(R(1,2)*R(3,2))/(2*1*2)
check("Ising odd level-two descendant",ising_a2-R(3,16))
# Coordinates psi'', phi' psi', phi'' psi, phi'^2 psi.
direct=s.Matrix([ising_a2,-ising_a1/2,-R(1,4),R(1,8)])
t=s.Matrix([-R(3,2),1,1,0])
l2=s.Matrix([R(3,4),0,0,-R(1,2)])
l11=s.Matrix([1,-2,-1,1])
jsfree=t+2*l2+R(3,2)*l11
check("direct-block superconformal coefficient",
      direct-(l2/4+3*l11/8-jsfree/4))
check("direct block leading exponent",h_g-2*(-R(9,16))-R(1,8))
check("level-two Virasoro L1 equation",3/R(4)-2*R(3,8))
check("level-two Virasoro L2 equation",
      R(1,4)*R(11,4)+R(3,8)*(-6)-(h_g-R(9,16)))
check("time scalar stress coefficient",2*R(1,8)-R(1,4))
check("spectator scalar stress coefficient",2*R(23,16)/R(23,2)-R(1,4))

# Independent meromorphic G Ward functions; the odd intertwiner N_d is
# odd and its descendant g has even intertwiner parity.
z,w,u,h,k=s.symbols("z w u h k", nonzero=True)
Hg=s.symbols("Hg")
Bz=z**(-Hg-R(1,2))
K=2*h+Hg-R(1,2)
check("external-descendant g Ward tensor",
      z*s.diff(Bz,z)+2*Hg*Bz+2*h*Bz-K*Bz)
Fp=Bz*(1/(w-z)-1/w)
Fd=Bz*(K/(w-z)-(Hg-R(1,2))/w-2*h*z/w**2)
def local_regular_coeff(F,j):
    pole=s.limit(u*F.subs(w,z+u),u,0)
    regular=s.cancel(F.subs(w,z+u)-pole/u)
    return s.simplify(s.diff(regular,u,j).subs(u,0)/s.factorial(j)/Bz)
check("primary G_-5/2 form factor",local_regular_coeff(Fp,1)-1/z**2)
check("descendant G_-5/2 form factor",
      local_regular_coeff(Fd,1)-(4*h+Hg-R(1,2))/z**2)
check("primary SL null decoupling",
      (local_regular_coeff(Fp,0)+s.diff(Bz,z)/Bz).subs(Hg,H))
check("descendant SL null decoupling",
      (local_regular_coeff(Fd,0)+K*s.diff(Bz,z)/Bz).subs(Hg,H))
check("descendant supercurrent pole at zero",
      s.limit(w*w*Fd/Bz,w,0)+2*h*z)
check("descendant supercurrent residue at infinity",
      s.limit(w*Fd/Bz,w,s.oo)-2*h)
check("second derivative form factor vanishes",
      (s.diff(Bz,z,2)/Bz).subs(Hg,H))
rgP=s.S.One
rgD=K.subs(Hg,H)
rtP=s.S.One
rtD=(4*h+Hg-R(1,2)).subs(Hg,H)
rlP=h+h_g
rlD=(h+R(1,2)+h_g)*rgD
check("descendant g coefficient",rgD-2*(h-1))
check("descendant L_-2 coefficient",rlD-(h-1)*(2*h-1))
rjP=rtP+2*rlP
rjD=rtD+2*rlD
check("same normalized scalar action",rjP-rjD/(2*h))
hphys=(1-k*k)/2
for label,rg,rl,rj,metric,hf in [
 ("V",rgP,rlP,rjP,1,R(1,2)),("S",rgD,rlD,rjD,2*h,0)]:
    check(label+" on-shell jSL action",(rj/metric).subs(h,hphys)+k*k)
    jx=(k*k*rg/2+R(2,25)*rl)/metric
    jf=(hf*rg+R(23,25)*rl)/metric
    check(label+" on-shell type-II decoupling",(jx+jf).subs(h,hphys))
    expected=(23*k*k-2)/50 if label=="V" else 23*k*k*(k*k+1)/(50*(k*k-1))
    check(label+" unused stress-transfer form factor",jx.subs(h,hphys)-expected)

result={"passed":True,"exact_checks":len(checks),"checks":checks,
        "independent_jSL_gram":[[str(x) for x in row] for row in gram.tolist()],
        "scope":"Relative chiral coefficients only; no paired-bulk/holomorphic O22/PCO normalization.",
        "normalization_domain":"2h != 0; physical k=iE with real E is nonsingular.",
        "method":"Direct SCA commutators, explicit ghost normal ordering, Ising/exponential direct block, meromorphic odd-intertwiner Ward functions. No repository algebra import."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_even_floor_independent_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

