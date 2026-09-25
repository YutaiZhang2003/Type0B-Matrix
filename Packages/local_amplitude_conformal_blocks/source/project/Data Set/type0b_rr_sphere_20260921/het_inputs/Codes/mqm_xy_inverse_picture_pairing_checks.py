"""Independent finite-level recombination and explicit Ising pairing diagnostic.

This file imports no peer BRST implementation.  The three differential images
are the independently ordered mode identities written in the companion memo.
The Ising calculation is an explicit locally consistent embedding; it does not
prove transport of the full resonant heterotic product or its contour action.
"""
from pathlib import Path
import json
import sympy as s

I=s.I
r=s.Rational
q=s.sqrt(2)
checks=[]

def eq(name,x,y=0):
    z=x-y
    vals=list(z) if isinstance(z,s.MatrixBase) else [z]
    assert all(s.simplify(v)==0 for v in vals),(name,z)
    checks.append(name)

# Ordered output basis:
# U,A,c beta B,c beta C,c beta D,c0 beta W,c beta_-3/2 W,
# c beta² gamma W,c c0 beta² U,c c0 beta² A.
O=s.Matrix([0,1,-r(1,2),r(1,2),-r(3,2),0,r(3,2),0,0,0])
R=s.Matrix([r(1,2),r(1,2),r(1,2),0,0,r(1,2),r(1,2),0,-r(1,4),r(1,4)])
Qbw=s.Matrix([1,-1,0,1,1,0,0,-1,0,0])
Qcb2U=s.Matrix([-2,0,-2,0,-2,0,0,-1,0,0])
Qcc0b3w=s.Matrix([0,0,0,0,0,-6,0,6,3,-3])
eq("short BRST preimage recombination",
   R-O/3,-Qbw/6-Qcb2U/3-Qcc0b3w/12)
z,a,b,c=s.symbols("z a b c")
solution=s.solve(list(R-z*O-a*Qbw-b*Qcb2U-c*Qcc0b3w),(z,a,b,c),dict=True)
assert solution==[{z:r(1,3),a:-r(1,6),b:-r(1,3),c:-r(1,12)}],solution
checks.append("unique coefficient in the displayed short decomposition")

# Inverse PCO's two unexpanded ghost pieces.  All powers below are fixed by
# e^{2varphi}(z)e^{-2varphi}(0)~z^4.
zz=s.symbols("z",positive=True)
first=zz**4 * (1/zz) * (-2/zz**3) * (-1)
undifferentiated= -zz**4*(1/zz)*(1/zz**2)
eq("inverse PCO first ghost term",first,2)
eq("inverse PCO derivative ghost term",s.diff(undifferentiated,zz),-1)
eq("inverse PCO identity normalization",first+s.diff(undifferentiated,zz),1)

# Ising spin labels +,- are even,odd.  B is the identity OPE coefficient,
# F and Fbar the coefficients of psi and bar-psi.  Tensor products are graded.
B=s.diag(1,I)
F=s.Matrix([[0,I/q],[1/q,0]])
Fb=s.Matrix([[0,-1/q],[-I/q,0]])
def paired(A,C):
    M=s.zeros(4)
    for a0 in range(2):
        for b0 in range(2):
            for c0 in range(2):
                for d0 in range(2):
                    M[2*a0+b0,2*c0+d0]=(-1)**(b0*c0)*A[a0,c0]*C[b0,d0]
    return M

BB=paired(B,B)
for ix,target in enumerate([1,I,I,1]):
    eq("FH paired free norm "+str(ix),BB[ix,ix],target)
FF=paired(F,Fb)
for ix,jx,target in [(0,3,1/(2*I)),(3,0,1/(2*I)),(1,2,-r(1,2)),(2,1,-r(1,2))]:
    eq("FH opposite-spin direct coefficient "+str((ix,jx)),FF[ix,jx],target)

# First insertion is in the root's C+ Clifford module; the second insertion
# uses its BPZ-dual C- module.  The last vector includes the -i Klein phase
# needed to reproduce the four ordinary bosonized H exponential products.
x=s.Matrix([1,0,0,I])/q       # first H-
xc=s.Matrix([0,I,1,0])/q     # first H+
y=s.Matrix([1,0,0,-I])/q     # second H+
yc=-I*s.Matrix([0,-I,1,0])/q # second H-

for label,v,w,bi,xx,xp,pp in [
    ("00",x,y,1,0,0,I/2),
    ("01",x,yc,0,1/q,-I/q,0),
    ("10",xc,y,0,1/q,I/q,0),
    ("11",xc,yc,1,0,0,-I/2)]:
    for name,A,C,target in [("identity",B,B,bi),("psiX",F,B,xx),("psiL",B,F,xp),("psiXpsiL",F,F,pp)]:
        eq("bosonized spin product "+label+" "+name,(v.T*paired(A,C)*w)[0],target)

# Fix the antiholomorphic Ramond signs to + at the first and - at the second
# insertion.  FH then fixes the two holomorphic opposite-sign OPE weights:
# Theta++ Theta-- -> 1/(2i), Theta-+ Theta+- -> -1/2.
FH=s.Matrix([[0,1/(2*I)],[-r(1,2),0]])
eq("FH block matrix equals F times fixed right odd coefficient",FH,F*Fb[0,1])
rho=s.simplify(Fb[0,1]/I) # bar g=i bar psi N
eq("right odd g normalization",rho,I/q)

v01=s.simplify((x.T*paired(B,FH)*yc)[0])
v10=s.simplify((xc.T*paired(B,FH)*y)[0])
eq("vacuum cross component 01",v01,I/2)
eq("vacuum cross component 10",v10,-I/2)
# Raw exponential coefficients are -1/sqrt2 in each ghost correction.
# BRST-selected component cocycle signs are (1,-1,+1,+1).
vac=s.simplify((v01-v10)/q)
eq("vacuum Y coefficient divided by tilde Cminus",vac,rho)

raw00=s.simplify((x.T*paired(F,FH)*y)[0])
# psiX g bar-g = -psiX psiL bar-psiL, at a=-1.
eq("direct leading A bar-g calibration",-raw00,rho/2)
eq("same-sign SL block ++ calibration",B[0,0]*Fb[0,1],-1/q)
eq("same-sign SL block -- calibration",B[1,1]*Fb[0,1],-I/q)

# This algebraic ratio assumes isolated leading-normalized exact-limit
# channels can be acted on separately; the memo explicitly does not assume
# that this is the completed resonant charge prescription.
k=s.symbols("k",nonzero=True)
isolated=s.simplify(r(1,3)*k**2/r(4)*(-4/k**2))
eq("conditional isolated fixed-dual ratio",isolated,-r(1,3))

out={"passed":True,"checks":checks,"rho":str(rho),
     "zeta_in_short_BRST_quotient":"1/3","inverse_PCO_identity":"1",
     "vacuum_Y_coefficient_over_tilde_Cminus":str(vac),
     "isolated_fixed_dual_ratio":str(isolated),
     "scope":"Local FH-compatible Ising/BPZ/Klein embedding and exact finite-level identities. No full resonant heterotic product, absolute charge cancellation, or contour-domain statement."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_xy_inverse_picture_pairing_results.json').write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
