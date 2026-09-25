"""Full-coordinate N=2 KZ/source first-order intertwiner checks."""
import json
from pathlib import Path
import sympy as s

passed = []


def check(name, value):
    if isinstance(value, s.MatrixBase):
        ok = all(s.factor(z) == 0 for z in value)
    elif isinstance(value, bool):
        ok = value
    else:
        ok = s.factor(value) == 0
    assert ok, name
    passed.append(name)


n, x, y = s.symbols("n x y", positive=True)
alpha = 1/(n-1)
r = x-y
I = s.eye(2)
G = s.diag(1,n-1)
O = s.diag(-(n-1),-1)
OS1 = s.Matrix([[0,n-1],[1,-(n-2)]])/2
OS2 = s.Matrix([[0,-(n-1)],[-1,-(n-2)]])/2
B = I+alpha*O
C1 = I/2+alpha*OS1
C2 = I/2+alpha*OS2
Ax = B/r+C1/x
Ay = -B/r+C2/y
V0 = ((Ax.diff(x)+Ay.diff(y)+Ax*Ax+Ay*Ay)/2).applyfunc(s.factor)
m = s.Matrix([x+y,x-y])
check("horizontal polynomial x", m.diff(x)-Ax*m)
check("horizontal polynomial y", m.diff(y)-Ay*m)
check("V0 null line", V0*m)
check("V0 spin metric", G*V0-V0.T*G)
check("V0 homogeneous", x*V0.diff(x)+y*V0.diff(y)+2*V0)
omega=s.symbols("omega",positive=True)
ground=s.exp(-omega*(x*x+y*y)/2)
radial=(1-omega*(x*x+y*y))*ground
commutator=-m.diff(x)*s.diff(radial,x)-m.diff(y)*s.diff(radial,y)
check("domain-admissible excited radial multiplier failure",
      commutator-omega*m*radial-2*omega*m*ground)
com=(x+y)*ground
check("center-of-mass multiplier interior failure",
      -m.diff(x)*s.diff(com,x)-m.diff(y)*s.diff(com,y)
      -omega*m*com+s.Matrix([2,0])*ground)


def pure(A, casimir):
    return 2*casimir*I/3+(n-2)*A/3+(n-4)*A*A/(3*(n-1))


def cross(A, B, other):
    return 4*other/3+(n-4)*(A*B+B*A)/(3*(n-1))


K = (2*pure(O,n-1)/r**2
     +pure(OS1,n*(n-1)/8)/x**2+pure(OS2,n*(n-1)/8)/y**2
     +cross(O,OS1,OS2)/(r*x)-cross(O,OS2,OS1)/(r*y)).applyfunc(s.factor)
ratio = 2*(n-1)*(n+2)/3
check("complete null potential proportionality", K+ratio*V0)
check("complete null potential kernel", K*m)
check("special positive coefficient", 1/ratio-3/(2*(n-1)*(n+2)))

# Translation coefficients of a first-order Euclidean Killing symbol.
u1,u2,v1,v2 = s.symbols("u1 u2 v1 v2")
u=s.Matrix([u1,u2]); v=s.Matrix([v1,v2])
curl=(V0*u).diff(y)-(V0*v).diff(x)
# Three leading pole constraints reduce u=A(1,1), v=A(1,-1).
candidate_curl=((V0*s.Matrix([1,1])).diff(y)
                -(V0*s.Matrix([1,-1])).diff(x)).applyfunc(s.factor)
expected=s.Matrix([
    -(n-2)*(x*x+y*y)/(4*x*x*y*y*(n-1)*(x-y)),
    (n-2)*(x+y)*(x*x+y*y)/(4*x*x*y*y*(n-1)**2*(x-y)**2),
])
check("translation residual curl", candidate_curl-expected)
coeffs=[]
for z in curl.subs(n,23):
    coeffs += s.Poly(s.together(z).as_numer_denom()[0],x,y).coeffs()
M,_=s.linear_eq_to_matrix(coeffs,[u1,u2,v1,v2])
check("n23 translation integrability rank", M.rank()==4)

# Rotation-symbol integrability follows from homogeneity, but its
# zero-order equation fails because the null line turns with angle.
w=s.Matrix(s.symbols("w1 w2"))
rot_curl=(-y*V0*w).diff(y)-(x*V0*w).diff(x)
check("rotation curl vanishes", rot_curl)
angular_m=-y*m.diff(x)+x*m.diff(y)
check("null line changes with angle",
      s.det(s.Matrix.hstack(m,angular_m))+2*(x*x+y*y))
check("rotating kernel metric identity",
      (m.T*G*(-y*V0.diff(x)+x*V0.diff(y))
       +angular_m.T*G*V0))

# Natural mixed boundary data at the free-interior exceptional coefficient.
v_origin=s.Matrix([1,1])
P_origin=v_origin*(v_origin.T*G)/(v_origin.T*G*v_origin)[0]
P_pair=s.diag(1,0)
check("origin singular kernel", C2*v_origin)
check("origin Robin coefficient",
      P_origin*B*v_origin-(n-2)*v_origin/n)
check("pair regular projected normal term", P_pair*(C1-C2)*P_pair)

# The scalar tensor domain has arbitrary jets at y=0. Together with
# its Neumann collision condition this kills all free first-order ladders.
ladder_boundary=s.Matrix([
    [0,1,0,1], [1,0,-1,0], [-1,1,0,0], [0,0,-1,1]
])
check("ladder boundary rank", ladder_boundary.rank()==4)
constant_rotation_boundary=s.Matrix([
    [0,1,0,0], [0,0,1,0], [-1,1,0,0], [0,0,-1,1]
])
check("constant rotation boundary rank", constant_rotation_boundary.rank()==4)

# Independent full-spin Ward-image check at n=3, x=3, y=1.
nn=3
sigmas=[
    s.Matrix([[0,1],[1,0]]),
    s.Matrix([[0,-s.I],[s.I,0]]),
    s.diag(1,-1),
]
dim=nn*nn*2
L1=[];L2=[];TS=[]
for a in range(nn):
    for b in range(a+1,nn):
        L=s.zeros(nn);L[a,b]=-s.I;L[b,a]=s.I
        T=-s.I*sigmas[a]*sigmas[b]/2
        L1.append(s.kronecker_product(L,s.eye(nn),s.eye(2)))
        L2.append(s.kronecker_product(s.eye(nn),L,s.eye(2)))
        TS.append(s.kronecker_product(s.eye(nn),s.eye(nn),T))
zero=s.zeros(dim)
OO=sum((a*b for a,b in zip(L1,L2)),zero)
O1=sum((a*b for a,b in zip(L1,TS)),zero)
O2=sum((a*b for a,b in zip(L2,TS)),zero)
E1=s.Matrix.vstack(*[s.eye(2)*int(a==b) for a in range(nn) for b in range(nn)])
E2=s.Matrix.vstack(*[
    s.zeros(2) if a==b else -sigmas[a]*sigmas[b]
    for a in range(nn) for b in range(nn)])
emb=[E1,E2]
norm=s.zeros(2)
for i,(Li,Lj,Oi) in enumerate([(L1,L2,O1),(L2,L1,O2)]):
    pos=s.Integer(3 if i==0 else 1)
    pair=s.Integer(2 if i==0 else -2)
    for l_i,l_j,t in zip(Li,Lj,TS):
        pairward=s.Rational(2,3)*l_j-l_i*OO/(nn-1)+OO*l_i/3
        srcward=s.Rational(2,3)*t-l_i*Oi/(nn-1)+Oi*l_i/3
        ward=pairward/pair+srcward/pos
        images=[ward*e for e in emb]
        for a in range(2):
            for b in range(2):
                norm[a,b]+=s.trace(images[a].conjugate().T*images[b])/2
full_coefficient=s.diag(nn,nn*(nn-1)).inv()*norm
check("full-spin null image matches reduced formula",
      full_coefficient-K.subs({n:3,x:3,y:1}))

results={
    "checks_passed":len(passed),
    "special_gamma_n23":str((1/ratio).subs(n,23)),
    "null_potential_over_KZ_potential":str(-ratio),
    "translation_integrability_rank_n23":M.rank(),
    "free_ladder_boundary_rank":ladder_boundary.rank(),
    "free_constant_rotation_boundary_rank":constant_rotation_boundary.rank(),
    "scope":"No nonzero first-order local differential intertwiner on the specified full tensor-to-source domains; no claim about nonlocal or higher-order maps.",
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_sector_intertwiner_results.json').write_text(
    json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
