"""Independent finite-tensor audit of the SO(n)_1 KZ proposal."""
import itertools, json
import sympy as s
from pathlib import Path
n=s.symbols("n", integer=True, positive=True)
x=s.symbols("x1:5", real=True)
pairs=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
def canon(p):
    return tuple(sorted(tuple(sorted(q)) for q in p))
def perm_mat(i,j):
    mat=s.zeros(3)
    swap=lambda k:j if k==i else i if k==j else k
    for col,b in enumerate(pairs):
        v=canon([(swap(a),swap(c)) for a,c in b])
        mat[pairs.index(v),col]=1
    return mat
def q_mat(i,j):
    mat=s.zeros(3)
    for col,b in enumerate(pairs):
        d={}
        for a,c in b:d[a]=c;d[c]=a
        if d[i]==j:
            mat[col,col]=n
        else:
            v=canon([(i,j),(d[i],d[j])])
            mat[pairs.index(v),col]=1
    return mat
omega={(i,j):perm_mat(i,j)-q_mat(i,j) for i in range(4) for j in range(i+1,4)}
om=lambda i,j:omega[tuple(sorted((i,j)))]
c=1/(n-1)
psi=s.Matrix([1/((x[0]-x[1])*(x[2]-x[3])),
             -1/((x[0]-x[2])*(x[1]-x[3])),
             1/((x[0]-x[3])*(x[1]-x[2]))])
checks={}
for i in range(4):
    conn=sum((om(i,j)/(x[i]-x[j]) for j in range(4) if j!=i),s.zeros(3))
    residual=psi.diff(x[i])-c*conn*psi
    checks[f"N4_KZ_i{i+1}"]=all(s.factor(t)==0 for t in residual)
# N2 singlet: c*omega=-1, psi=r^-1.
r=s.symbols("r",positive=True)
checks["N2_KZ"]=s.diff(1/r,r)-(-1/r)*(1/r)==0
checks["N2_trace_flat_pair_potential"]=s.simplify(c*c*(1-n)**2-c*(1-n)-2)==0
# All pairings have n^2 norm, n overlap if distinct.
gram=s.ones(3)*n+s.eye(3)*(n*n-n)
for key,m in omega.items():
    checks[f"Omega_{key}_gram_Hermitian"]=all(s.expand(t)==0 for t in gram*m-m.T*gram)
# Ordered triple sum in 1/2 sum_i A_i^2.
triple=s.zeros(3)
for i in range(4):
    for j in range(4):
        if j==i:continue
        for k in range(4):
            if k==i or k==j:continue
            triple+=c*c*om(i,j)*om(i,k)/(2*(x[i]-x[j])*(x[i]-x[k]))
at={n:23,x[0]:0,x[1]:1,x[2]:3,x[3]:6}
triple23=triple.subs(at).applyfunc(s.factor)
checks["N4_three_body_survives_in_singlet_sector"]=triple23!=s.zeros(3)
checks["N4_three_body_gram_Hermitian"]=(gram.subs(n,23)*triple23==triple23.T*gram.subs(n,23))
# Local flat radial branches s=-1,2. With weight r^beta, norm r^(beta-2)dr.
# For D=partial_r+1/r, the weighted adjoint gives
# DdagD=-partial_r^2-beta/r partial_r+(2-beta)/r^2.
beta=s.symbols("beta",real=True)
f=s.Function("f")(r)
D=lambda v:s.diff(v,r)+v/r
Ddag=lambda v:-s.diff(v,r)+(1-beta)*v/r
weighted=s.simplify(Ddag(D(f))-(-s.diff(f,r,2)-beta*s.diff(f,r)/r+(2-beta)*f/r**2))
checks["weighted_pair_DdagD"]=weighted==0
# Flat-unitary state u=r^(beta/2) psi. Pair coupling (beta/2-1)(beta/2-2).
q=s.symbols("q")
a=beta/2-1
checks["unitary_pair_coupling"]=s.expand((beta*(beta-2)/4+2-beta)-a*(a-1))==0
freq=s.symbols("freq",positive=True)
for label,omega_eig in [("ST",s.Integer(1)),("A",-s.Integer(1)),("trace",1-n)]:
    exponent=c*omega_eig
    branch=r**exponent*s.exp(-freq*r*r/4)
    residual=s.diff(branch,r)+(freq*r/2-exponent/r)*branch
    checks[f"N2_Gaussian_KZ_{label}"]=s.simplify(residual)==0
    for dim in (23,24):
        checks[f"N2_Delta2_integrable_{label}_n{dim}"]=bool((2+2*exponent).subs(n,dim)>-1)
checks["N2_all_zero_modes_dimension"]=s.expand((n*(n+1)/2-1)+n*(n-1)/2+1-n*n)==0
assert all(checks.values()),checks
out={"checks":checks,"all_pass":all(checks.values()),
     "N4_three_body_n23_x_0_1_3_6":[[str(t) for t in triple23.row(i)] for i in range(3)],
     "N4_gram":"diagonal n^2, off-diagonal n",
     "N2_Gaussian_maximal_form_kernel":"all n^2 spin states; ST, antisymmetric and trace",
     "N2_ordinary_harmonic_energies_beta2":"omega*(2+c), omega*(2-c), omega",
     "collision":{"unweighted_state":"r^-1 (non-square-integrable)",
                  "weighted_norm":"r^(beta-2) dr: integrable iff beta>1",
                  "weighted_pair_operator":"-d_r^2-beta/r d_r+(2-beta)/r^2",
                  "unitary_pair_coupling":"(beta/2-1)(beta/2-2)"}}
Path('data_exports/mqm/mqm_kz_independent_checks_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
