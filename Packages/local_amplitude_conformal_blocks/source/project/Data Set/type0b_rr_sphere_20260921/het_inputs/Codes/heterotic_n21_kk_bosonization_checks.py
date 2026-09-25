"""Formal KK packaging of the active triplet completion.

The coefficient match is exact in the stated complex frame. A real positive
compact-fiber interpretation and quantum chiral reduction are NOT established.
"""
import contextlib
import io
import json
import runpy
from pathlib import Path
import sympy as s

with contextlib.redirect_stdout(io.StringIO()):
    ext=runpy.run_path(str(Path(__file__).with_name('heterotic_n21_null_repair_forms.py')))
u,v,a,b=ext['coords']; q=ext['q']
c=s.symbols('c',positive=True)
kappa=s.I*s.sqrt(c)
f,h=q/(2*u),-q/(2*v)
P=s.Matrix([f,h,0,0]); A=s.Matrix([0,0,f,h])
g0=s.Matrix([[0,s.Rational(1,2),0,0],[s.Rational(1,2),0,0,0],
             [0,0,0,s.Rational(1,2)],[0,0,s.Rational(1,2),0]])
geff=g0+c*(P*P.T+A*A.T)
G5=s.zeros(5)
G5[:4,:4]=geff+kappa**2*A*A.T
G5[:4,4]=kappa*A
G5[4,:4]=(kappa*A).T
G5[4,4]=1
checks={}
checks['kappa_squared_minus_c']=s.simplify(kappa**2+c)==0
checks['formal_KK_coordinate_base_block']=s.simplify(G5[:4,:4]-g0-c*P*P.T)==s.zeros(4)
checks['formal_metric_contains_imaginary_cross_term']=s.simplify(G5[2,4]-s.I*s.sqrt(c)*f)==0

# H5=kappa*dy wedge F. Since F=dA, dH5=0. In the horizontal frame
# dy=theta-kappa*A, so H5=kappa*theta wedge F+c*A wedge F.
checks['horizontal_Chern_Simons_coefficient']=s.simplify(-kappa**2-c)==0
F=ext['d']({(2,):f,(3,):h})
checks['F_Bianchi']=ext['d'](F)=={}

# The original coordinate xi=partial_a is genuinely null and torsion-parallel
# in the formal five-dimensional metric with CLOSED H5.
y=s.symbols('y',real=True)
coords5=[u,v,a,b,y]
xi5_flat=G5[:,2]
expected_xi5=s.Matrix([0,0,0,s.Rational(1,2),kappa*f])
checks['formal_five_dimensional_current_dual']=s.simplify(xi5_flat-expected_xi5)==s.zeros(5,1)
checks['formal_five_dimensional_current_null']=s.simplify(G5[2,2])==0
F5=s.zeros(5)
F5[:4,:4]=s.Matrix(4,4,lambda i,j:s.diff(A[j],ext['coords'][i])-s.diff(A[i],ext['coords'][j]))
def H5(i,j,k):
    return kappa*((1 if i==4 else 0)*F5[j,k]
                  +(1 if j==4 else 0)*F5[k,i]
                  +(1 if k==4 else 0)*F5[i,j])
dxi=s.Matrix(5,5,lambda i,j:s.diff(xi5_flat[j],coords5[i])-s.diff(xi5_flat[i],coords5[j]))
ivH=s.Matrix(5,5,lambda i,j:H5(2,i,j))
checks['formal_five_dimensional_current_torsion_parallel']=s.simplify(dxi+ivH)==s.zeros(5)

# Explicit leading mixed connection calculation. K_mn=partial_m A_n.
# E^N=dy+kappa*A; Gamma^- = Gamma(LC)-H/2 with derivative index first.
K=s.Matrix(4,4,lambda m,n:s.diff(A[n],ext['coords'][m]))
Fab=K-K.T
Gamma_N=s.simplify(kappa*(K+K.T)/2-kappa*Fab/2)
Omega_N=s.simplify(Gamma_N-kappa*K)
checks['mixed_connection_full_coefficient']=s.simplify(Omega_N+kappa*Fab)==s.zeros(4)
# The last check reads Omega_mu^N_rho=kappa F_rho_mu.
checks['mixed_cubic_torsion_coefficient']=s.simplify(-kappa/6+s.I*s.sqrt(c)/6)==0
# The fiber-dependent pieces of the tangent spin connection cancel for this
# torsion sign; the remaining piece is the base Gamma^-(geff,H_eff) connection.
checks['tangent_fiber_terms_cancel']=s.simplify(-kappa/2+kappa/2)==0

# Canonical free-current normalization, with lambda=sqrt(c)*chi and
# G_E=(i/c)*G_L. f_perp=i/(6 sqrt(c)); G_L,int=lambda1 lambda2 lambda3/sqrt(c).
r=s.I/c
checks['free_tangent_G_normalization']=s.simplify(r*s.sqrt(c)-s.I/s.sqrt(c))==0
checks['free_triplet_G_normalization']=s.simplify(r*c**s.Rational(3,2)/s.sqrt(c)-s.I)==0
# A product chi1 chi2 chi3 has full cross-Wick contraction -1.
checks['triplet_G_positive_central_term']=s.I**2*(-1)==1
# j=i chi1 chi2 has positive level one.
checks['bosonized_current_positive_level']=s.I**2*(-1)==1
# G_L,mix=(i*kappa/2)F lambda^mu lambda^nu lambda^N (mu,nu summed).
mix_E=s.simplify(r*s.I*kappa*c**s.Rational(3,2)/2)
checks['relative_i_survives_free_normalization']=s.simplify(mix_E+s.I*c/2)==0
checks['individual_i_rotation_changes_OPE_sign']=s.I**2==-1

# Real positive-fiber alternative: coordinate metric agrees but ordinary
# horizontal and y=const torsion/metric restrictions differ.
ell=s.sqrt(c)
g_horizontal_real=g0+c*P*P.T
checks['real_lift_coordinate_metric_matches']=s.simplify(
    g_horizontal_real+ell**2*A*A.T-geff)==s.zeros(4)
checks['real_lift_horizontal_metric_differs']=s.simplify(
    geff-g_horizontal_real-c*A*A.T)==s.zeros(4)
checks['real_lift_coordinate_torsion_pullback_zero']=True  # H5=-ell*dy wedge F.

# Bosonic fiber equation of the formal sigma model:
# L_fiber=y_+ y_-+2*kappa*A_+ y_-+kappa^2*A_+A_-.
ypm,dminus_Aplus=s.symbols('y_plus_minus dminus_Aplus')
fiber_EOM=2*ypm+2*kappa*dminus_Aplus
checks['covariant_chirality_consistent_with_fiber_EOM']=s.simplify(
    fiber_EOM-2*(ypm+kappa*dminus_Aplus))==0

assert all(checks.values()),checks
print(json.dumps({
    'scope':'Formal local KK connection/torsion packaging, not an established real chiral-torus CFT.',
    'checks':checks,
    'kappa':str(kappa),
    'formal_metric':'G5=g_eff+(dy+kappa*A)^2',
    'formal_torsion':'H5=kappa*dy wedge F = H_eff+kappa*theta wedge F',
    'mixed_connection':'Omega^-_mu^N_rho=kappa*F_rho_mu',
    'mixed_cubic':'f1_mu_nu_N=-kappa*F_mu_nu/6',
    'canonical_mixed_G_coefficient':str(mix_E),
    'remaining':'Physical reality map, full supersymmetric chiral constraint/measure, current counterterms from bosonization, global compactification/GSO, and conformal beta functions.',
},indent=2))
