"""Leading Lawrence completion of one normal SU(2) triplet.

This computes necessary coefficients and selected quadratic terms. It does
not certify the full mixed-current algebra at O(alpha_prime q^2): the cited
Lawrence equations are only established through O(alpha_prime^0).
"""
from itertools import permutations, product, combinations
import json
import sympy as s

u,v,a,b,q,alpha=s.symbols('u v a b q alpha_prime', real=True, nonzero=True)
x=(u,v,a,b)
g=s.Matrix([[0,s.Rational(1,2),0,0],[s.Rational(1,2),0,0,0],
            [0,0,0,s.Rational(1,2)],[0,0,s.Rational(1,2),0]])
bundle=s.diag(g,s.eye(3))
finv=bundle.inv()
F=s.zeros(4)
F[0,2]=-q/(2*u**2)
F[2,0]=-F[0,2]
F[1,3]=q/(2*v**2)
F[3,1]=-F[1,3]
f_a=q/(2*u)
c=alpha/2
structure=s.I/(3*s.sqrt(2*alpha))
kappa=3*alpha*structure
checks={}
checks['normal_structure_normalization']=s.simplify(2*structure**2+1/(9*alpha))==0
checks['kappa_squared_minus_c']=s.simplify(kappa**2+c)==0
checks['Bianchi_dF_zero']=all(s.simplify(s.diff(F[j,k],x[i])+s.diff(F[k,i],x[j])
                              +s.diff(F[i,j],x[k]))==0
                              for i,j,k in product(range(4),repeat=3))

f0={}
f1={}
def antisym(target, indices, value):
    for permutation in permutations(range(3)):
        parity=(-1)**sum(permutation[i]>permutation[j] for i in range(3)
                         for j in range(i+1,3))
        target[tuple(indices[i] for i in permutation)]=parity*value

antisym(f0,(4,5,6),structure)
for i in range(4):
    for j in range(i+1,4):
        if F[i,j]!=0:
            antisym(f1,(i,j,6),-kappa*F[i,j]/6)

# Full connection matrices have one index raised with the bundle metric.
A1=[]
for mu in range(4):
    mat=s.zeros(7)
    for rho in range(4):
        mat[6,rho]=kappa*F[rho,mu]
        mat[rho,6]=-kappa*sum(g.inv()[rho,sigma]*F[sigma,mu]
                             for sigma in range(4))
    A1.append(mat)
checks['mixed_connection_metric_compatible']=all(
    (mat.T*bundle+bundle*mat).applyfunc(s.simplify)==s.zeros(7) for mat in A1)

Sigma0=s.zeros(7)
Sigma0[4,5]=s.I*f_a/2
Sigma0[5,4]=-s.I*f_a/2
Sigma1=s.zeros(7)
for rho in range(4):
    Sigma1[6,rho]=s.I*kappa*F[rho,2]/2
    Sigma1[rho,6]=-s.I*kappa*sum(g.inv()[rho,sigma]*F[sigma,2]
                                for sigma in range(4))/2
checks['mixed_Sigma_metric_compatible']=(Sigma1.T*bundle+bundle*Sigma1).applyfunc(
    s.simplify)==s.zeros(7)

# Eq91: j has one additional normal fermion coefficient. With Sigma^{BC}
# restricted to the normal block, upper and lower normal indices agree.
w=3*s.I*alpha*sum(f0.get((6,i,j),0)*Sigma0[i,j]
                  for i in range(7) for j in range(7))
checks['null_partner_normal_coefficient']=s.simplify(w+kappa*f_a)==0
checks['shifted_null_partner_norm_cancels']=s.simplify(c*f_a**2+w**2)==0

# Normal adjoint compatibility: Eq104 and the commuting-vector Eq105.
checks['normal_Sigma_preserves_structure']=all(s.simplify(sum(
    Sigma0[k,i]*f0.get((k,j,l),0)+Sigma0[k,j]*f0.get((i,k,l),0)
    +Sigma0[k,l]*f0.get((i,j,k),0) for k in range(7)))==0
    for i,j,l in product(range(4,7),repeat=3))
checks['F_structure_vector_commutes_with_Sigma']=all(Sigma0[6,i]==0 for i in range(7))

# Free Wick-contraction support for the proposed missing linear term.
# A cubic/ quadratic fermion OPE needs two shared indices to leave one
# fermion. f1(T,T,N3) and Sigma0(N1,N2) share none.
wick_f1_sigma0=[]
wick_f0_sigma1=[]
wick_f1_sigma1=[]
Sigma0up=Sigma0*finv
Sigma1up=Sigma1*finv
for i in range(7):
    wick_f1_sigma0.append(s.simplify(sum(f1.get((i,j,k),0)*Sigma0up[j,k]
                               for j,k in product(range(7),repeat=2))))
    wick_f0_sigma1.append(s.simplify(sum(f0.get((i,j,k),0)*Sigma1up[j,k]
                               for j,k in product(range(7),repeat=2))))
    wick_f1_sigma1.append(s.simplify(3*s.I*alpha*sum(f1.get((i,j,k),0)*Sigma1up[j,k]
                               for j,k in product(range(7),repeat=2))))
checks['f1_Sigma0_double_Wick_zero']=all(z==0 for z in wick_f1_sigma0)
checks['f0_Sigma1_double_Wick_zero']=all(z==0 for z in wick_f0_sigma1)

quadratic_curvature={}
for mu in range(4):
    for nu in range(mu+1,4):
        quadratic_curvature[mu,nu]=bundle*(A1[mu]*A1[nu]-A1[nu]*A1[mu])
curv=s.simplify(quadratic_curvature[0,1][2,3])
checks['nonzero_quadratic_tangent_curvature']=s.simplify(curv+c*q**2/(4*u**2*v**2))==0

def curvature(mu,nu,i,j):
    if mu==nu:return 0
    if mu>nu:return -curvature(nu,mu,i,j)
    return quadratic_curvature[mu,nu][i,j]

# Display only the two explicitly known quadratic contributions to eq103.
# Neither this partial sum nor its nonzero result is a full O(q^2) anomaly.
partial=0
for inds in permutations((0,1,3)):
    parity=(-1)**sum(inds[i]>inds[j] for i in range(3) for j in range(i+1,3))
    i,j,l=inds
    partial+=parity*(curvature(i,2,j,l)/2-6*s.I*sum(
        Sigma1[k,i]*f1.get((k,j,l),0) for k in range(7)))
partial=s.simplify(partial)
checks['specified_partial_GJ_quadratic_term']=s.simplify(partial-c*q**2/(4*u**2*v**2))==0

assert all(checks.values()),checks
report={
    'scope':'Natural e,f,A,Sigma completion: full cubic GJ tensor and bundle curvature tests through retained q^2 orders; one linear GJ tensor remains unresolved. Not full BRST closure.',
    'primary_source':'https://arxiv.org/pdf/hep-th/9605223 (eqs61,62,67,91,103,107; source controls leading alpha_prime order)',
    'order_bookkeeping':{
        'A0_Sigma0':'q',
        'A1_f1_Sigma1':'sqrt(alpha_prime) q',
        'g2_e2_H2_A2_Sigma2':'alpha_prime q^2',
        'kept_curvature':'dA0+dA1+dA2+A1 wedge A1; A0 commutes with A1 and A2',
        'discarded':'A1 A2 starts q^3, A2^2 starts q^4, e2 F starts q^3',
        'not_claimed':'All powers of alpha_prime at fixed q^2; the free f1 Sigma1 superpartner correction already starts alpha_prime^2 q^2.',
    },
    'checks':checks,
    'mixed_connection':'A1_mu^N_rho = kappa F_rho,mu, kappa^2=-alpha_prime/2',
    'mixed_cubic':'f1_mu,nu,N = -kappa F_mu,nu/6',
    'mixed_Sigma':'Sigma1^N_rho = i kappa F_rho,a/2',
    'null_partner_normal_coefficient':str(s.simplify(w)),
    'quadratic_F_uv_ab_from_A1_squared':str(curv),
    'GJ_lambda_u_lambda_v_lambda_b_before_tangent_completion':str(partial),
    'free_double_Wick_f1_Sigma1_superpartner_correction':list(map(str,wick_f1_sigma1)),
    'torsion_sign_control':{},
}

# Exploratory same-order tangent connection supplied by the shifted metric
# and Bismut torsion. This is an ansatz audit, not a derived counterterm map.
grad=s.Matrix([q/(2*u),-q/(2*v),0,0])
Aform=s.Matrix([0,0,q/(2*u),-q/(2*v)])
metric_shift=c*(grad*grad.T+Aform*Aform.T)
frame_shift=g.inv()*metric_shift/2
def torsion(i,j,k):
    return c*(Aform[i]*F[j,k]+Aform[j]*F[k,i]+Aform[k]*F[i,j])
J=s.Matrix([[0,0,-1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]])
for sign in (-1,1):
    A2=[]
    for mu in range(4):
        mat=s.zeros(4)
        for nu in range(4):
            for rho in range(4):
                christ=sum(g.inv()[rho,ell]*(s.diff(metric_shift[ell,nu],x[mu])
                    +s.diff(metric_shift[ell,mu],x[nu])-s.diff(metric_shift[mu,nu],x[ell]))/2
                    for ell in range(4))
                ht=sign*sum(g.inv()[rho,ell]*torsion(ell,mu,nu)/2 for ell in range(4))
                mat[rho,nu]=s.simplify(christ+ht-s.diff(frame_shift[rho,nu],x[mu]))
        A2.append(mat)
    full={}
    for mu in range(4):
        for nu in range(4):
            full[mu,nu]=(g*(A2[nu].diff(x[mu])-A2[mu].diff(x[nu]))
                          +s.Matrix(4,4,lambda i,j:curvature(mu,nu,i,j))).applyfunc(s.simplify)
    defects=[]
    for i,j in product(range(4),repeat=2):
        base=s.Matrix(4,4,lambda mu,nu:full[mu,nu][i,j])
        defect=(J.T*base*J-base).applyfunc(s.simplify)
        if defect!=s.zeros(4):defects.append((i,j,str(defect)))
    fullpartial=0
    for inds in permutations((0,1,3)):
        parity=(-1)**sum(inds[i]>inds[j] for i in range(3) for j in range(i+1,3))
        i,j,l=inds
        fullpartial+=parity*(full[i,2][j,l]/2-6*s.I*sum(
            Sigma1[k,i]*f1.get((k,j,l),0) for k in range(7)))
    report['torsion_sign_control'][str(sign)]={
        'tangent_holomorphy_tested_components':16,
        'tangent_holomorphy_nonzero_defects':len(defects),
        'GJ_lambda_u_lambda_v_lambda_b':str(s.simplify(fullpartial)),
    }
    if sign==-1:
        # Entire cubic-fermion tensor in Lawrence (103), to q^2. At this
        # order e2*F starts at q^3, Sigma2*f0 and Sigma0*f2 vanish by
        # tangent/normal support, so f0+f1 and Sigma0+Sigma1 suffice here.
        A0=[]
        for mu in range(4):
            mat=s.zeros(7)
            mat[4,5]=Aform[mu]
            mat[5,4]=-Aform[mu]
            A0.append(mat)
        Fcomplete={}
        for mu,nu in product(range(4),repeat=2):
            mat=bundle*((A0[nu]+A1[nu]).diff(x[mu])
                        -(A0[mu]+A1[mu]).diff(x[nu]))
            for i,j in product(range(4),repeat=2):mat[i,j]+=full[mu,nu][i,j]
            Fcomplete[mu,nu]=mat.applyfunc(s.simplify)
        GJ_defects=[]
        for inds0 in combinations(range(7),3):
            expression=0
            for perm in permutations(range(3)):
                parity=(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))
                i,j,l=(inds0[t] for t in perm)
                curvterm=Fcomplete[i,2][j,l]/2 if i<4 else 0
                productterm=-6*s.I*sum((Sigma0+Sigma1)[k,i]
                    *(f0.get((k,j,l),0)+f1.get((k,j,l),0)) for k in range(7))
                expression+=parity*(curvterm+productterm)
            expression=s.simplify(expression)
            if expression!=0:GJ_defects.append((inds0,str(expression)))
        assert GJ_defects==[]
        # Full bundle F remains (1,1), including mixed derivative terms.
        bundle_holomorphy=[]
        primitive=[]
        omega_inverse=(g*J).inv()
        for i,j in product(range(7),repeat=2):
            base=s.Matrix(4,4,lambda mu,nu:Fcomplete[mu,nu][i,j])
            defect=(J.T*base*J-base).applyfunc(s.simplify)
            if defect!=s.zeros(4):bundle_holomorphy.append((i,j,str(defect)))
            trace=s.simplify(sum(omega_inverse[mu,nu]*base[mu,nu]
                                for mu,nu in product(range(4),repeat=2)))
            if trace!=0:primitive.append((i,j,str(trace)))
        assert bundle_holomorphy==[]
        assert primitive==[]
        # The uncorrected geometric linear-lambda GJ relation is NOT imposed
        # blindly in shifted fields. Display its fixed-moment-map defect.
        sigma2_forced=s.I*g*A2[2]/2
        sigma2_geometric=s.Matrix(4,4,lambda i,j:s.I*torsion(i,j,2)/2)
        linear_defect=(sigma2_forced-sigma2_geometric).applyfunc(s.simplify)
        expected=(-s.I*c*f_a*F/2).applyfunc(s.simplify)
        assert linear_defect==expected
        report.update({
            'completed_tensor_checks':{
                'antisymmetric_GJ_cubic':{'tested':35,'nonzero':0},
                'full_bundle_curvature_type11':{'tested':49,'nonzero':0},
                'full_bundle_curvature_primitive':{'tested':49,'nonzero':0},
            },
            'natural_tangent_completion':{
                'metric_shift':'c(dphi tensor dphi+A tensor A)',
                'torsion':'c A wedge F',
                'frame_shift':'(1/2)g0^(-1)delta_g',
                'A2':'Gamma_minus(delta_g,H)-d(frame_shift)',
            },
            'remaining_linear_GJ_tensor':'-(i c/2) A(xi) F, c=alpha_prime/2',
            'remaining_linear_GJ_components':{
                str((i,j)):str(linear_defect[i,j]) for i,j in combinations(range(4),2)
                if linear_defect[i,j]!=0},
            'needed_OPE_contribution_in_Lawrence_free_propagator_convention':
                '+c^2 A(xi) F_(rho,a) partialX^rho lambda^a/(z-w)',
            'free_f1_Sigma0_diagram':'identically zero by fermion-index support',
            'candidate_interaction_diagram':'G_e - J_Sigma0 with one A_mu(X) partialbarX^mu lambda1 lambda2 action insertion: two normal fermion propagators and a boson/contact contraction',
            'interaction_diagram_coefficient':'not evaluated with a fixed regulator and current counterterm prescription',
            'remaining_full_closure':'Renormalized linear GJ relation, complete GG/GplusGminus/TG/TJ OPEs at the same order, higher-alpha terms and global/orbifold BRST data.',
        })

print(json.dumps(report,indent=2))
