"""Small exact checks for microscopic alternatives, not a duality test."""
import json
from pathlib import Path
import sympy as s

checks=[]

def check(name,lhs,rhs):
    difference=lhs-rhs
    if isinstance(difference,s.MatrixBase):
        assert difference.applyfunc(s.simplify)==s.zeros(*difference.shape),name
    else:
        assert s.simplify(difference)==0,(name,difference)
    checks.append(name)

# Contraction of the adjoint orbital gauge vector field with its coordinate.
size=4
basis=[]
for i in range(size):
    for j in range(i+1,size):
        generator=s.zeros(size)
        generator[i,j]=1
        generator[j,i]=-1
        basis.append(generator)
coordinates=s.symbols(f"a0:{len(basis)}")
matrix=sum((x*t for x,t in zip(coordinates,basis)),s.zeros(size))
orbital_contraction=sum((x*(t*matrix-matrix*t) for x,t in zip(coordinates,basis)),s.zeros(size))
check("adjoint_Gauss_contraction",orbital_contraction,s.zeros(size))

# Four flavor modes in one real O(2) eigenvalue block. Each complex mode
# comes from the two color components of a real fundamental Majorana.
number_flavors=4
identity=s.eye(2)
parity=s.diag(1,-1)
annihilator=s.Matrix([[0,1],[0,0]])
operators=[]
for flavor in range(number_flavors):
    factors=[parity if j<flavor else annihilator if j==flavor else identity
             for j in range(number_flavors)]
    operators.append(s.kronecker_product(*factors))
dimension=2**number_flavors
charge=s.zeros(dimension)
yukawa=s.zeros(dimension)
for f,c in enumerate(operators):
    majorana1=(c+c.H)/s.sqrt(2)
    majorana2=(c-c.H)/(s.I*s.sqrt(2))
    check(f"majorana_square_{f}",majorana1**2,s.eye(dimension)/2)
    yukawa+=s.I*majorana1*majorana2
    charge+=c.H*c-s.eye(dimension)/2
check("single_block_Yukawa_equals_Cartan_charge",yukawa,charge)
neutral=[i for i in range(dimension) if charge[i,i]==0]
check("neutral_state_count",len(neutral),s.binomial(number_flavors,number_flavors//2))
check("Yukawa_zero_on_neutral_states",yukawa.extract(neutral,neutral),s.zeros(len(neutral)))
check("SO23_middle_form_branching_dimensions",s.binomial(24,12),s.binomial(23,12)+s.binomial(23,11))

# Spin-factor closure fails after making its scalar coefficients noncommuting.
sx=s.Matrix([[0,1],[1,0]])
sz=s.diag(1,-1)
x=s.kronecker_product(sx,sx)+s.kronecker_product(sz,sz)
expected=2*s.eye(4)+s.kronecker_product(sx*sz-sz*sx,sx*sz)
check("matrix_coefficient_square",x*x,expected)
bivector=sx*sz
coefficient=s.zeros(2)
for i in range(2):
    for j in range(2):
        block=(x*x)[2*i:2*i+2,2*j:2*j+2]
        coefficient[i,j]=s.trace(bivector.H*block)/2
check("nonzero_bivector_coefficient",coefficient,sx*sz-sz*sx)
assert coefficient!=s.zeros(2)
checks.append("spin_factor_not_closed_with_noncommuting_coefficients")
t,v,w=s.symbols("t v w",real=True)
scalar_spin_factor=t*s.eye(2)+v*sx+w*sz
check("scalar_spin_factor_cubic",s.trace(scalar_spin_factor**3)/2,t**3+3*t*(v*v+w*w))

# Gauge anomaly in common Majorana-Weyl units, omitting a common factor1/2.
n=s.symbols("N")
check("standard_matrix_string_anomaly",8*(n+2)-8*(n-2)-32,0)
check("24_flavor_substitution_anomaly",8*(n+2)-8*(n-2)-24,8)
check("six_pair_24_flavor_anomaly",6*(n+2)-6*(n-2)-24,0)

result={"checks_passed":len(checks),"checks":checks,
        "24_flavor_single_block_neutral_dimension":int(s.binomial(24,12)),
        "scope":"Gauss-law and algebra identities only; no sea or scattering claim"}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_microscopic_alternatives_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
