"""Targeted coefficient and genuine clock-iteration checks of soft regularity.

The all-degree conclusion is the analytic proof in the companion memo.
Nilpotent external labels below extract exact multilinear clock kernels,
including repeated frequencies and internal subsets of total momentum zero.
"""
from pathlib import Path
from functools import lru_cache
from math import comb
import json
import sympy as s

checks=[]


def equal(name,a,b=0):
    residual=s.simplify(s.expand(a-b))
    if residual!=0:
        raise AssertionError((name,residual))
    checks.append(name)


g,w,A,B,scale=s.symbols("g w A B scale")
def falling(z,n):
    return s.prod(z-j for j in range(n))


def C(n):
    return sum(falling(-s.I*w,n-b)*A**(n-2*b)*B**b/
               (s.factorial(n-2*b)*s.factorial(b))
               for b in range(n//2+1))


# Independent exponential-log expansion of the exact clock factor.
log_coeff=s.series(s.log(1+g*A+g*g*B),g,0,8).removeO()
direct={0:s.Integer(1)}
for n in range(1,8):
    direct[n]=s.expand(sum((-s.I*w)**ell/s.factorial(ell)*
                           s.expand(log_coeff**ell).coeff(g,n)
                           for ell in range(1,n+1)))
    equal(f"exact exponential-log coefficient n{n}",C(n),direct[n])
    equal(f"weighted current homogeneity n{n}",
          C(n).subs({A:scale*A,B:scale**2*B}),scale**n*C(n))
    quotient=s.cancel(C(n)/(-s.I*w))
    s.Poly(quotient,w,A,B)  # No 1/w remains.
    checks.append(f"output-frequency quotient polynomial n{n}")
    equal(f"soft-root limit is the log coefficient n{n}",
          quotient.subs(w,0),log_coeff.coeff(g,n))

D=lambda k:1+s.I*k
R=lambda k:(1-s.I*k)/s.Integer(1+k*k)
# Conjugating the canonical z-map by u=Pz removes square roots:
# N_g(u)=P^2 beta(u), P^2=D(k)/D(-k).
phase_square=lambda k:(1+s.I*k)**2/s.Integer(1+k*k)


@lru_cache(None)
def scalar_log_vector(frequencies):
    size=len(frequencies)
    top=(1<<size)-1
    momenta={mask:sum(k for i,k in enumerate(frequencies) if mask>>i&1)
             for mask in range(top+1)}

    def add(*polys):
        out={}
        for poly in polys:
            for mask,value in poly.items():
                out[mask]=out.get(mask,0)+value
        return {mask:s.expand(value) for mask,value in out.items()
                if s.expand(value)!=0}

    def mul(left,right):
        out={}
        for a,va in left.items():
            for b,vb in right.items():
                if a&b:continue
                out[a|b]=out.get(a|b,0)+va*vb
        return {mask:s.expand(value) for mask,value in out.items()
                if s.expand(value)!=0}

    def multiply(poly,factor):
        return {mask:s.expand(factor*value) for mask,value in poly.items()}

    def resolvent(poly):
        return {mask:s.expand(R(momenta[mask])*value)
                for mask,value in poly.items()}

    def clock(poly):
        W=add(multiply(resolvent(poly),s.sqrt(2)),
              multiply(resolvent(mul(poly,poly)),s.Rational(1,2)))
        powers=[{0:s.Integer(1)}]
        for ell in range(1,size+1):
            powers.append(mul(powers[-1],W))
        j_powers=[mul(poly,power) for power in powers]
        out={}
        for mask in range(1,top+1):
            omega=momenta[mask]
            value=poly.get(mask,0)
            for ell in range(1,size+1):
                value+=falling(-s.I*omega,ell)/s.factorial(ell)*(
                    j_powers[ell].get(mask,0)
                    +s.sqrt(2)*powers[ell].get(mask,0))
            value=s.expand(phase_square(omega)*value)
            if value!=0:out[mask]=value
        return out

    initial={1<<i:s.Integer(1) for i in range(size)}
    iterates=[initial]
    for step in range(1,size):
        iterates.append(clock(iterates[-1]))
    logarithm={}
    for mask in range(1,top+1):
        value=0
        for order in range(1,size):
            difference=sum((-1)**(order-j)*comb(order,j)*
                           iterates[j].get(mask,0) for j in range(order+1))
            value+=s.Rational((-1)**(order+1),order)*difference
        value=s.expand(value)
        if value!=0:logarithm[mask]=value
    for index,it in enumerate(iterates):
        for mask in range(1,top+1):
            if momenta[mask]==0 and it.get(mask,0)!=0:
                raise AssertionError(("generated elementary zero mode",frequencies,index,mask))
    return logarithm.get(top,0),len([mask for mask in range(1,top+1)
                                   if momenta[mask]==0])


def h_from_log(all_momenta,outgoing):
    external=tuple(k for i,k in enumerate(all_momenta) if i!=outgoing)
    omega=sum(external)
    if omega==0:raise ValueError("The root quotient is checked analytically, not by 0/0.")
    vector,_=scalar_log_vector(external)
    return s.simplify(vector/(-s.I*omega))


for momenta in ((-3,1,2),(-2,1,1)):
    expected=s.sqrt(2)*(1+sum(s.Integer(k*k) for k in momenta)/2)/s.prod(D(k) for k in momenta)
    for out in range(3):
        equal(f"actual cubic log tensor {momenta} out{out}",
              h_from_log(momenta,out),expected)

for momenta in ((-4,1,1,2),(-1,-1,1,1),(-2,-1,1,2)):
    stripped=-(1+sum(s.Integer(k*k) for k in momenta)/2)
    stripped+=s.prod(momenta)*sum(
        1/s.Integer(1+(momenta[0]+momenta[j])**2) for j in range(1,4))
    expected=stripped/s.prod(D(k) for k in momenta)
    for out in range(4):
        equal(f"actual quartic log tensor {momenta} out{out}",
              h_from_log(momenta,out),expected)

# New quintic integrability checks, with no G5 formula used as input.
for momenta in ((-4,1,1,1,1),(-2,-1,1,1,1)):
    reference=h_from_log(momenta,0)
    for out in range(1,5):
        equal(f"actual quintic outgoing-slot symmetry {momenta} out{out}",
              h_from_log(momenta,out),reference)
    reversed_value=h_from_log(tuple(-k for k in momenta),0)
    equal(f"actual quintic Hermitian reality {momenta}",
          reversed_value,s.conjugate(reference))
    _,zeros=scalar_log_vector(tuple(momenta[1:]))
    checks.append(f"quintic clock compositions keep {zeros} internal-zero subsets fixed for {momenta}")

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "scope":"Finite targeted checks support the analytic all-degree soft-regularity proof; no convergence of the autonomous Hamiltonian logarithm is tested or claimed."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_all_degree_soft_results.json').write_text(
    json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
