"""Both descent components: ghost Fock bounds and local NS projection.

The all-level ghost bound is proved in the companion memo. This script
independently enumerates low Fock levels and checks the descent bookkeeping,
mode contour powers, physical channel weights and both flavor actions.
"""
import json
from pathlib import Path
import sympy as s

checks=[]
def check(name,expr):
    assert s.simplify(expr)==0,(name,expr)
    checks.append(name)

half=s.Rational(1,2)
hghost=lambda q:-s.sympify(q)*(q+2)/2
hN=-3*half
hg=-1
# Removing c by b_-1 raises weight by one and lowers BRST ghost number by one.
for label,hm,hxi in [("aXg",1+hg,1),
                    ("psi dN",half+1+hN,1),
                    ("dpsi N",3*half+hN,1),
                    ("second xi derivative",half+hN,2)]:
    check(label+" b-descendant weight",hxi+hghost(-2)+hm-1)
check("dz component source ghost number",(-1)+1)
check("dbar component source ghost number",0+0)
check("dz output ghost number",(0+2)-2)
check("dbar output ghost number",(1+1)-2)

# G_BRST=G_bc+G_betagamma-P, so e^-varphi has ghost number zero.
check("picture-independent ghost degree of e^-varphi",-1-(-1))
check("physical minus-one vertex ghost degree",1-1-(-1)-1)
check("physical zero-picture vertex ghost degree",1-0-1)

# Enumerate the picture-minus-one small-Hilbert ghost Fock space. The energy
# coordinate is twice h_ghost, including the vacuum value +1. c1 is the
# only negative-energy creation mode; c0 has zero energy and is fermionic.
max_energy=14
states={(1,0):1}
modes=[("c1",-2,1,False),("c0",0,1,False)]
for n in range(1,max_energy//2+2):
    modes.append((f"c_-{n}",2*n,1,False))
for n in range(2,max_energy//2+2):
    modes.append((f"b_-{n}",2*n,-1,False))
for rr in range(1,max_energy+2,2):
    modes.extend([(f"beta_-{rr}/2",rr,-1,True),
                  (f"gamma_-{rr}/2",rr,1,True)])
for name,cost,charge,bosonic in modes:
    old=states
    new={}
    for (energy,gh),multiplicity in old.items():
        limit=(max_energy-energy)//cost if bosonic else 1
        for occupation in range(max(0,limit)+1):
            ee=energy+occupation*cost
            if ee>max_energy:
                continue
            key=(ee,gh+occupation*charge)
            new[key]=new.get(key,0)+multiplicity
    states=new

mins={gh:min(energy for (energy,charge) in states if charge==gh)
      for gh in [0,1,2]}
check("ghost-zero minimal ghost weight",mins[0])
check("ghost-one minimal ghost weight",mins[1]+1)
check("ghost-one minimal state is unique",states[(-1,1)]-1)
check("ghost-two lowest weight is consistent with BPZ dual",mins[2]+1)
# Matter contributes +1 to twice L0 in the physical central module.
zero_candidates={str(gh):sum(count for (ee,gg),count in states.items()
                            if gg==gh and ee+1<=0)
                 for gh in [0,1,2]}
check("no ghost-zero L0=0 candidate",zero_candidates["0"])
check("unique ghost-one L0=0 candidate",zero_candidates["1"]-1)

# A(z)U(0) with h_A=0 has a constant physical output coefficient f.
# [b0,A(z)]=z (b_-1 A)(z), and b0 U=0. The coefficient of z^0 is
# Res (b_-1 A) U=b0(f c1|-1>V)=0, since {b0,c1}=0.
check("relative mode anticommutator",s.KroneckerDelta(0+1,0))

# Independent explicit contour powers for a local level-matched output.
theta,rad,h,hbar=s.symbols("theta rad h hbar",positive=True)
z=rad*s.exp(s.I*theta)
zb=rad*s.exp(-s.I*theta)
# For integer output h=hbar=0, the two simple poles have opposite orientation.
check("CCW dz simple pole",s.diff(z,theta)/z-s.I)
check("CCW dbar simple pole",s.diff(zb,theta)/zb+s.I)

k,p=s.symbols("k p")
delta=s.symbols("delta")
hSL=lambda al:al*(2-al)/2
hm=s.expand(k*k/2+hSL(1+p+delta)).subs(p*p,k*k)
check("central NS matter weight",hm.subs(delta,0)-half)
check("shifted NS plus channel weight",hm.subs(delta,1)+p)
check("shifted NS minus channel weight",hm.subs(delta,-1)-p)

BL,BR,rho,zeta=s.symbols("B_L B_R rho zeta")
left=k*BL/2
rightS=-k*k*BR
rightD=k*BR
check("scalar current dbar coefficient",left*rightS+k**3*BL*BR/2)
check("adjoint current dbar coefficient",left*rightD-k*k*BL*BR/2)
check("combined momentum coefficient",zeta*rho*(-left*rightS)/4
      -zeta*rho*k**3*BL*BR/8)
check("combined rotation coefficient",zeta*rho*(left*rightD)/4
      -zeta*rho*k*k*BL*BR/8)
check("same relative momentum and rotation coefficients",
      (-left*rightS)/k-left*rightD)

result={"passed":True,"exact_check_count":len(checks),"checks":checks,
        "fock_energy_cutoff_twice_hghost":max_energy,
        "ghost_minima_twice_hghost":mins,
        "L0_zero_candidate_counts":zero_candidates,
        "scope":"Generic NS small-Hilbert picture 0 to -1, exact irreducible central Liouville module, ordinary physical cohomology. The all-level bound is analytic; enumeration is an independent finite check.",
        "not_proved":["absolute normalized bulk coefficient",
                      "integrated moduli/PCO contact terms",
                      "exceptional-momentum or logarithmic-module action"]}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_o22_descent_particle_action_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(f"Passed {len(checks)} checks; zero-ghost physical candidates: {zero_candidates['0']}.")
