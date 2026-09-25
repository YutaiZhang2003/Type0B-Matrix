"""Null coalescence and fusion-channel checks for b=1 N=1 super-Liouville."""
import json
from pathlib import Path
import sympy as s
b,a=s.symbols("b a", nonzero=True)
Q=b+1/b
c=s.Rational(3,2)+3*Q**2
h=lambda alpha:s.Rational(1,16)+alpha*(Q-alpha)/2
checks=[]
def check(name, expression):
    val=s.simplify(expression)
    assert val==0,(name,val)
    checks.append(dict(name=name,passed=True))
kb=1+1/(2*b*b)
kd=1+b*b/2
for name,alpha,kappa in [("b",-b/2,kb),("dual",-1/(2*b),kd)]:
    hv=h(alpha); v=hv-c/24
    check(name+"_L1_null",2*kappa*hv-s.Rational(3,2)*v)
    check(name+"_G1_null",s.Rational(3,2)*kappa-2*hv-c/4)
check("null_operators_coalesce",kb.subs(b,1)-kd.subs(b,1))
check("common_null_kappa",kb.subs(b,1)-s.Rational(3,2))
hv=s.Rational(-9,16)
cv=s.Rational(27,2)
v=hv-cv/24
M=s.Matrix([[2*hv,s.Rational(3,2)*v],
            [s.Rational(3,2)*v,v*(2*hv+cv/4)]])
assert M.rank()==1
checks.append(dict(name="level_one_gram_rank_one",passed=True))
assert M*s.Matrix([s.Rational(3,2),-1])==s.zeros(2,1)
checks.append(dict(name="unique_even_null_direction",passed=True))
F=(a+1/(2*b))**2-b*b/4
G=(a+b/2)**2-1/(4*b*b)
check("fusion_polynomial_difference",F-G-(1/b-b)*(a+Q/2))
check("mixed_generic_common_root",F.subs(a,-Q/2))
check("dual_mixed_generic_common_root",G.subs(a,-Q/2))
check("b_one_fusion_b",F.subs(b,1)-a*(a+1))
check("b_one_fusion_dual",G.subs(b,1)-a*(a+1))
check("vacuum_is_allowed_at_b_one",F.subs({b:1,a:0}))
# The level-one determinant has a double zero in h without two null directions.
hh=s.symbols("h")
vv=hh-cv/24
MM=s.Matrix([[2*hh,s.Rational(3,2)*vv],
             [s.Rational(3,2)*vv,vv*(2*hh+cv/4)]])
check("double_kac_zero_not_double_kernel",
      MM.det()-4*(hh-cv/24)*(hh+s.Rational(9,16))**2)
out=dict(check_count=len(checks),checks=checks,
         gram_matrix=[[str(x) for x in row] for row in M.tolist()],
         fusion_difference=str(s.factor(F-G)),
         scope="Exact chiral null coalescence and channel admissibility; no claim of full local bulk crossing or heterotic BRST completion.")
Path('data_exports/mqm/mqm_ramond_null_coalescence_results.json').write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(dict(check_count=len(checks),all_passed=True),indent=2))

