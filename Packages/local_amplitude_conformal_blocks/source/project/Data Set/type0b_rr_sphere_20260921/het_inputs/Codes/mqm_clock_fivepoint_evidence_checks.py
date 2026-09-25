"""Exact and repository-code checks for one five-point evidence audit.

No worldsheet integral is evaluated and no incomplete resonance output
is treated as an amplitude datum.
"""
from pathlib import Path
import json
import sympy as s
from spin23_fivepoint_reconstruction import normalized_pole_part,raw_pole_part,contact_basis

checks=[]


def equal(name,a,b=0):
    residual=s.simplify(s.cancel(a-b))
    if residual!=0:
        raise AssertionError((name,residual))
    checks.append(name)


def close(name,a,b):
    if abs(complex(a)-complex(b))>1e-11*max(1,abs(complex(b))):
        raise AssertionError((name,a,b))
    checks.append(name)


A,B,lam=s.symbols("A B lambda")
prefactor=2*s.I*lam*s.pi**2
clock=prefactor*(1+A+B)**2/((1+A)*(1+B))
pole=prefactor*(A*A+B*B-1)/((1+A)*(1+B))
equal("clock contact in declared residue lift",clock-pole,4*s.I*lam*s.pi**2)
equal("A-pair residue",s.cancel((1+A)*clock).subs(A,-1),
      prefactor*B*B/(1+B))
equal("B-pair residue",s.cancel((1+B)*clock).subs(B,-1),
      prefactor*A*A/(1+A))
equal("compatible double residue",
      s.cancel((1+A)*(1+B)*clock).subs({A:-1,B:-1}),prefactor)
equal("pair-exchange symmetry",clock,clock.xreplace({A:B,B:A}))
equal("first resonance clock value",clock.subs(B,-2-A),
      -prefactor/(1+A)**2)
equal("first resonance pole value",pole.subs(B,-2-A),
      -2*prefactor-prefactor/(1+A)**2)
equal("restricted double-pole coefficient",
      s.cancel((1+A)**2*clock.subs(B,-2-A)),-prefactor)
equal("incoming-singlet null-plane zero",clock.subs(B,-1-A))

partitions=(
    (-s.Rational(1,8),-s.Rational(3,5),-s.Rational(5,8),-s.Rational(13,20)),
    (s.Rational(1,7),s.Rational(2,9),s.Rational(3,8),s.Rational(1,5)),
    (-s.Rational(1,5),s.Rational(1,3),-s.Rational(2,7),s.Rational(4,9)),
)
for number,xs in enumerate(partitions):
    aa,bb=xs[0]+xs[1],xs[2]+xs[3]
    expected=pole.subs({A:aa,B:bb,lam:1})
    close(f"repository pole function partition{number}",
          normalized_pole_part("S_to_VVVV",xs,tensor="delta12_delta34"),
          s.N(expected,18))
    if contact_basis("S_to_VVVV",xs)!=(1+0j,):
        raise AssertionError(("unexpected declared contact basis",xs))
    checks.append(f"declared contact basis is constant partition{number}")

xs=partitions[0]
aa,bb=xs[0]+xs[1],xs[2]+xs[3]
U=sum(xs)*s.prod(xs)
equal("pilot raw universal soft factor",U,-s.Rational(39,640))
raw_clock=s.factor(U*clock.subs({A:aa,B:bb,lam:1}))
raw_pole=s.factor(U*pole.subs({A:aa,B:bb,lam:1}))
equal("pilot exact raw clock prediction",raw_clock,s.Rational(195,121)*s.I*s.pi**2)
equal("pilot exact raw pole benchmark",raw_pole,s.Rational(35919,19360)*s.I*s.pi**2)
equal("pilot exact raw contact prediction",raw_clock-raw_pole,-s.Rational(39,160)*s.I*s.pi**2)
close("pilot reported benchmark reproduced",
      raw_pole_part("S_to_VVVV",xs,tensor="delta12_delta34"),18.31127688443847j)

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "pilot_raw_clock_at_lambda_one":str(s.N(raw_clock,18)),
        "pilot_raw_pole_at_lambda_one":str(s.N(raw_pole,18)),
        "scope":"One candidate tensor passes existing lower-point sewing and predicts its constant contact. No independently integrated five-point string datum has been supplied by the reviewed reports."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_fivepoint_evidence_results.json').write_text(
    json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
