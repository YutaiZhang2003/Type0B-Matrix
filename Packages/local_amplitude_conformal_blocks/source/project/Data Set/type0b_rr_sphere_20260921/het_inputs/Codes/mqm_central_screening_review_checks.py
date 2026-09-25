"""Independent one-screening residue check; no heterotic Ward coefficient inferred."""
import json
from pathlib import Path
import mpmath as m
m.mp.dps=80
gam=lambda x:m.gamma(x)/m.gamma(1-x)
checks=[]
def check(name,ok,detail=None):
    assert ok,(name,detail)
    checks.append(dict(name=name,passed=True,detail=detail))
for bs,As in [(".5",".6"),(".6",".7"),(".8",".8")]:
    b,A=m.mpf(bs),m.mpf(As)
    C=1-b*b
    assert b*b<A<1 and 0<C<1
    # Gaussian integration over the screening plane followed by the radial
    # Schwinger parameter integral leaves this ordinary convergent beta integral.
    beta=m.quad(lambda u:u**(-C)*(1-u)**(-A),[0,m.mpf(".5"),1])
    direct=m.pi*m.gamma(A+C-1)*beta/(m.gamma(A)*m.gamma(C))
    formula=m.pi*gam(1-A)*gam(b*b)*gam(A-b*b)
    check("convergent_beta_"+bs,abs(direct/formula-1)<m.mpf("1e-14"),
          dict(value=m.nstr(formula,35),relative_error=m.nstr(abs(direct/formula-1),8)))
b=1-m.mpf("1e-12")
check("renormalized_screening_ratio_two",
      abs(gam(b*b)/gam((1+b*b)/2)-2)<m.mpf("1e-10"))
for alpha in [m.mpf(".7"),m.mpf("1.2"),1+m.j*m.mpf(".8")]:
    exact=m.pi*gam(1-alpha*b)*gam(b*b)*gam(alpha*b-b*b)/gam((1+b*b)/2)
    target=-2*m.pi/(alpha-1)**2
    check("fixed_generic_alpha_"+str(alpha),abs(exact/target-1)<m.mpf("1e-9"),
          dict(value=str(exact),target=str(target)))
Path('data_exports/mqm/mqm_central_screening_review_results.json').write_text(
    json.dumps(dict(check_count=len(checks),checks=checks,
    scope="One-screening analytic residue only; overall action/descendant and heterotic BRST/PCO factors omitted."),indent=2)+"\n")
print(json.dumps(dict(check_count=len(checks),all_passed=True),indent=2))

