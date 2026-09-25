"""Exact background identity and soft limits of the canonical clock."""
import json
from pathlib import Path
import sympy as s

checks=[]
def check(name, expression):
    value=s.simplify(expression)
    if value!=0:
        raise AssertionError((name,value))
    checks.append(name)

g,shift,ru,rjj=s.symbols("g shift Ru Rjj", real=True)
rho=1+g*shift/s.sqrt(2)
geff=g/rho
q0=s.sqrt(2)/g
Ks=g*g*((q0+shift)**2/2+(q0+shift)*ru+rjj/2)
Keff=1+s.sqrt(2)*geff*ru+geff**2*rjj/2
check("exact shifted stress clock",Ks-rho**2*Keff)
check("effective background",q0+shift-s.sqrt(2)/geff)

E=s.symbols("E",positive=True)
for n in range(1,7):
    scale=rho**(-(n-1)-2*s.I*E)
    check(f"tree background derivative n{n}",
          s.diff(scale,shift).subs(shift,0)+g/s.sqrt(2)*(n-1+2*s.I*E))

# Independent symbolic limits of the full raw-current functions.
a,b,eps=s.symbols("a b epsilon",positive=True)
D=lambda x:1+s.I*x
P=lambda x,y:1+x*x+y*y+x*y
incoming=a+b+eps
W=incoming*a*b*eps
C3=-s.I*s.sqrt(2)*g
C4=s.I*g*g
threeVV=C3*(a+b)*a*b
fourSVV=C4*W*D(incoming)*(1+s.I*(2*incoming-eps))/D(a+b)
threeSS=threeVV*P(a,b)
fourSSS=C4*W*(W*(1/D(a+b)+1/D(a+eps)+1/D(b+eps))
                +(1+2*s.I*incoming)*(1+(incoming**2+a*a+b*b+eps**2)/2))
fourVSS=C4*W*(1+2*s.I*incoming+b*eps/D(b+eps))
expected=-g/s.sqrt(2)*(1+2*s.I*(a+b))
for name,four,three in (("S_SVV",fourSVV,threeVV),
                        ("S_SSS",fourSSS,threeSS),
                        ("V_VSS",fourVSS,threeVV)):
    # Cancel the explicit soft-current factor before taking the limit.
    check(f"soft ratio {name}",
          s.cancel(four/(eps*three)).subs(eps,0)-expected)
check("outgoing clock origin removes phase",
      s.diff(rho**(-1),shift).subs(shift,0)+g/s.sqrt(2))

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "scope":"Classical background transport and three tree soft identities; no matrix or integrated BRST wall-operator identification."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_background_results.json').write_text(
    json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
