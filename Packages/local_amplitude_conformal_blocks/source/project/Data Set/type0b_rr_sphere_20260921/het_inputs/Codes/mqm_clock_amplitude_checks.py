"""Independent CLOCK tree audit from a nilpotent-source expansion of the map.

K2/K3 are generated from (q0+j) K^(-iw), not transcribed from the report.
The source labels are nilpotent only to select a functional derivative.
No claim of a global quantum clock is tested here.
"""
from functools import lru_cache
from itertools import permutations
from pathlib import Path
import ast
import json
import sys
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Codes"))
from spin23_genuine_formulas import (s_to_svv_raw_candidate,s_to_sss_raw_candidate,
                                     v_to_vss_raw_candidate)
from so7e8_ns_threepoint import v_to_vs_raw

I=s.I
D=lambda x:1+I*x
P=lambda x,y:1+x*x+y*y+x*y
ef=lambda species,w:D(w) if species==0 else s.Integer(1)
lf=lambda species,w:D(-w) if species==0 else s.Integer(1)
tests=[]


def equal(name,x,y=0):
    if s.expand(s.together(x-y).as_numer_denom()[0]) != 0:
        raise AssertionError((name,s.factor(x-y)))
    tests.append(name)


def add(*polys):
    ans={}
    for poly in polys:
        for key,c in poly.items():
            ans[key]=ans.get(key,0)+c
    return ans


def mul(a,b):
    out={}
    for (ga,ma),ca in a.items():
        for (gb,mb),cb in b.items():
            if ma&mb:continue
            key=(ga+gb,ma|mb)
            out[key]=out.get(key,0)+ca*cb
    return out


def scale(a,c,degree=0):
    return {(g+degree,m):c*v for (g,m),v in a.items()}


@lru_cache(None)
def kernel(species,w,legs):
    n=len(legs)
    def current(sp):
        return {(0,1<<j):s.Integer(1) for j,(a,k) in enumerate(legs) if a==sp}
    def R(poly):
        return {(g,mask):coef/D(sum(legs[j][1] for j in range(n) if mask&(1<<j)))
                for (g,mask),coef in poly.items()}
    sourceA=scale(R(current(0)),s.sqrt(2),1)
    sourceB=scale(R(add(*(mul(current(a),current(a)) for a in range(3)))),s.Rational(1,2),2)
    x=add(sourceA,sourceB)
    power={(0,0):s.Integer(1)}
    multiplier={}
    for k in range(n+1):
        coefficient=s.prod(-I*w-j for j in range(k))/s.factorial(k)
        multiplier=add(multiplier,scale(power,coefficient))
        power=mul(power,x)
    j=current(species)
    if species==0:j=add(j,{(-1,0):s.sqrt(2)})
    beta=mul(j,multiplier)
    return ef(species,w)*beta.get((n-1,(1<<n)-1),0)


def forward3(incoming,outgoing,energies):
    a,b=outgoing
    x,y=energies
    E=x+y
    return ef(incoming,E)*E*lf(a,x)*x*kernel(b,y,((incoming,E),(a,-x)))


def forward4(incoming,outgoing,energies):
    a,b,c=outgoing
    x,y,z=energies
    E=x+y+z
    O=x+y
    direct=lf(a,x)*lf(b,y)*x*y*kernel(c,z,((incoming,E),(a,-x),(b,-y)))
    sequential=lf(a,x)*x*O*sum(
        kernel(b,y,((h,O),(a,-x)))*kernel(c,z,((incoming,E),(h,-O)))
        for h in range(3))
    return ef(incoming,E)*E*(direct+sequential)


def reverse(outgoing,incoming,energies):
    E=sum(energies)
    return kernel(outgoing,E,tuple(zip(incoming,energies)))*s.prod(
        ef(a,k)*k for a,k in zip(incoming,energies))


a,b,c=s.symbols("a b c",positive=True,real=True)
E=a+b+c
W=E*a*b*c
B4=1+(E*E+a*a+b*b+c*c)/2
target4v=W/D(b+c)
targetmix=W*D(E)*(1+I*(2*E-a))/D(b+c)
target4s=W*(W*(1/D(a+b)+1/D(a+c)+1/D(b+c))+(1+2*I*E)*B4)
targetvss=W*(1+2*I*E+b*c/D(b+c))

# These are all generated directly from the specified source expansion.
equal("generated_K2_SVV",kernel(0,a+b,((1,a),(1,b))),-I*s.sqrt(2)*(a+b))
equal("generated_K2_SSS",kernel(0,a+b,((0,a),(0,b))),
      -I*s.sqrt(2)*(a+b)*P(a,b)/(D(a)*D(b)))
equal("generated_K2_VVS",kernel(1,a+b,((1,a),(0,b))),
      -I*s.sqrt(2)*(a+b)/D(b))
for name,inc,out,ee,target in [
    ('local_archives/root_cleanup/SVV',0,(1,1),(a,b),-I*s.sqrt(2)*(a+b)*a*b),
    ("SSS",0,(0,0),(a,b),-I*s.sqrt(2)*(a+b)*a*b*P(a,b)),
    ("VVS_crossed",1,(1,0),(a,b),-I*s.sqrt(2)*(a+b)*a*b)]:
    equal(name,forward3(inc,out,ee),target)
    equal(name+"_reverse",reverse(inc,out,ee),target)

amplitudes={}
cases=[
    ("VVVV",1,(1,2,2),(a,b,c),-I*target4v),
    ("SSVV",0,(0,1,1),(a,b,c),I*targetmix),
    ("SSSS",0,(0,0,0),(a,b,c),I*target4s),
    ("VVSS_extra",1,(1,0,0),(a,b,c),I*targetvss)]
for name,inc,out,ee,target in cases:
    actual=forward4(inc,out,ee)
    equal(name,actual,target)
    amplitudes[name]=str(target)
    for perm in permutations(range(3)):
        equal(name+"_Bose_"+str(perm),
              forward4(inc,tuple(out[j] for j in perm),tuple(ee[j] for j in perm)),target)
    equal(name+"_reverse",reverse(inc,out,ee),target)

# Incoming-vector parity is a SINGLE unitary on the incoming Fock space.
# All displayed manuscript forward amplitudes then have common quartic i.
equal("phase_dictionary_VVVV",-forward4(1,(1,2,2),(a,b,c)),I*target4v)
equal("phase_dictionary_SSVV",forward4(0,(0,1,1),(a,b,c)),I*targetmix)
equal("phase_dictionary_SSSS",forward4(0,(0,0,0),(a,b,c)),I*target4s)
equal("phase_dictionary_crossed_cubic",-forward3(1,(1,0),(a,b)),
      I*s.sqrt(2)*(a+b)*a*b)

# Canonical VVVV real-energy off-diagonal unitarity at tree order:
# the reversed S2 equals S2; its real part is supplied by two cubics sewn
# in the common unit-delta basis, with normalized singlet completeness.
q=b+c
raw=forward4(1,(1,2,2),(a,b,c))
cut=-2*W*q/(1+q*q)
equal("VVVV_real_energy_unitarity_raw",raw+s.conjugate(raw),cut)
canonical=raw/s.sqrt(W)
equal("VVVV_real_energy_unitarity_canonical",
      canonical+s.conjugate(canonical),-2*s.sqrt(W)*q/(1+q*q))
# After U_in=(-1)^NV, S0 is -1 in both these odd-vector sectors:
# S0†S2'+S2'†S0 retains the same cut.
equal("unitarity_after_incoming_parity",(-1)*(-raw)+s.conjugate(-raw)*(-1),cut)

# Continued bilinear sewing uses the internal current metric q(1+q^2).
# Residues below are coefficients of 1/D(q), not residues in the q variable.
q0=s.symbols("q0")
Wq=(a+q0)*a*b*(q0-b)
basic=-2*Wq*q0/(1-I*q0)
equal("null_pole_VVVV",basic.subs(q0,I),(-I*Wq).subs(q0,I))
equal("null_pole_SSVV",(basic*P(a,q0)).subs(q0,I),
      (-I*Wq*(a+q0)*a).subs(q0,I))
equal("null_pole_SSSS",(basic*P(a,q0)*P(b,q0-b)).subs(q0,I),
      (I*Wq**2).subs(q0,I))
equal("null_pole_extra_VVSS",(basic*P(b,q0-b)).subs(q0,I),
      (I*Wq*b*(q0-b)).subs(q0,I))

# A precise finite-mode quantum limitation. Normal ordering gives a nonzero
# two-particle component but no order-g^2 one-particle renormalization.
pair_creation=kernel(1,-2,((1,-1),(0,-1)))
equal("quantum_mode2_pair_creation",pair_creation,s.sqrt(2)*(-1+I))
equal("quantum_mode2_CCR_excess",pair_creation*s.conjugate(pair_creation),4)
for w in range(2,9):
    generated_excess=sum(
        s.expand_complex(kernel(1,-w,((1,-(w-n)),(0,-n)))
                         *s.conjugate(kernel(1,-w,((1,-(w-n)),(0,-n)))))
        *n*(w-n) for n in range(1,w))
    expected=2*w*w*sum(s.Rational(n*(w-n),1+n*n) for n in range(1,w))
    equal("transverse_CCR_excess_w"+str(w),generated_excess,expected)
    correction=-expected/(2*w)
    equal("required_linear_loop_real_part_w"+str(w),2*w*correction+generated_excess)
x,w=s.symbols("x w",positive=True,real=True)
primitive=w*s.log(1+x*x)/2-x+s.atan(x)
equal("continuum_CCR_integrand",s.diff(primitive,x),x*(w-x)/(1+x*x))
equal("continuum_CCR_integral",primitive.subs(x,w)-primitive.subs(x,0),
      w*s.log(1+w*w)/2-w+s.atan(w))

# Direct source evidence: read ONLY the simple raw VVVV ansatz with AST.
bundle=ROOT/"Codes/heterotic_so23_1to3_vvvv_fit_bundle/heterotic_so23_1to3_fast.py"
tree=ast.parse(bundle.read_text())
func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="pair_channel_ansatz")
import math
import numpy as np
ns={"math":math,"np":np,"Sequence":list}
exec(compile(ast.Module(body=[func],type_ignores=[]),str(bundle),"exec"),ns)
for aa,bb,cc in [(0.2,0.3,0.7),(0.61,0.43,0.91),(1.2,0.7,0.4)]:
    subs={a:aa,b:bb,c:cc}
    raw_vvv=ns["pair_channel_ansatz"]((aa,bb,cc,aa+bb+cc))[2]
    values={"VVVV":raw_vvv,"SSVV":s_to_svv_raw_candidate(aa,bb,cc),
            "SSSS":s_to_sss_raw_candidate(aa,bb,cc),
            "VVSS_extra":v_to_vss_raw_candidate(aa,bb,cc)}
    for name,raw_code in values.items():
        value=complex(s.sympify(amplitudes[name],locals={"a":a,"b":b,"c":c}).subs(subs).evalf())
        if abs(value-1j*raw_code/math.pi)>2e-11*max(1,abs(value)):
            raise AssertionError(("raw_code",name,subs,value,raw_code))
        tests.append("raw_repository_"+name+"_"+str((aa,bb,cc)))
    vvs=complex(forward3(1,(1,0),(a,b)).subs({a:aa,b:bb}).evalf())
    if abs(vvs+1j*math.sqrt(2)*v_to_vs_raw(aa,bb))>1e-12:
        raise AssertionError("crossed_cubic_code")
    tests.append("raw_crossed_cubic_"+str((aa,bb)))

output={"status":"passed","checks":len(tests),"amplitudes":amplitudes,
        "common_clock_raw_code_quartic_multiplier":"i*g^2/pi",
        "manuscript_basis":"U_in=(-1)^(number of vectors), U_out=1",
        "manuscript_C3":"-i*sqrt(2)*g","manuscript_C4":"i*g^2",
        "vector_elastic_in_manuscript_basis":-1,
        "scope":"tree kernels and one real-energy cut, not global quantum unitarity",
        "named_checks":tests}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_amplitude_results.json').write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k!="named_checks"},indent=2))
