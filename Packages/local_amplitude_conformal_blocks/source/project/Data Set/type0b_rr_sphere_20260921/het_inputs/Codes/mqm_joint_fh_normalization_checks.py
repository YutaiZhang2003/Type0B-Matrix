"""Joint b=1 RR/NS normalization from the FH degenerate four-point function.

Uses FH hep-th/0202032 eqs. 2.31, 2.51, 2.53, 2.58; independent
Barnes-G implementation of exact-b=1 Upsilon verifies the crossing ratio.
No odd-charge Ward cancellation is imposed.
"""
import json
from pathlib import Path
import sympy as s
import mpmath as m
m.mp.dps=85
checks=[]

def check(name,x):
    assert s.simplify(x)==0,(name,x)
    checks.append(name)

k,v,ep=s.symbols("k v epsilon",nonzero=True)
aR=-s.Rational(1,2)+ep
a2=1+k
a3=1-k+v*ep
args=[
(aR+a2+a3)/2-s.Rational(3,4),
(aR+a2-a3)/2+s.Rational(1,4),
(aR-a2+a3)/2+s.Rational(1,4),
(aR-a2-a3)/2+s.Rational(5,4)
]
wanted=[(1+v)*ep/2,k+(1-v)*ep/2,-k+(1+v)*ep/2,(1-v)*ep/2]
for j,(a,b) in enumerate(zip(args,wanted)):
    check("FH Gamma argument "+str(j),a-b)
# gamma(k) gamma(-k)=-1/k^2 and gamma(2-epsilon)~-epsilon.
check("joint leading crossing ratio",
      -4/(1-v*v)*(-1/k**2)-4/((1-v*v)*k*k))
rvac=2/(1-v*v)
rdir=16*s.I/((1-v*v)**2*k*k)
ctilde=-2*s.I
check("joint residue ratio",
      rdir/(ctilde*rvac)+4/((1-v*v)*k*k))

def gg(x):return m.gamma(x)/m.gamma(1-x)
def U(x):return m.barnesg(x)*m.barnesg(2-x)
def NS(x):return U(x/2)*U((x+2)/2)
def RR(x):return U((x+1)/2)**2
def C(aa,bb,cc,mu,odd=False):
    f=RR if odd else NS
    top=NS(2*aa)*NS(2*bb)*NS(2*cc)
    bottom=f(aa+bb+cc-2)*f(aa+bb-cc)*f(bb+cc-aa)*f(cc+aa-bb)
    return (1j if odd else m.mpf(".5"))*mu**(2-aa-bb-cc)*top/bottom
def ratio_gamma(aR,aa,bb):
    ar=[(aR+aa+bb)/2-m.mpf(".75"),
        (aR+aa-bb)/2+m.mpf(".25"),
        (aR-aa+bb)/2+m.mpf(".25"),
        (aR-aa-bb)/2+m.mpf("1.25")]
    return -gg(m.mpf("1.5")-aR)**2/(aR-m.mpf(".5"))**2*m.fprod(gg(x) for x in ar)
def blocks(aR,aa,bb,z):
    gd=z**(aR/2+m.mpf(".375"))*(1-z)**(aa/2)*m.hyp2f1(
        (aR+aa+bb)/2-m.mpf(".75"),
        (aR+aa-bb)/2+m.mpf(".25"),aR+m.mpf(".5"),z)
    ar=2-aR
    gv=z**(ar/2-m.mpf(".125"))*(1-z)**(aa/2)*m.hyp2f1(
        (ar+aa+bb)/2-m.mpf("1.25"),
        (ar+aa-bb)/2-m.mpf(".25"),ar-m.mpf(".5"),z)
    return gd,gv

mu=m.mpf("1.7")
numeric=[]
for kval in [m.mpc(".3",".7"),m.mpc(0,".8"),m.mpc(".37",0)]:
    for slope in [m.mpf(0),m.mpf(1)/3,-m.mpf(".5")]:
        errs=[]
        for e in [m.mpf("1e-5"),m.mpf("1e-9"),m.mpf("1e-13")]:
            ar=-m.mpf(".5")+e;aa=1+kval;bb=1-kval+slope*e
            rg=ratio_gamma(ar,aa,bb)
            c1=C(e,aa,bb,mu)
            c3=C(-1+e,aa,bb,mu,True)
            cm=-2j*mu/(ar-m.mpf(".5"))**2
            ru=-c3/((-1+e)**2*cm*c1)
            assert abs(ru-rg)<m.mpf("1e-55")*max(1,abs(rg))
            target=4/((1-slope*slope)*kval*kval)
            rv=2/(1-slope*slope)
            rd=16j*mu/((1-slope*slope)**2*kval*kval)
            errors=[abs(rg-target),abs(e*c1-rv),abs(e*c3-rd)]
            errs.append(errors)
        assert max(errs[-1])<m.mpf("1e-9")
        numeric.append({"k":str(kval),"dual_drift":str(slope),
                        "crossing_target":str(target),
                        "vacuum_residue":str(rv),"direct_residue":str(rd),
                        "final_errors":[str(x) for x in errs[-1]]})

# Limiting blocks at fixed exact BPZ-dual external momenta.
block_tests=[]
for kval in [m.mpc(".3",".7"),m.mpc(0,".8")]:
    aa=1+kval;bb=1-kval
    for z in [m.mpf(".15"),m.mpf(".55"),m.mpc(".2",".17")]:
        fplus=z**m.mpf(".125")*(1-z)**((1+kval)/2)
        fminus=z**m.mpf(".125")*(1-z)**((1-kval)/2)
        gd0=(fplus+fminus)/2
        gv0=(fminus-fplus)/kval
        e=m.mpf("1e-15")
        gd,gv=blocks(-m.mpf(".5")+e,aa,bb,z)
        assert max(abs(gd-gd0),abs(gv-gv0))<m.mpf("1e-12")
        block_tests.append({"k":str(kval),"z":str(z),
                            "errors":[str(abs(gd-gd0)),str(abs(gv-gv0))]})

# Independent tests of the special function used for the structure constants.
for x in [m.mpc(".6",".3"),m.mpc("-1.3",".4"),m.mpc("2.4",".7")]:
    assert abs(U(x+1)-gg(x)*U(x))<m.mpf("1e-65")*max(1,abs(U(x+1)))
    assert abs(U(x)-U(2-x))<m.mpf("1e-65")*max(1,abs(U(x)))
checks += ["Barnes-G Upsilon shift and reflection at three complex points",
           "FH crossing ratio equals joint Upsilon structure-constant ratio",
           "nine correlated Gamma/Upsilon residue sequences",
           "six limiting hypergeometric block tests"]
result={"passed":True,"exact_and_family_checks":checks,
        "joint_sequences":numeric,"limiting_blocks":block_tests,
        "fixed_dual_result":"P_direct/P_vacuum -> 4/k^2",
        "FH_structure_ratio":"C3/(tildeC_minus C1) -> -4/k^2",
        "important_scope":"b=1 fixed first, exact NS BPZ-dual momenta held fixed while Ramond momentum tends -1/2. This is a common meromorphic four-point normalization; it is not yet the BRST/PCO projection of the heterotic charge."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_joint_fh_normalization_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"passed":True,"families":len(numeric),
                  "block_tests":len(block_tests),"result":result["fixed_dual_result"]},indent=2))

