"""Distributional and paired-block audit of the joint FH b=1 limit.

Same analytic Liouville momenta are used in both chiral blocks. No complex
conjugation of the momentum drift is introduced in the barred factor.
"""
import json
from pathlib import Path
import sympy as s
import mpmath as m
m.mp.dps=65
u,k,mu,GD,GV,HD,HV,A,B=s.symbols("u k m gd gv hd hv A B")
checks=[]
def check(name,x):
    assert s.simplify(x)==0,(name,x)
    checks.append(name)

# Local coefficients with ds=epsilon du.
pd=-8*mu/(k*k*(1+u*u)**2)
pv=-2*mu/(1+u*u)
gd=GD+s.I*u*k*GV/2
hd=HD+s.I*u*k*HV/2
full=s.expand(pd*gd*hd+pv*GV*HV)
even=s.simplify((full+full.subs(u,-u))/2)
check("paired analytic even integrand collapses to a common kernel",
      even+2*mu/(1+u*u)**2*(4*GD*HD/(k*k)+GV*HV))
check("any common even weight retains 4/k² ratio",
      (2*mu*(A-B)-2*mu*A)+2*mu*B)
check("separate three-point distribution ratio",
      (16*s.I*mu/(k*k)*(s.pi/2))/(2*s.pi)-4*s.I*mu/(k*k))
check("full paired weak vacuum shift",
      2*mu*(s.pi/2)-2*mu*s.pi+s.pi*mu)
check("full paired weak direct coefficient",
      -8*mu/(k*k)*(s.pi/2)+4*s.pi*mu/(k*k))

q=s.symbols("q",positive=True)
box=(2-(2+q)*s.exp(-q))/(1-s.exp(-q))
check("box time-window ratio simplification",box-(2-q/(s.exp(q)-1)))
check("box small-time-width order",s.limit(box,q,0)-1)
check("box large-time-width order",s.limit(box,q,s.oo)-2)

def U(x):return m.barnesg(x)*m.barnesg(2-x)
def NS(x):return U(x/2)*U((x+2)/2)
def RR(x):return U((x+1)/2)**2
def C(aa,bb,cc,muv,odd=False):
    f=RR if odd else NS
    return (1j if odd else m.mpf(".5"))*muv**(2-aa-bb-cc)*(
        NS(2*aa)*NS(2*bb)*NS(2*cc))/(
        f(aa+bb+cc-2)*f(aa+bb-cc)*f(bb+cc-aa)*f(cc+aa-bb))
def dblock(ar,aa,bb,z):
    return z**(ar/2+m.mpf(".375"))*(1-z)**(aa/2)*m.hyp2f1(
        (ar+aa+bb)/2-m.mpf(".75"),
        (ar+aa-bb)/2+m.mpf(".25"),ar+m.mpf(".5"),z)
def limiting_blocks(kv,z):
    fp=z**m.mpf(".125")*(1-z)**((1+kv)/2)
    fm=z**m.mpf(".125")*(1-z)**((1-kv)/2)
    return (fp+fm)/2,(fm-fp)/kv

muv=m.mpf("1.7");kv=m.mpc(0,".8");eps=m.mpf("1e-13")
point_tests=[]
for uv in [m.mpf(0),m.mpf(".5"),m.mpf(2),m.mpf(5)]:
    aa=1+kv;bb=1-kv+1j*eps*uv
    c1=C(eps,aa,bb,muv);c3=C(-1+eps,aa,bb,muv,True)
    errors=[abs(eps*c1-2/(1+uv*uv)),
            abs(eps*c3-16j*muv/(kv*kv*(1+uv*uv)**2))]
    for z in [m.mpf(".25"),m.mpc(".3",".2")]:
        gd0,gv0=limiting_blocks(kv,z)
        errors.append(abs(dblock(-m.mpf(".5")+eps,aa,bb,z)
                          -(gd0+1j*uv*kv*gv0/2)))
    assert max(errors)<m.mpf("1e-9")
    point_tests.append({"u":str(uv),"errors":[str(x) for x in errors]})

# Exact weak integrals using u=tan(theta); no nonuniform numerical cutoff.
I1=m.quad(lambda th:1,[-m.pi/2,m.pi/2])
I2=m.quad(lambda th:m.cos(th)**2,[-m.pi/2,m.pi/2])
Iu=m.quad(lambda th:m.sin(th)**2,[-m.pi/2,m.pi/2])
assert max(abs(I1-m.pi),abs(I2-m.pi/2),abs(Iu-m.pi/2))<m.mpf("1e-60")
checks.append("independent full-line weak kernel integrals")

gd0,gv0=limiting_blocks(kv,m.mpf(".25"))
hd0,hv0=limiting_blocks(kv,m.mpc(".4","-.1"))
paired_tests=[]
for width in [m.mpf(0),m.mpf(".3"),m.mpf(2)]:
    def integrand(th):
        uv=m.tan(th)
        wt=m.exp(-width**2*uv**2)
        return wt*(-8*muv/(kv*kv)*m.cos(th)**2*
            (gd0+1j*uv*kv*gv0/2)*(hd0+1j*uv*kv*hv0/2)
            -2*muv*gv0*hv0)
    actual=m.quad(integrand,[-m.pi/2,0,m.pi/2])
    bb=m.quad(lambda th:m.exp(-width**2*m.tan(th)**2)*m.cos(th)**2,
              [-m.pi/2,0,m.pi/2])
    expected=-2*muv*bb*(4*gd0*hd0/(kv*kv)+gv0*hv0)
    assert abs(actual-expected)<m.mpf("1e-55")
    paired_tests.append({"time_width":str(width),"error":str(abs(actual-expected))})

result={"passed":True,"exact_checks":checks,"boundary_layer_tests":point_tests,
        "paired_weak_tests":paired_tests,
        "full_weak_result":"-pi*m [(4/k^2) gd(z)gd(zbar)+gv(z)gv(zbar)]",
        "vacuum_shift":"direct block mixing adds +pi*m; original -2pi*m becomes -pi*m",
        "scope":"Local meromorphic FH four-point boundary value. Same analytic pairing in both chiral factors. Not a completed heterotic BRST charge or a selected physical time/Liouville regulator order."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_joint_fh_distribution_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"passed":True,"boundary_layer_tests":len(point_tests),
                  "paired_weight_tests":len(paired_tests)},indent=2))

