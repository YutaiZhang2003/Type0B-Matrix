"""Finite-time sinc test of the two candidate identity-resonance profiles."""
import json
from pathlib import Path
import sympy as s
import mpmath as mp

u=s.symbols("u",positive=True)
ratio=2-u/(s.exp(u)-1)
assert s.limit(ratio,u,0)==1
assert s.limit(ratio,u,s.oo)==2
assert s.simplify((2-(2+u)*s.exp(-u))/(1-s.exp(-u))-ratio)==0
assert s.series(ratio,u,0,4).removeO()==1+u/2-u*u/12

mp.mp.dps=60
rows=[]
for ep,T in [("0.3","2"),("0.7","5"),("1.1","0.8")]:
    ep,T=mp.mpf(ep),mp.mpf(T)
    tau=T/2
    def dt(x):
        return mp.sin(tau*x)/(mp.pi*x) if x else tau/mp.pi
    def f1(x): return dt(x)*2*ep/(ep*ep+x*x)
    # Strip the common factor i*m/k^2 from K3.
    def f3(x): return dt(x)*16*ep**3/(ep*ep+x*x)**2
    v1=2*mp.quadosc(f1,[0,mp.inf],omega=tau)
    v3=2*mp.quadosc(f3,[0,mp.inf],omega=tau)
    uu=ep*tau
    e1=2*(-mp.expm1(-uu))/ep
    e3=8*(2-(2+uu)*mp.exp(-uu))/ep
    err=max(abs(v1-e1),abs(v3-e3))
    assert err<mp.mpf("1e-45"),(ep,T,err)
    assert abs(v3/v1-4*(2-uu/mp.expm1(uu)))<mp.mpf("1e-45")
    rows.append({"epsilon":str(ep),"T":str(T),"u":str(uu),
                 "direct_I1":str(v1),"direct_I3_stripped":str(v3),
                 "ratio_stripped":str(v3/v1),"maximum_error":str(err)})

out={"passed":True,"symbolic_checks":4,"direct_oscillatory_checks":rows,
     "ratio":"(4 i m/k^2) [2-u/(exp(u)-1)], u=epsilon*T/2",
     "scope":"Assumed leading resonant profiles and s equal to energy mismatch after mass-shell conversion; no assertion that the worldsheet chooses a joint limit."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_resonant_time_order_results.json').write_text(json.dumps(out,indent=2)+"\n")
print("Passed four symbolic checks and three direct oscillatory quadratures.")
