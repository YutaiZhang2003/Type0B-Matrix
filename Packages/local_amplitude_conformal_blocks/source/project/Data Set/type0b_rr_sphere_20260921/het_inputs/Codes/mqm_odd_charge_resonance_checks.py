"""Independent checks of the b=1 Ramond fusion prescription.
Sources: hep-th/0307195 (3.35)-(3.41), hep-th/0202032 (2.30)-(2.32).
No scattering target or inverse-fitted coefficient is used.
"""
from fractions import Fraction as F
import json
from pathlib import Path
import mpmath as mp
mp.mp.dps=70
checks=[]
def record(name, ok, detail=None):
    assert ok, (name,detail)
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
def gam(x):
    return mp.gamma(x)/mp.gamma(1-x)
def coeff(alpha,b):
    x=alpha*b-b*b/2
    y=alpha*b+mp.mpf(".5")
    # 1/gamma(y) vanishes at y=0, so retain the regularized reciprocal.
    return b*b*mp.gamma(x)*mp.rgamma(1-x)*mp.gamma(1-y)*mp.rgamma(y)
eps=mp.mpf("1e-12")
b=1+eps
# gamma(z)gamma(1-z)=1 cancels the two cosmological renormalization factors.
z=(1+b*b)/2
record("cosmological_gamma_factors_cancel",abs(gam(z)*gam(1-z)-1)<mp.mpf("1e-65"))
val=coeff(-mp.mpf(".5"),b)
record("fixed_alpha_limit_is_minus_one_third",abs(val+mp.mpf(1)/3)<mp.mpf("1e-10"),mp.nstr(val,55))
# For alpha=-1/(2b), y=0 exactly and x=-(1+b^2)/2 is finite off b=1.
x=-(1+b*b)/2
cross=b*b*mp.gamma(x)*mp.rgamma(1-x)*mp.gamma(1)*mp.rgamma(0)
record("cross_degenerate_path_zero",cross==0)
alpha=-mp.mpf(".47")
record("generic_b_one_gamma_recurrence",
       abs(coeff(alpha,mp.mpf(1))+1/(alpha-mp.mpf(".5"))**2)<mp.mpf("1e-65"))
alpha=-mp.mpf(".5")+eps
generic=-1/(alpha-mp.mpf(".5"))**2
record("generic_b_then_alpha_limit_minus_one",abs(generic+1)<mp.mpf("1e-10"),mp.nstr(generic,55))
same=coeff(-b/2,b)
record("same_degenerate_path_limit_minus_one_half",abs(same+mp.mpf(".5"))<mp.mpf("1e-10"),mp.nstr(same,55))
slopes=[]
for u in [mp.mpf("-1"),mp.mpf("0"),mp.mpf("1"),mp.mpf("2")]:
    val=coeff(-mp.mpf(".5")+u*eps,b)
    predicted=-(u-mp.mpf(".5"))/(u-mp.mpf("1.5"))
    record("linear_path_slope_"+str(u),abs(val-predicted)<mp.mpf("1e-9"))
    slopes.append(dict(slope=str(u),limit=mp.nstr(predicted,30),value=mp.nstr(val,40)))
# Dimension and local simple-pole kinematics, independent of the disputed coefficient.
h=lambda a:a*(2-a)/2
hR=F(1,16)+h(F(-1,2))
record("Ramond_weight_minus_nine_sixteenths",hR==F(-9,16))
record("spectator_spinor_dimension",2**11==2048)
record("antiholomorphic_current_weight_one",F(1,8)+hR+F(23,16)==1)
record("vacuum_block_double_pole",F(-1,4)+F(9,8)-F(23,8)==-2)
record("vacuum_rank_two_simple_pole",F(-1,4)+F(9,8)-F(15,8)==-1)
record("direct_same_sign_rank_four_simple_pole",F(1,4)-F(3,8)-F(7,8)==-1)
record("direct_opposite_sign_rank_four_simple_pole",F(-1,4)+F(1,8)-F(7,8)==-1)
out=dict(check_count=len(checks),checks=checks,linear_paths=slopes,
         scope="Checks analytic fusion coefficient and local weights; not bulk BRST completion, PCO cohomology, Hilbert-domain action, or full charge closure.")
Path('data_exports/mqm/mqm_odd_charge_resonance_results.json').write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(dict(check_count=len(checks),all_passed=True),indent=2))

