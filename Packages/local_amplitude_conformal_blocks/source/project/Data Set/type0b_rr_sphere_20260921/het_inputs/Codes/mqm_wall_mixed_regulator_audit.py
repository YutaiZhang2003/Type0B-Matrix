"""Independent cutoff integration of one SSVV sample.

Uses the audited signed vertex generator but replaces the spectral
pole-subtraction quadrature by symmetric pole excision, and evaluates
the parent quartic by a direct spatial cutoff instead of F4 transforms.
"""
import json
from pathlib import Path
import mpmath as m
import sympy as s
from mqm_wall_mixed_quartic_checks import make_vertex

m.mp.dps=55
w1,w2,w3=s.Rational(2,5),s.Rational(3,5),s.Rational(7,10)
w0=w1+w2+w3
q=w2+w3
W=m.mpf(str(w0*w1*w2*w3))
uv=m.pi**2*W*m.mpf(str(1+w0*w1))/4

def richardson_odd(values):
    a=2*values[1]-values[0]
    b=2*values[2]-values[1]
    return (8*b-a)/7

def integrate_exchange(left,right,e,constrained,uv_constant):
    e=m.mpf(str(e))
    lv,_=left;rv,_=right
    def numerator(p):
        return lv(p)*rv(p)/(1+p*p if constrained else 1)
    def integrand(p):
        return numerator(p)/(e*e-p*p)
    L=m.mpf(35)
    # The only nonzero polynomial tail belongs to the scalar channel;
    # its full rational tail is known analytically from the UV audit.
    tail=-uv_constant*(1+e*e)*(m.pi/2-m.atan(L)) if constrained else 0
    values=[]
    for eps in [m.mpf('0.0001'),m.mpf('0.00005'),m.mpf('0.000025')]:
        value=m.quad(integrand,[0,e/2,e-eps])+m.quad(integrand,[e+eps,e+1,e+6,L])
        values.append(value-uv_constant*L+tail)
    return -(2/m.pi)*richardson_odd(values)+1j*numerator(e)/e

left=make_vertex(['S','S','S'],[w0,w1],[-w0,w1,q])
right=make_vertex(['a','a','S'],[w2,w3],[w2,w3,-q])
exchange=integrate_exchange(left,right,q,True,uv)
for j,k in [(w2,w3),(w3,w2)]:
    e=w0-j
    left=make_vertex(['S','a','a'],[w0,j],[-w0,j,e])
    right=make_vertex(['S','a','a'],[w1,k],[w1,k,-e])
    exchange+=integrate_exchange(left,right,e,False,m.mpf(0))

qq=m.mpf(str(q));r=m.mpf(str(w0+w1));d=m.mpf(str(w2-w3))
P2=-qq**2+(r*r+d*d)/2
def parent_integrand(t):
    P=-m.mpf('1.5')+m.cos(2*qq*t)/2-m.cos((r+d)*t)/2-m.cos((r-d)*t)/2
    return P/m.sinh(t)**4
parent_values=[]
for eps in [m.mpf('0.0001'),m.mpf('0.00005'),m.mpf('0.000025')]:
    value=m.quad(parent_integrand,[eps,1,4,10,m.inf])
    parent_values.append(value+2/(3*eps**3)-(P2+m.mpf(4)/3)/eps)
parent=W*richardson_odd(parent_values)
bulk_overlap=m.quad(lambda t:(m.sin(qq*t)/m.sinh(t))**2,[0,1,4,10,m.inf])
deformation=-W*m.mpf(str(w0*w1))*(2*bulk_overlap+1)
result=-(exchange+parent+deformation)/W
reference=m.mpc('-6.5637423209597483028075064062816678','3.7286245353159851301115241635687732')
target=1+2j*m.mpf(str(w0))-m.mpf(str(w0*w1))/(1+1j*qq)
assert abs(result-reference)<m.mpf('1e-16'),(result,result-reference)
out={'sample':['0.4','0.6','0.7'],
     'method':'common spectral cutoff with symmetric pole excision and odd-power extrapolation; direct spatial contact cutoff',
     'result':m.nstr(result,35),'difference_from_original_quadrature':m.nstr(result-reference,20),
     'difference_from_target':m.nstr(result-target,35),
     'scope':'same specified nested finite parts, not a derived common finite-matrix regulator limit'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_mixed_regulator_audit_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
