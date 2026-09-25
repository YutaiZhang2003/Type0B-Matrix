"""Assemble charged-primary -> charged-primary + two singlets.

This checks the complete order-gamma^2 outgoing-operator assembly in the
long-string dictionary, not an independently constructed worldsheet dual.
External charged states are the normalized primary modes specified in the
companion charged-primary note. No manuscript files are modified.
"""
import json
from pathlib import Path
import sympy as s
x,y,z,h = s.symbols('x y z h', real=True)
E=x+y+z
Y=y+z
CE,Cxy,Cxz,Cx=s.symbols('C_E C_xy C_xz C_x')
D=lambda w:1+s.I*w
R0=lambda w:D(-w)/D(w)
A=lambda w,r:w-(h-1)*r
B=lambda n,r:n+(h-1)*r

def primary1(w,k):
    return -2*s.I*(w+(h-1)*k)/D(k)

def primary2(w,k,l):
    a=1-h
    b=h-1-s.I*w
    return a*(2*a-1)+b/D(k+l)+2*b*(b-1)/(D(k)*D(l))+2*a*b*(1/D(k)+1/D(l))

def singlet1_boson(w,k,l):
    b=-s.I*w
    return b/D(w)+2*b*(b-1)/(D(k)*D(l))+b*(1/D(k)+1/D(l))

def singlet1_stress(w):
    return -2*s.I*w/D(w)

def singlet2_stress_boson(w,k,r):
    b=-s.I*w
    return 4*b*(b-1)/(D(k)*D(r))+2*b/D(r)

# <T_r O_w Odag_-n>, n=w+r.
def one_T(w,r,Cn,Cw):
    n=w+r
    return -A(w,r)*Cn+B(n,r)*Cw

# <T_z T_y O_x Odag_-E>.
two_T=(-A(x,y)*one_T(x+y,z,CE,Cxy)
       +B(E,y)*one_T(x,z,Cxz,Cx))

terms={}
terms['primary_quadratic']=R0(y)*R0(z)*2*y*z*primary2(x,-y,-z)*CE
terms['singlet_boson_sequential']=R0(z)*2*z*Y*singlet1_boson(y,-z,Y)*primary1(x,-Y)*CE
terms['singlet_y_stress_primary']=R0(z)*z*singlet1_stress(y)*primary1(x,-z)*one_T(x+z,y,CE,Cxz)
terms['singlet_z_stress_primary']=R0(y)*y*singlet1_stress(z)*primary1(x,-y)*one_T(x+y,z,CE,Cxy)
terms['singlet_y_cubic']=R0(z)*z*singlet2_stress_boson(y,-z,Y)*one_T(x,Y,CE,Cx)
terms['two_singlet_stresses']=singlet1_stress(z)*singlet1_stress(y)*two_T
total=s.factor(sum(terms.values()))
prefactored=s.factor(total*D(y)*D(z)/(2*y*z))

# For h=1, gamma^2/2 and canonical S normalization leave expected
# i*gamma^2*sqrt(y*z)*sqrt(x/E)*E*[H_E+y*z/D(Y)]/[Q(y)Q(z)].
expected=s.I*E*(1+2*s.I*E+y*z/D(Y))*Cx
base=s.I*B(E,Y)*(1+2*s.I*E+y*z/D(Y))*Cx
trial=base-2*h*(h-1)*y*z*Cx

zero=lambda expr:s.simplify(s.expand(s.cancel(expr)))==0
checks={}
for C in [CE,Cxy,Cxz]:
    checks['cancel_'+str(C)]=zero(s.diff(prefactored,C))
checks['full_kernel']=zero(prefactored-trial)
checks['Cartan_h1']=zero(prefactored.subs(h,1)-expected)
checks['outgoing_singlet_exchange']=zero(
    trial-trial.xreplace({y:z,z:y}))
checks['h2_contact']=zero((prefactored-base).subs(h,2)+4*y*z*Cx)
checks['Virasoro_ordering']=zero(
    two_T-two_T.xreplace({y:z,z:y,Cxy:Cxz,Cxz:Cxy})
    -(z-y)*one_T(x,y+z,CE,Cx))

# Derive the kernels independently as coefficients in the literal
# primary map P^(1-h) K^(h-1-i*x), P=(1+lambda*j)^2.
# Nonlocal R multiplies a Fourier mode k by 1/D(k).
l,j1,j2=s.symbols('lambda j1 j2')
k1,k2=s.symbols('k1 k2')
def polynomial_power(a0,a1,a2,power):
    return a0**power*(1+l*power*a1/a0+
        l*l*(power*a2/a0+power*(power-1)*a1*a1/(2*a0*a0)))
Ppower=polynomial_power(1,2*(j1+j2),(j1+j2)**2,1-h)
Kpower=polynomial_power(1,2*(j1/D(k1)+j2/D(k2)),
    j1*j1/D(2*k1)+2*j1*j2/D(k1+k2)+j2*j2/D(2*k2),h-1-s.I*x)
literal=s.Poly(s.expand(Ppower*Kpower),l,j1,j2)
checks['primary_linear_from_literal_map']=zero(
    literal.coeff_monomial(l*j1)-primary1(x,k1))
checks['primary_quadratic_from_literal_map']=zero(
    literal.coeff_monomial(l*l*j1*j2)-2*primary2(x,k1,k2))

assert all(checks.values()),checks
result={
    'status':'passed',
    'scope':'Complete two-singlet outgoing-operator assembly for normalized charged primary states in the proposed ripple dictionary; not an independent worldsheet amplitude.',
    'checks':checks,
    'normalization':'Unit-delta states, energy delta removed; E=x+y+z and physical Rindler frequencies in D,Q.',
    'amplitude':'gamma^2*sqrt(y*z)/(Q(y)*Q(z))*(x/E)^(h-1/2)*{i*[E+(h-1)*(y+z)]*[1+2*i*E+y*z/D(y+z)]-2*h*(h-1)*y*z}',
    'h2_contact_inside_braces':'-4*y*z',
    'outgoing_operator_contributions':list(terms),
}
output=Path(__file__).resolve().parents[1]/'data_exports/heterotic_long_string_trees_20260912'/'charged_primary_two_singlets_results.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
