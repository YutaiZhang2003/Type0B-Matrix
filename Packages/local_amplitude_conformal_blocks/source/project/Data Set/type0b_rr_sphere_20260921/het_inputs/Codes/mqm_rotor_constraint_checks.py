"""Exact local checks for the engineered, regulated matrix–rotor construction."""
import json
from pathlib import Path
import sympy as s

checks = {}

def check(name, expression):
    value = s.simplify(s.trigsimp(expression))
    assert value == 0, (name, value)
    checks[name] = True

a, b, tau = s.symbols('a b tau', positive=True)
x = b * s.cosh(tau)
v = b * s.sinh(tau)
rho = v / a
count = b**2 * (s.cosh(tau)*s.sinh(tau)-tau)/(2*a)
check('wall_time_of_flight', s.diff(x, tau)-v)
check('cumulative_count_density', s.diff(count, tau)-rho*v)
check('nonzero_wall_identity', x*x-v*v-b*b)

eps, L = s.symbols('eps L', positive=True)
up, um, rp, rm, rpp, rpm, vp, vm = s.symbols('up um rp rm rpp rpm vp vm', nonzero=True)
A1 = rp*up-rm*um
A2 = rpp*up**2-rpm*um**2
B1 = up/vp-um/vm
finite_difference = -s.sqrt(a)*(eps*A1+eps**2*A2/2)/(L+eps*B1)
check('constraint_linear_displacement',
      s.diff(finite_difference,eps).subs(eps,0)+s.sqrt(a)*A1/L)
check('constraint_quadratic_displacement',
      s.diff(finite_difference,eps,2).subs(eps,0)/2
      -s.sqrt(a)*(-A2/(2*L)+A1*B1/L**2))

z, theta = s.symbols('z theta', real=True)
n = s.Matrix([z,s.sqrt(1-z*z)*s.cos(theta),s.sqrt(1-z*z)*s.sin(theta)])
nz, nt = n.diff(z), n.diff(theta)
check('sphere_latitude_metric', nz.dot(nz)-1/(1-z*z))
check('sphere_fiber_metric', nt.dot(nt)-(1-z*z))
check('sphere_cross_metric', nz.dot(nt))

hp, hm = s.symbols('hp hm', positive=True)
y1,y2,y3 = s.symbols('y1 y2 y3')
secant=(y1*(hp+hm)+y2*(hp**2-hm**2)/2+y3*(hp**3+hm**3)/6)/(hp+hm)
check('unequal_grid_error', secant-y1-(hp-hm)*y2/2-(hp**2-hp*hm+hm**2)*y3/6)

v0 = s.symbols('v0', positive=True)
et, ex, Qt, Qx = s.symbols('eta_t eta_tau Q_t Q_tau', real=True)
r = v0/a+eps*ex/(s.sqrt(a)*v0)
flow = -eps*et/(s.sqrt(a)*r)
charge = v0*((eps*et)**2/(2*a*r)-a*a*r**3/6)
spin = v0*((eps*Qt+flow*eps*Qx/v0)**2/(a*r)-a*r*(eps*Qx/v0)**2)/2

def coefficient(expr, order):
    return s.diff(expr,eps,order).subs(eps,0)/s.factorial(order)

check('charge_quadratic', coefficient(charge,2)-(et*et-ex*ex)/2)
check('charge_cubic', coefficient(charge,3)+s.sqrt(a)*(ex**3+3*ex*et**2)/(6*v0**2))
check('director_quadratic', coefficient(spin,2)-(Qt*Qt-Qx*Qx)/2)
check('director_cubic', coefficient(spin,3)+s.sqrt(a)*(et*Qt*Qx+ex*(Qt**2+Qx**2)/2)/v0**2)

beta, J, kappa = s.symbols('beta J kappa', positive=True)
check('rotor_nematic_kinetic_coefficient',
      (beta**2/(2*a)).subs({beta**2:s.sqrt(2)*kappa,a:2*s.sqrt(2)*J})-kappa/(4*J))
check('rotor_nematic_gradient_coefficient',
      (a*beta**2/2).subs({beta**2:s.sqrt(2)*kappa,a:2*s.sqrt(2)*J})-2*J*kappa)

out = {'scope':'Exact local identities only; no quantum sea or scattering established',
       'checks_passed':len(checks),'checks':checks}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_rotor_constraint_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
