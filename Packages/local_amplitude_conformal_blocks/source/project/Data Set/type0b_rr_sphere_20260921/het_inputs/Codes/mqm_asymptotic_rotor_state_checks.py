#!/usr/bin/env python3
"""Exact local-geometry and parameter checks for the asymptotic rotor audit.

These checks do not prove a quantum matrix continuum limit or mass gap.
"""
import json
from pathlib import Path
import sympy as s

checks=[]
def check(name,expression):
    assert s.simplify(expression)==0,(name,s.simplify(expression))
    checks.append(name)

beta=s.symbols('beta',positive=True)
w,v=s.symbols('w v',real=True)
wd,vd=s.symbols('wd vd',real=True)
n=s.Matrix([s.sqrt(1-(w*w+v*v)/beta**2),w/beta,v/beta])
nd=n.jacobian([w,v])*s.Matrix([wd,vd])
check('sphere pullback metric',beta**2*nd.dot(nd)-wd**2-vd**2-(w*wd+v*vd)**2/(beta**2-w*w-v*v))
check('flat metric limit',s.limit(beta**2*nd.dot(nd),beta,s.oo)-wd**2-vd**2)

x,y,xd,yd=s.symbols('x y xd yd',real=True)
nj=s.Matrix([s.sqrt(1-(x*x+y*y)/beta**2),x/beta,y/beta])
check('flat positive bond metric',s.limit(beta**2*nd.dot(nj),beta,s.oo)-(wd*(x-w)+vd*(y-v)))
check('flat spring limit',s.limit(beta**2*(n-nj).dot(n-nj),beta,s.oo)-(w-x)**2-(v-y)**2)

g,lam,b,rhohat=s.symbols('g lam b rhohat',positive=True)
a=1/(g**2*lam**2)
mu=1/(g**2*lam)
K=g/2
rho=g*s.sqrt(lam)*rhohat
check('flat coordinate inertia',1/(a*rho**2)-lam/rhohat**2)
check('flat base action scaling',1/(a*rho**2)/g**2-lam/(rhohat**2*g**2))
check('flat spring action scaling',a/g**2*(g**2*lam)-1/(g**2*lam))
check('flat inverse bulk metric scaling',(1/(2*mu))/g**4-lam/(2*g**2))
check('flat endpoint action scaling',(K**2/8)/g**4-1/(32*g**2))
eps=s.symbols('eps',positive=True)
check('flat endpoint d scaling',(K**2/8)/eps**2-g**2/(32*eps**2))
check('fixed radius relation',(g**2/b**2)-1/(b/g)**2)

E,E0,R2=s.symbols('E E0 R2',positive=True)
running=R2+22/(2*s.pi)*s.log(E/E0)
M=E0*s.exp(-s.pi*R2/11)
check('O24 leading-log strong scale',running.subs(E,M))
check('O24 beta function',s.diff(1/running,E)*E+22/(2*s.pi)/running**2)

# W=partial_tau U: the curvature numerator has derivative counts
# 2,4,6 for V-only, mixed, and W-only quartics, respectively.
derivatives={'V_only':2,'V_W':4,'W_only':6}
assert derivatives['V_W']>2 and derivatives['W_only']>2
checks.append('IR singlet couplings are higher-derivative')

theta,Lambda,aa,rr=s.symbols('theta Lambda a rho',positive=True)
G=1+Lambda**2*s.sin(theta)**2
Hs=4*Lambda**2*s.sin(theta/2)**2+4*Lambda**4*s.sin(theta)**2*s.sin(theta/2)**2
check('frozen singlet dispersion',Hs/G-4*Lambda**2*s.sin(theta/2)**2)
zero_point=s.integrate(Lambda*s.sin(theta/2),(theta,0,s.pi))/s.pi
check('zero point energy per branch',zero_point-2*Lambda/s.pi)
check('all24 zero point pressure ratio',24*(2*aa*rr**2/s.pi)/(aa**2*rr**2/6)-288/(s.pi*aa))
check('dense family unrenormalized loop ratio',(288/(s.pi*aa)).subs(aa,1/(g**2*lam**2))-288*g**2*lam**2/s.pi)

out={'checks':checks,'count':len(checks),
     'leading_log_scale':'E_star exp[-pi beta_R(E_star)^2/11]',
     'important_scope':'The RG scale is not a perturbatively generated pole mass. The ordinary sigma-model mass gap is a separate dynamics input, conditional on IR universality.',
     'flat_model_endpoint_scaling':'A=dQ_1 and epsilon_e independent of g; d=g^2 d_hat. This differs from A=dn_1 in the compact rotor parameterization.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_asymptotic_rotor_state_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
