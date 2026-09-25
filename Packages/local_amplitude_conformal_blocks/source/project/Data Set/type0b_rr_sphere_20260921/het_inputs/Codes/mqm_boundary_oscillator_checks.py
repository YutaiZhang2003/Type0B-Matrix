"""Normalizable full-Legendre/AW boundary oscillator, in momentum space.

Dimensionless operator: -epsilon*d_z^2 + Gaussian[t_mu](z),
where u^3+mu*u=z and t_mu=mu*u^2/2+3*u^4/4.
No matrix bath, continuum state, or string amplitude is computed.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gamma, hyp1f1, roots_legendre
from scipy.optimize import brentq

MOMENT=gamma(7/6)/np.sqrt(np.pi)


def legendre_u(z,mu):
    z=np.asarray(z)
    if mu==0:return np.cbrt(z)
    return 2*np.sqrt(mu/3)*np.sinh(np.arcsinh(3*np.sqrt(3)*z/(2*mu**1.5))/3)


def potential(z,mu,quad_order=20):
    z=np.asarray(z,dtype=float)
    if mu==0:
        g=.75*MOMENT*hyp1f1(-2/3,.5,-z*z)
        gp=2*MOMENT*z*hyp1f1(1/3,1.5,-z*z)
        return g,gp
    maximum=float(np.max(abs(z)))
    # Resolve the narrow cubic-inversion crossover near s=0 as mu->0.
    edges=np.unique(np.r_[0,1e-5,1e-4,1e-3,.01,.03,.1,
                            np.arange(.25,maximum+10.25,.25)])
    nodes,weights=roots_legendre(quad_order)
    s=((edges[:-1,None]+edges[1:,None])/2
       +(edges[1:,None]-edges[:-1,None])*nodes/2).ravel()
    weights=((edges[1:,None]-edges[:-1,None])*weights/2).ravel()/np.sqrt(np.pi)
    u=legendre_u(s,mu)
    t=.5*mu*u*u+.75*u**4
    out=np.zeros_like(z);deriv=np.zeros_like(z)
    for start in range(0,len(z),192):
        zz=z[start:start+192,None]
        a=np.exp(-(zz-s)**2);b=np.exp(-(zz+s)**2)
        out[start:start+192]=(a+b)@(weights*t)
        deriv[start:start+192]=(a-b)@(weights*u)
    return out,deriv


def solve(mu,epsilon,n=4095,L=12.,quad_order=20,raw_massless=False):
    dz=2*L/(n+1);z=np.linspace(-L+dz,L-dz,n)
    if raw_massless:
        g=.75*abs(z)**(4/3);gp=np.sign(z)*abs(z)**(1/3);threshold=0.
    else:
        positive=z[n//2:]
        gplus,gpplus=potential(positive,mu,quad_order)
        g=np.r_[gplus[:0:-1],gplus]
        gp=np.r_[-gpplus[:0:-1],gpplus]
        threshold=float(gplus[0])
    kinetic_diag=2*epsilon/dz**2
    energy,vec=eigh_tridiagonal(kinetic_diag+g,
                    np.full(n-1,-epsilon/dz**2),
                    select='i',select_range=(0,1),lapack_driver='stebz')
    state=vec[:,0];prob=state*state
    # eigh uses Euclidean normalization; prob already carries grid weight.
    K0=np.sum(np.diff(np.r_[0.,state,0.])**2)/dz**2
    z2=float(prob@(z*z));v2=float(prob@(gp*gp))
    virial=float(prob@(z*gp)-2*epsilon*K0)
    return {"mu":mu,"epsilon":epsilon,"grid_points":n,"half_box":L,
            "ground_energy":float(energy[0]),"free_threshold":threshold,
            "ground_above_threshold":float(energy[0]-threshold),
            "gap":float(energy[1]-energy[0]),
            "z_variance":z2,"minus_d2_expectation":float(K0),
            "gprime_squared_expectation":v2,
            "virial_residual":virial,
            "boundary_probability":float(prob[0]+prob[-1])}


def gaussian_upper_bound(epsilon):
    x=brentq(lambda x:MOMENT*x*x/(1+x)**(1/3)-epsilon,
             1e-16,max(1.,(2*epsilon/MOMENT)**.6+1.))
    return epsilon/(2*x)+.75*MOMENT*(1+x)**(2/3)


def richardson_pair(a,b):
    keys=["ground_energy","ground_above_threshold","gap","z_variance",
          "minus_d2_expectation","gprime_squared_expectation"]
    return {key:(4*b[key]-a[key])/3 for key in keys}


def main():
    out={"definition":"H/Ec=-epsilon*d_z^2+g_mu; reported ground_above_threshold subtracts only g_mu(0)."}
    # Finite-mass convolution is independently compared at two quadrature orders.
    qerrs=[]
    for mu in [4.,1.,.1,.01]:
        points=np.array([0.,.03,.4,1.3,3.7,8.])
        g1,d1=potential(points,mu,16)
        g2,d2=potential(points,mu,28)
        qerrs.append(float(max(np.max(abs(g1-g2)),np.max(abs(d1-d2)))))
    assert max(qerrs)<2e-10
    out["quadrature_order_discrepancies"]=qerrs

    masses=[4.,1.,.3,.1,.03,.01,0.]
    out["fixed_epsilon_mass_limit"]=[solve(mu,1.) for mu in masses]
    for result in out["fixed_epsilon_mass_limit"]:
        mu=result["mu"]
        if mu>0:
            result["Gaussian_ground_above_threshold"]=np.sqrt(1/(2*mu))
            result["Gaussian_gprime_squared"]=np.sqrt(.5)*mu**(-1.5)
    for a,b in zip(out["fixed_epsilon_mass_limit"],out["fixed_epsilon_mass_limit"][1:]):
        assert b["ground_energy"]>a["ground_energy"]
    out["grid_convergence"]=[solve(0.,1.,n=n) for n in [2047,4095,8191]]
    out["box_convergence"]=[solve(0.,1.,n=4095,L=L) for L in [10.,12.,14.]]
    fine=out["grid_convergence"][-1]
    assert fine["boundary_probability"]<1e-20
    assert abs(fine["virial_residual"])<2e-6
    # Raw grid errors are second order; Richardson extrapolation is reported.
    coarse,medium,fine=out["grid_convergence"]
    extrap={}
    for name in ["ground_energy","gap","z_variance","minus_d2_expectation","gprime_squared_expectation"]:
        e1=(4*medium[name]-coarse[name])/3
        e2=(4*fine[name]-medium[name])/3
        assert abs(e2-e1)<2e-8,(name,e2-e1)
        extrap[name]={"extrapolated":e2,"successive_extrapolation_difference":abs(e2-e1)}
    out["massless_epsilon1_Richardson"]=extrap

    # Gaussian-ordering harmonic regime epsilon->0.
    small=[]
    for eps in [1e-2,1e-3,1e-4]:
        result=solve(0.,eps,n=4095,L=3.)
        result["ground_harmonic_plus_quartic"]=np.sqrt(MOMENT*eps)-eps/12
        result["gap_harmonic_plus_quartic"]=2*np.sqrt(MOMENT*eps)-eps/3
        result["z_variance_leading"]=.5*np.sqrt(eps/MOMENT)
        result["Q_variance_over_h_leading"]=.5*np.sqrt(MOMENT/eps)
        result["gprime_squared_leading"]=2*MOMENT**1.5*np.sqrt(eps)
        small.append(result)
    out["small_epsilon_crossover"]=small
    assert abs(small[-1]["ground_above_threshold"]/small[-1]["ground_harmonic_plus_quartic"]-1)<2e-5

    # Explicit h sequence at FIXED c=1/4,k=2,m=0:
    # Ec=h^(2/3), epsilon=h^(1/3). Adaptive box tracks the state width.
    hrows=[]
    for hb in [1e-3,1e-6,1e-9,1e-12]:
        eps=hb**(1/3);Ec=hb**(2/3)
        box=10*(eps/MOMENT)**.25
        r1=solve(0.,eps,n=1023,L=box)
        r2=solve(0.,eps,n=2047,L=box)
        r3=solve(0.,eps,n=4095,L=box)
        ext1=richardson_pair(r1,r2);ext2=richardson_pair(r2,r3)
        bigger1=solve(0.,eps,n=2047,L=1.2*box)
        bigger2=solve(0.,eps,n=4095,L=1.2*box)
        bigger_ext=richardson_pair(bigger1,bigger2)
        error=max(abs(ext2[key]-ext1[key]) for key in ext2)
        tail_error=max(abs(ext2[key]-bigger_ext[key]) for key in ext2)
        upper=gaussian_upper_bound(eps)
        assert ext2["ground_energy"]<upper
        assert error<3e-8 and tail_error<3e-8
        assert r3["boundary_probability"]<1e-20
        row={"hbar":hb,"fixed_c":.25,"fixed_k":2.,"m":0.,
             "epsilon":eps,"dimensionless_half_box":box,
             "raw_ground_energy":Ec*ext2["ground_energy"],
             "free_threshold":Ec*.75*MOMENT,
             "ground_above_threshold":Ec*ext2["ground_above_threshold"],
             "gap":Ec*ext2["gap"],
             "P_variance":hb*ext2["z_variance"],
             "Q_variance":hb*ext2["minus_d2_expectation"],
             "true_velocity_variance":Ec*Ec/hb*ext2["gprime_squared_expectation"],
             "ground_divided_by_h5over6":Ec*ext2["ground_above_threshold"]/hb**(5/6),
             "gap_divided_by_h5over6":Ec*ext2["gap"]/hb**(5/6),
             "velocity_variance_divided_by_sqrt_h":Ec*Ec/hb**1.5*ext2["gprime_squared_expectation"],
             "dimensionless_variational_upper":upper,
             "dimensionless_grid_extrapolation_error":error,
             "dimensionless_box_extrapolation_difference":tail_error,
             "boundary_probability":r3["boundary_probability"],
             "ground_asymptotic_with_first_correction":np.sqrt(MOMENT)*hb**(5/6)-hb/12,
             "gap_asymptotic_with_first_correction":2*np.sqrt(MOMENT)*hb**(5/6)-hb/3}
        hrows.append(row)
    out["explicit_h_sequence"]=hrows
    out["h_sequence_predicted_limits"]={"ground_over_h5over6":np.sqrt(MOMENT),
      "gap_over_h5over6":2*np.sqrt(MOMENT),
      "velocity_variance_over_sqrt_h":2*MOMENT**1.5}
    out["massless_epsilon1_gaussian_variational_upper"]=gaussian_upper_bound(1.)

    # Large epsilon approaches the homogeneous, unsmoothed oscillator.
    raw=solve(0.,1.,n=8191,L=12.,raw_massless=True)
    large=[]
    for eps in [10.,100.,1000.]:
        result=solve(0.,eps,n=4095,L=12*eps**.3)
        result["raw_ground_scaled"]=raw["ground_energy"]*eps**.4
        result["raw_gap_scaled"]=raw["gap"]*eps**.4
        result["raw_z_variance_scaled"]=raw["z_variance"]*eps**.6
        result["raw_gprime_squared_scaled"]=raw["gprime_squared_expectation"]*eps**.2
        large.append(result)
    out["raw_massless_epsilon1"]=raw
    out["large_epsilon_crossover"]=large
    out["scope"]="One-coordinate confined quantum benchmark; no coupled eigenvalue bath or MQM scattering limit."
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_boundary_oscillator_results.json').write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k not in ["box_convergence","small_epsilon_crossover","large_epsilon_crossover"]},indent=2))


if __name__=="__main__":
    main()
