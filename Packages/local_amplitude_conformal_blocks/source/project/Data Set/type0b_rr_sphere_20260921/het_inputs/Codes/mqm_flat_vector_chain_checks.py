#!/usr/bin/env python3
"""Exact frozen Gaussian vector chains; no coupled-state claim.

The common coordinate is kept as a separate free particle. All covariances
reported below belong only to the relative oscillator vacuum, at P_common=0.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


def reference(lam, N, cstar=1.0):
    c=np.arange(N+2,dtype=float)+cstar
    # Bracketed inversion avoids small-tau Newton overshoot.
    def fn(t): return np.sinh(t)*np.cosh(t)-t
    tau=np.array([brentq(lambda t:fn(t)-z/lam,0.,
                       max(2.,np.log(4*(z/lam+1))),xtol=5e-15)
                  for z in c])
    return np.sqrt(2.)*np.cosh(tau),tau


def chain(lam,N,cstar=1.,positions=None):
    y,tau=reference(lam,N,cstar)
    if positions is not None:
        y=np.asarray(positions,dtype=float)
        tau=np.arccosh(y/np.sqrt(2.))
    gaps=np.diff(y)
    rhohat=2/(y[2:]-y[:-2])
    mass=lam/rhohat**2
    endpoint_mass_addition=0.
    if positions is not None:
        # At V=0 the selective boundary metric is the only repair that
        # changes the frozen vector quadratic form. Here f=g F.
        c=np.arange(N+2,dtype=float)+cstar
        e=c/lam-(y*np.sqrt(y*y-2.)/2.-tau)
        e[0]=e[-1]=0.
        f=(e[2:]-e[:-2])/(tau[2:]-tau[:-2])
        endpoint_mass_addition=5*f[0]**2/24.
        mass[0]+=endpoint_mass_addition
    spring=1/(lam*gaps[1:-1]**2)
    diag=np.r_[spring[0],spring[:-1]+spring[1:],spring[-1]]/mass
    off=-spring/np.sqrt(mass[:-1]*mass[1:])
    vals,vec=eigh_tridiagonal(diag,off,lapack_driver='stemr')
    zero=np.sqrt(mass/mass.sum())
    assert abs(abs(vec[:,0]@zero)-1)<2e-7
    assert np.min(vals[1:])>0
    omega=np.sqrt(vals[1:])
    endpoint=vec[0,1:]/np.sqrt(mass[0])
    Cq=np.sum(endpoint**2/(2*omega))
    Cv=np.sum(endpoint**2*omega/2)
    # Moment interpolation and concavity give N-independent local bounds
    # on <e1,sqrt(B)e1>, using only B_11 and (B^2)_11.
    Cv_lower=diag[0]**1.5/np.sqrt(diag[0]**2+off[0]**2)/(2*mass[0])
    Cv_upper=np.sqrt(diag[0])/(2*mass[0])
    assert Cv_lower <= Cv*(1+1e-10) <= Cv_upper*(1+1e-10)
    # Exact high moment identity: <(B^1/2)^2> = B_11.
    square=np.sum(vec[0,1:]**2*omega**2)
    assert abs(square/diag[0]-1)<1e-8
    M=mass.sum()
    T=tau[-2] # This approaches the outer Neumann wall at fixed N/lam.
    count=min(6,N-1)
    low=[]
    for j in range(count):
        n=j+1
        u=vec[:,n]/np.sqrt(mass)
        if u[0]<0:u=-u
        cos=np.cos(n*np.pi*tau[1:-1]/T)
        cos/=np.sqrt(np.sum(mass*cos*cos))
        if cos[0]<0:cos=-cos
        error=np.sqrt(np.sum(mass*(u-cos)**2))
        shift=T-M
        effective_cos=np.cos(n*np.pi*(tau[1:-1]-shift)/M)
        effective_cos/=np.sqrt(np.sum(mass*effective_cos**2))
        if effective_cos[0]<0:effective_cos=-effective_cos
        effective_error=np.sqrt(np.sum(mass*(u-effective_cos)**2))
        spacing=(omega[j+1]-(omega[j-1] if j else 0.))/2
        low.append({'n':n,'omega':float(omega[j]),
          'omega_over_box':float(omega[j]*T/(n*np.pi)),
          'endpoint_over_box':float(abs(endpoint[j])*np.sqrt(T/2)),
          'weighted_mode_error':float(error),
          'omega_over_mass_box':float(omega[j]*M/(n*np.pi)),
          'endpoint_over_mass_box':float(abs(endpoint[j])*np.sqrt(M/2)),
          'weighted_shifted_mode_error':float(effective_error),
          'endpoint_unit_delta_ratio':float(abs(endpoint[j])*np.sqrt(np.pi/(2*spacing)))})
    # Site-1 UV spectral moments should have powers lam^(2/3) and log lam.
    # No common-coordinate position variance is included in Cq.
    result={'Lambda_star':float(lam),'N':N,'N_over_Lambda':N/lam,
      'tau_first':float(tau[1]),'tau_last':float(T),
      'total_mass':float(M),'first_mass':float(mass[0]),
      'effective_missing_wall_length':float(T-M),
      'missing_wall_length_scaled':float((T-M)*lam**(1/3)),
      'repair_endpoint_mass_addition':float(endpoint_mass_addition),
      'first_mass_scaled':float(mass[0]*lam**(1/3)),
      'relative_position_variance':float(Cq),
      'relative_velocity_variance':float(Cv),
      'velocity_variance_lower_bound':float(Cv_lower),
      'velocity_variance_upper_bound':float(Cv_upper),
      'velocity_variance_scaled':float(Cv/lam**(2/3)),
      'boundary_tadpole_ratio_per_g_squared':float(25*Cv/(8*mass[0])),
      'relative_sector_boundary_tadpole_ratio_per_g_squared':float(25*Cv*(1/mass[0]-1/M)/8),
      'boundary_vacuum_energy_shift_per_g_squared':float(-23*25*Cv**2/32),
      'lowest_modes':low}
    return result


def edge_limit(N,cstar=1.):
    # Exact homogeneous wall-edge limit at fixed labels as lam->infinity.
    c=np.arange(N+2,dtype=float)+cstar
    A=(1.5)**(2/3)/np.sqrt(2.)
    ybar=A*c**(2/3)
    mass=(ybar[2:]-ybar[:-2])**2/4
    spring=1/np.diff(ybar)[1:-1]**2
    diag=np.r_[spring[0],spring[:-1]+spring[1:],spring[-1]]/mass
    off=-spring/np.sqrt(mass[:-1]*mass[1:])
    vals,vec=eigh_tridiagonal(diag,off,lapack_driver='stemr')
    omega=np.sqrt(vals[1:])
    u=vec[0,1:]/np.sqrt(mass[0])
    return {'N':N,'first_scaled_mass':float(mass[0]),
      'velocity_variance_coefficient':float(np.sum(u*u*omega/2)),
      'velocity_coefficient_lower_bound':float(diag[0]**1.5/np.sqrt(diag[0]**2+off[0]**2)/(2*mass[0])),
      'velocity_coefficient_upper_bound':float(np.sqrt(diag[0])/(2*mass[0])),
      'position_variance':float(np.sum(u*u/(2*omega)))}


def main():
    out={'fixed_ratio':[],'growing_interval':[],'homogeneous_edge':[]}
    for lam in [16,64,256,1024,4096]:
        row=chain(lam,lam)
        out['fixed_ratio'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='lowest_modes'}),flush=True)
    for ratio in [1,4,16,64]:
        row=chain(64,64*ratio)
        out['growing_interval'].append(row)
        print('interval',json.dumps({k:v for k,v in row.items() if k!='lowest_modes'}),flush=True)
    for N in [64,256,1024,4096]:
        row=edge_limit(N)
        out['homogeneous_edge'].append(row)
        print('edge',json.dumps(row),flush=True)
    # Repaired rotor stationary coordinates provide a separate nearby profile.
    source=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_repaired_potential_stability_results.json')
    if source.exists():
        data=json.loads(source.read_text())
        y0,tau=reference(20,15)
        # For a=1, mu=20: y=x/sqrt(mu); g=1/mu.
        y0[1:-1]=np.array(data['continuation'][-1]['positions'],float)/np.sqrt(20.)
        out['nearby_repaired_profile']=chain(20,15,positions=y0)
        out['same_reference_profile']=chain(20,15)
    fixed=out['fixed_ratio']
    out['log_variance_slopes']=[
      (fixed[i+1]['relative_position_variance']-fixed[i]['relative_position_variance'])/
      np.log(fixed[i+1]['Lambda_star']/fixed[i]['Lambda_star'])
      for i in range(len(fixed)-1)]
    assert fixed[-1]['lowest_modes'][0]['weighted_mode_error']<fixed[0]['lowest_modes'][0]['weighted_mode_error']
    assert abs(fixed[-1]['lowest_modes'][0]['omega_over_box']-1)<.1
    assert abs(fixed[-1]['lowest_modes'][0]['endpoint_over_box']-1)<.1
    out['zero_mode']='Common coordinate R=sum_i m_i V_i/sum_i m_i, H0=P_R^2/(2 sum_i m_i); its variance is not included. P_R=0 is a generalized state, not a normalizable full vacuum.'
    out['boundary_scope']='Isolated perturbative Hamiltonian contribution H4=-g^2/32 (sum_a dot V_1^a dot V_1^a)^2, evaluated in 23 relative Gaussian flavors. The exact positive boundary Hamiltonian has not been replaced by this unbounded truncation.'
    out['status']='all checks pass'
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_flat_vector_chain_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print('all checks pass')


if __name__=='__main__':main()
