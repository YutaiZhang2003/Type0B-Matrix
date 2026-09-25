"""Exact Gaussian bath for a nonlinear endpoint of the frozen vector chain.

Eigenvalues are frozen and the bulk nonlinear metric interactions are
omitted; every vector site of the Gaussian bulk remains dynamical.
This is not the complete frozen-vector Hamiltonian of the repaired model.
No coupled singlet state or matrix/string scattering is computed.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from mqm_flat_vector_chain_checks import reference


def weights(lam, n):
    y, tau = reference(lam, n)
    gaps = np.diff(y)
    return lam*(y[2:]-y[:-2])**2/4, 1/(lam*gaps[1:-1]**2), tau


def bath_kernel(mass, spring, frequencies):
    nu = np.asarray(frequencies, dtype=float)
    impedance = np.full_like(nu, 1/mass[-1])
    for i in range(len(mass)-2, 0, -1):
        impedance = 1/(mass[i] + 1/(nu*nu/spring[i] + impedance))
    return nu*nu/spring[0] + impedance


def spring_matrix(spring):
    diagonal = np.r_[spring[0],spring[:-1]+spring[1:],spring[-1]]
    return np.diag(diagonal)+np.diag(-spring,1)+np.diag(-spring,-1)


def independent_checks():
    m, k, _ = weights(20, 15)
    K = spring_matrix(k)
    omega2, u = eigh(K, np.diag(m))
    Kq = K[:-1,:-1]
    inverse_Kq = np.linalg.inv(Kq)
    C = np.diag(1/m[:-1])+np.ones((len(m)-1,len(m)-1))/m[-1]
    errors=[]
    for nu in [0.,.01,.1,.7,2.,7.,50.]:
        answer=float(bath_kernel(m,k,nu))
        precision=nu*nu*inverse_Kq+C
        schur=precision[0,0]-precision[0,1:]@np.linalg.solve(precision[1:,1:],precision[1:,0])-1/m[0]
        variance=np.sum(m[0]**2*u[0,1:]**2*omega2[1:]/(nu*nu+omega2[1:]))
        spectral=1/variance-1/m[0]
        errors.append(max(abs(schur/answer-1),abs(spectral/answer-1)))
    assert max(errors)<2e-9, errors
    assert abs(float(bath_kernel(m,k,0.))*np.sum(m[1:])-1)<1e-14
    # The nonlinear site's quadratic inertia is absent from the bath kernel.
    modified=m.copy();modified[0]*=1000
    assert np.array_equal(bath_kernel(m,k,np.array([0.,1.,10.])),bath_kernel(modified,k,np.array([0.,1.,10.])))
    return errors


def covariance_integral(m,k,curvature):
    # Per-component zero-temperature Gaussian comparison, with hbar=1.
    value,error=quad(lambda frequency: 1/(float(bath_kernel(m,k,frequency))+curvature),
                     0.,np.inf,epsabs=2e-9,epsrel=2e-9,limit=200)
    return value/np.pi,error/np.pi


def main():
    result={'independent_Schur_and_spectral_errors':independent_checks()}
    frequencies=np.array([0.,.1,.3,1.,3.,10.])
    result['frequencies']=frequencies.tolist()
    rows=[]
    for lam in [16,64,256,1024,4096,16384]:
        m,k,tau=weights(lam,lam)
        actual=bath_kernel(m,k,frequencies)
        T=float(m.sum())
        comparison=np.r_[1/T,frequencies[1:]/np.tanh(frequencies[1:]*T)]
        physical_T=float(tau[-2])
        original_comparison=np.r_[1/physical_T,frequencies[1:]/np.tanh(frequencies[1:]*physical_T)]
        upper,upper_error=covariance_integral(m,k,0.)
        lower,lower_error=covariance_integral(m,k,2.)
        rows.append({'Lambda_star':lam,'N':lam,'total_mass':T,'outer_tau':physical_T,
                     'bath_kernel':actual.tolist(),'mass_box_ratio':(actual/comparison).tolist(),
                     'wall_box_ratio':(actual/original_comparison).tolist(),
                     'Gaussian_covariance_curvature0':upper,'Gaussian_covariance_curvature2':lower,
                     'quadrature_errors':[upper_error,lower_error]})
        print(json.dumps(rows[-1]),flush=True)
    result['wall_sequence']=rows
    result['scope']='Gaussian vector bulk plus full nonlinear endpoint benchmark. Bulk nonlinear metric interactions of the final repaired matrix model are omitted. Gaussian comparison integrals do not solve the nonlinear endpoint ground state.'
    result['status']='all checks pass'
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_boundary_bath_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
