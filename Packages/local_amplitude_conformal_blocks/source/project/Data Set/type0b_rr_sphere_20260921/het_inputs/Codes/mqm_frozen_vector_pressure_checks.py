"""Exact conditional Gaussian energies/forces of the flat matrix vector chain.

The common translation momentum is fixed to zero; this is a relative
Gaussian fibre, not a normalizable state of the coupled matrix model.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, helmert
from scipy.optimize import brentq


def positive_power(matrix, power):
    values, vectors = eigh(matrix)
    assert values[0] > 0
    return (vectors * values**power) @ vectors.T


def reference(n, a=1., mu=20., cstar=1.):
    labels = cstar + np.arange(n+2)
    tau = np.array([brentq(lambda t: mu/a*(np.sinh(t)*np.cosh(t)-t)-ci,
                          1e-8, 10.) for ci in labels])
    return np.sqrt(2*mu)*np.cosh(tau)


def fibre(full_x, a, flavours=23):
    n = len(full_x)-2
    # Fixed, x-independent coordinates q=U^T V on translation orbits.
    U = helmert(n, full=False).T
    edge = np.diff(np.eye(n), axis=0)
    span = full_x[2:]-full_x[:-2]
    gaps = np.diff(full_x[1:-1])
    cdiag = 4*a/span**2
    kbond = a/gaps**2
    C = U.T @ np.diag(cdiag) @ U
    K = U.T @ (edge.T @ np.diag(kbond) @ edge) @ U
    rootC = positive_power(C, .5)
    inverseC = positive_power(C, -.5)
    omega = positive_power(rootC @ K @ rootC, .5)
    A = inverseC @ omega @ inverseC
    inverseA = np.linalg.inv(A)
    assert np.linalg.norm(A @ C @ A-K)/np.linalg.norm(K) < 1e-10
    energy = flavours*np.trace(omega)/2
    # Hellmann--Feynman: dE=d/4 tr(A dC+A^-1 dK).
    momentum_variance = np.diag(U @ A @ U.T)
    bond_variance = np.diag(edge @ U @ inverseA @ U.T @ edge.T)
    force_gradient = np.zeros(n)
    for site in range(n):
        full_index = site+1
        for j in range(n):
            derivative_span = int(j+1 == full_index+1)-int(j+1 == full_index-1)
            force_gradient[j] += flavours/4*momentum_variance[site]*(
                -2*cdiag[site]/span[site]*derivative_span)
    for i in range(n-1):
        derivative = flavours/4*bond_variance[i]*(-2*kbond[i]/gaps[i])
        force_gradient[i] -= derivative
        force_gradient[i+1] += derivative
    # If Cartesian AW quantization precedes restriction to P_cm=0,
    # its constant includes the full common-mode momentum variance.
    aw_shift = flavours*np.sum(cdiag)/4
    aw_grad = np.zeros(n)
    for site in range(n):
        for j in range(n):
            derivative_span = int(j+1 == site+2)-int(j+1 == site)
            aw_grad[j] += flavours/4*(-2*cdiag[site]/span[site]*derivative_span)
    return energy, force_gradient, A, np.linalg.eigvalsh(omega), aw_shift, aw_grad


def main():
    out = {"scope": "frozen quadratic relative vector fibre only; no coupled state"}
    samples = []
    for n in [7, 15, 31]:
        x = reference(n)
        energy, grad, A, frequencies, aw_shift, aw_grad = fibre(x, 1.)
        errors = []
        aw_errors = []
        for j in sorted(set([0, 1, n//2, n-2, n-1])):
            step = min(np.diff(x))*1e-4
            values = [];aw_values = []
            for shift in [-2, -1, 1, 2]:
                y=x.copy();y[j+1]+=shift*step
                row=fibre(y, 1.)
                values.append(row[0]);aw_values.append(row[4])
            numerical=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
            errors.append(abs(numerical-grad[j])/max(1.,abs(grad[j])))
            aw_numerical=(aw_values[0]-8*aw_values[1]+8*aw_values[2]-aw_values[3])/(12*step)
            aw_errors.append(abs(aw_numerical-aw_grad[j])/max(1.,abs(aw_grad[j])))
        assert max(errors)<2e-7
        assert max(aw_errors)<2e-7
        # At fixed Lambda_*=mu/a, x->s*x and a->s²*a change g->g/s.
        # Frequencies/energy stay fixed, physical x-gradients scale as 1/s.
        scaled = fibre(7*x, 49.)
        assert abs(scaled[0]/energy-1)<1e-10
        assert np.linalg.norm(7*scaled[1]-grad)/np.linalg.norm(grad)<1e-9
        assert np.linalg.norm(scaled[2]-A)/np.linalg.norm(A)<1e-9
        samples.append({"N":n,"conventional_Gaussian_vector_energy":float(energy),
                        "full_Cartesian_AW_kinetic_shift":float(aw_shift),
                        "AW_shift_gradient_norm":float(np.linalg.norm(aw_grad)),
                        "gradient_norm":float(np.linalg.norm(grad)),
                        "gradient_first_last":grad[[0,-1]].tolist(),
                        "lowest_frequencies":frequencies[:3].tolist(),
                        "HF_max_relative_error":float(max(errors)),
                        "AW_shift_gradient_check_error":float(max(aw_errors))})
    out["inhomogeneous_wall_fibres"] = samples
    # Uniform open chain, effective continuum length L=N*delta.
    # E=d/(2delta)*(cot(pi/(4N))-1). Subtract bulk and endpoint powers.
    import mpmath as mp
    mp.mp.dps=50
    length=mp.mpf('3.7'); d=mp.mpf(23)
    uniform=[]
    for n in [8,16,32,64,128,256,512]:
        delta=length/n
        exact=d/(2*delta)*(mp.cot(mp.pi/(4*n))-1)
        spectral=d/delta*sum(mp.sin(mp.pi*k/(2*n)) for k in range(1,n))
        assert abs(exact-spectral)<mp.mpf('1e-40')
        bulk=2*d*n/(mp.pi*delta)
        endpoint=-d/(2*delta)
        renormalized=exact-bulk-endpoint
        casimir=-d*mp.pi/(24*length)
        # First omitted term is -d*pi³/(5760*L*N²).
        remainder_coefficient=(renormalized-casimir)*n*n
        uniform.append({"N":n,"energy_after_two_subtractions":float(renormalized),
                        "Casimir_limit":float(casimir),
                        "N2_times_remainder":float(remainder_coefficient)})
    expected=-d*mp.pi**3/(5760*length)
    assert abs(uniform[-1]['N2_times_remainder']/float(expected)-1)<1e-6
    out["uniform_subtraction_test"] = uniform
    out["uniform_next_coefficient"] = float(expected)
    out["counterterm_scope"] = (
        "Exact subtraction in conditional Gaussian theory. A quantum MQM choice, "
        "not a derivation of heterotic renormalization or the full density force. "
        "Full Cartesian AW ordering adds d tr(C_full)/4; quantizing after "
        "a relative quotient gives a distinct d tr(C_rel)/4 prescription.")
    out["all_checks_passed"] = True
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_frozen_vector_pressure_results.json').write_text(
        json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
