#!/usr/bin/env python3
"""Shared-endpoint comparison for the constrained wall-fluid exchange.

The free half-line Green functions remain fixed. Cubic boundary
counterterms subtract p/eps and q/eps before convolution. This is a
continuum regulator comparison, not a derived finite-MQM endpoint limit.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import sici
import mpmath as mp
import math

TAYLOR=[(-1.)**j/math.factorial(2*j) for j in range(36)]


def thermal_transform(k):
    k=abs(float(k))
    return -1.0 if k < 1e-9 else -np.pi*k/(2*np.tanh(np.pi*k/2))


def cos_moment(k,eps,power):
    # Taylor evaluation is stable at small z; the recurrence is stable
    # at large z. All requested powers are even and at most six.
    z=k*eps
    if power==0:
        return eps*np.sinc(z/np.pi)
    if abs(z)<8:
        return eps**(power+1)*sum(
            TAYLOR[j]*z**(2*j)/(power+2*j+1)
            for j in range(36))
    c=np.sin(z)/k
    ss=(1-np.cos(z))/k
    for n in range(1,power+1):
        c,ss=eps**n*np.sin(z)/k-n*ss/k, -eps**n*np.cos(z)/k+n*c/k
    return c


def endpoint_transform(k,eps):
    # Exact singular part, plus the convergent csch^2 expansion through
    # tau^6. The omitted local integral is O(eps^9) uniformly in real k.
    z=k*eps
    si=sici(z)[0]
    singular=(np.cos(z)-1)/eps+k*si
    return (thermal_transform(k)+singular
            +cos_moment(k,eps,0)/3
            -cos_moment(k,eps,2)/15
            +2*cos_moment(k,eps,4)/189
            -cos_moment(k,eps,6)/675)


def cutoff_jk(p,q,eps):
    minus=endpoint_transform(p-q,eps)
    plus=endpoint_transform(p+q,eps)
    return ((p+q)*minus+(p-q)*plus)/2, ((p+q)*minus-(p-q)*plus)/2


def uv_functions(z):
    if z==0:return -np.pi/2,0.0
    # (F(z)-1)/z, where F=cos(z)-z*(pi/2-Si(z)).
    a=(np.cos(z)-1)/z-(np.pi/2-sici(z)[0])
    b=(np.cos(z)-1)/z
    return a,b


def closed_j(q):
    with mp.workdps(40):
        return complex(q/2+q/(1+1j*q)+(1+q*q)/(8j)*(
            mp.polygamma(1,(1+1j*q)/2)-mp.polygamma(1,(1-1j*q)/2)))


def common_cutoff_kernel(q,eps,which='J',zmax=250):
    je,ke=cutoff_jk(q,q,eps)
    if which=='J':pole=q*je*ke/(1+q*q)
    elif which=='I':pole=je*je/(1+q*q)
    elif which=='V':pole=ke*ke
    else:raise ValueError(which)
    def integrand(p):
        j,k=cutoff_jk(p,q,eps)
        a,b=uv_functions(eps*p)
        if which=='J':numerator,uv=p*j*k/(1+p*p),q*a*b
        elif which=='I':numerator,uv=j*j/(1+p*p),a*a
        else:numerator,uv=k*k,q*q*b*b
        return (numerator-pole)/(q*q-p*p)+uv
    # The UV term integrates to +q/eps exactly. The real pole-subtraction
    # term has zero principal-value integral on the full half-line.
    L=zmax/eps
    points=[0,q/2,q,q+1,10]
    points+=list(np.arange(max(10,np.pi/eps),L,np.pi/eps))
    points.append(L)
    points=sorted(set(points))
    total=0.;error=0.
    for a,b in zip(points[:-1],points[1:]):
        val,err=quad(integrand,a,b,epsabs=2e-7,epsrel=2e-7,limit=80)
        total+=val;error+=err
    # Leading nonoscillatory tail of the pole-subtracted integrand.
    total+=pole/L
    return 2*total/np.pi-1j*pole/q,error


def common_cutoff_j(q,eps,zmax=250):
    return common_cutoff_kernel(q,eps,'J',zmax)


def main():
    q=1.3
    target=closed_j(q)
    rows=[]
    previous=float('inf')
    for eps in [.08,.04,.02,.01,.005,.002,.001,.0005]:
        measured,error=common_cutoff_j(q,eps)
        row={'epsilon':eps,'J_epsilon_plus_q_over_epsilon':str(measured),
             'difference_from_J':str(measured-target),
             'quadrature_estimate':error}
        rows.append(row)
        distance=abs(measured-target)
        assert distance<previous
        previous=distance
        print(row,flush=True)
    assert previous<.009
    # Positive double-cosine identity evaluates all three UV constants:
    # integral [(1-cos uz)(1-cos vz)]/z^2 dz = pi/2 min(u,v).
    # It gives I_eps~ -2/eps, J_eps~ -q/eps, V_eps~ -q^2/eps.
    other=[]
    for which,closed in [('I',np.pi*q/np.tanh(np.pi*q)+1/(1+1j*q)),
                         ('V',((1+q*q)*np.pi*q/np.tanh(np.pi*q)-1)/3-1j*q)]:
        for eps in [.02,.005,.001]:
            value,error=common_cutoff_kernel(q,eps,which)
            other.append({'kernel':which,'epsilon':eps,'renormalized_value':str(value),
                          'difference_from_nested':str(value-closed),'quadrature_estimate':error})
    tails=[]
    for zmax in [150,300,600]:
        value,error=common_cutoff_j(q,.02,zmax)
        tails.append({'zmax':zmax,'value':str(value),'quadrature_estimate':error})
    result={'scope':__doc__,'convergence_checks':'pass','q':q,'J':str(target),'samples':rows,'other_kernel_checks':other,'tail_checks':tails,
            'leading_divergences':{'I':'-2/epsilon','J':'-q/epsilon','V':'-q^2/epsilon'},
            'numerical_scope':'convergence diagnostic; csch series and large-z tail approximated as documented'}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_endpoint_order_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
