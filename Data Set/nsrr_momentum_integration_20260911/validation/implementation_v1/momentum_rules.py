"""Positive-momentum Gaussian rules with explicit SCFT propagator scale.

Here a=-log|q| because the SCFT weight contains P^2/2; the nonchiral
primary is exp(-a P^2). No factor of two from the bosonic convention.
"""
import math
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_genlaguerre, roots_legendre


@lru_cache(None)
def half_gaussian_nodes(order, beta):
    """Gauss rule for u^beta exp(-u^2) du on the positive half line.

Stieltjes recurrence from analytic moments in high precision, then the
Jacobi eigensystem. Moments are Gamma((k+beta+1)/2)/2.
"""
    if type(order) is not int or not 1 <= order <= 32 or beta not in (0,2):
        raise ValueError('supported orders 1..32 and beta=0 or 2')
    with mp.workdps(120):
        moments=[mp.gamma(mp.mpf(k+beta+1)/2)/2 for k in range(2*order+2)]
        def inner(p,q,shift=0):
            return mp.fsum(p[i]*q[j]*moments[i+j+shift]
                           for i in range(len(p)) for j in range(len(q)))
        p=[mp.mpf(1)/mp.sqrt(moments[0])];old=[];b_prev=mp.mpf(0)
        diagonal=[];off=[]
        for k in range(order):
            a=inner(p,p,1);diagonal.append(float(a))
            r=[mp.mpf(0)]+p[:]
            for j in range(len(p)):r[j]-=a*p[j]
            for j in range(len(old)):r[j]-=b_prev*old[j]
            if k+1<order:
                norm=inner(r,r)
                if norm<=0:raise ArithmeticError('nonpositive orthogonal-polynomial norm')
                b=mp.sqrt(norm);off.append(float(b));old,p=p,[v/b for v in r];b_prev=b
        x,v=eigh_tridiagonal(diagonal,off)
        w=float(moments[0])*v[0]**2
        if np.any(x<=0) or np.any(w<=0):raise ArithmeticError('invalid Gaussian rule')
        return x,w


def rule(order, a, beta, scheme='half-gaussian', p_max=3.5):
    if a<=0:raise ValueError('positive Gaussian decay required')
    if scheme in ('laguerre','threshold-laguerre'):
        alpha=-.5 if scheme=='laguerre' else (beta-1)/2
        x,w=roots_genlaguerre(order,alpha)
        scale=1/math.sqrt(a)
        p=scale*np.sqrt(x)
        weights=scale*w*np.exp(x)*x**(-alpha-.5)/(2*math.pi)
    elif scheme=='half-gaussian':
        u,w=half_gaussian_nodes(order,beta)
        p=u/math.sqrt(a)
        weights=w*np.exp(u*u)*u**(-beta)/(math.pi*math.sqrt(a))
    elif scheme=='legendre':
        x,w=roots_legendre(order)
        p=(x+1)*p_max/2
        weights=w*p_max/(2*math.pi)
    else:raise ValueError(f'unknown quadrature scheme {scheme}')
    return p,weights


def channel_rules(config, channel, order, scheme='half-gaussian', p_max=3.5):
    betas=(0,0,2) if channel=='source' else (2,2,2)
    return [rule(order,-math.log(q),b,scheme,p_max)
            for q,b in zip(config['q_envelope'][channel],betas)]
