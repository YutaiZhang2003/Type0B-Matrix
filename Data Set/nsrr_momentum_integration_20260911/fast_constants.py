"""Real-continuum structure constants using the existing Upsilon integral.

This evaluates the same normalization as GenericSuperLiouvilleConstants.
It changes only the numerical quadrature of log Upsilon in its defining
strip. Reference comparisons are required before production use.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import roots_legendre


class FastPositiveConstants:
    def __init__(self, b=1.4, *, order=128, max_momentum=6.0):
        self.b = float(b)
        self.q = self.b + 1 / self.b
        self.max_momentum = float(max_momentum)
        self.order = int(order)
        if self.b != 1.4:
            raise ValueError('This numerical calibration is restricted to b=1.4')
        if self.order < 32:
            raise ValueError('at least 32 integration nodes per panel are required')
        x, w = roots_legendre(self.order)
        panels = (0., 1., 4., 12., 32., 64., 120.)
        self.t = np.concatenate([.5*(b-a)*x+.5*(a+b) for a,b in zip(panels[:-1],panels[1:])])
        self.w = np.concatenate([.5*(b-a)*w for a,b in zip(panels[:-1],panels[1:])])
        self.denominator = np.sinh(self.b*self.t/2)*np.sinh(self.t/(2*self.b))
        self.exp_minus_one = np.expm1(-self.t)
        self.log_prime = -math.log(2)+float(self.log_upsilon_real(self.b, 0.))

    def log_upsilon_real(self, offset, imaginary):
        """Real part of log Upsilon_b(offset+i*imaginary), 0<offset<Q."""
        if not 0 < offset < self.q:
            raise ValueError('the real offset must lie in the Upsilon strip')
        y = np.asarray(imaginary, dtype=float)
        a = self.q/2-float(offset)
        yt = y[...,None]*self.t
        leading = a*a-y[...,None]**2
        ratio = (np.sinh(a*self.t/2)**2*np.cos(yt)-np.sin(yt/2)**2)/self.denominator
        integrand = (leading*self.exp_minus_one + leading-ratio)/self.t
        return np.sum(integrand*self.w,axis=-1)

    def log_leg(self, p, sector):
        p = np.asarray(p,dtype=float)
        if np.any(p <= 0) or np.any(p > self.max_momentum):
            raise ValueError('positive momentum outside the calibrated range')
        if sector == 'NS':
            # Upsilon_b(iP)=Upsilon_b(b+iP)/(gamma(i b P)*b^(1-2 i b P));
            # |gamma(i b P)|=1/(b P), and the NS leg metric is 1/b.
            return np.log(p/self.b)+self.log_upsilon_real(self.b,p)+self.log_upsilon_real(self.q/2,p)
        if sector == 'R':
            return self.log_upsilon_real(self.b/2,p)+self.log_upsilon_real(1/(2*self.b),p)
        raise ValueError('unknown sector')

    def log_denominator_factor(self, p, sector):
        offset = self.q/4 if sector == 'NS' else self.q/4+self.b/2
        # The two Upsilon factors are conjugates by Upsilon_b(x)=Upsilon_b(Q-x).
        return 2*self.log_upsilon_real(offset,np.asarray(p,dtype=float)/2)

    @staticmethod
    def combinations(p1,p2,p3):
        p1,p2,p3=np.broadcast_arrays(p1,p2,p3)
        return np.stack((p1+p2+p3,p2+p3-p1,p1+p3-p2,p1+p2-p3),axis=-1)

    def ns_constants(self,p1,p2,p3):
        values = self.combinations(p1,p2,p3)
        log_n = self.log_prime+sum(self.log_leg(p,'NS') for p in (p1,p2,p3))
        bottom=np.exp(log_n-np.sum(self.log_denominator_factor(values,'NS'),axis=-1))
        top=2*np.exp(log_n-np.sum(self.log_denominator_factor(values,'R'),axis=-1))
        return bottom,top

    def rr_ns_constants(self,p_r1,p_r2,p_ns):
        values=self.combinations(p_r1,p_r2,p_ns)
        log_n=(self.log_prime-2*math.log(self.b)+self.log_leg(p_r1,'R')
               +self.log_leg(p_r2,'R')+self.log_leg(p_ns,'NS'))
        ns=self.log_denominator_factor(values,'NS')
        rr=self.log_denominator_factor(values,'R')
        even=np.exp(log_n-rr[...,0]-rr[...,3]-ns[...,1]-ns[...,2])
        odd=np.exp(log_n-ns[...,0]-ns[...,3]-rr[...,1]-rr[...,2])
        return even,odd
