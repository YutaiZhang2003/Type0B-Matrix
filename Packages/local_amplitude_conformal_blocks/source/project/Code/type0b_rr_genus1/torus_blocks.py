"""Flat RR descendant blocks and both marked long-handle limits.

All quantities are chiral, with the original tube lifts explicit. No map
from these lifts to a physical flat-torus spin projection is assumed.
"""
from __future__ import annotations

import cmath
from functools import lru_cache
from itertools import product

import numpy as np

from geometry import marked_coordinates
from mixed_blocks import MixedNSRamondPlumbingBlock, RamondState

K = 2j*np.pi


class MarkedRRBlocks:
    def __init__(self, **kwargs):
        self.block = MixedNSRamondPlumbingBlock(**kwargs)

    def _terms(self, mode, ground, derivative, second):
        if ground not in (0, 1) or mode not in (None, 0, 1):
            raise ValueError('invalid Ramond mode or ground label')
        if mode is None:
            return ((1., RamondState((), ground)),)
        module = self.block.form(0, 1).modules[2]
        zero = [(complex(c), RamondState(w, g))
                for (w, g), c in module.act('G', 0, (), ground).items()]
        if mode == 0:
            return tuple(zero)
        return ((1/derivative, RamondState((('G', -1),), ground)),
                *[(-.75*second/derivative**2*c, state) for c, state in zero])

    @lru_cache(None)
    def coefficient(self, n, r, left, right, forms, etas, ns_lift, r_lift):
        return self.block.coefficient(n, r, left, right, forms=forms, etas=etas,
            ns_lift=ns_lift, r_parity_insertion=r_lift)

    def _series(self, log_ns, log_r, pairs, terms, *, forms, etas, ns_lift, r_lift):
        total = 0j
        for (cl, left), (cr, right) in product(*terms):
            for n, r in pairs:
                total += cl*cr*self.coefficient(n, r, left, right, tuple(forms),
                    tuple(etas), ns_lift, r_lift)*np.exp(n*log_ns/2+r*log_r)
        return total

    def evaluate(self, tau, z, *, max_twice_level=6, grounds=(0, 0),
                 modes=(None, None), forms=(0, 0), etas=(1, 1),
                 ns_lift=1, r_lift=1, long_ground=None):
        """Full finite bulk tensor, or its own long-ground-state term.

        The punctures are (z,0). `long_ground='R'` retains r=0;
        `long_ground='NS'` retains n=0. At fixed cutoff the difference
        subtracts the bank's own cusp, as in the TT matched calculation.
        """
        if long_ground not in (None, 'NS', 'R'):
            raise ValueError('long_ground must be None, NS or R')
        if not isinstance(max_twice_level, int) or max_twice_level < 0:
            raise ValueError('nonnegative twice-level cutoff required')
        g = marked_coordinates(tau, z)
        terms = (self._terms(modes[0], grounds[0], g.derivative_w, g.second_derivative_w),
                 self._terms(modes[1], grounds[1], g.derivative_v, g.second_derivative_v))
        pairs = tuple((n, r) for n in range(max_twice_level+1)
                      for r in range((max_twice_level-n)//2+1)
                      if (long_ground != 'R' or r == 0) and (long_ground != 'NS' or n == 0))
        ln, lr = cmath.log(g.q_ns), cmath.log(g.q_r)
        b = self.block
        prefactor = np.exp(-complex(b.c)/24*K*g.tau
            -complex(b.h_ext)*cmath.log(g.derivative_w*g.derivative_v)
            +complex(b.h_ns)*ln+complex(b.h_r)*lr)
        return prefactor*self._series(ln, lr, pairs, terms, forms=forms, etas=etas,
                                     ns_lift=ns_lift, r_lift=r_lift)

    def cusp(self, z, *, sector, short_twice_cutoff=6, grounds=(0, 0),
             modes=(None, None), forms=(0, 0), etas=(1, 1), ns_lift=1, r_lift=1):
        """Leading Q coefficient with the long-handle Casimir stripped.

        R chart: marked puncture z; remove Q^(h_R-c/24).
        NS chart: marked puncture tau-z; remove Q^(h_NS-c/24).
        These are different marked charts, not an asserted spin transport.
        The short series has the requested cutoff and is not resummed.
        """
        if sector not in ('NS', 'R'):
            raise ValueError('long sector must be NS or R')
        if not isinstance(short_twice_cutoff, int) or short_twice_cutoff < 0:
            raise ValueError('nonnegative short twice cutoff required')
        if complex(z).imag <= 0:
            raise ValueError('positive short-cylinder height required')
        b = self.block
        s, logs = cmath.exp(K*z), K*z
        if sector == 'R':
            dw, dv = (s-1)/K, (1-s)/K
            second_w, second_v = dw*(1+s), dv*(1+s)
            prefactor = np.exp(2*complex(b.h_ext)*np.log(2*np.pi)
                +(complex(b.h_ns)-complex(b.h_r))*logs
                +2*(complex(b.h_r)-complex(b.h_ext))*cmath.log(1-s))
            pairs = tuple((n, 0) for n in range(short_twice_cutoff+1))
        else:
            dw, dv, second_w, second_v = -1/K, 1/K, -1/K, 1/K
            prefactor = np.exp(2*complex(b.h_ext)*np.log(2*np.pi)
                +(complex(b.h_r)-complex(b.h_ns))*logs)
            pairs = tuple((0, r) for r in range(short_twice_cutoff//2+1))
        terms = (self._terms(modes[0], grounds[0], dw, second_w),
                 self._terms(modes[1], grounds[1], dv, second_v))
        ln, lr = (logs, 0) if sector == 'R' else (0, logs)
        return prefactor*self._series(ln, lr, pairs, terms, forms=forms, etas=etas,
                                     ns_lift=ns_lift, r_lift=r_lift)

    @lru_cache(None)
    def elliptic_stripe(self, sector, order, left, right, forms, etas, ns_lift, r_lift):
        from elliptic_series import EllipticShortSeries
        coefficients = np.zeros(2*order+2, complex)
        for n in range(2*order+2):
            if sector == 'NS' and n % 2:
                continue
            levels = (n,0) if sector == 'R' else (0,n//2)
            coefficients[n] = self.coefficient(*levels,left,right,forms,etas,ns_lift,r_lift)
        return EllipticShortSeries(coefficients,order=order)

    def elliptic_cusp(self, z, *, sector, order=4, grounds=(0,0),
                      modes=(None,None), forms=(0,0), etas=(1,1), ns_lift=1, r_lift=1):
        """Same marked degeneration, with independently converted short stripes.

        The exact elementary geometry and descendant transport stay outside
        the conversion. Collision matching must use the back-expansion of
        this chosen finite nome bank, not a different truncation.
        """
        if sector not in ('NS','R') or complex(z).imag <= 0:
            raise ValueError('valid long sector and positive short height required')
        b = self.block
        logs, s = K*z, cmath.exp(K*z)
        if sector == 'R':
            dw,dv = (s-1)/K,(1-s)/K
            sw,sv = dw*(1+s),dv*(1+s)
            prefactor = np.exp(2*complex(b.h_ext)*np.log(2*np.pi)
                +(complex(b.h_ns)-complex(b.h_r))*logs
                +2*(complex(b.h_r)-complex(b.h_ext))*cmath.log(1-s))
        else:
            dw,dv,sw,sv = -1/K,1/K,-1/K,1/K
            prefactor = np.exp(2*complex(b.h_ext)*np.log(2*np.pi)
                +(complex(b.h_r)-complex(b.h_ns))*logs)
        terms = (self._terms(modes[0],grounds[0],dw,sw),self._terms(modes[1],grounds[1],dv,sv))
        result = 0j
        for (cl,left),(cr,right) in product(*terms):
            stripe = self.elliptic_stripe(sector,order,left,right,tuple(forms),tuple(etas),ns_lift,r_lift)
            result += cl*cr*stripe.evaluate(s,log_s=logs)
        return prefactor*result

    def long_exponent(self, sector):
        b = self.block
        if sector not in ('NS', 'R'):
            raise ValueError('long sector must be NS or R')
        return complex((b.h_ns if sector == 'NS' else b.h_r)-b.c/24)
