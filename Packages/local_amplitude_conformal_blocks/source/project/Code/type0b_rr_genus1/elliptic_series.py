"""Coefficient-preserving elliptic re-expansion of an RR short stripe.

Only the modular lambda coordinate is reused from TT. No NSNS external
weight, coupling, or block prefactor is imported. Both integer and
half-integer powers are retained. This is a re-expansion of a finite bank,
not a claim that its adjacent-order difference bounds the omitted tail.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT/'handoffs/reference/HetSO23_1to1_20260920/Codes/heterotic_so23_1to3_vvvv_fit_bundle/ns_elliptic_conversion.py'
spec = importlib.util.spec_from_file_location('_rr_lambda_coordinate', CONVERTER)
conversion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion)


def _mul(a, b, n):
    return [sum(a[j]*b[k-j] for j in range(max(0,k-len(b)+1), min(k+1,len(a))))
            for k in range(n+1)]


def _power(a, exponent, n):
    if a[0] != 1:
        raise ValueError('unit constant required')
    out = [mp.mpc(1)]+[mp.mpc(0)]*n
    for k in range(1,n+1):
        out[k] = sum(((exponent+1)*j-k)*a[j]*out[k-j] for j in range(1,k+1))/k
    return out


def _compose(a, x, n):
    result = [mp.mpc(0)]*(n+1)
    for v in reversed(a):
        result = _mul(result, x, n)
        result[0] += v
    return result


class EllipticShortSeries:
    def __init__(self, twice_coefficients, *, order, dps=70):
        raw = np.asarray(twice_coefficients, complex)
        if not isinstance(order,int) or order < 0 or raw.shape != (2*order+2,):
            raise ValueError('supply integer and half-integer coefficients through order+1/2')
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite short-channel coefficients')
        self.order, self.raw = order, raw.copy()
        with mp.workdps(dps):
            x, u, _ = [[mp.mpc(complex(v)) for v in row] for row in conversion.lambda_series(order)]
            self._exact = tuple(tuple(_mul(_power(u, mp.mpf(f)/2, order),
                _compose([mp.mpc(v) for v in raw[f::2]], x, order), order)) for f in (0,1))
            self.coefficients = np.array(self._exact,complex)

    def evaluate(self, s, *, log_s=None):
        s = complex(s)
        q, _, _ = conversion.nome_geometry(np.array([s]))
        q = complex(q[0])
        logs = np.log(s) if log_s is None else complex(log_s)
        # This form keeps the chosen sqrt(s) branch instead of resetting it
        # through an unrelated principal sqrt(q).
        half = np.exp(.5*(logs+np.log(16*q/s)))
        powers = q**np.arange(self.order+1)
        return self.coefficients[0]@powers+half*(self.coefficients[1]@powers)

    def back_expansion(self, dps=70):
        with mp.workdps(dps):
            q, v, _ = [[mp.mpc(complex(a)) for a in row]
                       for row in conversion.inverse_lambda_series(self.order)]
            out = np.zeros_like(self.raw)
            for f in (0,1):
                coefficients = _mul(_power(v,mp.mpf(f)/2,self.order),
                                    _compose(self._exact[f],q,self.order),self.order)
                out[f::2] = list(map(complex,coefficients))
        return out
