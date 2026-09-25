"""High-precision analytic limits of coincident necklace c poles.

The SAME CCY recurrence is evaluated with mpmath arithmetic. Independently
detuning the internal weights separates coincident poles; averaging the
complete answer on two small circles verifies the analytic limit.
"""
from functools import lru_cache
from types import FunctionType,SimpleNamespace
import mpmath as mp

import spin23_virasoro_torus_recursion as base
import spin23_genus1_virasoro_necklace as necklace


def _clone(function,scope):
    result=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


@lru_cache(maxsize=1)
def high_precision_engine():
    names=('_normalized_rho_necklace_two_edge','_b_square_rs_from_h','_c_rs_from_h',
        '_momentum_from_weight','_fusion_polynomial','_minus_dc_dh_times_a_rs',
        '_rising_pochhammer','_falling_pochhammer')
    scope=dict(vars(base),complex=mp.mpc,cmath=SimpleNamespace(sqrt=mp.sqrt))
    for name in names:scope[name]=_clone(getattr(base,name),scope)
    outer=dict(vars(necklace),complex=mp.mpc,cmath=SimpleNamespace(sqrt=mp.sqrt))
    outer.update({name:scope[name] for name in names})
    return _clone(necklace.coefficient_table,outer)


def coefficient_limit(c,hi,he,cutoffs,total_cutoff=None):
    engine=high_precision_engine()
    def average(radius):
        values={}
        with mp.workdps(80):
            for j in range(8):
                delta=mp.mpf(radius)*mp.exp(2j*mp.pi*(mp.mpf(j)+mp.mpf('.5'))/8)
                shifted=tuple(mp.mpc(h)+(i+1)*delta for i,h in enumerate(hi))
                table=engine(c=mp.mpc(c),internal_weights=shifted,external_weights=he,
                    maximum_levels=cutoffs,maximum_total_level=total_cutoff)
                for k,v in table.items():values[k]=values.get(k,mp.mpc(0))+v/8
            return {k:complex(v) for k,v in values.items()}
    primary=average('0.001');check=average('0.002')
    error=max(abs(v-check[k]) for k,v in primary.items())
    scale=max(1.,max(abs(v) for v in primary.values()))
    if error>1e-9*scale:
        raise ArithmeticError(f'Virasoro coincident-pole limit failed: {error/scale:.3e}')
    return primary
