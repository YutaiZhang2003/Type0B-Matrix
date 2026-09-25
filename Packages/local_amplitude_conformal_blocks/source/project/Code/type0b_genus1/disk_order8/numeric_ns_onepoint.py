"""Numerical use of the frozen NS Ward recursion for one-point traces.

Function bodies are cloned from the reference, with ordinary numerical
arithmetic in place of symbolic expansion. The fixed-parity global terminal
and all Ward identities are retained. No reference module is modified.
"""
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bulk_order8'))
import numeric_ramond as nr
from ns_algebra import ns_middle_fusion as middle
from ns_algebra import ns_three_point_tensor as ward

M=nr._clone_actions(middle)
W=nr._clone_actions(ward)
for scope in (M,W):
    number=scope['sp']
    number.binomial=lambda a,k:nr.binomial(float(a),int(k))
    number.prod=math.prod
    number.floor=math.floor
    scope.update(G=nr.N['G'],L=nr.N['L'],act_mode=nr.N['act_mode'],_HALF=.5)
W['_rho_global_exact']=M['_rho_global_exact']


@lru_cache(maxsize=256)
def edge(level,loop):
    basis=tuple(nr.nsca.pbw_basis(level));h=(1+loop*loop)/2
    gram=np.array([[nr.N['descendant_inner_product'](a,b,h=h,c=13.5)
        for b in basis] for a in basis],float)
    gram=(gram+gram.T)/2
    scale=1/np.sqrt(np.diag(gram))
    chol=np.linalg.cholesky(gram*scale[:,None]*scale[None,:])
    return basis,scale,chol


def descendant(loop,bridge,first=5,cutoff=16):
    h=(1+loop*loop)/2;external=(1+bridge*bridge)/2
    evaluate=W['_three_point_ward_cached'];evaluate.cache_clear()
    word=(nr.nsca.Mode('G',-1),);result={}
    try:
        for level in range(first,cutoff+1):
            basis,scale,chol=edge(level,float(loop))
            vertex=np.array([[evaluate(a,word,b,h,external,h,13.5)
                for b in basis] for a in basis],complex)
            vertex*=scale[:,None]*scale[None,:]
            vertex=solve_triangular(chol,vertex,lower=True,check_finite=False)
            vertex=solve_triangular(chol,vertex.T,lower=True,check_finite=False).T
            result[level]=complex(np.trace(vertex))
        if not np.isfinite(list(result.values())).all():raise ArithmeticError('Nonfinite NS one-point trace')
        return result,'numeric evaluation of frozen fixed-parity NS Ward recursion'
    finally:evaluate.cache_clear()
