#!/usr/bin/env python3
"""Explicit homogeneous Ramond sewing, separate from the existing assembler.

The legacy sign filter was validated for bottom-component one/two-point
blocks. This audit retains the even/odd three-form label at each vertex.
Its nonchiral contraction is being checked before use in a string integral.
Reference: Hadasz--Jaskolski--Suchanek, arXiv:1207.5740, (2.6)--(2.11).
"""

from functools import lru_cache
from itertools import product
import math

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import fermion_parity
from ramond_algebra.nrr_three_point_tensor import hjs_to_polynomial_ground_tensor
from spin23_genus1_blocks import TorusNecklaceBlockSeries, direct_ramond_torus_necklace_series
from spin23_super_liouville_data import ns_weight, rr_ns_structure_constants
from spin23_ramond_fast import (
    _fast_edge_factor,
    _prewhitened_vertex_sign_coefficients,
    _evaluate_matrix_polynomial,
)


def _cutoffs(value, size):
    values = (value,) * size if isinstance(value, int) else tuple(value)
    if len(values) != size or any(isinstance(v, bool) or not isinstance(v, int)
                                  or v < 0 for v in values):
        raise ValueError('one nonnegative twice-level cutoff is needed per edge')
    return tuple(v-v % 2 for v in values)


def homogeneous_series(*, internal_momenta, external_momenta, signs,
                       form_parities, external_words,
                       maximum_twice_levels=0, temporal_lift_sign=1,
                       backend='fast', digits=40, condition_limit=1e11):
    """Keep one homogeneous three-form at each vertex before sewing."""
    internal = tuple(float(p) for p in internal_momenta)
    external = tuple(complex(p) for p in external_momenta)
    words = tuple(tuple(w) for w in external_words)
    forms = tuple(form_parities)
    signs = tuple(signs)
    size = len(internal)
    if not size or not all(len(v) == size for v in (external, words, forms, signs)):
        raise ValueError('necklace input lengths must agree')
    if any(p <= 0 or not math.isfinite(p) for p in internal):
        raise ValueError('internal Ramond momenta must be positive and finite')
    if any(f not in (0, 1) for f in forms) or any(s not in (-1, 1) for s in signs):
        raise ValueError('invalid three-form label or HJS sign')
    if temporal_lift_sign not in (-1, 1):
        raise ValueError('temporal lift must be a sign')
    cutoffs = _cutoffs(maximum_twice_levels, size)
    lifts = (temporal_lift_sign,) + (1,) * (size-1)
    if backend == 'direct':
        tensors = []
        for v in range(size):
            tensor = hjs_to_polynomial_ground_tensor(
                sp.I*sp.Float(internal[v-1],digits)/sp.sqrt(2),
                sp.I*sp.Float(internal[v],digits)/sp.sqrt(2),
                structure_sign=signs[v])
            tensors.append(sp.Matrix(2,2,lambda i,j: tensor[i,j]
                                     if (i+j) % 2 == forms[v] else 0))
        return direct_ramond_torus_necklace_series(
            c=13.5, internal_weights=tuple(13.5/24+p*p/2 for p in internal),
            external_weights=tuple(ns_weight(p) for p in external),
            ground_tensors=tensors, maximum_twice_levels=cutoffs,
            edge_lift_signs=lifts, external_words=words, digits=digits,
            condition_limit=condition_limit, vertex_backend='template')
    if backend != 'fast':
        raise ValueError('backend must be fast or direct')
    coefficients = {}
    conditions = {}
    for levels in product(*(range(0,n+1,2) for n in cutoffs)):
        matrices = []
        for v in range(size):
            previous = (v-1) % size
            left = _fast_edge_factor(levels[previous],internal[previous],
                                     lifts[previous],condition_limit)
            right = _fast_edge_factor(levels[v],internal[v],lifts[v],condition_limit)
            coefficients_by_sign = _prewhitened_vertex_sign_coefficients(
                levels[previous],words[v],levels[v],internal[previous],internal[v],
                lifts[previous],lifts[v],condition_limit)
            matrix = _evaluate_matrix_polynomial(coefficients_by_sign,
                                                 complex(ns_weight(external[v])))[
                                                     0 if signs[v] == -1 else 1]
            # Parity-preserving Gram whitening commutes with this projection.
            parity = np.asarray([[((l.parity+r.parity+
                                    fermion_parity(words[v])) % 2 == forms[v])
                                  for r in right.basis] for l in left.basis])
            matrices.append(matrix*parity)
            conditions[(v,levels[v])] = right.condition_number
        contracted = matrices[0]
        for matrix in matrices[1:]:
            contracted = contracted @ matrix
        coefficients[levels] = complex(np.trace(contracted))
    return TorusNecklaceBlockSeries(
        coefficients=coefficients,sector='R',c=13.5,
        internal_weights=tuple(13.5/24+p*p/2 for p in internal),
        external_weights=tuple(ns_weight(p) for p in external),external_words=words,
        edge_lift_signs=lifts,maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions)


@lru_cache(maxsize=2048)
def _cached_homogeneous_series(internal,external,signs,forms,words,cutoffs,lift,
                               backend,digits,condition_limit):
    return homogeneous_series(internal_momenta=internal,external_momenta=external,
        signs=signs,form_parities=forms,external_words=words,
        maximum_twice_levels=cutoffs,temporal_lift_sign=lift,backend=backend,
        digits=digits,condition_limit=condition_limit)


def paired_momentum_integrand(internal_momenta, *, external_momenta,
                              plumbing_parameters, holomorphic_words,
                              antiholomorphic_words, maximum_twice_levels=0,
                              temporal_lift_sign=1, structure_precision=32,
                              block_digits=40, condition_limit=1e11,
                              strip_internal_gaussian=False, backend='fast'):
    """Audit HJS homogeneous nonchiral pairing, without dP/pi measures.

    This implementation deliberately does not apply a primary-only HJS
    sign-product shortcut. The traces decide which sectors vanish.
    Each BRY structure coefficient supplies C_HJS=C_BRY/2.
    """
    internal = tuple(internal_momenta)
    external = tuple(external_momenta)
    holo = tuple(tuple(w) for w in holomorphic_words)
    anti = tuple(tuple(w) for w in antiholomorphic_words)
    size = len(internal)
    if not size or not all(len(v) == size for v in
                           (external,holo,anti,plumbing_parameters)):
        raise ValueError('necklace input lengths must agree')
    par_h = tuple(fermion_parity(w) for w in holo)
    par_a = tuple(fermion_parity(w) for w in anti)
    if sum(par_h) % 2 != sum(par_a) % 2:
        return 0j
    constants = tuple(rr_ns_structure_constants(internal[v-1],internal[v],external[v],
                                                precision=structure_precision)
                      for v in range(size))
    q = tuple(complex(v) for v in plumbing_parameters)
    primary = 1. if strip_internal_gaussian else math.exp(sum(
        math.log(abs(v))*float(p)**2 for p,v in zip(internal,q)))
    total = 0j
    for signs in product((-1,1),repeat=size):
        weight = math.prod(constants[v][0 if sign == 1 else 1]/2
                           for v,sign in enumerate(signs))
        for forms in product((0,1),repeat=size):
            if sum(forms) % 2 != sum(par_h) % 2:
                continue
            # Represent a graded tensor A (x) B as A P_left^|B| tensor B.
            # Move the P_left factors to the end of the closed necklace.
            # The right trace enforces even total |B|, so no P remains.
            # Each right superdescendant contributes (-i)(-1)^form_parity
            # in the HJS convention; n=1 recovers their +i starred pairing.
            left_parity = tuple((f+a) % 2 for f,a in zip(forms,par_h))
            right_parity = tuple((f+b) % 2 for f,b in zip(forms,par_a))
            crossings = sum(right_parity[v]*left_parity[w]
                            for v in range(size) for w in range(v+1,size))
            phase = (-1j)**sum(par_a)*(-1)**(
                sum(f*b for f,b in zip(forms,par_a))+crossings)
            cutoffs = _cutoffs(maximum_twice_levels,size)
            left = _cached_homogeneous_series(internal,external,signs,forms,holo,
                cutoffs,temporal_lift_sign,backend,block_digits,condition_limit)
            right = left if holo == anti else _cached_homogeneous_series(
                internal,external,signs,forms,anti,cutoffs,temporal_lift_sign,
                backend,block_digits,condition_limit)
            total += weight*phase*left.descendant_value(q)*right.descendant_value(
                tuple(v.conjugate() for v in q))
    return complex(primary*total)
