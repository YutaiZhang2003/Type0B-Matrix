"""PBW preparation of the additional odd-spin Liouville insertions.

This extends the transferred algebraic oracle to fixed homogeneous forms
and an explicit full-state Ramond parity insertion. No heterotic integrand
or trace weight is imported. This is a chiral-block tool, not a completed
odd-spin string density.
"""
from __future__ import annotations

from itertools import product
import sys

import numpy as np

from even import REFERENCE


def enable_oracle():
    for directory in (REFERENCE,REFERENCE/"Codes"):
        if str(directory) not in sys.path: sys.path.append(str(directory))


def chiral_table(momenta,omega,words,*,cutoff=4,supertrace=False,long_ground=False):
    """[form, sign0/sign1, level] with words 0,1,3 for P,G_-1/2,G_-3/2.

    Both ground parities are retained. Multiplying a sign by (-1)^level is
    NOT a Ramond supertrace: integer-level descendants contain both parities.
    """
    enable_oracle()
    from ns_algebra.ns_sca import Mode
    from spin23_ramond_fast import (
        _fast_edge_factor,_prewhitened_vertex_sign_coefficients,
        _evaluate_matrix_polynomial,
    )
    p = tuple(map(float,momenta)); words = tuple(words)
    if len(p)!=2 or len(words)!=2 or any(w not in (0,1,3) for w in words):
        raise ValueError("two positive momenta and supported NS words required")
    if min(p)<=0 or cutoff<0 or cutoff%2:
        raise ValueError("positive momenta and even twice-level cutoff required")
    signs = tuple(product((1,-1),repeat=2))
    forms = tuple(f for f in product((0,1),repeat=2)
                  if sum(f)%2 == sum(w!=0 for w in words)%2)
    levels = np.array([n for n in product(range(0,cutoff+1,2),repeat=2)
                       if sum(n)<=cutoff and (not long_ground or n[1]==0)],int)
    values = np.zeros((len(forms),len(signs),len(levels)),complex)
    for li,lev in enumerate(levels):
        edge = [_fast_edge_factor(int(l),x,1,1e11) for l,x in zip(lev,p)]
        par = [np.array([s.parity for s in e.basis]) for e in edge]
        vertex = []
        for v in range(2):
            word = () if words[v]==0 else (Mode("G",-words[v]),)
            table = _prewhitened_vertex_sign_coefficients(
                int(lev[v-1]),word,int(lev[v]),p[v-1],p[v],1,1,1e11)
            vertex.append(_evaluate_matrix_polynomial(table,(1+complex(omega)**2)/2))
        for fi,f in enumerate(forms):
            masks = [(par[v-1][:,None]+par[v][None,:]+(words[v]!=0))%2 == f[v]
                     for v in range(2)]
            for si,s in enumerate(signs):
                a,b = [vertex[v][0 if s[v]==-1 else 1]*masks[v] for v in range(2)]
                if supertrace: b=b*(-1.)**par[1][None,:]
                values[fi,si,li] = np.einsum("ij,ji->",a,b)
    return levels,values,forms,signs
