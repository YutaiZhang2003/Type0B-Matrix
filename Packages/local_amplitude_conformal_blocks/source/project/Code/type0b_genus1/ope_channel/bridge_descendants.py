"""Finite collision-bridge sewing using the existing NS/R Ward tensors.

This expands the fusion bridge, independently of the handle truncation.
All NS PBW states at each retained grade are included. The odd contribution
here extends the existing parity-even BB contact family only; it does not
complete the separately unresolved odd noncontact boundary prescription.
"""
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'bulk_order8'))
sys.path.insert(0,str(HERE.parent/'disk_order8'))
import numeric_ramond as nr
from numeric_ns_onepoint import W, edge as ns_edge

ORDER=4
K=2j*math.pi
G1=(nr.nsca.Mode('G',-1),)


def multiply(a,b,order=ORDER):
    a,b=np.asarray(a),np.asarray(b)
    out=np.zeros(np.broadcast_shapes(a.shape[:-1],b.shape[:-1])+(order+1,),complex)
    for n in range(order+1):
        for j in range(max(0,n-b.shape[-1]+1),min(n+1,a.shape[-1])):
            out[...,n]+=a[...,j]*b[...,n-j]
    return out


def power(a,exponent,order=ORDER):
    a=np.asarray(a,complex);exponent=np.asarray(exponent,complex)
    shape=np.broadcast_shapes(a.shape[:-1],exponent.shape)
    out=np.zeros(shape+(order+1,),complex);out[...,0]=1
    if not np.all(a[...,0]==1):raise ValueError('unit-constant power required')
    for n in range(1,order+1):
        for j in range(1,n+1):
            out[...,n]+=((exponent+1)*j-n)*a[...,j]*out[...,n-j]/n
    return out


def cylinder_matrix(h,d1,d2,family,order=ORDER):
    """v=exp(kz)-1 to the flat local coordinate; no fitted coefficients."""
    a=np.array([K**j/math.factorial(j+1) for j in range(order+1)])
    exponential=np.array([(K*d1)**j/math.factorial(j) for j in range(order+1)])
    out=np.zeros((order+1,order+1),complex)
    for j in range(order+1):
        p=multiply(exponential,power(a,h+family/2-d1-d2+j,order),order)
        for n in range(j,order+1):out[n,j]=K**j*p[n-j]
    return out


@lru_cache(maxsize=256)
def bridge_factor(level,p):
    basis=tuple(nr.nsca.pbw_basis(level));h=(1+p*p)/2
    gram=np.array([[nr.N['descendant_inner_product'](a,b,h=h,c=13.5)
                    for b in basis] for a in basis],float)
    scale=1/np.sqrt(np.diag(gram))
    chol=np.linalg.cholesky((gram+gram.T)/2*scale[:,None]*scale[None,:])
    return basis,scale,chol


def sew_node(p,omega,even_ns,even_r,odd_contact):
    bridge,loop=map(float,p);h=(1+bridge*bridge)/2;d=(1+omega*omega)/2
    vertex=nr.Vertex(loop,loop,h)
    evaluate=W['_three_point_ward_cached'];evaluate.cache_clear()
    nsmax=even_ns.shape[-1]-1;rmax=2*(even_r.shape[-1]-1)
    omax=2*(len(odd_contact)-1)

    @lru_cache(maxsize=None)
    def trace(word,sector):
        lev=nr.nsca.twice_level(word) if word else 0
        if not word:
            return even_ns[0] if sector=='ns' else even_r[0] if sector=='r' else odd_contact
        if word==G1:
            if sector=='ns':return even_ns[1]
            if sector=='r':return even_r[1]
            raise ValueError('odd-parity contact family is outside this extension')
        if word[0]==nr.nsca.Mode('L',-2):
            return -(h+(lev-2)/2)*trace(word[1:],sector)
        if sector=='ns':
            hh=(1+loop*loop)/2;values=[]
            for level in range(nsmax+1):
                basis,scale,chol=ns_edge(level,loop)
                matrix=np.array([[evaluate(a,word,b,hh,h,hh,13.5)
                                  for b in basis] for a in basis],complex)
                matrix*=scale[:,None]*scale[None,:]
                matrix=solve_triangular(chol,matrix,lower=True,check_finite=False)
                matrix=solve_triangular(chol,matrix.T,lower=True,check_finite=False).T
                values.append(np.trace(matrix))
            return np.asarray(values)
        values=[]
        for level in range(0,(rmax if sector=='r' else omax)+1,2):
            edge=nr.edge(level,loop);matrix=vertex.matrix(edge,edge,word)
            if sector=='r':values.append(np.trace(matrix,axis1=1,axis2=2)[::-1])
            else:values.append(np.diag(matrix[0])@((-1.)**edge[3]))
        return np.asarray(values).T

    ns=np.zeros((2,2,ORDER+1,nsmax+1),complex)
    r=np.zeros((2,2,2,ORDER+1,rmax//2+1),complex)
    contact=np.zeros((ORDER+1,omax//2+1),complex)
    # Slots are (bridge at infinity, external at W, external at 1).
    external_words=((G1,G1),((),()),(G1,()))
    try:
        for family in (0,1):
            for j in range(ORDER+1):
                level=family+2*j;basis,scale,chol=bridge_factor(level,bridge)
                sphere=np.array([[evaluate(state,a,b,h,d,d,13.5)
                                  for state in basis] for a,b in external_words])
                solved=solve_triangular(chol,sphere.T*scale[:,None],lower=True,check_finite=False)
                solved=solve_triangular(chol.T,solved,lower=False,check_finite=False)*scale[:,None]
                solved=solved.T*(2*h)**family
                tn=np.array([trace(word,'ns') for word in basis])
                tr=np.array([trace(word,'r') for word in basis])
                ns[:,family,j]=np.einsum('wb,bk->wk',solved[:2],tn)
                r[:,family,:,j]=np.einsum('wb,bsk->wsk',solved[:2],tr)
                if family==0:
                    tc=np.array([trace(word,'contact') for word in basis])
                    contact[j]=solved[2]@tc
        ns_local=np.zeros_like(ns);r_local=np.zeros_like(r)
        for w in range(2):
            external=d+(.5 if w==0 else 0)
            for f in range(2):
                matrix=cylinder_matrix(h,external,external,f)
                # Ordered sphere Ward GG sign in the existing collision basis.
                if w==0:matrix*=(-1 if f==0 else 1)
                ns_local[w,f]=matrix@ns[w,f]
                r_local[w,f]=np.einsum('ij,sjk->sik',matrix,r[w,f])
        contact_local=cylinder_matrix(h,d+.5,d,0)@contact
        arrays=dict(ns_bridge=ns,r_bridge=r,contact_bridge=contact,
                    ns_local=ns_local,r_local=r_local,contact_local=contact_local)
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ArithmeticError('non-finite bridge descendant bank')
        return arrays
    finally:
        evaluate.cache_clear();vertex.evaluate.cache_clear();vertex.middle.cache_clear()
