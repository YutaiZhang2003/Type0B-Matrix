"""Research-only diagonal primary R+ V V V R+ five-point sewing test.

No GSO, PCO, moduli integration, physical heterotic amplitude, or permission
to use this kernel in production. The ordered-primary nonchiral extension
must pass the independent integrated crossing/identity tests. The stored
BRY three-point functions are not changed. A conditional metric is explicit.
"""
from itertools import product
import cmath
import math

import numpy as np

from so7e8_fivepoint_blocks import EDGES
from so7e8_fivepoint_uniformization import fivepoint_b1_bank, to_nome_series
from so7e8_ramond_metric import inverse_pairing_factor
from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants

PERMUTATIONS={"A":(3,2,1,0,4),"B":(0,1,4,2,3),"C":(0,1,2,3,4)}
REAL_MOMENTA=(.21,.32,.43,.54,.65)
COMPLEX_MOMENTA=(.02+.08j,.03+.09j,.04+.10j,.05+.11j,.14+.38j)
POINTS=((.25+.12j,.65+.18j),(.35+.18j,.62-.15j),(.30+.10j,.65+.12j))


def primary_labels(channel):
    if channel=="A":
        return tuple((p,(1,1,s)) for p in product((0,1),repeat=2) for s in (-1,1))
    if channel=="B":
        return tuple(((0,p),(s,t,1)) for p,s,t in product((0,1),(-1,1),(-1,1)))
    if channel=="C":
        return tuple(((0,0),s) for s in product((-1,1),repeat=3))
    raise ValueError("channel must be A, B, or C")


def primary_sewing_weights(channel,internal_momenta,external_momenta):
    """Coefficient product occurs ONCE, never squared between chiralities.

    Positive-P representatives are used on both ends of a tube; Ramond
    reflection swaps E/O and the sign label together, leaving this formula
    equivalent to the signed-momentum four-point convention. The i phases
    analytically continue the real-momentum anti-chiral primary interface:
    A: i^p_b, B: (-i)^p_b, C: 1. Momenta are never conjugated.
    This routine is ONLY for the all-unstarred scalar primary observable.
    """
    a,b=internal_momenta
    p1,p2,p3,p4,p5=external_momenta
    ns=ns_structure_constants
    def rr(x,y,z):
        return dict(zip((1,-1),rr_ns_structure_constants(x,y,z)))
    if channel=="A":
        r,m,l=ns(a,p2,p1),ns(b,p3,a),rr(p4,p5,b)
        return {(p,s):r[p[0]]*m[p[0]^p[1]]*l[s[2]]*1j**p[1]/2
                for p,s in primary_labels(channel)}
    if channel=="B":
        r,m,l=rr(a,p1,p2),rr(p3,a,b),ns(p5,p4,b)
        return {(p,s):r[s[0]]*m[s[1]]*l[p[1]]*(-1j)**p[1]/4
                for p,s in primary_labels(channel)}
    if channel=="C":
        r,m,l=rr(a,p1,p2),rr(b,a,p3),rr(p5,b,p4)
        return {(p,s):r[s[0]]*m[s[1]]*l[s[2]]/8 for p,s in primary_labels(channel)}
    raise ValueError("channel must be A, B, or C")


def sewing_coordinates(channel,z2,z3):
    if channel=="A":
        return (1-z3)/(1-z2),1-z2
    if channel=="B":
        return z2/(z2-1),(z3-1)/z3
    if channel=="C":
        return z2/z3,z3
    raise ValueError("channel must be A, B, or C")


def continued_logs(channel,point,*,anchor=POINTS[2],samples=201):
    path=[sewing_coordinates(channel,*(x+t*(y-x) for x,y in zip(anchor,point)))
          for t in np.linspace(0,1,samples)]
    q=np.asarray(path)
    if np.any(abs(q)==0) or np.any(abs(q)>=1):
        raise ValueError("path leaves the noncolliding sewing bidisk")
    angles=np.unwrap(np.angle(q),axis=0)
    return tuple(np.log(abs(q[-1]))+1j*angles[-1])


def original_frame_factor(channel,point,momenta):
    if channel in ("A","C"):
        return 1.+0j
    if channel!="B":
        raise ValueError("channel must be A, B, or C")
    z2,z3=point
    h1,h2,h3,h4,h5=tuple(.5+p*p/2+(1/16 if i in (0,4) else 0)
                           for i,p in enumerate(momenta))
    k=(z3-1)/z3
    return cmath.exp(2*(h1+h2+h3-h4+h5)*math.log(abs(k))
                     -4*h2*math.log(abs(z2-1))-4*h3*math.log(abs(z3-1)))


def primary_density(internal_momenta,*,channel,maximum_twice_levels=(4,4),
                    momenta=REAL_MOMENTA,points=POINTS,convention,contour=None,
                    resummation='type0b_joint'):
    if resummation not in ('type0b_joint','independent_nome_diagnostic'):
        raise ValueError('unknown five-point resummation')
    external=tuple(momenta[i] for i in PERMUTATIONS[channel])
    bank,worst=fivepoint_b1_bank(primary_labels(channel),channel=channel,
        internal_momenta=tuple(internal_momenta),external_momenta=external,
        maximum_twice_levels=tuple(maximum_twice_levels),**(contour or {}))
    weights=primary_sewing_weights(channel,internal_momenta,external)
    multiplier=inverse_pairing_factor(EDGES[channel],convention=convention)
    cuts=tuple(product(*(range(2,n+1,2) for n in maximum_twice_levels)))
    values={cut:np.zeros(len(points),complex) for cut in cuts}
    raw=np.zeros(len(points),complex)
    for label,source in bank.items():
        if resummation=='type0b_joint':
            from so7e8_fivepoint_collar import JointEllipticBlock
            nome=JointEllipticBlock(source)
        else:
            nome=to_nome_series(source)
        for j,point in enumerate(points):
            qs=sewing_coordinates(channel,*point)
            logs=continued_logs(channel,point)
            # Coordinate conjugation only, including the continued logarithms.
            qbars=tuple(q.conjugate() for q in qs)
            logbars=tuple(l.conjugate() for l in logs)
            scalar=weights[label]*multiplier*original_frame_factor(channel,point,momenta)
            raw[j]+=scalar*source.value(*qs,logarithms=logs)*source.value(*qbars,logarithms=logbars)
            for cut in cuts:
                hol=nome.value(*qs,logarithms=logs,maximum_twice_levels=cut)
                anti=(nome.value(*qs,logarithms=logs,maximum_twice_levels=cut,antiholomorphic=True)
                      if resummation=='type0b_joint' else nome.value(
                          *qbars,logarithms=logbars,maximum_twice_levels=cut))
                values[cut][j]+=complex(scalar*hol*anti)
    if not np.all(np.isfinite(raw)) or any(not np.all(np.isfinite(v)) for v in values.values()):
        raise ArithmeticError("nonfinite five-point primary density")
    return values,dict(raw_sewing=raw,maximum_radius_defect=worst,
                      metric_multiplier=multiplier,physical_amplitude=False,resummation=resummation)
