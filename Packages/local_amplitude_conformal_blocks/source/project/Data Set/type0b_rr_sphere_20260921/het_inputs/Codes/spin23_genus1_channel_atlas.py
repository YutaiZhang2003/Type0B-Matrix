"""Candidate chart geometry and overlap diagnostics for the SVV amplitude.

Only necklace coefficients are implemented. Modular S gives another necklace
chart, not a pair-OPE block. Candidates here are *diagnostics*, never silently
used to enlarge the production integration domain. Pair/comb graphs record the
required internal sectors; their mixed superconformal sewing is still needed.
"""
from dataclasses import dataclass
import math
import numpy as np

from spin23_genus1_banked_geometry import GeometryBatch,evaluate_density


@dataclass(frozen=True)
class NecklaceChart:
    name: str
    tau: complex
    points: tuple
    permutation: tuple
    spin_pullback: tuple
    density_pullback: float
    minimum_gap: float
    maximum_plumbing_radius: float


def necklace_charts(tau, points):
    """Original and S-transformed geometry with explicit labels and Jacobian.

    After S, NS stays NS and NS_tilde exchanges with R. The full density has
    modular weight |tau|^8, cancelling the six-real-dimensional measure
    Jacobian. No coefficient or spectral truncation is modular invariant by
    itself, so comparing fixed internal momenta would be an invalid test.
    """
    tau=complex(tau); points=tuple(map(complex,points))
    if len(points)!=3 or abs(points[0])>1e-14 or tau.imag<=0:
        raise ValueError('fix the singlet at zero on an upper-half-plane torus')
    result=[]
    for name,t,z,spins,factor in (
        ('identity',tau,points,(0,1,2),1.),
        ('S',-1/tau,tuple(p/tau for p in points),(0,2,1),abs(tau)**-8),
    ):
        # Physical picture-zero vertices are even, but each chiral term still
        # needs its spin/half-level lift; GeometryBatch supplies the latter.
        wrapped=[]
        for p in z:
            b=p.imag/t.imag; shifted=p-math.floor(b+1e-14)*t
            wrapped.append(shifted-math.floor(shifted.real+1e-14))
        permutation=(0,)+tuple(sorted((1,2),key=lambda i:wrapped[i].imag))
        ordered=tuple(wrapped[i] for i in permutation)
        increments=(ordered[1],ordered[2]-ordered[1],t-ordered[2])
        gap=min(d.imag/t.imag for d in increments)
        radius=max(math.exp(-2*math.pi*d.imag) for d in increments)
        result.append(NecklaceChart(name,t,ordered,permutation,spins,factor,gap,radius))
    return tuple(result)


def candidate_choice(tau,points,*,minimum_gap=.12,maximum_radius=.6):
    """Rank usable necklace geometries; ranking is not an accuracy certificate."""
    candidates=necklace_charts(tau,points)
    usable=[c for c in candidates if c.minimum_gap>=minimum_gap and c.tau.imag>=.15
            and c.maximum_plumbing_radius<=maximum_radius]
    return min(usable,key=lambda c:c.maximum_plumbing_radius) if usable else None


def required_sectors(topology,loop_sector):
    """All external Liouville fields are NS, including their G descendants.

    Fusing two external NS fields inserts an NS internal representation. A
    Ramond loop remains Ramond at each external NS insertion. Consequently
    mixed NS-bridge/R-loop graphs cannot use a uniform-R necklace adapter.
    """
    if loop_sector not in ('NS','R'): raise ValueError('unknown loop sector')
    if topology=='necklace': return (loop_sector,)*3
    if topology=='pair_ope': return ('NS',loop_sector,loop_sector)
    if topology=='comb_ope': return ('NS','NS',loop_sector)
    raise ValueError('unknown topology')


def evaluate_chart(chart,energy,banks,cutoffs):
    """Pull a spectrally integrated candidate into the original density frame."""
    if chart.minimum_gap<=0 or chart.tau.imag<.15:
        raise ValueError('candidate lies outside the numerical necklace strip')
    swapped=chart.permutation==(0,2,1)
    key=tuple(energy[::-1] if swapped else energy)
    geometry=GeometryBatch.build([chart.tau],[chart.points])
    values=evaluate_density(banks[key],geometry,cutoffs)[0]
    values=values[:,chart.spin_pullback,:]
    if swapped: values=values[:,:,(0,2,1,3)]
    return chart.density_pullback*values


def overlap_diagnostic(tau,points,energy,banks,cutoffs=(6,8)):
    """Compare complete spectral sums, all spins and components, in two charts.

    These residuals combine spectral and level error. No chart is marked
    certified based on this single test, and no divergent samples are dropped.
    """
    charts=necklace_charts(tau,points)
    values=np.stack([evaluate_chart(c,energy,banks,cutoffs) for c in charts])
    difference=values[1]-values[0]
    scale=np.maximum(abs(values[0]),abs(values[1]))
    relative=np.divide(abs(difference),scale,out=np.full(scale.shape,np.nan),where=scale>0)
    totals=values.sum(axis=(-1,-2))
    return dict(values=values,absolute_component_difference=abs(difference),
                relative_component_difference=relative,
                relative_total_difference=abs(totals[1]-totals[0])/np.maximum(abs(totals[0]),abs(totals[1])),
                production_enabled=False,charts=charts)
