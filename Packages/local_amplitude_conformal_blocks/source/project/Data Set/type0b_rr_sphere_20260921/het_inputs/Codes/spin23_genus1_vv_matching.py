"""Integrate actual overlap shells before certifying a matched VV amplitude.

The leading cusp has already integrated tau1 and the semi-infinite height.
Here the remaining puncture integral includes both halves of the torus and
restores ONE full collision disk. No recursion runs during integration.
"""
from dataclasses import dataclass
import math
import json
import hashlib
import numpy as np

from spin23_genus1_vv_boundaries import primary_disc, tube_propagator_integral
from spin23_genus1_vv_tail import integrated_cusp_density
from spin23_genus1_vv_conventions import collision_ward_coefficients


def gauss_interval(a, b, order):
    x, w = np.polynomial.legendre.leggauss(order)
    return a+(b-a)*(x+1)/2, (b-a)*w/2


@dataclass
class CuspCollisionBank:
    metadata: dict
    momenta: np.ndarray
    weights: np.ndarray
    coefficients: np.ndarray

    def disc_tail(self, start, radius):
        h = (1+self.momenta[:, 0]**2)/2
        radial = np.column_stack((primary_disc(h, radius),
            math.pi*(h+.5)/(h-.5)*np.exp((2*h-1)*math.log(radius))))
        propagated = tube_propagator_integral(2*math.pi*self.momenta[:, 1]**2, 0., start)
        return np.einsum('mf,mf,m,m->f', self.coefficients, radial,
                         propagated, self.weights)/(8*np.sqrt(8*math.pi**2))

    def integrated_density_components(self, radius, start):
        h = (1+self.momenta[:, 0]**2)/2
        powers = np.column_stack((radius**(2*h-4), radius**(2*h-3)))
        ward = collision_ward_coefficients(complex(*self.metadata['energy']), h)
        propagated = tube_propagator_integral(2*math.pi*self.momenta[:, 1]**2, 0., start)
        return np.einsum('mf,mf,mfc,m,m->c', self.coefficients, powers, ward,
                         propagated, self.weights)/(8*np.sqrt(8*math.pi**2))


def load_cusp_collision(path, expected_energy):
    """Read the new hashed format or the explicitly identified .25i pilot."""
    with np.load(path,allow_pickle=False) as a:
        arrays={k:a[k].copy() for k in ('momenta','weights','coefficients')}
        if 'metadata' in a:
            metadata=json.loads(str(a['metadata']))
            if metadata.get('energy')!=list(expected_energy):
                raise ValueError('cusp collision energy mismatch')
            for k,v in arrays.items():
                if metadata.get('array_sha256',{}).get(k)!=hashlib.sha256(v.tobytes()).hexdigest():
                    raise ValueError('cusp collision array hash mismatch')
        else:
            if list(expected_energy)!=[0.,.25]:
                raise ValueError('legacy collision pilot is only for omega=0.25i')
            metadata=dict(energy=[0.,.25],legacy_energy_assumption=True)
    n=len(arrays['momenta'])
    for name,shape in [('momenta',(n,2)),('weights',(n,)),('coefficients',(n,2))]:
        if arrays[name].shape!=shape or not np.isfinite(arrays[name]).all():
            raise ValueError('malformed cusp collision array')
    if not n or np.any(arrays['momenta']<=0) or np.any(arrays['weights']<=0):
        raise ValueError('positive collision spectral rule required')
    return CuspCollisionBank(metadata,**arrays)


def prepare_cusp_collision(omega, momenta, weights):
    from audit_spin23_genus1_vv_collision import collision_coefficients
    momenta, weights = np.asarray(momenta), np.asarray(weights)
    if momenta.ndim != 2 or momenta.shape[1] != 2 or weights.shape != (len(momenta),):
        raise ValueError('two momenta and one weight per node required')
    if not np.isfinite(momenta).all() or not np.isfinite(weights).all() or np.any(momenta <= 0) or np.any(weights <= 0):
        raise ValueError('positive finite quadrature required')
    values = np.array([collision_coefficients(float(p), float(l), complex(omega))
                       for p, l in momenta])
    return CuspCollisionBank(dict(energy=[complex(omega).real, complex(omega).imag]),
                             momenta, weights, values)


def cusp_annulus(bank, inner, outer, start, order=16, q_order=8):
    """Actual recursive cusp over a puncture annulus, including both halves."""
    if not 0 < inner < outer < .5 or start <= 1:
        raise ValueError('local radii and cusp height>1 required')
    r, rw = gauss_interval(inner, outer, order)
    theta, tw = gauss_interval(0., math.pi, order)
    result = np.zeros(2, complex)
    for ri, wi in zip(r, rw):
        for ti, vi in zip(theta, tw):
            result += 2*ri*wi*vi*integrated_cusp_density(bank, ri*np.exp(1j*ti), start, q_order)
    return result


def puncture_tail(bank, collision, *, start=4., radius=.05, s_max=32., order=12, q_order=8):
    """Leading cusp on 0<s<s_max, with the analytic full disk restored.

    The explicit s_max is a second cutoff; vary it and the short-momentum
    rule before using this as an approximation to the full puncture tail.
    """
    if bank.metadata['energy'] != collision.metadata['energy']:
        raise ValueError('bulk and disk external energies must agree')
    if not 0 < radius < .2 or not radius < s_max < 90 or start <= 1:
        raise ValueError('local disk, finite puncture range<90, and start>1 required')
    intervals = sorted(set([radius, s_max]+[s for s in (.2,.5,1.,2.,4.,8.,16.,32.,64.)
                                                  if radius < s < s_max]))
    bands = []
    angle, aw = gauss_interval(0., math.pi/2, order)
    s = radius*np.sin(angle); sw = aw*radius*np.cos(angle)
    bands.append((s, sw, 'below_radius'))
    for a, b in zip(intervals[:-1], intervals[1:]):
        s, sw = gauss_interval(a, b, order)
        bands.append((s, sw, [a,b]))
    totals = []
    for svalues, sweights, label in bands:
        subtotal = np.zeros(2, complex)
        for s, sw in zip(svalues, sweights):
            minimum = math.sqrt(max(0., radius**2-s**2))
            breaks = sorted(set([minimum, .5]+[x for x in (.05,.15) if minimum < x < .5]))
            for a, b in zip(breaks[:-1], breaks[1:]):
                x, xw = gauss_interval(a, b, order)
                for xx, ww in zip(x, xw):
                    subtotal += 2*sw*ww*(integrated_cusp_density(bank, xx+1j*s, start, q_order)
                                       +integrated_cusp_density(bank, -xx+1j*s, start, q_order))
        totals.append((label, subtotal))
    disk = collision.disc_tail(start, radius)
    bulk = sum((value for _, value in totals), np.zeros(2, complex))
    return dict(bulk_pco=bulk, disk_families=disk, total=bulk.sum()+disk.sum(),
                bands=totals, radius=radius, start=start, s_max=s_max, order=order)


def puncture_rule(height, radius, order):
    """Half torus outside a circular disk; weights include the other half."""
    if not 0 < radius < min(.5,height/2):
        raise ValueError('the disk must fit in the fundamental rectangle')
    angle, aw = gauss_interval(0., math.pi/2, order)
    s1, w1 = radius*np.sin(angle), aw*radius*np.cos(angle)
    s2, w2 = gauss_interval(radius, height/2, order)
    points, weights = [], []
    for s, sw in zip(np.r_[s1,s2], np.r_[w1,w2]):
        minimum = math.sqrt(max(0.,radius**2-s**2))
        edges=sorted(set([minimum,.5]+[x for x in (.05,.15) if minimum<x<.5]))
        for a,b in zip(edges[:-1],edges[1:]):
            x,xw=gauss_interval(a,b,order)
            for xx,ww in zip(x,xw):
                points.extend([xx+1j*s,-xx+1j*s]);weights.extend([2*sw*ww]*2)
    return np.array(points),np.array(weights)


def best_necklace_density(bank, tau, points, cutoffs=(6,8)):
    """Choose a faster identity/S chart geometrically, retaining all spins.

    No equality is imposed between charts. Their independently evaluated
    overlap must be audited separately. This is a diagnostic atlas until
    spectral and level convergence are demonstrated for the integral.
    """
    from spin23_genus1_vv_bank import VVGeometry,evaluate_density
    tau, points = np.broadcast_arrays(np.asarray(tau,complex),np.asarray(points,complex))
    ts=-1/tau;zs=points/tau
    zs-=np.floor(zs.imag/ts.imag)*ts;zs-=np.floor(zs.real)
    gap=np.minimum(points.imag,tau.imag-points.imag)
    sgap=np.minimum(zs.imag,ts.imag-zs.imag)
    use=(ts.imag>=.15)&(sgap>gap)&(sgap>1e-12)
    t=np.where(use,ts,tau);z=np.where(use,zs,points)
    result=evaluate_density(bank,VVGeometry(t,np.column_stack((np.zeros(len(t)),z))),cutoffs)
    result[use]=result[use][:,:,(0,2,1),:]*abs(tau[use,None,None,None])**-6
    return result, int(use.sum())


def compact_integral(bank, collision, *, start=3., radius=.12, order=8, cutoffs=(6,8)):
    """Actual four-dimensional compact torus integral plus ONE collision disk.

    Restored disks use the leading two bridge families; this is a matched
    approximation with explicit radius and recursion-order controls.
    """
    if bank.metadata['energy'] != collision.metadata['energy']:
        raise ValueError('bulk and collision energies differ')
    if start<=1 or not 0<radius<.2:
        raise ValueError('height>1 and local collision radius required')
    tx,tw=gauss_interval(-.5,.5,order)
    bulk=np.zeros((len(cutoffs),3,2),complex);disk=np.zeros((3,2),complex)
    charts=0;samples=0
    for x,xw in zip(tx,tw):
        ys,yw=gauss_interval(math.sqrt(1-x*x),start,order)
        for y,ww in zip(ys,yw):
            z,zw=puncture_rule(y,radius,order);tau=x+1j*y
            for begin in range(0,len(z),128):
                values,used=best_necklace_density(bank,np.full(len(z[begin:begin+128]),tau),
                                                 z[begin:begin+128],cutoffs)
                bulk+=xw*ww*np.einsum('glsc,g->lsc',values,zw[begin:begin+128])
                charts+=used;samples+=len(values)
            disk+=xw*ww*collision.disc(tau,radius,collision.metadata['cutoff'])
    return dict(bulk_components=bulk,disk_families=disk,
                total=bulk.sum(axis=(1,2))+disk.sum(),start=start,radius=radius,
                order=order,cutoffs=list(cutoffs),samples=samples,S_chart_samples=charts)


def compact_cusp_subtracted(bank, tail, collision, *, start=3., radius=.08,
                            order=8, angle_order=16, cutoffs=(6,8), q_order=8):
    """Compact integral with a resummed leading-cusp control variate.

    Above tau2=1 the tau1 range is complete. Evaluate the full recursive
    density minus its OWN leading cusp, then add the independently prepared
    elliptic cusp. Only the massive remainder retains the short-edge error.
    The curved bottom of the fundamental domain uses the ordinary atlas.
    """
    from spin23_genus1_vv_bank import VVGeometry,evaluate_density
    from spin23_genus1_vv_tail import leading_cusp_density
    if not 1<start<=8 or not 0<radius<.2 or angle_order<8:
        raise ValueError('finite compact height, local radius, and >=8 torus angles required')
    if bank.metadata['energy'] != tail.metadata['energy'] or bank.metadata['energy'] != collision.metadata['energy']:
        raise ValueError('all three banks must have the same external energy')
    tx,tw=gauss_interval(-.5,.5,order)
    cap=np.zeros(len(cutoffs),complex);disk=0j
    for x,xw in zip(tx,tw):
        ys,yw=gauss_interval(math.sqrt(1-x*x),1.,order)
        for y,ww in zip(ys,yw):
            z,zw=puncture_rule(y,radius,order);tau=x+1j*y
            for begin in range(0,len(z),128):
                values,_=best_necklace_density(bank,np.full(len(z[begin:begin+128]),tau),
                                               z[begin:begin+128],cutoffs)
                cap+=xw*ww*np.einsum('gl,g->l',values.sum(axis=(2,3)),zw[begin:begin+128])
            disk+=xw*ww*collision.disc(tau,radius,collision.metadata['cutoff']).sum()
    angles=(np.arange(angle_order)+.5)/angle_order-.5
    ys,yw=gauss_interval(1.,start,order)
    massive=np.zeros(len(cutoffs),complex);resummed=0j
    for y,ww in zip(ys,yw):
        z,zw=puncture_rule(y,radius,order)
        for begin in range(0,len(z),16):
            points=z[begin:begin+16];weights=zw[begin:begin+16]
            taus=np.tile(angles+1j*y,len(points));zz=np.repeat(points,angle_order)
            values=evaluate_density(bank,VVGeometry(taus,np.column_stack((np.zeros(len(zz)),zz))),cutoffs)
            values=values.sum(axis=(2,3)).reshape(len(points),angle_order,len(cutoffs)).mean(axis=1)
            leading=np.array([[leading_cusp_density(bank,point,y,c).sum() for c in cutoffs]
                               for point in points])
            massive+=ww*np.einsum('gl,g->l',values-leading,weights)
            exact=np.array([leading_cusp_density(tail,point,y,q_order).sum() for point in points])
            resummed+=ww*np.dot(exact,weights)
        disk+=ww*np.mean([collision.disc(x+1j*y,radius,collision.metadata['cutoff']).sum()
                          for x in angles])
    return dict(total=cap+massive+resummed+disk,cap=cap,massive_remainder=massive,
                resummed_cusp=resummed,disk=disk,start=start,radius=radius,order=order,
                angle_order=angle_order,cutoffs=list(cutoffs),
                method='full bulk minus its leading cusp plus elliptic cusp')
