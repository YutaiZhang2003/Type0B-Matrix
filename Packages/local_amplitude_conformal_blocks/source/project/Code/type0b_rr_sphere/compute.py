"""Reuse frozen CFT banks for Type 0B sphere assembly and diagnostics."""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

from frozen import (ALGEBRA, MANIFEST, ROOT, BlockGeometry, SelfDualConstants,
                    block_grid_values, correlators, pair, pairs, records,
                    sign_pairs, source_identity, unpairs)
from projection import chart_terms, four_r_terms, ising_blocks

BRANCHES = ('W', 'psi_psibar', 'psi_Lambdabar', 'psibar_Lambda')


def four_r_grid(rows, z, constants, order=5):
    z = np.asarray(z, complex)
    geometry = BlockGeometry.build(z)
    tim = ising_blocks(z)
    p = tuple(unpairs(rows[0][0]['momenta']))
    k = (*p[:3], -p[3])
    lz, l1 = np.log(z), np.log(1-z)
    common = np.exp((-k[0]*k[1]-.25)*(lz+lz.conjugate())
                    +(-k[1]*k[2]-.25)*(l1+l1.conjugate()))
    out = np.zeros(len(z), complex)
    signs = {s: i for i, s in enumerate(sign_pairs('rrrr'))}
    for task, payload in rows:
        h, a = pair(task, payload, order)
        hv = block_grid_values(h, geometry)
        av = block_grid_values(a, geometry).conjugate()
        densities = ALGEBRA.density_map(constants, h)
        for qh, qt, qa, qb, sl, sr, coefficient in four_r_terms():
            s = signs[sl, sr]
            out += (task['dP_weight']/np.pi*coefficient*densities[0, sl, sr]
                    *hv[0, qh, s]*av[0, qa, s]*tim[qt]*tim[qb].conjugate()*common)
    return out


def mixed_grid(rows, z, outer, constants, order=5):
    ext = sorted({t.external for ts in outer.values() for t in ts}, key=repr)
    values = correlators(rows, z, ext, order, constants)
    out = {}
    for key, terms in outer.items():
        out[key] = np.zeros((len(z), len(BRANCHES)), complex)
        for term in terms:
            out[key][:, BRANCHES.index(term.branch)] += term.value(z)*values[term.external]
    return out


def quadrature(radius=.18, radial=24, angular=64):
    """D={|z|<1, Re(z)<1/2}, omitting a complete disk at zero.

    C is partitioned by D, 1-D, and 1/D. Each native chart is integrated
    over D; the conformal factor cancels the inversion's area Jacobian.
    Both slit lips are evaluated, without assuming complex conjugation.
    """
    if not 0 < radius < .5:
        raise ValueError('radius must lie in (0,1/2)')
    x, w = leggauss(radial)
    a, aw = leggauss(angular)
    points, weights = [], []
    for lo, hi in ((radius, .5), (.5, 1.)):
        for r, rw in zip((hi-lo)*x/2+(hi+lo)/2, (hi-lo)*w/2):
            start = 0 if r <= .5 else math.acos(1/(2*r))
            # Separate the negative-real cut, preserving each lip.
            for lower, upper in ((start, math.pi), (math.pi, 2*math.pi-start)):
                theta = (upper-lower)*a/2+(upper+lower)/2
                points.extend(r*np.exp(1j*theta))
                weights.extend(r*rw*(upper-lower)*aw/2)
    return np.asarray(points), np.asarray(weights)


def anti_local(local):
    return {key: [ALGEBRA.Local(v.exponent.conjugate(), v.coefficients.conjugate())
                  for v in vals] for key, vals in local.items()}


def disk_diagnostic(h, local, radius):
    error = 0.
    for z in (radius*np.exp(.71j), radius*np.exp(-.71j)):
        direct = block_grid_values(h, BlockGeometry.build([z]))[..., 0]
        expanded = np.zeros_like(direct)
        for key, vals in local.items():
            expanded[key] = sum(v.value(z) for v in vals)
        error = max(error, float(np.max(abs(direct-expanded))/max(np.max(abs(direct)), 1e-300)))
    return error


def mixed_disk(rows, outer, constants, radius=.18, degree=23, order=5):
    out = {key: np.zeros(len(BRANCHES), complex) for key in outer}
    radial_min, boundary_error = math.inf, 0.
    for task, payload in rows:
        h, a = pair(task, payload, order)
        hl = ALGEBRA.local_blocks(h, degree)
        al = anti_local(ALGEBRA.local_blocks(a, degree))
        densities = ALGEBRA.density_map(constants, h)
        for key, terms in outer.items():
            for term in terms:
                ho = ALGEBRA.Local(term.hz, ALGEBRA.binomial(term.h1, degree))
                ao = ALGEBRA.Local(term.az, ALGEBRA.binomial(term.a1, degree))
                for hi, ai, dp, sl, sr, c in ALGEBRA.sewing_terms(h.family, term.external):
                    val = 0j
                    for hs, ass in product(hl[hi], al[ai]):
                        hh, aa = hs.times(ho, degree), ass.times(ao, degree)
                        radial_min = min(radial_min, float((hh.exponent+aa.exponent+2).real))
                        val += ALGEBRA.disk_pair(hh, aa, radius)
                    out[key][BRANCHES.index(term.branch)] += task['dP_weight']/np.pi*term.coefficient*c*densities[dp, sl, sr]*val
        boundary_error = max(boundary_error, disk_diagnostic(h, hl, radius))
    return out, dict(minimum_leading_radial_real_part=radial_min,
                     local_series_boundary_relative_error=boundary_error)


def four_r_disk(rows, constants, radius=.18, degree=23, order=5):
    n = degree
    root = ALGEBRA.binomial(.5, n)
    plusbase = root/2
    plusbase[0] += .5
    from literature_component_blocks import _series_power
    timepoly = [_series_power(plusbase, .5), _series_power(plusbase, -.5)]
    signs = {s: i for i, s in enumerate(sign_pairs('rrrr'))}
    result = 0j
    radial_min, boundary_error = math.inf, 0.
    for task, payload in rows:
        h, a = pair(task, payload, order)
        hl, al = ALGEBRA.local_blocks(h, n), anti_local(ALGEBRA.local_blocks(a, n))
        k = (*h.p[:3], -h.p[3])
        densities = ALGEBRA.density_map(constants, h)
        for qh, qt, qa, qb, sl, sr, coefficient in four_r_terms():
            s = signs[sl, sr]
            ho = ALGEBRA.Local(-k[0]*k[1]-.375+qt/2,
                    np.convolve(ALGEBRA.binomial(-k[1]*k[2]-.375, n), timepoly[qt])[:n+1])
            ao = ALGEBRA.Local(-k[0]*k[1]-.375+qb/2,
                    np.convolve(ALGEBRA.binomial(-k[1]*k[2]-.375, n), timepoly[qb])[:n+1])
            val = 0j
            for hs, ass in product(hl[0, qh, s], al[0, qa, s]):
                hh, aa = hs.times(ho, n), ass.times(ao, n)
                radial_min = min(radial_min, float((hh.exponent+aa.exponent+2).real))
                val += ALGEBRA.disk_pair(hh, aa, radius)
            result += task['dP_weight']/np.pi*coefficient*densities[0, sl, sr]*val
        boundary_error = max(boundary_error, disk_diagnostic(h, hl, radius))
    return result, dict(minimum_leading_radial_real_part=radial_min,
                        local_series_boundary_relative_error=boundary_error)


def overlap(energy='t0250', order=5):
    constants = SelfDualConstants(70)
    z = np.array([.35+.1j, .5+.1j, .65+.1j, .35-.1j])
    m = tuple(unpairs(MANIFEST['energies'][energy]))
    times = (m[0], -m[3], m[2], m[1])
    r4 = records(energy, 'rrrr')
    v = four_r_grid(r4, z, constants, order)
    vt = four_r_grid(r4, 1-z, constants, order)
    vu = four_r_grid(r4, 1/z, constants, order)/abs(z)**4
    out = dict(energy=energy, order=order, points=pairs(z), four_r=dict(
        s=pairs(v), t=pairs(vt), u=pairs(vu),
        relative_t=float(np.max(abs(v-vt)/np.maximum(abs(v), abs(vt)))),
        relative_u=float(np.max(abs(v-vu)/np.maximum(abs(v), abs(vu))))))
    out['mixed'] = {}
    for ch in ('s', 't', 'u'):
        zz = z if ch == 's' else 1-z if ch == 't' else 1/(1-z)
        rows = records(energy, 'mixed_ns' if ch == 's' else 'mixed_r')
        outer = {p: chart_terms(times, ch, p) for p in ('one', 'infinity')}
        vals = mixed_grid(rows, zz, outer, constants, order)
        if ch == 'u':
            vals = {k: v/abs(1-z)[:, None]**4 for k, v in vals.items()}
        out['mixed'][ch] = {p: dict(branches=pairs(v), sum=pairs(v.sum(axis=1))) for p, v in vals.items()}
    return out


def compute(energy, *, radius=.18, radial=24, angular=64, degree=23, orders=(4, 5)):
    started = time.monotonic()
    constants = SelfDualConstants(70)
    z, weights = quadrature(radius, radial, angular)
    m = tuple(unpairs(MANIFEST['energies'][energy]))
    times = (m[0], -m[3], m[2], m[1])
    out = dict(schema='type0b-rr-sphere-integrals-v1', energy=energy,
               source=source_identity(), status='worldsheet_integrals_computed',
               numerical_error_certified=False,
               projection='mixed graded PCO; four-R chiral GSO v1',
               code_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                            for name in ('compute.py','projection.py','frozen.py','normalization.py')},
               settings=dict(radius=radius,
               radial=radial, angular=angular, local_degree=degree,
               momentum_nodes=32, orders=list(orders), points_per_chart=len(z)),
               mixed={}, four_r={})
    rows = records(energy, 'rrrr')
    for order in orders:
        bulk = np.dot(weights, four_r_grid(rows, z, constants, order))
        disk, diagnostic = four_r_disk(rows, constants, radius, degree, order)
        out['four_r'][str(order)] = dict(bulk=pairs(3*bulk), disk=pairs(3*disk),
                                               value=pairs(3*(bulk+disk)), diagnostic=diagnostic)
        print(f'{energy} four-R order {order} integrated', flush=True)
    for order in orders:
        totals = {p: np.zeros(len(BRANCHES), complex) for p in ('one', 'infinity')}
        pieces = {}
        for ch in ('s', 't', 'u'):
            rows = records(energy, 'mixed_ns' if ch == 's' else 'mixed_r')
            outer = {p: chart_terms(times, ch, p) for p in ('one', 'infinity')}
            vals = mixed_grid(rows, z, outer, constants, order)
            disks, diagnostic = mixed_disk(rows, outer, constants, radius, degree, order)
            pieces[ch] = {'diagnostic': diagnostic}
            for p in outer:
                bulk = weights @ vals[p]
                totals[p] += bulk+disks[p]
                pieces[ch][p] = dict(bulk=pairs(bulk), disk=pairs(disks[p]))
            print(f'{energy} mixed/{ch} order {order} integrated', flush=True)
        out['mixed'][str(order)] = dict(branches=list(BRANCHES), pieces=pieces,
                 pictures={p: dict(branches=pairs(v), value=pairs(v.sum())) for p, v in totals.items()})
    from normalization import four_r_amplitude, mixed_amplitude
    mixed_p = (m[0], m[3], m[2], m[1])
    out['normalized_amplitudes'] = dict(convention='mu_F^2 M; energy delta omitted',
        four_r={n: pairs(four_r_amplitude(complex(*v['value']), m)) for n,v in out['four_r'].items()},
        mixed={n: {p: pairs(mixed_amplitude(complex(*v['value']), mixed_p))
                   for p,v in row['pictures'].items()} for n,row in out['mixed'].items()})
    out['seconds'] = time.monotonic()-started
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('overlap', 'integrate'))
    ap.add_argument('--energy', default='t0250', choices=list(MANIFEST['energies']))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--radial', type=int, default=24)
    ap.add_argument('--angular', type=int, default=64)
    ap.add_argument('--radius', type=float, default=.18)
    ap.add_argument('--degree', type=int, default=23)
    args = ap.parse_args()
    out = overlap(args.energy) if args.command == 'overlap' else compute(
        args.energy, radius=args.radius, radial=args.radial, angular=args.angular, degree=args.degree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
