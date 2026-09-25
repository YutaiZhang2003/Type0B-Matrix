#!/usr/bin/env python3
"""Moduli integrals of the frozen literature candidate/reference tensors.

No new recursion and no fitted channel normalization. Run with --frozen-root
pointing at the verified v3 source archive. The output is a reduced amplitude,
not a certification of the microscopic GSO dictionary or sphere constant.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

# The executing script is new; every scientific dependency is frozen.
if __name__ == '__main__':
    _bootstrap = argparse.ArgumentParser(add_help=False)
    _bootstrap.add_argument('--frozen-root', type=Path, required=True)
    _boot, _unused = _bootstrap.parse_known_args()
    sys.path.insert(0, str(_boot.frozen_root.resolve()/'Codes'))

import numpy as np
from scipy.linalg import solve_triangular
from literature_component_blocks import (
    SECTORS, _components, _series_power, crossing_phase, ramond_vertex,
)
from literature_self_dual_correlator import SelfDualConstants
from so7e8_literature_campaign import (
    FrozenBlocks, _matching_record, atomic_json, chiral_components, pairs,
    read_manifest, sign_pairs, unpairs,
)
from so7e8_literature_atlas import (
    BlockGeometry, block_grid_values, channel_coordinates, four_ns_moduli_layout,
    geometric_assignment, exchange_mixed_ns_tensor, four_ramond_channel_to_base,
)
from so7e8_literature_amplitude import (
    FAMILIES, four_ramond_reference_terms, g0_phase,
)
from so7e8_four_ramond_assembly import spin7_spectator_block_series
from so7e8_direct_domain import spin7_global_value
from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3_fast as fast


CHARTS = ('s', 't', 'u')
SPECIES = ('SS', 'SV', 'VS', 'VV')
PICTURES = ('one', 'infinity')


@lru_cache(None)
def sewing_terms(family, external):
    """Compile the literal sewn_integrand into coordinate-independent terms.

    Tuple entries: hol-index, anti-index, density parity/signs, coefficient.
    The reflected R sign and graded pairing are retained independently.
    """
    sectors = SECTORS[family]
    indices = {e: i for i, e in enumerate(chiral_components(family))}
    signs = {s: i for i, s in enumerate(sign_pairs(family))}
    ends = [_components(sectors[i], external[i]) for i in (0, 3)]
    out = []
    for ((a0, b0), ket), ((a4, b4), bra) in product(*ends):
        a3, b3 = (0, 0) if sectors[2] == 'R' else external[2]
        h, a = (a0, 0, a3, a4), (b0, 0, b3, b4)
        for qh, qa, sl, sr in product((0, 1), (0, 1), (-1, 1), (-1, 1)):
            if family == 'mixed_ns' and sl != 1:
                continue
            lf, rf, blf, brf = (a4+qh)%2, (qh+a0)%2, (b4+qa)%2, (qa+b0)%2
            right = ramond_vertex(external[1], sr, rf, brf)
            if not right:
                continue
            dp = 0
            if family == 'mixed_ns':
                dp, bt = (lf+a3)%2, (blf+b3)%2
                if dp != bt:
                    continue
                left = (-1)**(dp+dp*b3)*(1j if dp else 1)
            else:
                left = ramond_vertex(external[2], sl, lf, blf)
            if not left:
                continue
            factor = ket*np.conj(bra)*left*right*(-1)**(brf*a0+blf*qh)
            bs = signs[sl, -sr if family == 'mixed_r' else sr]
            if family == 'mixed_r':
                factor *= (-1)**(rf+brf)
            out.append(((indices[h], qh, bs), (indices[a], qa, bs), dp, sl, sr, factor))
    return tuple(out)


@dataclass(frozen=True)
class Outer:
    external: tuple
    rank: int
    coefficient: complex
    hz: complex
    h1: complex
    az: complex
    a1: complex
    polynomial: tuple

    def reflected(self):
        e = self.external
        poly = self.polynomial
        # These free-field polynomials have degree at most one.
        poly = (sum(poly), -poly[1]) if len(poly) == 2 else poly
        return Outer((e[2], e[1], e[0], e[3]), self.rank,
            self.coefficient*crossing_phase(SECTORS['mixed_ns'], e),
            self.h1, self.hz, self.a1, self.az, poly)

    def value(self, z):
        lz, l1 = np.log(z), np.log(1-z)
        return self.coefficient*np.exp(self.hz*lz+self.h1*l1
            +self.az*lz.conjugate()+self.a1*l1.conjugate())*np.polynomial.polynomial.polyval(z.conjugate(), self.polynomial)


def mixed_outer_terms(species, times, picture):
    """Separable exact spectator/ghost/PCO factors in the canonical z frame."""
    k = tuple(times)
    v = tuple(s == 'V' for s in species)
    anti = tuple(int(s == 'S') for s in species)
    X, I = np.array([[0, 1], [1, 0]]), np.eye(2)
    if not any(v):
        free = [(X, 1., 0., 0., (1.,))]
    elif sum(v) == 1:
        free = [(I, 1/math.sqrt(2), .5, -.5 if v[0] else 0., (1.,))]
    else:
        free = [(X, .5, 0., -.5, (2., -1.)), (X, .5, 0., -.5, (0., -1.))]
    out = []
    for r, s, primary in product((0, 1), (0, 1), (False, True)):
        ss = 1-s if primary else s
        c = (-1)**r*np.exp(-1j*math.pi/4*(r-ss))/2*(-1)**(v[0]*(r+ss))
        if primary:
            c *= (-1)**anti[0]*g0_phase(1-s)
        e = (r, s, (0 if primary else int(picture == 'one'), anti[0]),
             (0 if primary else int(picture == 'infinity'), anti[1]))
        hz, h1 = -k[0]*k[1]-.375, -k[1]*k[2]-(.5 if picture == 'infinity' else 0.)
        if primary:
            hz += .5
            if picture == 'one':
                h1 -= .5
                c *= k[2]
            else:
                c *= -k[3]
        for rank, (matrix, scale, az, a1, poly) in enumerate(free):
            if matrix[ss, r]:
                out.append(Outer(e, rank, c*scale*matrix[ss, r], hz, h1,
                    -k[0]*k[1]-.875+az, -k[1]*k[2]+a1, poly))
    return tuple(out)


def chart_outer(species, times, channel, picture):
    if channel == 's':
        return mixed_outer_terms(species, times, picture)
    if channel == 'u':
        species = species[::-1]
        times = (times[0], times[1], times[3], times[2])
        picture = 'infinity' if picture == 'one' else 'one'
    return tuple(t.reflected() for t in mixed_outer_terms(species, times, picture))


def native_pair(task, payload, order):
    p = tuple(unpairs(task['momenta']))
    table = unpairs(payload['coefficients'])[..., :2*order+1]
    return (FrozenBlocks(task['family'], p, task['P'], table[0]),
            FrozenBlocks(task['family'], tuple(x.conjugate() for x in p), task['P'], table[1]))


def density_map(constants, block):
    return {(q, sl, sr): constants.density(block.family, block.p, block.internal.p, q, sl, sr)
            for q in ((0, 1) if block.family == 'mixed_ns' else (0,))
            for sl, sr in sign_pairs(block.family)}


def mixed_grid(records, native, outer, constants, orders=(4, 5)):
    geometry = BlockGeometry.build(native)
    exts = sorted({t.external for terms in outer.values() for t in terms}, key=repr)
    result = {n: {e: np.zeros(len(native), complex) for e in exts} for n in orders}
    for task, payload in records:
        for n in orders:
            h, a = native_pair(task, payload, n)
            hv, av = block_grid_values(h, geometry), block_grid_values(a, geometry).conjugate()
            densities = density_map(constants, h)
            weight = task['dP_weight']/math.pi
            for e in exts:
                for hi, ai, dp, sl, sr, c in sewing_terms(h.family, e):
                    result[n][e] += weight*c*densities[dp, sl, sr]*hv[hi]*av[ai]
    answers = {}
    for n in orders:
        answers[n] = {}
        for key, terms in outer.items():
            rank = 2 if key[0] == 'VV' else 1
            values = np.zeros((len(native), rank), complex)
            for t in terms:
                values[:, t.rank] += t.value(native)*result[n][t.external]
            answers[n][key] = values
    return answers


def spin_grid(native, cache_dir):
    signature = hashlib.sha256(np.asarray(native, '<c16').tobytes()).hexdigest()
    path = cache_dir/f'spin7_{signature}.npz'
    if path.exists():
        with np.load(path) as f:
            if not np.array_equal(f['points'], native):
                raise ValueError('spectator cache coordinate mismatch')
            return f['values']
    # Polynomial evaluation is accurate away from the unit circle; continue
    # the same KZ solution for the remaining points, on their original lips.
    out = np.empty((2, len(native), 4), complex)
    mask = abs(native) <= .65
    for j, channel in enumerate(('vacuum', 'vector')):
        series = spin7_spectator_block_series(channel, maximum_order=90)
        coeff = np.array(list(series.coefficients.values()))
        out[j, mask] = (np.exp(series.leading_exponent*np.log(native[mask].conjugate()))[:, None]
            *np.polynomial.polynomial.polyval(native[mask].conjugate(), coeff).T)
        for i in np.flatnonzero(~mask):
            out[j, i] = spin7_global_value(channel, native[i].conjugate())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+f'.{__import__("os").getpid()}.npz')
    np.savez_compressed(temp, points=native, values=out)
    temp.replace(path)
    return out


def four_r_grid(records, native, constants, cache_dir, orders=(4, 5)):
    geometry = BlockGeometry.build(native)
    spin = spin_grid(native, cache_dir)
    lz, l1 = np.log(native), np.log(1-native)
    plus = np.sqrt((1+np.exp(.5*l1))/2)
    tim = np.array([np.exp(-(lz+l1)/8)*plus,
                    np.exp(-(lz+l1)/8+.5*lz)/plus])
    p = tuple(unpairs(records[0][0]['momenta']))
    k = (*p[:3], -p[3])
    common = np.exp(-k[0]*k[1]*(lz+lz.conjugate())-k[1]*k[2]*(l1+l1.conjugate())-.25*(lz+l1))
    result = {n: np.zeros((len(native), 4), complex) for n in orders}
    signs = {s: i for i, s in enumerate(sign_pairs('rrrr'))}
    for task, payload in records:
        for n in orders:
            h, a = native_pair(task, payload, n)
            hv, av = block_grid_values(h, geometry), block_grid_values(a, geometry).conjugate()
            densities = density_map(constants, h)
            for qh, qt, qa, q7, sl, sr, weight in four_ramond_reference_terms(FAMILIES):
                s = signs[sl, sr]
                scalar = (task['dP_weight']/math.pi*weight*densities[0, sl, sr]
                    *hv[0, qh, s]*av[0, qa, s]*tim[qt]*common)
                result[n] += scalar[:, None]*spin[q7]
    return result


@dataclass
class Local:
    exponent: complex
    coefficients: np.ndarray

    def times(self, other, order):
        return Local(self.exponent+other.exponent,
                     np.convolve(self.coefficients, other.coefficients)[:order+1])

    def value(self, z):
        return np.exp(self.exponent*np.log(z))*np.polynomial.polynomial.polyval(z, self.coefficients)


def binomial(exponent, order):
    return fast._binomial_minus_power(-exponent, order)


def local_blocks(block, total_order):
    """Expand the retained elliptic polynomial, with exact prefactors.

    This is NOT padding a native order-5 z series with zeros. Invert the
    formal elliptic transform through local degree 17 with H coefficients
    above the retained order set to zero, as in numerical block evaluation.
    """
    n = 2*total_order
    table = np.zeros((*block.table.shape[:-1], n+1), complex)
    extended = FrozenBlocks(block.family, block.p, block.internal.p, table)
    output = {}
    for ie, e in enumerate(chiral_components(block.family)):
        T = extended.elliptic_transform(e)
        scales = 4.**np.arange(n+1)
        T = T/scales[:, None]
        ah, _, _ = block.prefactor_exponents(e)
        base = ah+block.internal.p**2/2
        for q in (0, 1):
            for s, (sl, sr) in enumerate(sign_pairs(block.family)):
                H = np.zeros(n+1, complex)
                hh = block.elliptic_coefficients(e, q, sl, sr)
                H[:len(hh)] = hh
                c = solve_triangular(T, H/scales, lower=True, unit_diagonal=True)
                values = []
                for parity in (0, 1):
                    cc = c[parity::2]
                    if np.any(cc):
                        values.append(Local(base+parity/2, cc))
                output[ie, q, s] = values
    return output


def disk_pair(hol, anti, radius):
    # The physical products are single-valued; reject a hidden branch change
    # rather than using an angular formula in a different slit convention.
    delta = hol.exponent-anti.exponent
    if abs(delta-round(delta.real)) > 2e-8:
        raise ArithmeticError(f'nonintegral disk spin {delta}')
    return fast._integrate_series_disk(hol.coefficients, anti.coefficients,
                                      hol.exponent, anti.exponent, radius)


def mixed_disk(records, outer, constants, radius, total_order, orders=(4, 5)):
    result = {n: {key: np.zeros(2 if key[0] == 'VV' else 1, complex) for key in outer} for n in orders}
    boundary_error, minimum_radial = 0., math.inf
    for task, payload in records:
        for n in orders:
            h, a = native_pair(task, payload, n)
            hl, al = local_blocks(h, total_order), local_blocks(a, total_order)
            al = {i: [Local(v.exponent.conjugate(), v.coefficients.conjugate()) for v in vs] for i, vs in al.items()}
            densities = density_map(constants, h)
            for key, terms in outer.items():
                for t in terms:
                    ho = Local(t.hz, binomial(t.h1, total_order))
                    ao = Local(t.az, np.convolve(binomial(t.a1, total_order), t.polynomial)[:total_order+1])
                    for hi, ai, dp, sl, sr, c in sewing_terms(h.family, t.external):
                        integral = 0j
                        for hs, ass in product(hl[hi], al[ai]):
                            hh, aa = hs.times(ho, total_order), ass.times(ao, total_order)
                            minimum_radial = min(minimum_radial, float((hh.exponent+aa.exponent+2).real))
                            integral += disk_pair(hh, aa, radius)
                        result[n][key][t.rank] += task['dP_weight']/math.pi*t.coefficient*c*densities[dp, sl, sr]*integral
            # Re-expansion check on both lips, retaining all actual variants.
            for z in (radius*np.exp(.71j), radius*np.exp(-.71j)):
                exact = block_grid_values(h, BlockGeometry.build([z]))[..., 0]
                values = np.zeros_like(exact)
                for i, terms in hl.items():
                    values[i] = sum(t.value(z) for t in terms)
                scale = max(float(np.max(abs(exact))), 1e-300)
                boundary_error = max(boundary_error, float(np.max(abs(values-exact))/scale))
    return result, dict(local_series_boundary_relative_error=boundary_error, minimum_leading_radial_real_part=minimum_radial)


def four_r_disk(records, constants, radius, total_order, orders=(4, 5)):
    n = total_order
    root = binomial(.5, n)
    plusbase = root/2
    plusbase[0] += .5
    timepoly = [_series_power(plusbase, .5), _series_power(plusbase, -.5)]
    spin = [spin7_spectator_block_series(c, maximum_order=n) for c in ('vacuum', 'vector')]
    result = {order: np.zeros(4, complex) for order in orders}
    minimum_radial, boundary_error = math.inf, 0.
    signs = {s: i for i, s in enumerate(sign_pairs('rrrr'))}
    for task, payload in records:
        for order in orders:
            h, a = native_pair(task, payload, order)
            hl, al = local_blocks(h, n), local_blocks(a, n)
            al = {i: [Local(v.exponent.conjugate(), v.coefficients.conjugate()) for v in vs] for i, vs in al.items()}
            k = (*h.p[:3], -h.p[3])
            densities = density_map(constants, h)
            for qh, qt, qa, q7, sl, sr, weight in four_ramond_reference_terms(FAMILIES):
                s = signs[sl, sr]
                ho = Local(-k[0]*k[1]-.375+qt/2,
                    np.convolve(binomial(-k[1]*k[2]-.375, n), timepoly[qt])[:n+1])
                ss = spin[q7]
                for rank in range(4):
                    spoly = np.array([ss.coefficients[j][rank] for j in range(n+1)])
                    ao = Local(-k[0]*k[1]+ss.leading_exponent,
                        np.convolve(binomial(-k[1]*k[2], n), spoly)[:n+1])
                    integral = 0j
                    for hs, ass in product(hl[0, qh, s], al[0, qa, s]):
                        hh, aa = hs.times(ho, n), ass.times(ao, n)
                        minimum_radial = min(minimum_radial, float((hh.exponent+aa.exponent+2).real))
                        integral += disk_pair(hh, aa, radius)
                    result[order][rank] += task['dP_weight']/math.pi*weight*densities[0, sl, sr]*integral
            for z in (radius*np.exp(.71j), radius*np.exp(-.71j)):
                exact = block_grid_values(h, BlockGeometry.build([z]))[..., 0]
                values = np.zeros_like(exact)
                for i, terms in hl.items():
                    values[i] = sum(t.value(z) for t in terms)
                boundary_error = max(boundary_error, float(np.max(abs(values-exact))/max(np.max(abs(exact)), 1e-300)))
    return result, dict(local_series_boundary_relative_error=boundary_error, minimum_leading_radial_real_part=minimum_radial)


def load_records(run, energy):
    manifest = read_manifest(run)
    parent = read_manifest(run/'parent', check_sources=False)
    integrity = json.loads((run/'integrity_audit.json').read_text())
    hashes = {(r['parent'], r['index']): r['sha256'] for r in integrity['records']}
    records = {}
    for is_parent, m, folder in ((True, parent, run/'parent'), (False, manifest, run)):
        for i, task in enumerate(m['bank_tasks']):
            if task['energy'] != energy:
                continue
            path = folder/'banks'/f'{i:04d}.json'
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[is_parent, i]:
                raise ValueError(f'bank differs from audited record: {path}')
            payload = _matching_record(path, m, task)
            if payload['order'] != 5:
                raise ValueError('this completed atlas selects order 5')
            records.setdefault((task['observable'], task['channel']), []).append((task, payload))
    for rows in records.values():
        if len(rows) != 32 or {t['momentum_index'] for t, p in rows} != set(range(32)):
            raise ValueError('32 complete, unique internal momenta required')
    completion = json.loads((run/'atlas_completion.json').read_text())
    if hashlib.sha256((run/'atlas_completion.json').read_bytes()).hexdigest() != integrity['atlas_completion_sha256']:
        raise ValueError('regional completion differs from audit')
    return manifest, records, completion


def prepared_layout(run, manifest, completion, energy, observable):
    settings = manifest['config']['moduli']['settings']
    grids, disks = four_ns_moduli_layout(settings)
    names, xs, weights = [], [], []
    for name, (x, w) in grids.items():
        names.extend([name]*len(x)); xs.extend(x); weights.extend(w)
    count = len(xs)
    n = 2*settings['disk_total_order']+4
    boundary = settings['ope_radius']*np.exp(2j*math.pi*(np.arange(n)+.5)/n)
    xs.extend(boundary); xs.extend(1/boundary)
    x = np.array(xs)
    z = 1-x if observable == 'mixed' else x
    owner, _ = geometric_assignment(z, observable)
    coords = channel_coordinates(z, observable)
    point_sets = json.loads((run/manifest['point_sets_file']).read_text())
    original = {c: np.flatnonzero(owner == j) for j, c in enumerate(CHARTS)}
    for c in CHARTS:
        if not np.allclose(coords[c][original[c]], unpairs(point_sets[observable+'/'+c]), rtol=1e-14, atol=1e-14):
            raise ValueError('integration grid differs from audited point ordering')
    row = next(r for r in completion['rows'] if (r['energy'], r['observable']) == (energy, observable))
    if row['unresolved']:
        raise ValueError('unresolved atlas')
    for o in row['overrides']:
        index = original[o['original_channel']][o['original_point_index']]
        if abs(z[index]-complex(*o['canonical_z'])) > 1e-12:
            raise ValueError('override point mismatch')
        owner[index] = CHARTS.index(o['selected_channel'])
    return np.array(names), z[:count], np.array(weights), owner[:count], row, disks


def summarize_pieces(pieces):
    p = {k: np.asarray(v, complex) for k, v in pieces.items()}
    total = sum(p.values())
    scale = max(float(np.linalg.norm(total)), 1e-300)
    return dict(value=pairs(total), pieces={k: pairs(v) for k, v in p.items()},
        piece_cancellation_ratio=float(sum(np.linalg.norm(v) for v in p.values())/scale))


def compute_energy(run, output, energy):
    started = time.monotonic()
    manifest, records, completion = load_records(run, energy)
    constants = SelfDualConstants(manifest['config']['precision'])
    m = tuple(unpairs(manifest['energies'][energy]))
    times = (m[0], -m[3], m[2], m[1])
    settings = manifest['config']['moduli']['settings']
    orders = (4, 5)
    report = dict(schema='so7e8-reduced-moduli-amplitudes-v1', energy=energy,
        momenta=manifest['energies'][energy], manifest_sha256=manifest['manifest_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalization='dP/pi; common external literature Upsilon factor stripped; sphere constant, string coupling, energy delta and external reflection phases factored out',
        physical_amplitude_certified=False, numerical_error_certified=False,
        families=list(FAMILIES), orders=list(orders), disk_total_order=settings['disk_total_order'],
        momentum_nodes_per_channel=32, mixed={}, four_ramond={}, diagnostics={})
    for observable in ('mixed', 'rrrr'):
        names, z, weights, owner, coverage, disks = prepared_layout(run, manifest, completion, energy, observable)
        outkeys = list(product(SPECIES, PICTURES)) if observable == 'mixed' else [('rrrr', 'reference')]
        pieces = {n: {key: {name: np.zeros(2 if key[0] == 'VV' else 4 if observable == 'rrrr' else 1, complex)
                    for name in (*dict.fromkeys(names), 'original_zero_disk', 'exterior_zero_disk')}
                    for key in outkeys} for n in orders}
        for j, chart in enumerate(CHARTS):
            mask = owner == j
            zz, ww = z[mask], weights[mask]
            native = channel_coordinates(zz, observable)[chart]
            rows = records[observable, chart]
            if observable == 'mixed':
                # Scan labels are NS_x, NS_1; canonical labels are NS_1, NS_infinity.
                outer = {key: chart_outer(key[0][::-1], times, chart, key[1]) for key in outkeys}
                values = mixed_grid(rows, native, outer, constants)
                if chart == 'u':
                    for n in orders:
                        for key in outkeys:
                            values[n][key] = exchange_mixed_ns_tensor(values[n][key], key[0][::-1])/abs(1-zz)[:, None]**4
            else:
                raw = four_r_grid(rows, native, constants, output/'spectators')
                values = {n: {outkeys[0]: np.array([four_ramond_channel_to_base(v, p, chart) for v, p in zip(raw[n], zz)])}
                          for n in orders}
            for n in orders:
                for key in outkeys:
                    for name in dict.fromkeys(names):
                        local = names[mask] == name
                        pieces[n][key][name] += np.sum(ww[local, None]*values[n][key][local], axis=0)
            print(f'{energy} {observable}/{chart}: {sum(mask)} moduli points integrated', flush=True)
        diagnostics = []
        for exterior, disk in enumerate(disks):
            chart = ('t' if observable == 'mixed' else 's') if not exterior else 'u'
            rows = records[observable, chart]
            if observable == 'mixed':
                outer = {key: chart_outer(key[0][::-1], times, chart, key[1]) for key in outkeys}
                vals, diagnostic = mixed_disk(rows, outer, constants, disk['radius'], disk['total_order'])
                if exterior:
                    for n in orders:
                        for key in outkeys:
                            # Full-integrand conformal factor |u|^4 cancels inversion area |u|^-4.
                            vals[n][key] = exchange_mixed_ns_tensor(vals[n][key], key[0][::-1])
            else:
                vals, diagnostic = four_r_disk(rows, constants, disk['radius'], disk['total_order'])
                if exterior:
                    # A unit-modulus nonsingular point extracts only the fixed
                    # tensor map; the inversion measure has cancelled already.
                    vals = {n: four_ramond_channel_to_base(v, 1j, 'u') for n, v in vals.items()}
                vals = {n: {outkeys[0]: v} for n, v in vals.items()}
            for n in orders:
                for key in outkeys:
                    pieces[n][key][disk['name']] = vals[n][key]
            diagnostic['piece'] = disk['name']
            diagnostics.append(diagnostic)
            print(f'{energy} {observable}/{disk["name"]}: analytic disk integrated', flush=True)
        destination = report['mixed' if observable == 'mixed' else 'four_ramond']
        for key in outkeys:
            values = {n: summarize_pieces(pieces[n][key]) for n in orders}
            lo, hi = (unpairs(values[n]['value']) for n in orders)
            destination['/'.join(key)] = dict(orders={str(n): v for n, v in values.items()},
                adjacent_integrated_relative_change=float(np.linalg.norm(hi-lo)/max(np.linalg.norm(hi), np.linalg.norm(lo), 1e-300)))
        report['diagnostics'][observable] = dict(disks=diagnostics,
            numerical_moduli_points=len(z), maximum_selected_block_change=coverage['maximum_selected_change'],
            channel_overrides=len(coverage['overrides']))
        atomic_json(output/f'{energy}.partial.json', report)
    report['seconds'] = time.monotonic()-started
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['status'] = 'reduced_candidate_reference_amplitudes_computed'
    atomic_json(output/f'{energy}.json', report)
    print(json.dumps(dict(energy=energy, status=report['status'], seconds=report['seconds'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--energy', required=True)
    args = parser.parse_args()
    compute_energy(args.run.resolve(), args.output.resolve(), args.energy)


if __name__ == '__main__':
    main()
