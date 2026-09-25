"""Independent evaluation-only review of the saved genus-one VV computation.

No fit or predicted amplitude is imported, no coefficient bank is changed,
and the production integrator is left intact. The retained shell differences
and cross-chart disagreements are diagnostics, not certified error bounds.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_spin23_genus1_vv_completion import encode
from refine_spin23_genus1_vv_disks import correction
from spin23_genus1_vv_bank import VVBank, VVGeometry, evaluate_density
from spin23_genus1_vv_factored import FactoredVVBank
from spin23_genus1_vv_matching import (
    best_necklace_density, compact_cusp_subtracted, load_cusp_collision,
    puncture_tail,
)
from spin23_genus1_vv_ope import OPEBank
from spin23_genus1_vv_tail import integrated_cusp_density, leading_cusp_density
from spin23_genus1_vv_tail_bank import TailBank

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_exports/spin23_genus1_vv"
OUT = DATA / "review_20260920"
BULK12 = DATA / "gaussian_rc/gaussian/banks/e0.npz"
BULK16 = DATA / "completion/bank_order16.npz"
OPE = DATA / "completion/ope_24_16.npz"
CUSP = DATA / "completion/cusp_collision_40_32.npz"
TAIL = DATA / "tail_spectral_cache/4f9328bc44527d6e1a02323f289ca36900f66890a01d6302f6960159340658ef.npz"
CUTOFFS = (2, 4, 6, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = list((ROOT / "Codes").glob("spin23_genus1_vv_*.py"))
    paths += [Path(__file__), ROOT / "Codes/audit_spin23_genus1_vv_completion.py",
              ROOT / "Codes/refine_spin23_genus1_vv_disks.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def save(name, results, started, inputs, initial):
    if sources() != initial:
        raise RuntimeError("review source changed while evaluating")
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(energy=[0., .25], results=results,
                 seconds=time.perf_counter()-started,
                 input_hashes={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                 source_hashes=initial,
                 error_interpretation="Sensitivity measurements, not a total error bound.")
    path = OUT / name
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False)+"\n")
    print("saved", path, "seconds", value["seconds"], flush=True)


def l1_relative(a, b):
    return float(np.sum(abs(a-b))/max(np.sum(abs(a)), np.sum(abs(b)), 1e-300))


def batch_density(bank, taus, zs, cutoffs=CUTOFFS):
    parts = []
    for begin in range(0, len(zs), 64):
        t, z = taus[begin:begin+64], zs[begin:begin+64]
        parts.append(evaluate_density(bank, VVGeometry(t, np.column_stack((np.zeros(len(z)), z))), cutoffs))
    return np.concatenate(parts)


def pointwise():
    started = time.perf_counter(); initial = sources()
    banks = {12: FactoredVVBank(VVBank.load(BULK12)),
             16: FactoredVVBank(VVBank.load(BULK16))}
    # The first three tori cover the curved cap actually integrated in the atlas.
    tori = (1j, .45+.90j, .45+.97j, .1+1.25j)
    radii = (.04, .06, .08, .12, .16, .24)
    angles = (np.arange(24)+.5)*math.pi/24
    locations = [(tau, r, theta, r*np.exp(1j*theta))
                 for tau in tori for r in radii for theta in angles]
    taus = np.array([v[0] for v in locations]); zs = np.array([v[3] for v in locations])
    transformed_tau = -1/taus
    transformed_z = zs/taus
    transformed_z -= np.floor(transformed_z.imag/transformed_tau.imag)*transformed_tau
    transformed_z -= np.floor(transformed_z.real)
    original = batch_density(banks[16], taus, zs)
    transformed = batch_density(banks[16], transformed_tau, transformed_z)
    transformed = transformed[:, :, (0, 2, 1), :]*abs(taus[:, None, None, None])**-6
    atlas = {}
    for order, bank in banks.items():
        atlas[order] = np.concatenate([best_necklace_density(bank, taus[a:a+64], zs[a:a+64], CUTOFFS)[0]
                                       for a in range(0, len(zs), 64)])
    rows = []
    for n, (tau, radius, theta, z) in enumerate(locations):
        a, b = original[n], transformed[n]; chosen = atlas[16][n]
        rows.append(dict(tau=tau, z=z, radius=radius, theta=theta,
            original=a, transformed=b, atlas=chosen,
            level_3_to_4_l1=l1_relative(chosen[-1], chosen[-2]),
            level_3_to_4_total=float(abs(chosen[-1].sum()-chosen[-2].sum())/max(abs(chosen[-1].sum()), 1e-300)),
            modular_l1=l1_relative(a[-1], b[-1]),
            momentum_12_to_16_l1=l1_relative(chosen[-1], atlas[12][n, -1])))
    summaries = []
    for tau in tori:
        for radius in radii:
            group = [r for r in rows if r['tau'] == tau and r['radius'] == radius]
            summaries.append(dict(tau=tau, radius=radius,
                max_level_3_to_4_l1=max(r['level_3_to_4_l1'] for r in group),
                max_modular_l1=max(r['modular_l1'] for r in group),
                max_momentum_12_to_16_l1=max(r['momentum_12_to_16_l1'] for r in group)))
    save("pointwise.json", dict(cutoffs=CUTOFFS, rows=rows, summaries=summaries), started,
         [BULK12, BULK16], initial)
    for s in summaries:
        if s['radius'] in (.08, .12): print(encode(s), flush=True)


def strip():
    started = time.perf_counter(); initial = sources()
    bank = FactoredVVBank(VVBank.load(BULK16)); tail = TailBank.load(TAIL)
    rows = []
    for y in (1.+1e-8, 1.25, 2., 3., 4.):
        points = [.08*np.exp(1j*t) for t in (.03, .3, .8, 1.5)]
        points += [.12j, .3+.02j, .23+.4j]
        for z in points:
            values = {}
            for n in (16, 32, 64):
                angles = (np.arange(n)+.5)/n-.5
                full = batch_density(bank, angles+1j*y, np.full(n, z)).sum(axis=2).mean(axis=0)
                own = np.array([leading_cusp_density(bank, z, y, c) for c in CUTOFFS])
                restored = leading_cusp_density(tail, z, y, 8)
                values[n] = dict(full=full, own_cusp=own, massive=full-own,
                                 corrected=full-own+restored, restored_cusp=restored)
            fine = values[64]['corrected']
            rows.append(dict(height=y, z=z, values=values,
                corrected_level_3_to_4_l1=l1_relative(fine[-1], fine[-2]),
                corrected_level_3_to_4_total=float(abs(fine[-1].sum()-fine[-2].sum())/max(abs(fine[-1].sum()), 1e-300)),
                angle_16_to_64_l1=l1_relative(values[16]['corrected'][-1], fine[-1]),
                angle_32_to_64_l1=l1_relative(values[32]['corrected'][-1], fine[-1])))
    save("strip.json", dict(cutoffs=CUTOFFS, rows=rows), started, [BULK16, TAIL], initial)
    for y in (1.+1e-8, 1.25, 2., 3., 4.):
        group = [r for r in rows if r['height'] == y]
        print(y, {key:max(r[key] for r in group) for key in
              ('corrected_level_3_to_4_l1','angle_16_to_64_l1','angle_32_to_64_l1')}, flush=True)


def tail_review():
    started = time.perf_counter(); initial = sources()
    tail = TailBank.load(TAIL); cusp = load_cusp_collision(CUSP, [0., .25])
    rows = []
    for order, q_order in ((12, 8), (16, 8), (24, 8), (16, 4), (16, 6)):
        row = puncture_tail(tail, cusp, start=1.+1e-12, radius=.08, s_max=64., order=order, q_order=q_order)
        rows.append(dict(quadrature_order=order, q_order=q_order, **row))
        print('tail', order, q_order, row['total'], flush=True)
    cache_banks = []
    endpoint_rows = []
    for p in sorted((DATA / 'tail_spectral_cache').glob('*.npz')):
        b = TailBank.load(p); config = b.metadata.get('spectral_audit_key', {})
        if config.get('long_max') is None: continue
        cache_banks.append(p)
        for z in (.23+.65j, .35+.05j, .04+.12j, .08*np.exp(.3j)):
            endpoint_rows.append(dict(config={k:config[k] for k in ('short_order','long_order','long_max')},
                z=z, value=integrated_cusp_density(b,z,1.+1e-12,8)))
    save("tail.json", dict(rows=rows, endpoint_rows=endpoint_rows), started,
         [TAIL, CUSP, *cache_banks], initial)


def integrated(radius, order, angle_order, q_order, tail_order, start):
    started = time.perf_counter(); initial = sources()
    bank = FactoredVVBank(VVBank.load(BULK16)); tail = TailBank.load(TAIL)
    ope = OPEBank.load(OPE); cusp = load_cusp_collision(CUSP, [0., .25])
    compact = compact_cusp_subtracted(bank, tail, ope, start=start, radius=radius,
        order=order, angle_order=angle_order, cutoffs=CUTOFFS, q_order=q_order)
    reference = 1.+1e-12
    long = puncture_tail(tail, cusp, start=reference, radius=radius, order=tail_order,
                         s_max=64., q_order=q_order)
    transfer = cusp.disc_tail(start,radius).sum()-cusp.disc_tail(reference,radius).sum()
    angular = correction(ope,cusp,radius=radius,start=start)
    total = compact['cap']+compact['massive_remainder']+compact['disk']+long['total']+transfer+angular
    name = f"integral_r{radius:g}_n{order}_a{angle_order}_q{q_order}_t{tail_order}_Y{start:g}.json"
    result = dict(compact=compact, long=long, collision_transfer=transfer,
                  angular_correction=angular, total=total, cutoffs=CUTOFFS)
    save(name, result, started, [BULK16, OPE, CUSP, TAIL], initial)
    print('total', total, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('pointwise','strip','tail','integrated'))
    p.add_argument('--radius', type=float, default=.08)
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--angle-order', type=int, default=16)
    p.add_argument('--q-order', type=int, default=8)
    p.add_argument('--tail-order', type=int, default=16)
    p.add_argument('--start', type=float, default=3.)
    a = p.parse_args()
    if a.mode == 'pointwise': pointwise()
    elif a.mode == 'strip': strip()
    elif a.mode == 'tail': tail_review()
    else: integrated(a.radius,a.order,a.angle_order,a.q_order,a.tail_order,a.start)


if __name__ == '__main__': main()
