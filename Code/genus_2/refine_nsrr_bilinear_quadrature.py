#!/usr/bin/env python3
"""Refine both momentum integrals with the new NSRR pairing held fixed.

Only complete tensor grids enter the convergence table. Each new source
node uses the current native L5 engine; L3 is retained as a separate
block-cutoff diagnostic. The target uses R12 throughout. Stop only after
two successive refinements stabilize each channel and their ratios.
"""
from __future__ import annotations

import argparse
import cmath
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in ('Code', 'Code/genus_2', 'Code/c_Recursion',
                  'Code/genus_2_cross_channel', 'Code/full_ramond_block_runtime'):
    sys.path.insert(0, str(ROOT / directory))

import numpy as np
from scipy.special import roots_genlaguerre
from all_ns_reflected_sewing import LIFTS, lift_conversion
from compare_nsrr_nsnsns_theta import NSGenus2CRecursion, _rules, _measure
from generic_super_liouville_structure_constants import GenericSuperLiouvilleConstants
from nsrr_bilinear_sewing import CHANNELS, projected_components, contract_nsrr_bilinear
from nsrr_cpp_backend import NativeNSRR, implementation_hashes
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
from recombine_saved_genus2_coefficient_ledger import decode, encode, csum, object_digest, write_csv
from test_nsrr_bilinear_cross_channel import save

BASE = ROOT / 'Data Set/nsrr_bilinear_cross_channel_20260915'
OUT = ROOT / 'Data Set/nsrr_bilinear_quadrature_20260915'
LABELS = tuple((f, e, e) for f in (0, 1) for e in (1, -1))
VECTORS = {'s=+1': np.array([1, -1j]) / math.sqrt(2),
           's=-1': np.array([1, 1j]) / math.sqrt(2),
           'resolved_00': np.array([1, 0]), 'resolved_11': np.array([0, 1]),
           'spin_sum': None}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def matrix(row):
    return np.array([[decode(z) for z in r] for r in row['density']])


def encoded_matrix(m):
    return [[encode(z) for z in row] for row in m]


def scalar(m, vector):
    z = complex(np.trace(m) if vector is None else vector @ m @ vector.conjugate())
    assert abs(z.imag) < 1e-12 * abs(z.real) and z.real > 0
    return z.real


def prepare(output, maximum_order, tolerance):
    old = read(BASE / 'config.json')
    for name in ('provenance.json', 'target_provenance.json', 'report_provenance.json'):
        for path, digest in read(BASE / name).items():
            assert sha(ROOT / path) == digest, path
    source_config = read(ROOT / 'Data Set/nsrr_trial_L5_N3_local_20260830/summary.json')['config']
    names = [Path(__file__), ROOT / 'Code/genus_2/nsrr_bilinear_sewing.py',
             ROOT / 'Code/genus_2/all_ns_reflected_sewing.py',
             ROOT / 'Code/genus_2/nsrr_plumbing_adapter.py',
             ROOT / 'Code/c_Recursion/ns_genus2_partition.py',
             ROOT / 'Code/c_Recursion/generic_super_liouville_structure_constants.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in names}
    config = dict(schema='nsrr-bilinear-quadrature-v1', b=old['b'], parameters=old['parameters'],
        point=next(p for p in old['geometry'] if p['t'] == .6),
        source_envelope=source_config['q_envelope'], target_envelope=old['target_envelope'],
        source_levels=[3, 5], target_order=12, native_precision=40,
        block_precision=old['block_precision'], structure_precision=old['structure_precision'],
        global_tolerance=old['global_tolerance'], global_max_occupation=60,
        quadrature_orders=list(range(3, maximum_order + 1)),
        relative_stability_tolerance=tolerance, required_successive_steps=2,
        stability_observables='source Z, target Z, and their normalized ratio for both s signs, both resolved spins, and the spin sum',
        primary_policy=old['primary_policy'], transport_policy=old['transport_policy'],
        implementation_sha256={**hashes, **implementation_hashes()},
        baseline_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            (BASE / 'config.json', BASE / 'source.json', BASE / 'target_summary.json',
             BASE / 'provenance.json', BASE / 'target_provenance.json', BASE / 'report_provenance.json')})
    if (output / 'config.json').exists():
        assert read(output / 'config.json') == config, 'Existing calculation has a different design or implementation'
    save(output / 'config.json', config)
    return config


def grid(config, channel, n, index):
    envelope = config[f'{channel}_envelope']
    rules = _rules(envelope, n)
    ids = np.unravel_index(index, (n,) * 3)
    momenta = tuple(float(rules[e][0][j]) for e, j in enumerate(ids))
    measure = float(_measure(rules, ids))
    # Independent reconstruction of d^3p/pi^3 from the Laguerre weights.
    x, w = roots_genlaguerre(n, -.5)
    check = math.prod(w[j] * math.exp(x[j]) / (2 * math.pi * math.sqrt(-math.log(abs(q))))
                      for q, j in zip(envelope, ids))
    assert abs(measure / check - 1) < 1e-12
    return momenta, measure


def source_node(config, n, index):
    momenta, measure = grid(config, 'source', n, index)
    b = config['b']
    constants = GenericSuperLiouvilleConstants(b, dps=config['structure_precision'],
        mu=complex(config['parameters']['mu']),
        include_cosmological_prefactor=config['parameters']['include_cosmological_prefactor'])
    couplings = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
    runtime = NativeNSRR(b, momenta[::-1], max(config['source_levels']), dps=config['native_precision'])
    components = {k: runtime.physical_components(*k) for k in LABELS}
    assert runtime.ward_residual_maximum < 1e-8
    q = tuple(map(complex, config['point']['q_source']))
    plumbing = NSRRPlumbingInputs(q, (1, 1, 1), GEOMETRY_SECTORS)
    primary = plumbing.primary(b, momenta)
    rows = []
    for level in config['source_levels']:
        blocks = {k: csum(projected_components(v) * math.prod(z ** (a / 2) for z, a in zip(q[::-1], ex))
                         for ex, v in table.items() if sum(ex) <= 2 * level)
                  for k, table in components.items()}
        density = np.zeros((2, 2), complex)
        for i, eta in enumerate((1, -1)):
            d = np.array([blocks[0, eta, eta], 1j * blocks[1, eta, eta]])
            density += couplings[i] ** 2 / 4 * np.outer(d, d.conjugate())
        density *= measure * abs(primary) ** 2
        # Mixed eta blocks are annihilated by M for r=-1, and are not evaluated.
        full = {k: blocks.get(k, 0j) for k in CHANNELS}
        for sign in (1, -1):
            z = measure * contract_nsrr_bilinear(descendant_blocks=full,
                antiholomorphic_blocks={k: z.conjugate() for k, z in full.items()},
                left_bry=couplings, right_bry=couplings, physical_lifts_slots=(sign, -1, 1),
                primary=primary, antiholomorphic_primary=primary.conjugate())['total']
            v = VECTORS[f's={sign:+d}']
            assert abs(z - v @ density @ v.conjugate()) < 3e-13 * max(np.max(abs(density)), 1e-300)
        rows.append(dict(L=level, blocks=[encode(blocks[k]) for k in LABELS],
                         density=encoded_matrix(density)))
    return dict(momenta=momenta, measure=measure, constants=list(map(encode, couplings)),
                primary=encode(primary), log_q=list(map(lambda z: encode(cmath.log(z)), q)),
                channels=LABELS, rows=rows, native_checks=runtime.diagnostics())


def target_node(config, n, index):
    momenta, measure = grid(config, 'target', n, index)
    b = config['b']; Q = b + 1 / b
    weights = tuple(Q * Q / 8 + p * p / 2 for p in momenta)
    constants = GenericSuperLiouvilleConstants(b, dps=config['structure_precision'],
        mu=complex(config['parameters']['mu']),
        include_cosmological_prefactor=config['parameters']['include_cosmological_prefactor'])
    couplings = constants.ns_constants(*momenta)
    q = tuple(map(complex, config['point']['q_target']))
    primary = cmath.exp(sum(h * cmath.log(z) for h, z in zip(weights, q)))
    recursion = NSGenus2CRecursion(channel='theta', q_values=q, global_method='resummed',
        global_tolerance=config['global_tolerance'], global_max_total_occupation=config['global_max_occupation'],
        vacuum_word_length=7, vacuum_max_mode=50)
    raw_indices = [LIFTS.index(tuple(r['target_raw_lift'])) for r in config['point']['spin_rows']]
    density = np.zeros((2, 2), complex); sectors = []
    for sector in (0, 1):
        f = np.array([recursion.collision_aware_block_mp(weights=weights, sector=sector,
            recursion_order=config['target_order'], lifts=lift, central_charge=1.5 + 3 * Q * Q,
            working_precision=config['block_precision']) for lift in LIFTS], complex)
        raw = np.asarray(lift_conversion(sector)) @ f
        d = raw[raw_indices]
        density += measure * abs(primary) ** 2 * couplings[sector] ** 2 * np.outer(d, d.conjugate())
        sectors.append(dict(sector=sector, literal_blocks=list(map(encode, f)), raw_blocks=list(map(encode, raw))))
    assert recursion.global_nonconverged_calls == 0, 'Unconverged global block'
    return dict(momenta=momenta, measure=measure, constants=list(map(encode, couplings)),
        primary=encode(primary), weights=weights, log_q=list(map(lambda z: encode(cmath.log(z)), q)),
        sectors=sectors, density=encoded_matrix(density), global_nonconverged_calls=0,
        global_max_occupation_used=recursion.global_max_used)


def worker(output, channel, n, index):
    config = read(output / 'config.json'); digest = object_digest(config)
    path = output / channel / f'N{n}' / f'node-{index:04d}.json'
    if path.exists():
        row = read(path)
        assert (row['config_digest'], row['channel'], row['N'], row['index']) == (digest, channel, n, index)
        return
    start = time.monotonic()
    result = (source_node if channel == 'source' else target_node)(config, n, index)
    save(path, dict(config_digest=digest, channel=channel, N=n, index=index,
                    seconds=time.monotonic() - start, **result))


def baseline(config):
    s = [r for r in read(BASE / 'source.json') if r['t'] == config['point']['t'] and r['N'] == 3]
    source = {int(r['L']): matrix(r) for r in s}
    target = {r['N']: matrix(r) for r in read(BASE / 'target_summary.json')
              if r['t'] == config['point']['t'] and r['order'] == config['target_order']}
    return source, target


def reduce_order(output, config, n):
    source0, target0 = baseline(config)
    digest = object_digest(config)
    matrices = {}; hashes = {}
    for channel in ('source', 'target'):
        if channel == 'source' and n == 3:
            matrices[channel] = source0
            continue
        if channel == 'target' and n in target0:
            matrices[channel] = target0[n]
            continue
        records = []
        for index in range(n ** 3):
            path = output / channel / f'N{n}' / f'node-{index:04d}.json'
            node = read(path)
            assert (node['config_digest'], node['channel'], node['N'], node['index']) == (digest, channel, n, index)
            records.append(node); hashes[str(path.relative_to(ROOT))] = sha(path)
        def add(ms):
            return np.array([[csum(m[i, j] for m in ms) for j in (0, 1)] for i in (0, 1)])
        if channel == 'source':
            matrices[channel] = {level: add([matrix(next(r for r in node['rows'] if r['L'] == level))
                                             for node in records]) for level in config['source_levels']}
        else:
            matrices[channel] = add([matrix(node) for node in records])
    source = matrices['source'][5]; target = matrices['target']
    for m in (source, target):
        scale = np.max(abs(m))
        assert np.max(abs(m - m.conjugate().T)) < 1e-12 * scale
        assert np.min(np.linalg.eigvalsh(m)) > -1e-12 * scale
    kappa = 1 + 2 * (config['b'] + 1 / config['b']) ** 2
    frame = (config['point']['source_free'] / config['point']['target_free']) ** kappa
    values = {}
    for label, v in VECTORS.items():
        zs, zt = scalar(source, v), scalar(target, v)
        values[label] = dict(source_Z=zs, target_Z=zt, ratio=zs / zt / frame,
            source_L3_Z=scalar(matrices['source'][3], v),
            source_L3_to_L5_relative_change=zs / scalar(matrices['source'][3], v) - 1)
    result = dict(N=n, nodes_per_channel=n ** 3, source_L=5, target_R=config['target_order'],
        t=config['point']['t'], source_density=encoded_matrix(source), target_density=encoded_matrix(target),
        free_frame_power=frame, values=values, complete=True)
    save(output / f'order-N{n}.json', result)
    save(output / f'provenance-N{n}.json', hashes)
    return result


def report(output, config, orders):
    rows = []; steps = []; streak = 0
    for i, r in enumerate(orders):
        changes = []
        for label, values in r['values'].items():
            row = dict(N=r['N'], comparison=label, **values,
                       source_Z_relative_step=None, target_Z_relative_step=None, ratio_relative_step=None)
            if i:
                previous = orders[i - 1]['values'][label]
                for key in ('source_Z', 'target_Z', 'ratio'):
                    change = values[key] / previous[key] - 1
                    row[f'{key}_relative_step'] = change; changes.append(abs(change))
            rows.append(row)
        maximum = max(changes) if changes else None
        if maximum is not None:
            streak = streak + 1 if maximum < config['relative_stability_tolerance'] else 0
        steps.append(dict(N=r['N'], maximum_relative_step=maximum, consecutive_stable_steps=streak))
    stable = streak >= config['required_successive_steps']
    summary = dict(status='quadrature_stabilized' if stable else 'refinement_in_progress',
        completed_orders=[r['N'] for r in orders], tolerance=config['relative_stability_tolerance'],
        required_successive_steps=config['required_successive_steps'], steps=steps, results=orders,
        scope='Central saved surface t=0.60, fixed source L5 and target R12. Stability is empirical and does not bound block truncation or establish crossing.')
    save(output / 'summary.json', summary); write_csv(output / 'convergence.csv', rows)
    lines = ['# NSRR/all-NS momentum quadrature refinement', '',
        f'Status: **{summary["status"]}**. Both channels use the same number N of quadrature nodes per direction.', '',
        'The pairing, spin transport, BRY constants, and primary factors are unchanged from the preceding cross-channel test. '
        'Source total descendant cutoff L=5 and target recursion twice-level R=12 are fixed. '
        'The central surface is t=0.60 and b=1.4. Agreement requires normalized ratio one.', '',
        '| N | Nodes/channel | Source spin sum | Target spin sum | Ratio s=+ | Ratio s=- | Spin-sum ratio | Max relative step |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r, step in zip(orders, steps):
        v = r['values']; z = v['spin_sum']; change = step['maximum_relative_step']
        step_text = '—' if change is None else f'{change:.3e}'
        lines.append(f'| {r["N"]} | {r["nodes_per_channel"]} | {z["source_Z"]:.12e} | {z["target_Z"]:.12e} | '
            f'{v["s=+1"]["ratio"]:.10f} | {v["s=-1"]["ratio"]:.10f} | {z["ratio"]:.10f} | {step_text} |')
    lines += ['', 'The maximum step checks **each channel integral and its ratio**, for s=±1, '
        'each resolved spin, and their sum. Stabilization requires two successive steps below '
        f'{config["relative_stability_tolerance"]:.1e} relative. This is not a rigorous error bound.', '',
        'The existing N3 source and N3/N4 target are reused with verified input hashes. All new source nodes '
        'retain complex projected blocks from the current 40-digit native engine, at both L3 and L5. '
        'All new target nodes retain four complex lift blocks in each form sector. '
        'Each node independently checks the momentum measure. Only complete grids enter this table.', '',
        'The target global series retains its 2e-9 tolerance; its allowed occupation ceiling is increased '
        'from 36 to 60 to accommodate larger tail momenta. The recursion order remains R12. '
        'No unconverged global sum is accepted.', '',
        'See `convergence.csv` for separate signs and resolved spins; `config.json` for frozen conventions '
        'and implementation hashes; `source/`, `target/`, and `provenance-N*.json` for complete node data.', '',
        'Reproduce with `python Code/genus_2/refine_nsrr_bilinear_quadrature.py --workers 6`.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    return stable, steps[-1]


def run(output, workers, maximum_order, tolerance):
    config = prepare(output, maximum_order, tolerance)
    orders = []; env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def dispatch(task):
        channel, n, index = task
        log = output / 'logs' / f'{channel}-N{n}-{index:04d}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                '--node', channel, str(n), str(index)], stdout=stream, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f'Node failed: {task}. See {log}')
    for n in config['quadrature_orders']:
        tasks = [(channel, n, i) for channel in ('source', 'target')
                 if not (channel == 'source' and n == 3 or channel == 'target' and n in (3, 4))
                 for i in range(n ** 3)]
        print(f'Starting N={n}: {len(tasks)} node evaluations.', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(dispatch, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done % 25 == 0 or done == len(tasks):
                    print(f'N={n}: {done}/{len(tasks)} complete.', flush=True)
        result = reduce_order(output, config, n); orders.append(result)
        stable, step = report(output, config, orders)
        print(json.dumps(dict(ratios={k: v['ratio'] for k, v in result['values'].items()}, **step)), flush=True)
        if stable:
            print('Quadrature stability criterion satisfied.', flush=True)
            return
    print('Maximum requested order reached; stability criterion not satisfied.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--maximum-order', type=int, default=10)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--node', nargs=3, metavar=('CHANNEL', 'N', 'INDEX'))
    args = parser.parse_args()
    if args.node:
        channel, n, index = args.node
        assert channel in ('source', 'target')
        worker(args.output, channel, int(n), int(index))
    else:
        assert args.workers > 0 and args.maximum_order >= 5 and args.tolerance > 0
        run(args.output, args.workers, args.maximum_order, args.tolerance)


if __name__ == '__main__':
    main()
