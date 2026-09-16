#!/usr/bin/env python3
"""Retain complex all-NS blocks while varying only the null-recursion order."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for relative in ('Code', 'Code/genus_2', 'Code/genus_2_cross_channel', 'Code/c_Recursion',
                 'Code/double_virasoro/nsrr', 'Code/full_ramond_block_runtime'):
    sys.path.insert(0, str(ROOT/relative))
BASE = ROOT/'Data Set/nsrr_double_virasoro_N7_L5_20260911'
GRID = ROOT/'Data Set/nsrr_threshold_order_scan_20260911/grid_N7'
OUTPUT = ROOT/'Data Set/nsrr_block_convergence_20260911'
spec = importlib.util.spec_from_file_location('convergence_endpoint', BASE/'validation/audit_all_ns_depth.py')
depth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', type=int, required=True)
    parser.add_argument('--points', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads((BASE/'config.json').read_text())
    saved_path = GRID/'target'/f'node-{args.node:05d}.json'
    node = json.loads(saved_path.read_text())
    saved = {r['point_id']: r for r in node['rows']}
    b = config['b']; Q = b+1/b; central = 1.5+3*Q*Q
    weights = tuple(Q*Q/8+p*p/2 for p in node['momenta'])
    for name in ('osp_two_chain_kernel', 'osp_norm', 'theta_orientation_sign'):
        fn = getattr(depth.core, name)
        if not hasattr(fn, 'cache_clear'):
            setattr(depth.core, name, lru_cache(maxsize=200000)(fn))
    out = args.output/'all_ns'
    out.mkdir(parents=True, exist_ok=True)
    path = out/f'node-{args.node:05d}.json'
    result = {'schema': 'all-ns-complex-order-convergence-v1', 'status': 'running',
              'node': args.node, 'momenta_geometry': node['momenta'],
              'settings': {'null_levels': [5, 6, 7, 8, 9, 10], 'endpoint_cap': 8,
                           'vacuum_word_length': 7, 'vacuum_max_mode': 50,
                           'working_precision': 50, 'global_tolerance': 2e-8,
                           'middle_chain': 'analytic', 'primary_parities': [0, 0, 0]},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), BASE/'config.json', saved_path,
                                          BASE/'validation/audit_all_ns_depth.py')}, 'rows': []}
    def save():
        path.write_text(json.dumps(result, indent=2)+'\n')
    start = time.monotonic()
    for point in config['points']:
        pid = point['point_id']
        if pid not in args.points:
            continue
        target = point['target']
        recursion = depth.FixedEndpointRecursion(
            channel='theta', q_values=tuple(map(complex, target['q_values'])),
            global_method='resummed', global_tolerance=2e-8, global_max_total_occupation=36,
            vacuum_word_length=7, vacuum_max_mode=50)
        recursion.endpoint_cap = 8
        for sector in (0, 1):
            row = {'point_id': pid, 'sector': sector, 'q_geometry': target['q_values'],
                   'lifts_geometry': target['lifts'], 'blocks': {}}
            result['rows'].append(row)
            for level in result['settings']['null_levels']:
                tick = time.monotonic()
                value = complex(recursion.collision_aware_block_mp(
                    weights=weights, sector=sector, recursion_order=2*level,
                    lifts=target['lifts'], central_charge=central, working_precision=50,
                    primary_parities=(0, 0, 0)))
                row['blocks'][str(level)] = {'real': value.real, 'imag': value.imag,
                                             'seconds': time.monotonic()-tick}
                save()
                print(args.node, pid, 'sector', sector, 'L', level, 'block', value,
                      'seconds', row['blocks'][str(level)]['seconds'], flush=True)
            val = lambda level: complex(row['blocks'][str(level)]['real'], row['blocks'][str(level)]['imag'])
            row['steps'] = [{'from': lo, 'to': hi,
                             'relative_complex_change': abs(val(hi)-val(lo))/abs(val(hi)),
                             'relative_norm_change': abs(abs(val(hi))**2/abs(val(lo))**2-1)}
                            for lo, hi in zip(range(5, 10), range(6, 11))]
            errors = [abs((abs(val(level))/abs(val(8)))**2 /
                          (saved[pid]['values'][str(2*level)][sector]/saved[pid]['values']['16'][sector])-1)
                      for level in (5, 6, 7)]
            row['maximum_saved_norm_ratio_residual'] = max(errors)
            if max(errors) > 1e-11:
                raise AssertionError('fresh blocks do not reproduce saved order changes')
            save()
        recursion.components.clear()
    result['status'] = 'complete'
    result['seconds'] = time.monotonic()-start
    save()


if __name__ == '__main__':
    main()
