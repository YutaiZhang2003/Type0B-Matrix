#!/usr/bin/env python3
"""Prepare fixed-spin geometric-BPZ configurations for matched surfaces.

Momentum nodes and three-point constants are separate generated inputs. No
saved chiral block or previous comparison result is read.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / 'Data Set/nsrr_nsnsns_generic_10point_N4_20260904/config.json'


def signs(marked, branch, source):
    beta = marked[1]
    if source:
        alpha = marked[0]
        charge = [
            (beta[i] + sum(branch[i][j] * alpha[j] for j in range(2)) + branch[i][i]) % 2
            for i in range(2)
        ]
        if charge not in ([0, 0], [1, 1]):
            raise ValueError(f'odd source charge-frame characteristic {charge}')
        return [(-1) ** charge[0], -1, 1]
    charge = [(beta[i] + branch[i][i]) % 2 for i in range(2)]
    return [-1, (-1) ** charge[1], (-1) ** charge[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, default=GEOMETRY)
    args = parser.parse_args()
    geometry = json.loads(args.geometry.read_text())
    points = geometry['points']
    if len(points) != 10:
        raise ValueError('Expected ten matched surfaces')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'partition-ten-geometric-fixed-input-v1',
                'geometry': str(args.geometry),
                'point_ids': []}
    for point in points:
        name = point['point_id']
        marked_source = [point['source']['characteristic'], [[1, 1], [1, 1]]]
        marked_target = [point['target']['characteristic'], [[0, 0], [1, 0]]]
        source_branches = [point['source']['period_branch']] * 2
        target_branches = [point['target']['period_branch']] * 2
        data = {'schema': 'paper-partition-fixed-spin-input-v2',
                'b': geometry['b'], 'edge_order': [1, 2, 3],
                'source_to_target': geometry['source_to_target'],
                'point': {
            'source_free': point['source']['Z_free'],
            'target_free': point['target']['Z_free'],
            'q_source': list(reversed(point['source']['q_values'])),
            'q_target': list(reversed(point['target']['q_values'])),
        }}
        data['source_marked_spins'] = marked_source
        data['target_marked_spins'] = marked_target
        data['source_period_branch'] = point['source']['period_branch']
        data['target_period_branch'] = point['target']['period_branch']
        data['source_eta_e'] = [signs(s, b, True) for s, b in zip(marked_source, source_branches)]
        data['target_eta_e'] = [signs(s, b, False) for s, b in zip(marked_target, target_branches)]
        data['provenance'] = {
            'geometry': str(args.geometry), 'point_id': name,
            'physical_spin': 0,
            'policy': 'fixed marked spin transported by the archived symplectic map; no fit',
        }
        path = args.output / f'{name}.json'
        path.write_text(json.dumps(data, indent=2) + '\n')
        manifest['point_ids'].append(name)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
