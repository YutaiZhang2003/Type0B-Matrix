#!/usr/bin/env python3
"""Sum the paper's literal all-NS blocks diagonally as a failure diagnostic.

The source sums are read from the saved run; their Ramond sewing remains
under review. This script does not claim a physical fixed-spin partition.
"""
import argparse
import json
import math
from pathlib import Path


def complex_number(value):
    return complex(float(value['real']), float(value['imag']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    terms = [[] for _ in range(4)]
    nodes = sorted((args.run / 'target').glob('node-*.json'))
    for path in nodes:
        node = json.loads(path.read_text())
        row = node['rows'][-1]
        C_a = [complex_number(value) for value in node['C_a']]
        F = [[complex_number(value) for value in form] for form in row['F']]
        measure = float(node['measure']['real'])
        primary = complex_number(node['primary'])
        for spin in range(4):
            trial = measure * abs(primary)**2 * sum(
                (-1)**a * C_a[a]**2 * abs(F[a][spin])**2 for a in range(2))
            if abs(trial.imag) > 1e-12 * max(abs(trial.real), 1e-300):
                raise AssertionError(('nonreal diagonal trial', path, spin))
            terms[spin].append(trial.real)
    if len(nodes) != 1000:
        raise AssertionError(('incomplete saved target grid', len(nodes)))
    target = [math.fsum(values) for values in terms]
    saved = json.loads((args.run / 'comparison.json').read_text())
    source = [complex_number(value).real for value in saved['comparisons'][-1]['source_Z'][:2]]
    frame = complex_number(saved['free_frame_power']).real
    result = {
        'status': 'diagnostic_only_literal_paper_F',
        'source_R_sewing_independently_verified': False,
        'target_nodes': len(nodes),
        'target_diagonal_trial': target,
        'source_values_from_saved_run': source,
        'ratios_if_diagonal_trial_were_fixed_spin': [
            source[0] / (frame * target[2]),
            source[1] / (frame * target[0]),
        ],
        'sum_ratio_if_diagonal_trial_were_fixed_spin':
            sum(source) / (frame * (target[2] + target[0])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
