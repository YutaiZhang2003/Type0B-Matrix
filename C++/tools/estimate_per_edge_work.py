"""Count independent-cutoff work and apply saved PBW fits; never evaluate PBW."""
import hashlib
import json
from math import isqrt
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'C++/results/per_edge_level10_2026-09-11'
CALIBRATION = ROOT / 'C++/results/current_algorithm_2026-09-11/pbw_estimates.json'


def character(cutoff, fermions, bosons):
    values = [1] + [0] * cutoff
    for step in bosons:
        for n in range(step, cutoff + 1):
            values[n] += values[n - step]
    for step in fermions:
        for n in range(cutoff, step - 1, -1):
            values[n] += values[n - step]
    return values


def main():
    level = 10
    ns = character(2 * level, range(1, 2 * level + 1, 2), range(2, 2 * level + 1, 2))
    ramond = character(level, range(1, level + 1), range(1, level + 1))
    monomials = len(ns) * len(ramond) ** 2
    counts = {
        'monomials': monomials,
        'gram_entries': sum(a * a for a in ns) + 4 * sum(a * a for a in ramond),
        'sum_dimension_cubed': sum(a ** 3 for a in ns) + 4 * sum(a ** 3 for a in ramond),
        'requested_tensor_entries': sum(4 * a * b * c for a in ns for b in ramond for c in ramond),
        'dense_contraction_products': sum(2 * a * b * c * (a + b + c + 1)
                                          for a in ns for b in ramond for c in ramond),
        'json_record_writes_proxy': monomials * (monomials + 1) // 2,
    }
    calibration = json.loads(CALIBRATION.read_text())
    assert ns == calibration['levels']['10']['pbw']['ns_dimensions_by_twice_level']
    assert ramond == calibration['levels']['10']['pbw']['ramond_dimensions_per_parity_by_level']
    models = {}
    for name, backend in calibration['calibrations'].items():
        models[name] = []
        for model in backend['models']:
            pieces = {feature: rate * counts[feature] for feature, rate in
                      zip(model['features'], model['fitted_seconds_per_feature_unit'])}
            total = sum(pieces.values())
            models[name].append({
                'precision_bits': backend['precision_bits'],
                'model': 'full' if 'dense_contraction_products' in pieces else 'lean',
                'negative_sector_seconds': total,
                'positive_sector_seconds_range': [total - pieces['requested_tensor_entries'] / 2, total],
                'fitted_feature_contributions_seconds': pieces,
            })

    # Replay the label order and transposition reuse, without numerical vertices.
    t = lambda n: n + 1 + n * (n - 1) // 2
    labels = [n for n in range(-4 * level - 1, 4 * level + 2, 2)
              if (n * n - 1) // 8 <= level]
    pairs = [(a, b) for a in labels for b in (a - 2, a + 2) if b in labels]
    cumulative, seen = [0], set()
    for n in range(-isqrt(2 * level), isqrt(2 * level) + 1):
        first = level - (n * n + 1) // 2
        for a, b in pairs:
            for c in labels:
                key = (n, min(a, b), max(a, b), c)
                cost = 0 if key in seen else (2 * t(first) * t(level - (a * a - 1) // 8)
                                               * t(level - (b * b - 1) // 8)
                                               * t(level - (c * c - 1) // 8))
                seen.add(key)
                cumulative.append(cumulative[-1] + cost)
    log_path = FOLDER / 'full_pipeline_40dps/inserted_per_edge_level10_40dps.log'
    log = log_path.read_text()
    branching = float(re.search(r'branching: ([\d.]+) s;', log).group(1))
    samples = []
    for number, elapsed in re.findall(r'numerator: (\d+) branches, ([\d.]+) s', log):
        number, elapsed = int(number), float(elapsed)
        if number > 0 and cumulative[number]:
            predicted = branching + (elapsed - branching) * cumulative[-1] / cumulative[number] + 80
            samples.append({'completed_branch_tuples': number, 'elapsed_seconds': elapsed,
                            'prefix_seed_work_bound': cumulative[number],
                            'estimated_full_seconds': predicted})
    report = {
        'q_level_cutoffs': [level] * 3, 'new_pbw_evaluations': 0,
        'pbw_integer_work_counts': counts,
        'largest_single_three_point_tensor_entries': max(ns) * max(ramond) ** 2,
        'pbw_cost_models': models,
        'matched_40_digit_pbw_calibration_available': False,
        'inserted_ccy_seed_work_bound': cumulative[-1],
        'inserted_scheduled_branch_tuples_before_zero_skips': len(cumulative) - 1,
        'inserted_progress_estimates': samples,
        'limitations': [
            'PBW fits are for the existing Python implementations, not a hypothetical C++ port.',
            'No numerical physical PBW, Gram matrix, or Ward identity was evaluated for these estimates.',
            'The PBW model spread is sensitivity to modelling choices, not a confidence interval.',
            'Native and FLINT fits use 53 and 384 bits; neither is a precision-matched 136-bit estimate.',
            'Large-matrix BLAS costs, Ward recursion costs, and memory pressure can differ from calibration.',
            'Gram dimensions are unchanged, but the persistent Ward cache is vastly larger; sufficient memory is assumed.',
            'CCY progress assumes the scheduled prefix approximates actual branch order and the average seconds per seed term remains similar.',
            'Exactly zero prefactors and residues can reduce actual CCY work; transition and product costs are included only through elapsed time.',
            'The CCY forecast adds about 80 seconds for post-numerator work, based on the completed ordinary run.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (CALIBRATION, Path(__file__), log_path)},
    }
    output = FOLDER / 'work_estimates.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pbw_counts': counts, 'latest_ccy_estimate': samples[-1],
                      'pbw_models': models}, indent=2))


if __name__ == '__main__':
    main()
