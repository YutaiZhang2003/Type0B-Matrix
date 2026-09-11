"""Precision calibration of the PBW arithmetic backend, with no PBW evaluation.

Only synthetic scalar operations and synthetic dense matrix products are timed.
Saved 384-bit PBW feature fits are rescaled by measured 136/384-bit ratios.
This is a cost estimate, not a measured physical-block runtime.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/ramond_zero_mode_recovery'))
import complex_arithmetic as ca
import numpy as np

FOLDER = ROOT / 'C++/results/per_edge_level10_2026-09-11'


def inputs(bits):
    ca.ctx.prec = bits
    x = [ca.F(ca.acb(11 + j, 7 - j) / ca.acb(37 + j, 19)) for j in range(64)]
    y = [ca.F(ca.acb(3 + 2 * j, 5 + j) / ca.acb(29 + j, 13)) for j in range(64)]
    return x, y


def scalar_time(x, y, mix):
    value = ca.F(1)
    start = time.perf_counter()
    for j in range(4000):
        k = j % 64
        value = (value * x[k] + y[k]) / (x[k] + 5) if mix else value + x[k] * y[k]
    elapsed = time.perf_counter() - start
    assert bool(value)
    return elapsed


def matrix_time(x, y, n, columns):
    a = np.array([x[(7 * i + 3 * j) % 64] for i in range(n) for j in range(n)],
                 dtype=object).reshape(n, n)
    b = np.array([y[(5 * i + 11 * j) % 64] for i in range(n) for j in range(columns)],
                 dtype=object).reshape(n, columns)
    start = time.perf_counter()
    result = ca.mm(a, b)
    elapsed = time.perf_counter() - start
    assert result.shape == (n, columns)
    return elapsed


def main():
    started = time.perf_counter()
    measurements = {str(bits): {'scalar_multiply_add': [], 'scalar_mixed': [],
                               'matrix_32x32x64': [], 'matrix_232x232x64': []}
                    for bits in (136, 384)}
    # Warm both precisions, then alternate the order to limit first-run bias.
    for bits in (136, 384):
        x, y = inputs(bits)
        scalar_time(x, y, False)
        matrix_time(x, y, 16, 8)
    for order in ((136, 384), (384, 136), (136, 384)):
        for bits in order:
            x, y = inputs(bits)
            row = measurements[str(bits)]
            row['scalar_multiply_add'].append(scalar_time(x, y, False))
            row['scalar_mixed'].append(scalar_time(x, y, True))
            row['matrix_32x32x64'].append(matrix_time(x, y, 32, 64))
            row['matrix_232x232x64'].append(matrix_time(x, y, 232, 64))
    ratios = {k: statistics.median(measurements['136'][k]) /
                 statistics.median(measurements['384'][k]) for k in measurements['136']}
    scalar_range = [min(1., ratios['scalar_multiply_add'], ratios['scalar_mixed']),
                    max(1., ratios['scalar_multiply_add'], ratios['scalar_mixed'])]
    matrix_range = [min(1., ratios['matrix_32x32x64'], ratios['matrix_232x232x64']),
                    max(1., ratios['matrix_32x32x64'], ratios['matrix_232x232x64'])]
    work_path = FOLDER / 'work_estimates.json'
    work = json.loads(work_path.read_text())
    estimates = []
    for model in work['pbw_cost_models']['flint_high_precision']:
        pieces = model['fitted_feature_contributions_seconds']
        candidates = []
        for sr in scalar_range:
            for mr in matrix_range:
                adjusted = {name: value * (sr if name in ('gram_entries', 'requested_tensor_entries')
                                           else mr if name in ('sum_dimension_cubed', 'dense_contraction_products')
                                           else 1.) for name, value in pieces.items()}
                total = sum(adjusted.values())
                candidates.append((total, total - adjusted['requested_tensor_entries'] / 2))
        estimates.append({'model': model['model'],
                          'negative_sector_seconds_range': [min(x[0] for x in candidates), max(x[0] for x in candidates)],
                          'positive_sector_seconds_range': [min(x[1] for x in candidates), max(x[0] for x in candidates)]})
    result = {
        'target_precision_bits': 136, 'calibration_precision_bits': 384,
        'new_pbw_evaluations': 0, 'new_physical_gram_or_ward_evaluations': 0,
        'arithmetic_backend': 'same complex_arithmetic.F and mm as saved physical PBW',
        'measured_seconds': measurements, 'median_136_over_384_ratios': ratios,
        'scalar_cost_ratio_range': scalar_range, 'matrix_cost_ratio_range': matrix_range,
        'estimated_136_bit_pbw': estimates,
        'benchmark_wall_seconds': time.perf_counter() - started,
        'limitations': [
            'These are synthetic arithmetic ratios applied to existing PBW cost models, not measured full-block runtimes.',
            'The ratio ranges include one because Python, cache, and conversion overhead need not speed up with precision.',
            'Matrix operands are synthetic and have 64 columns; the largest real contractions have many more columns.',
            'Feature-fit uncertainty, changing cache misses, BLAS regimes, and memory pressure remain unmeasured.',
            'The scalar threshold is unchanged; no benchmark operand approaches that threshold.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / 'Code/ramond_zero_mode_recovery/complex_arithmetic.py', work_path)},
    }
    path = FOLDER / 'pbw_40digit_arithmetic_calibration.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
