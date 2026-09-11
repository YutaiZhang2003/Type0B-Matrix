"""Inspect saved optimized outputs and compare the saved ordinary baseline.

No conformal blocks are evaluated. Run after the timing driver has completed.
"""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'C++/results/per_edge_level10_2026-09-11'


def magnitude(value):
    real, imag = Decimal(value['real']), Decimal(value['imag'])
    assert real.is_finite() and imag.is_finite()
    return (real * real + imag * imag).sqrt()


def compare(reference, current):
    before = {tuple(row['exponents']): row['values'] for row in reference['coefficients']}
    maximum_absolute = Decimal(0)
    maximum_scaled = Decimal(0)
    worst_scaled = None
    components = 0
    for row in current['coefficients']:
        key = tuple(row['exponents'])
        for parity, (a, b) in enumerate(zip(before[key], row['values'])):
            difference = magnitude({
                'real': str(Decimal(a['real']) - Decimal(b['real'])),
                'imag': str(Decimal(a['imag']) - Decimal(b['imag']))})
            scale = max(Decimal(1), magnitude(a), magnitude(b))
            scaled = difference / scale
            maximum_absolute = max(maximum_absolute, difference)
            if scaled > maximum_scaled:
                maximum_scaled = scaled
                worst_scaled = {'exponents': list(key), 'parity_component': parity}
            components += 1
    return {'components': components,
            'maximum_absolute_difference': str(maximum_absolute),
            'maximum_scaled_difference': str(maximum_scaled),
            'scale_definition': 'max(1, abs(reference), abs(current))',
            'worst_scaled_component': worst_scaled,
            'interpretation': 'Same-precision implementation comparison; not an independent accuracy certificate.'}


def main():
    directory = FOLDER / 'optimized_full_pipeline_40dps'
    timings = json.loads((directory / 'timings.json').read_text())
    assert timings['status'] == 'completed'
    expected = {(a, b, b, d) for a in range(21) for b in range(11)
                for d in range(0, 21, 2)}
    hashes_match = all(hashlib.sha256((ROOT / 'C++' / name).read_bytes()).hexdigest() == digest
                       for name, digest in timings['source_sha256'].items())
    binary_matches = hashlib.sha256((ROOT / 'C++/bin/ramond').read_bytes()).hexdigest() == timings['binary_sha256']
    result = {'source_hashes_match': hashes_match, 'binary_hash_matches': binary_matches,
              'runs': []}
    assert hashes_match and binary_matches
    for run in timings['runs']:
        assert run['status'] == 'completed' and run['exit_code'] == 0
        data = json.loads(Path(run['result']).read_text())
        rows = data['coefficients']
        assert len(rows) == len(expected) and {tuple(row['exponents']) for row in rows} == expected
        assert data['q_level_cutoffs'] == [10, 10, 10]
        assert data['dps'] == 40 and data['precision_bits'] == 136
        assert data['truncation'] == 'per-edge'
        assert data['etas'] == ([1, 1] if run['mode'] == 'ordinary' else [1, -1])
        assert all(len(row['values']) == 8 for row in rows)
        largest = max(magnitude(value) for row in rows for value in row['values'])
        entry = {'mode': run['mode'], 'monomials': len(rows), 'components': 8 * len(rows),
                 'all_components_finite': True, 'maximum_coefficient_magnitude': str(largest)}
        if run['mode'] == 'ordinary':
            reference = json.loads((FOLDER / 'full_pipeline_40dps/ordinary_per_edge_level10_40dps.json').read_text())
            entry['saved_baseline_comparison'] = compare(reference, data)
        result['runs'].append(entry)
    target = FOLDER / 'optimized_output_inspection.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with localcontext() as context:
        context.prec = 60
        main()
