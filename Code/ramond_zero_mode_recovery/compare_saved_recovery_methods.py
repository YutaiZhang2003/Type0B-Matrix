"""Compare saved restricted-inverse and zero-mode physical coefficients.

No block is recomputed. Decimal midpoints are parsed at 160 decimal digits,
without the original backend's 1e-80 small-term clipping or binary64 rounding.
The +/- radii in saved FLINT displays are not treated as certified error
bounds on the underlying block computation.
"""

import argparse
import json
from pathlib import Path
import re
import time

import mpmath as mp


NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
SIGNED_NUMBER = rf'[+-]?{NUMBER}'


def midpoint(text):
    text = re.sub(r'\[([^\[\]]*?)\s+\+/-\s+[^\[\]]*\]', r'\1', text)
    text = text.replace(' ', '').replace('*', '').replace('I', 'j')
    if re.fullmatch(SIGNED_NUMBER, text):
        return mp.mpc(mp.mpf(text), 0)
    match = re.fullmatch(rf'({SIGNED_NUMBER})([+-]{NUMBER})j', text)
    if match:
        return mp.mpc(mp.mpf(match[1]), mp.mpf(match[2]))
    match = re.fullmatch(rf'({SIGNED_NUMBER})j', text)
    if match:
        return mp.mpc(0, mp.mpf(match[1]))
    raise ValueError(f'Unrecognized coefficient display: {text}')


def load_series(record):
    series = {}
    for row in record['coefficients']:
        levels = tuple(row['twice_levels'])
        if levels in series or len(levels) != 3 or len(row['values']) != 8:
            raise ValueError('Invalid or duplicate coefficient record.')
        series[levels] = tuple(midpoint(value) for value in row['values'])
    return series


def display(value):
    return mp.nstr(value, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--restricted', type=Path, default=here/'restricted_level10.json')
    parser.add_argument('--zero-mode', type=Path, default=here/'complex_level10.cold.json')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--tolerance', default='1e-70')
    args = parser.parse_args()
    mp.mp.dps = 160
    started = time.perf_counter()
    restricted_record = json.loads(args.restricted.read_text())
    zero_record = json.loads(args.zero_mode.read_text())
    if restricted_record['total_level'] != zero_record['total_level']:
        raise ValueError('The saved total-level cutoffs differ.')
    restricted = load_series(restricted_record)
    zero = load_series(zero_record)
    if restricted.keys() != zero.keys():
        raise ValueError('The saved sets of plumbing monomials differ.')
    cutoff = 2 * restricted_record['total_level']
    expected = {(a, b, c) for a in range(cutoff + 1)
                for b in range(0, cutoff - a + 1, 2)
                for c in range(0, cutoff - a - b + 1, 2)}
    if restricted.keys() != expected:
        raise ValueError('The saved coefficient set is not the full total-level truncation.')

    tolerance = mp.mpf(args.tolerance)
    if tolerance <= 0:
        raise ValueError('Tolerance must be positive.')
    differences = []
    by_level = {}
    failures = []
    for levels in sorted(restricted, key=lambda n: (sum(n), n)):
        for parity, (a, b) in enumerate(zip(restricted[levels], zero[levels])):
            absolute = abs(a - b)
            scaled = absolute / max(mp.mpf(1), abs(a), abs(b))
            entry = (absolute, scaled, levels, parity, a, b)
            differences.append(entry)
            by_level.setdefault(sum(levels), []).append(entry)
            if scaled > tolerance:
                failures.append(dict(twice_levels=levels,parity_index=parity,
                                     absolute_difference=display(absolute),scaled_difference=display(scaled)))

    def worst_record(entry):
        absolute, scaled, levels, parity, a, b = entry
        return dict(twice_levels=levels,levels=[n/2 for n in levels],
                    eta_parity=[(parity>>i)&1 for i in range(3)],
                    absolute_difference=display(absolute),scaled_difference=display(scaled),
                    restricted=dict(real=display(a.real),imag=display(a.imag)),
                    zero_mode=dict(real=display(b.real),imag=display(b.imag)))

    worst_absolute = max(differences,key=lambda row:row[0])
    worst_scaled = max(differences,key=lambda row:row[1])
    report = dict(status='passed' if not failures else 'failed',
                  comparison='User-requested comparison of two saved physical blocks; no recomputation or additional cross-checks',
                  restricted_file=str(args.restricted),zero_mode_file=str(args.zero_mode),
                  block=restricted_record['block'],parameters=restricted_record['parameters'],
                  zero_mode_parameter_provenance='LEVEL10_VALIDATION.md and check_level10_complex.py: canonical p=f=0, eta=eta_prime=1, first Ramond cut, same rational b and P',
                  total_level=restricted_record['total_level'],monomials_compared=len(restricted),
                  parity_components_compared=len(differences),
                  restricted_precision_bits=restricted_record['precision_bits'],
                  zero_mode_precision_bits=zero_record['precision_bits'],comparison_decimal_digits=mp.mp.dps,
                  scaled_difference_definition='abs(a-b)/max(1,abs(a),abs(b))',tolerance=args.tolerance,
                  maximum_absolute_difference=display(worst_absolute[0]),
                  maximum_scaled_difference=display(worst_scaled[1]),
                  worst_absolute=worst_record(worst_absolute),worst_scaled=worst_record(worst_scaled),
                  by_total_level=[dict(total_level=n/2,parity_components=len(rows),
                      maximum_absolute_difference=display(max(row[0] for row in rows)),
                      maximum_scaled_difference=display(max(row[1] for row in rows)))
                      for n,rows in sorted(by_level.items())],
                  failures=failures,elapsed_seconds=time.perf_counter()-started,
                  limitations='Comparison of saved decimal midpoints; the computations share primary and Ward routines and use a 1e-80 small-term cutoff. Agreement is not an independent bound on their common error.')
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['by_total_level','failures']},indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
