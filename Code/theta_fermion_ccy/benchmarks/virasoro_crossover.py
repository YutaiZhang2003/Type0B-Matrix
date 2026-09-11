"""Time one full ordinary Virasoro block, with no coefficient comparison.

Each invocation starts fresh: one method, one copy, one total cutoff.
The two methods use the same exact rational inputs and 100-digit mpmath
working precision. All four integer descendant levels enter total degree.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import mpmath as mp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from punctured_ccy import PuncturedCCY, downward_indices
from direct_virasoro_pbw import DirectVirasoroPBW


def inputs(copy):
    b = Fraction(7, 5)
    Q = b+1/b
    momenta = tuple(map(Fraction, ("11/23", "13/29", "13/29", "17/31")))
    labels = tuple(map(Fraction, ("-1/2", "-1/4", "1/4", "-1/4")))
    embedding_b = b if copy == 1 else 1/b
    central = 1+3*Q*Q/(1-embedding_b*embedding_b)
    weights = tuple((Q*Q/4-(P+2*n*embedding_b)**2)/(2*(1-embedding_b*embedding_b))
                    for P, n in zip(momenta, labels))
    external = -(1+2*embedding_b*embedding_b)/(2*(1-embedding_b*embedding_b))
    return central, weights, external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('ccy', 'pbw'), required=True)
    parser.add_argument('--copy', type=int, choices=(1, 2), required=True)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--dps', type=int, default=100)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    if args.level < 0:
        parser.error('The total descendant cutoff must be nonnegative')
    mp.mp.dps = args.dps
    central, weights, external = inputs(args.copy)
    started = time.perf_counter()
    indices = downward_indices(args.level)
    engine_type = PuncturedCCY if args.method == 'ccy' else DirectVirasoroPBW
    engine = engine_type(central, weights, external, dps=args.dps)
    coefficients = engine.full_series(indices=indices)
    elapsed = time.perf_counter()-started
    diagnostic = engine.cache_info() if args.method == 'ccy' else engine.diagnostics()
    files = [Path(__file__), HERE/'direct_virasoro_pbw.py', HERE.parent/'punctured_ccy.py',
             HERE.parent/'schottky_vacuum.py', ROOT/'Code/ramond_branching_recursion/compute_target.py']
    report = {'status': 'computed_for_timing_only', 'method': args.method, 'copy': args.copy,
              'total_descendant_level': args.level, 'dps': args.dps, 'arithmetic': 'mpmath',
              'computation_seconds': elapsed, 'coefficient_count': len(coefficients),
              'scope': 'Fresh engine, index generation and full ordinary block through cutoff; excludes imports and serialization',
              'truncation': 'a+b+c+d<=N, all four integer Virasoro descendant levels',
              'parameters': {'central_charge': str(central), 'weights': list(map(str, weights)),
                             'external_weight': str(external)},
              'diagnostics': diagnostic,
              'source_hashes': {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'coefficients': [{'levels': key,
                                'value': {'real': mp.nstr(mp.re(value), args.dps),
                                          'imag': mp.nstr(mp.im(value), args.dps)}}
                               for key, value in coefficients.items()]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('method','copy','total_descendant_level',
                                                  'computation_seconds','coefficient_count')}), flush=True)


if __name__ == '__main__':
    main()
