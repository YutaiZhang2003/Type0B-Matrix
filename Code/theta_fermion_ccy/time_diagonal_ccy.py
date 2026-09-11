"""Time actual Python CCY series with the proposed diagonal-target budgets.

This measures the recursion stage, not a recovered physical block. Sample
mode runs one actual branching tuple per distinct budget, both Virasoro
copies, and weights those times by the number of tuples with that budget.
Computed series are stored in a pickle stream for reuse, separately timed.
No PBW comparison or coefficient-accuracy test is performed.
"""

import argparse
from collections import defaultdict
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import hashlib
import json
import pickle
import resource
import sys
import time

import pipeline
from punctured_ccy import PuncturedCCY


def cases(level):
    ramond_level = lambda n: int(2*n*n-F(1, 8))
    for k in range(-4*level-1, 4*level+2, 2):
        incoming, outgoing = F(k, 4), F(k+2, 4)
        for twice_ns in range(-isqrt(2*level), isqrt(2*level)+1):
            for z in range(-4*level-1, 4*level+2, 2):
                last = F(z, 4)
                budget = (2*level-twice_ns**2-2*ramond_level(last))//2
                left, right = budget-ramond_level(incoming), budget-ramond_level(outgoing)
                if min(left, right) >= 0:
                    yield (left, right), (F(twice_ns, 2), incoming, outgoing, last)


def indices(left, right):
    return tuple(sorted(((a, b, c, d)
        for a in range(min(left, right)+1)
        for d in range(min(left, right)-a+1)
        for b in range(left-a-d+1)
        for c in range(right-a-d+1)), key=lambda row: (sum(row), row)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--scope', choices=('all', 'sample'), default='sample')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--series', type=Path, required=True)
    parser.add_argument('--algorithm', choices=('backward', 'forward'), default='backward')
    parser.add_argument('--reference-series', type=Path,
                        help='Compare these same ordinary coefficients with a saved timing stream')
    args = parser.parse_args()
    started = time.perf_counter()
    engine_type = PuncturedCCY
    if args.algorithm == 'forward':
        from forward_ccy import ForwardPuncturedCCY
        engine_type = ForwardPuncturedCCY
    reference = {}
    if args.reference_series:
        with args.reference_series.open('rb') as stream:
            while True:
                try:
                    row = pickle.load(stream)
                except EOFError:
                    break
                record = row['record']
                reference[tuple(record['labels']), record['copy']] = row['coefficients']
    pipeline.set_arithmetic(0)
    b = pipeline.numeric(F(7, 5))
    momenta = tuple(pipeline.numeric(F(x)) for x in ('11/23', '13/29', '17/31'))
    q = b+1/b
    central = (1+3*q*q/(1-b*b), 1+3*q*q/(1-1/(b*b)))
    external = (-(1+2*b*b)/(2*(1-b*b)), (b*b+2)/(2*(1-b*b)))
    groups = defaultdict(list)
    for budget, labels in cases(args.level):
        groups[budget].append(labels)
    records = []
    report = dict(status='running', level=args.level, scope=args.scope, algorithm=args.algorithm,
        numerical_backend='native binary64', groups=len(groups),
        total_series_required=2*sum(map(len, groups.values())),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('punctured_ccy.py', 'forward_ccy.py', 'time_diagonal_ccy.py')}, records=records)
    comparison = dict(coefficients=0, maximum_absolute_difference=0.0, maximum_scaled_difference=0.0)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.series.parent.mkdir(parents=True, exist_ok=True)
    save_seconds = 0.0
    with args.series.open('wb') as output:
        for budget, branch_labels in sorted(groups.items(), key=lambda item: (sum(item[0]), item[0])):
            selected = branch_labels if args.scope == 'all' else [branch_labels[len(branch_labels)//2]]
            requested = indices(*budget)
            for labels in selected:
                edge_weights = tuple(pipeline.weights(b, momentum, n) for momentum, n in
                    zip((momenta[0], momenta[1], momenta[1], momenta[2]), labels))
                for copy in (0, 1):
                    tick = time.perf_counter()
                    engine = engine_type(central[copy], tuple(pair[copy] for pair in edge_weights),
                                          external[copy], dps=0)
                    series = engine.reduced_series(indices=requested)
                    stats = engine.cache_info()
                    engine.clear_caches()
                    elapsed = time.perf_counter()-tick
                    record = dict(budget=budget, labels=list(map(str, labels)), copy=copy+1,
                        coefficients=len(series), seconds=elapsed,
                        multiplicity=1 if args.scope == 'all' else len(branch_labels), cache=stats)
                    records.append(record)
                    if args.reference_series:
                        previous = reference[tuple(record['labels']), copy+1]
                        if previous.keys() != series.keys():
                            raise AssertionError('reference and candidate index sets differ')
                        for key, value in series.items():
                            delta = abs(value-previous[key])
                            scaled = delta/max(1, abs(previous[key]))
                            comparison['coefficients'] += 1
                            if delta > comparison['maximum_absolute_difference']:
                                comparison.update(maximum_absolute_difference=float(delta),
                                    worst_absolute=dict(labels=record['labels'], copy=copy+1, levels=key))
                            if scaled > comparison['maximum_scaled_difference']:
                                comparison.update(maximum_scaled_difference=float(scaled),
                                    worst_scaled=dict(labels=record['labels'], copy=copy+1, levels=key))
                        report['ordinary_coefficient_comparison'] = comparison
                    tick = time.perf_counter()
                    pickle.dump(dict(record=record, coefficients=series), output, protocol=5)
                    save_seconds += time.perf_counter()-tick
                    del series, engine
            report['measured_recursion_seconds'] = sum(row['seconds'] for row in records)
            report['weighted_recursion_estimate_seconds'] = sum(row['seconds']*row['multiplicity'] for row in records)
            report['sampled_series'] = len(records)
            args.json.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(budget=budget, sampled_series=len(records),
                measured_seconds=report['measured_recursion_seconds'],
                weighted_estimate_seconds=report['weighted_recursion_estimate_seconds'])), flush=True)
    report.update(status=('timed and compared with saved ordinary CCY coefficients; no physical-block recovery'
                          if args.reference_series else 'timed; no physical-block recovery or accuracy comparison performed'),
                  internal_wall_seconds=time.perf_counter()-started, series_write_seconds=save_seconds,
                  series_file=str(args.series),
                  peak_resident_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024))
    args.json.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
