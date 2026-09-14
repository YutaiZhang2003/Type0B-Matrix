"""Sequential fresh factor timings and directed same-precision comparisons."""
from collections import defaultdict
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
CPP = HERE.parents[1]
PROFILES = {'largest': (0, -1, 1, 1), 'medium': (4, 3, 5, -5), 'small': (6, 5, 7, 7)}
VARIANTS = ('baseline', 'grouped', 'dense', 'combined')


def magnitude(z):
    a, b = Decimal(z['real']), Decimal(z['imag'])
    assert a.is_finite() and b.is_finite()
    return (a * a + b * b).sqrt()


def difference(reference, result):
    assert reference['seed_terms'] == result['seed_terms']
    assert reference['transitions'] == result['transitions']
    assert len(reference['coefficients']) == len(result['coefficients'])
    worst = Decimal(0)
    for a, b in zip(reference['coefficients'], result['coefficients']):
        assert a['exponents'] == b['exponents']
        diff = {k: str(Decimal(a['value'][k]) - Decimal(b['value'][k])) for k in ('real', 'imag')}
        scaled = magnitude(diff) / max(Decimal(1), magnitude(a['value']), magnitude(b['value']))
        worst = max(worst, scaled)
    assert worst < Decimal('1e-20'), str(worst)
    return str(worst)


def main():
    directory = HERE / 'results'
    directory.mkdir(exist_ok=True)
    report = {'status': 'running', 'dps': 40, 'precision_bits': 136,
              'protocol': 'two rounds, reversed order in second round; sequential fresh processes; no PBW',
              'production_modified': False, 'runs': []}
    start = time.perf_counter()
    for repeat in range(2):
        profiles = list(PROFILES.items())
        variants = list(VARIANTS)
        if repeat:
            profiles.reverse()
            variants.reverse()
        for profile, labels in profiles:
            for copy in range(2):
                for variant in variants:
                    output = directory / f'{profile}_copy{copy}_round{repeat}_{variant}.json'
                    command = [str(HERE / variant), '10', str(copy), *map(str, labels), str(output)]
                    tick = time.perf_counter()
                    run = subprocess.run(command, capture_output=True, text=True)
                    if run.returncode:
                        print(run.stdout, run.stderr, flush=True)
                        raise SystemExit(run.returncode)
                    data = json.loads(output.read_text())
                    record = {'profile': profile, 'copy': copy, 'round': repeat, 'variant': variant,
                              'wall_seconds': time.perf_counter() - tick, 'command': command,
                              **{k: v for k, v in data.items() if k != 'coefficients'}}
                    report['runs'].append(record)
                    (HERE / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(f'{profile} copy {copy} round {repeat} {variant}: {data["seconds"]:.3f} s '
                          f'(propagation {data["propagation"]:.3f}, assembly {data["assembly"]:.3f})', flush=True)
    report['timing_wall_seconds'] = time.perf_counter() - start
    report['comparisons'] = []
    with localcontext() as context:
        context.prec = 60
        for profile in PROFILES:
            for copy in range(2):
                for repeat in range(2):
                    prefix = f'{profile}_copy{copy}_round{repeat}_'
                    reference = json.loads((directory / (prefix + 'baseline.json')).read_text())
                    for variant in VARIANTS[1:]:
                        data = json.loads((directory / (prefix + variant + '.json')).read_text())
                        report['comparisons'].append({'profile': profile, 'copy': copy, 'round': repeat,
                                                      'variant': variant,
                                                      'maximum_scaled_difference': difference(reference, data)})
    median = {}
    for profile in PROFILES:
        for copy in range(2):
            for variant in VARIANTS:
                rows = [r for r in report['runs'] if (r['profile'], r['copy'], r['variant']) == (profile, copy, variant)]
                median[(profile, copy, variant)] = statistics.median(r['seconds'] for r in rows)
    profile_seed = {profile: next(r['seed_terms'] for r in report['runs'] if r['profile'] == profile)
                    for profile in PROFILES}
    t = lambda cap: cap + 1 + cap * (cap - 1) // 2
    labels = [n for n in range(-41, 42, 2) if (n * n - 1) // 8 <= 10]
    cap = lambda n: 10 - (n * n - 1) // 8
    weights = defaultdict(int)
    branches = defaultdict(int)
    for ns in range(-4, 5):
        first = 10 - (ns * ns + 1) // 2
        for left in labels:
            right = left + 2
            if right not in labels:
                continue
            for third in labels:
                seed = t(first) * t(cap(left)) * t(cap(right)) * t(cap(third))
                profile = min(PROFILES, key=lambda p: abs(math.log(seed / profile_seed[p])))
                for copy in range(2):
                    weights[(profile, copy)] += seed
                    branches[(profile, copy)] += 1
    old = json.loads((CPP / 'results/per_edge_level10_2026-09-11/optimized_full_pipeline_40dps/timings.json').read_text())
    old = next(r for r in old['runs'] if r['mode'] == 'inserted')
    assert sum(weights.values()) == old['counts']['ccy_seed_terms']
    extrapolated = {variant: sum(weight * median[(profile, copy, variant)] / profile_seed[profile]
                                 for (profile, copy), weight in weights.items()) for variant in VARIANTS}
    report['projection'] = {}
    for variant in VARIANTS[1:]:
        ratio = extrapolated[variant] / extrapolated['baseline']
        projected = old['wall_seconds'] - old['timing_seconds']['ccy'] * (1 - ratio)
        report['projection'][variant] = {'ccy_ratio': ratio, 'ccy_speedup': 1 / ratio,
                                         'projected_full_wall_seconds': projected,
                                         'projected_savings_seconds': old['wall_seconds'] - projected,
                                         'sample_ccy_speedups': [median[(p, c, 'baseline')] / median[(p, c, variant)]
                                                                  for p in PROFILES for c in range(2)]}
    report['projection_method'] = ('Assign all 1620 factors to the nearest sampled seed-work size; '
                                  'scale sample time by exact seed counts in each group and copy; '
                                  'normalize baseline projection to the measured 1460.243-second CCY stage; '
                                  'leave all other full-pipeline stages unchanged. Not a full-pipeline measurement.')
    report['profile_weights'] = [{'profile': p, 'copy': c, 'seed_terms': n,
                                 'factor_count': branches[(p, c)]} for (p, c), n in weights.items()]
    report['status'] = 'completed'
    (HERE / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['projection'], indent=2), flush=True)


if __name__ == '__main__':
    main()
