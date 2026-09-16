"""Compare saved coefficients only; never recompute a block."""
from collections import Counter
from decimal import Decimal, localcontext
from hashlib import sha256
import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ZERO = (Decimal(0), Decimal(0))


def rows(path, field):
    out = {}
    for line in path.open():
        x = json.loads(line)
        key = tuple(x['level2'])
        assert key not in out, ('duplicate', key)
        parity = sum((k % 2) << e for e, k in enumerate(key))
        row = x[field]
        assert len(row) <= 1
        if row:
            assert row[0][0] == parity
            value = Decimal(row[0][1]), Decimal(row[0][2])
        else:
            value = ZERO
        out[key] = x['case'], value
    return out


def norm(z):
    return (z[0] ** 2 + z[1] ** 2).sqrt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=10)
    args = parser.parse_args()
    level = args.level
    suffix = f'_L{level}'
    dv = rows(ROOT / f'coefficients{suffix}.jsonl', 'double_virasoro')
    ns = rows(ROOT / f'c_recursion{suffix}.jsonl', 'c_recursion')
    assert set(dv) == set(ns)
    assert len(dv) == comb(2 * level + 6, 6)
    assert all(len(k) == 6 and min(k) >= 0 and sum(k) <= 2 * level for k in dv)
    slots = ((0, 1, 2), (0, 3, 5), (1, 4, 3), (2, 5, 4))
    shell = {}
    cases = Counter()
    bad = []
    with localcontext() as ctx:
        ctx.prec = 60
        for k in sorted(dv, key=lambda x: (sum(x), x)):
            c, x = dv[k]
            cn, y = ns[k]
            expected_case = sum((sum(k[e] for e in slots[v]) % 2) << (v - 1)
                                for v in range(1, 4))
            assert c == cn == expected_case
            delta = norm((x[0] - y[0], x[1] - y[1]))
            scaled = delta / max(Decimal(1), norm(x), norm(y))
            record = shell.setdefault(sum(k), {'count': 0, 'absolute': Decimal(0),
                                                'scaled': Decimal(0), 'worst': None})
            record['count'] += 1
            record['absolute'] = max(record['absolute'], delta)
            if scaled > record['scaled']:
                record['scaled'] = scaled
                record['worst'] = {'case': c, 'level2': k, 'double_virasoro': tuple(map(str, x)),
                                   'c_recursion': tuple(map(str, y))}
            cases[c] += 1
            if scaled > Decimal('1e-18'):
                bad.append({'case': c, 'level2': k, 'absolute': str(delta), 'scaled': str(scaled),
                            'double_virasoro': tuple(map(str, x)), 'c_recursion': tuple(map(str, y))})
    result = {
        'passed': not bad, 'dps': 40, 'total_level': level,
        'multidegrees': len(dv), 'case_multidegrees': dict(sorted(cases.items())),
        'max_absolute': str(max(x['absolute'] for x in shell.values())),
        'max_scaled': str(max(x['scaled'] for x in shell.values())),
        'failed_coefficients': len(bad),
        'shells': [{'total_level': n / 2, 'count': v['count'],
                    'max_absolute': str(v['absolute']), 'max_scaled': str(v['scaled']),
                    'worst': v['worst']} for n, v in sorted(shell.items())],
        'timing_double_virasoro': json.loads((ROOT / f'timing_dv{suffix}.json').read_text()),
        'timing_ns_recursion': json.loads((ROOT / f'timing_ns{suffix}.json').read_text()),
    }
    (ROOT / f'results{suffix}.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / f'discrepancies{suffix}.json').write_text(json.dumps(bad, indent=2) + '\n')
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.cpp', '.hpp', '.inc', '.py', '.md')]
    paths += [ROOT / 'Makefile']
    paths += [ROOT / f'{x}{suffix}{ext}' for x, ext in
              [('coefficients', '.jsonl'), ('c_recursion', '.jsonl'), ('results', '.json'),
               ('timing_dv', '.json'), ('timing_ns', '.json'), ('schottky_seed', '.jsonl')]]
    paths += list((REPO / 'C++/include/ramond').glob('*.hpp'))
    paths += [ROOT.parent / 'total_level6_2026-09-16/ns_local.hpp', REPO / 'Human Notes/SCblock.tex']
    manifest = {str(p.relative_to(REPO)): sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    (ROOT / f'manifest{suffix}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'multidegrees', 'max_absolute', 'max_scaled', 'failed_coefficients')}))


if __name__ == '__main__':
    main()
