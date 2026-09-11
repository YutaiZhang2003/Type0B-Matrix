"""Structural counts for computing only the diagonal enlarged numerator.

This is a proposed index-set change, not a production implementation or a
numerical block test. Required ordinary Virasoro indices obey
a+d+max(r_level(incoming)+b, r_level(outgoing)+c) <= N-base_outer.
The set is downward closed, so the existing c-recursion can use it.
"""

from collections import Counter
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse
import json

from count_level_work import moment_product


def combine(a, b):
    x, i, o, p = a
    y, j, q, r = b
    return x*y, i*y+x*j, o*y+x*q, p*y+i*q+o*j+x*r


def prefix(rows):
    answer, value = [], [0]*4
    for row in rows:
        value = [x+y for x, y in zip(value, row)]
        answer.append(tuple(value))
    return answer


def count_tables(maximum):
    labels = lambda n: [r*s for r in range(2, n+1) for s in range(1, n//r+1)]
    outgoing = [len(labels(n)) for n in range(maximum+1)]
    incoming = [sum(n-d != 1 for d in labels(n)) for n in range(maximum+1)]
    polynomials = {}
    for mode in ('all', 'terminal', 'root', 'root_terminal'):
        rows = []
        for total in range(maximum+1):
            row = [0]*4
            for shift in range(total+1):
                level = total-shift
                if shift == 1 or (mode.startswith('root') and shift):
                    continue
                if 'terminal' in mode and level >= 2:
                    continue
                value = (1, incoming[shift], outgoing[level], incoming[shift]*outgoing[level])
                row = [x+y for x, y in zip(row, value)]
            rows.append(row)
        # The two outer edges enter both budgets. Each middle edge enters
        # only its own budget, allowing a product of the two prefix sums.
        polynomials[mode] = moment_product(rows, rows, maximum), prefix(rows)

    def moments(mode, left, right):
        outer, side = polynomials[mode]
        answer = [0]*4
        for i in range(min(left, right)+1):
            row = combine(combine(outer[i], side[left-i]), side[right-i])
            answer = [x+y for x, y in zip(answer, row)]
        return answer

    result = {}
    for left in range(maximum+1):
        for right in range(maximum+1):
            a, t, r, rt = [moments(mode, left, right)
                          for mode in ('all', 'terminal', 'root', 'root_terminal')]
            result[left, right] = dict(
                rational_blocks=a[0]-t[0], terminal_globals=t[0],
                requested_coefficients=r[0], numeric_evaluations=a[1]-t[1]+r[0]-rt[0],
                stored_pole_terms=a[2], pole_additions=a[3]+r[2])
    return result


def histogram(level):
    result = Counter()
    ramond_level = lambda n: int(2*n*n-F(1, 8))
    for k in range(-4*level-1, 4*level+2, 2):
        left, right = F(k, 4), F(k+2, 4)
        for twice_ns in range(-isqrt(2*level), isqrt(2*level)+1):
            for z in range(-4*level-1, 4*level+2, 2):
                last = F(z, 4)
                budget = (2*level-twice_ns**2-2*ramond_level(last))//2
                first_budget = budget-ramond_level(left)
                second_budget = budget-ramond_level(right)
                if min(first_budget, second_budget) >= 0:
                    result[first_budget, second_budget] += 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', nargs='+', type=int, default=[10, 15])
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    table = count_tables(max(args.levels))
    report = dict(status='structural count for proposed diagonal-target closure; no numerical validation', levels={})
    for level in args.levels:
        budgets, total = histogram(level), Counter()
        for budget, copies in budgets.items():
            for key, value in table[budget].items():
                total[key] += copies*value
        report['levels'][str(level)] = dict(
            series=sum(budgets.values()), counts=dict(total), largest_series=table[level, level],
            budget_histogram=[dict(left=l, right=r, series=v) for (l, r), v in sorted(budgets.items())])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({level: {k: v for k, v in data.items() if k != 'budget_histogram'}
                      for level, data in report['levels'].items()}, indent=2))


if __name__ == '__main__':
    main()
