"""Time one fresh action solve and optionally compare its saved coefficients."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mpmath as mp
from action_optimization import CachedActionModule, solve_ramond_lminus, solve_ramond_lplus, solve_ns_l1
import compute_target as br


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', type=F, default=F(9,4))
    parser.add_argument('--sector', choices=('R','NS'), default='R')
    parser.add_argument('--action', choices=('minus','plus'), default='minus')
    parser.add_argument('--parity', type=int, default=0)
    parser.add_argument('--dps', type=int, default=0)
    parser.add_argument('--order', choices=('by_copy','largest_mode'), default='largest_mode')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--with-plus', action='store_true')
    parser.add_argument('--tuple-states', action='store_true')
    parser.add_argument('--complex-descendants', action='store_true')
    parser.add_argument('--scalar-native', action='store_true')
    args = parser.parse_args()
    br.set_multiprecision(args.dps)
    start = perf_counter()
    module = CachedActionModule(args.sector, F(7,5), F(13,29), descendant_order=args.order,
                                indexed_descendants=not args.tuple_states,
                                real_descendants=not args.complex_descendants,
                                sparse_native=not args.scalar_native)
    plus_result = [] if args.with_plus else None
    if args.with_plus and (args.sector != 'R' or args.action != 'minus'):
        parser.error('--with-plus requires a Ramond minus action')
    if args.sector == 'NS':
        terms, fit = solve_ns_l1(module, args.label)
    else:
        solver = solve_ramond_lminus if args.action == 'minus' else solve_ramond_lplus
        if args.with_plus:
            terms, fit = solver(module, args.label, args.parity, plus_result=plus_result)
        else:
            terms, fit = solver(module, args.label, args.parity)
    seconds = perf_counter()-start
    def encoded(value):
        return {'real': str(value.real), 'imag': str(value.imag)}
    records = [{'label': str(term.label), 'first': term.first, 'second': term.second,
                'coefficient': encoded(term.coefficient)} for term in terms]
    report = dict(status='computed', sector=args.sector, action=args.action, label=str(args.label),
                  parity=args.parity, dps=args.dps, order=args.order, seconds=seconds,
                  indexed_descendants=module.indexed_descendants,
                  real_descendants=module.real_descendants, sparse_native=module.sparse_native,
                  diagnostic={k:v for k,v in fit.items() if k != 'coefficients'}, coefficients=records,
                  peak_resident_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    if plus_result:
        plus_terms, plus_fit = plus_result[0]
        report['plus'] = dict(diagnostic={k:v for k,v in plus_fit.items() if k != 'coefficients'},
            coefficients=[{'label': str(t.label), 'first': t.first, 'second': t.second,
                           'coefficient': encoded(t.coefficient)} for t in plus_terms])
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for key in ('sector','action','label','parity','dps'):
            if reference[key] != report[key]:
                raise ValueError('reference parameters differ: '+key)
        ctx = mp.mp.clone(); ctx.dps = max(60,args.dps)
        key = lambda r: (r['label'],tuple(r['first']),tuple(r['second']))
        number = lambda r: ctx.mpc(r['coefficient']['real'],r['coefficient']['imag'])
        def compare(old_records, new_records):
            old, new = ({key(r):number(r) for r in rs} for rs in (old_records,new_records))
            if old.keys() != new.keys():
                raise AssertionError('action span labels differ')
            return dict(coefficients=len(old),
                maximum_absolute_difference=str(max((abs(new[k]-v) for k,v in old.items()), default=0)),
                maximum_scaled_difference=str(max((abs(new[k]-v)/max(1,abs(v)) for k,v in old.items()), default=0)))
        report['comparison'] = dict(reference=str(args.reference),
                                   **compare(reference['coefficients'], records))
        if plus_result:
            report['comparison']['plus'] = compare(reference['plus']['coefficients'],
                                                    report['plus']['coefficients'])
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2,default=str)+'\n')
    printed = {k:v for k,v in report.items() if k not in ('coefficients','plus')}
    if plus_result:
        printed['plus'] = report['plus']['diagnostic']
    print(json.dumps(printed,indent=2,default=str))


if __name__ == '__main__':
    main()
