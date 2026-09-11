"""Fresh branching-only timing; compare saved branching data, without blocks."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mpmath as mp
import pipeline
from outer_branching import OuterBranching, middle_pairs
from middle_branching import MiddleBranching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--dps', type=int, default=0)
    parser.add_argument('--f', type=int, choices=(0,1), default=0)
    parser.add_argument('--p', type=int, choices=(0,1), default=0)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    pipeline.set_arithmetic(args.dps)
    b, momenta = F(7,5), tuple(map(F, ('11/23','13/29','17/31')))
    started = perf_counter()
    outer = OuterBranching(b, momenta, args.level, f=args.f, p=args.p,
                           dps=args.dps, machine=not args.dps, precompute_middle=True)
    outer.prepare((1,-1))
    outer_seconds = perf_counter()-started
    pipeline.set_arithmetic(args.dps)
    middle = MiddleBranching(pipeline.numeric(b),pipeline.numeric(momenta[1]),
                             dps=args.dps, actions=outer.middle_actions(args.dps))
    tick = perf_counter()
    values = {(outgoing,incoming,alpha): middle.raw(outgoing,incoming,alpha)
              for incoming,outgoing in middle_pairs(args.level) for alpha in (0,1)}
    middle_seconds = perf_counter()-tick
    report = dict(status='computed', scope='branching only; no Virasoro or physical block',
        parameters=dict(total_q_level=args.level,b=str(b),momenta=list(map(str,momenta)),
                        p=args.p,f=args.f,etas=[1,-1],ccy_dps=args.dps,ward_dps=args.dps),
        outer_seconds=outer_seconds,middle_seconds=middle_seconds,
        construction_seconds=perf_counter()-started,
        outer_tables=[dict(alpha=alpha,gamma=gamma,eta=eta,
            coefficients=[dict(labels=list(map(str,labels)),value=pipeline.encoded(value))
                          for labels,value in sorted(table.items())])
            for (alpha,gamma,eta),table in sorted(outer.tables.items())],
        middle_coefficients=[dict(outgoing=str(o),incoming=str(i),parity=a,value=pipeline.encoded(v))
                             for (o,i,a),v in sorted(values.items())],
        outer_diagnostics=outer.diagnostics,
        source_hashes=pipeline.source_hashes(),
        peak_resident_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    if args.reference:
        old = json.loads(args.reference.read_text())
        for key,value in report['parameters'].items():
            if old['parameters'][key] != value:
                raise ValueError('reference parameters differ: '+key)
        ctx = mp.mp.clone(); ctx.dps = max(60,args.dps)
        number = lambda value: ctx.mpc(value['real'],value['imag'])
        def unpack(data,part):
            if part == 'outer':
                return {(t['alpha'],t['gamma'],t['eta'],tuple(v['labels'])):number(v['value'])
                        for t in data['outer_tables'] for v in t['coefficients']}
            return {(v['outgoing'],v['incoming'],v['parity']):number(v['value'])
                    for v in data['middle_coefficients']}
        report['comparison'] = dict(reference=str(args.reference), arithmetic_digits=ctx.dps)
        for part in ('outer','middle'):
            a,b = unpack(old,part),unpack(report,part)
            if a.keys() != b.keys():
                raise AssertionError(part+' coefficient support changed')
            worst = max(a,key=lambda k:abs(a[k]-b[k])/max(1,abs(a[k])))
            report['comparison'][part] = dict(coefficients=len(a),
                maximum_absolute_difference=str(max(abs(a[k]-b[k]) for k in a)),
                maximum_scaled_difference=str(abs(a[worst]-b[worst])/max(1,abs(a[worst]))),
                worst_scaled_key=worst)
    report['total_seconds_through_encoding_and_comparison'] = perf_counter()-started
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ('outer_tables','middle_coefficients','outer_diagnostics','source_hashes')},indent=2))


if __name__ == '__main__':
    main()
