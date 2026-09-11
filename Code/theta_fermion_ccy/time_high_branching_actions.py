"""Small timing sample of new level-15 primary action coefficients.

Use the production machine-precision action solvers and their normal error
guards. This is not a physical-block or PBW comparison. Store each result.
"""

from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import pickle
import time

from middle_branching import RamondActions
from action_optimization import CachedActionModule, solve_ns_l1
import compute_target as br


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path, required=True)
    args = parser.parse_args()
    br.set_multiprecision(0)
    b, momentum = float(F(7, 5)), complex(F(13, 29))
    provider = RamondActions(b, momentum)
    tasks = [('R', 'minus', F(9, 4)), ('R', 'minus', F(11, 4)),
             ('R', 'plus', F(11, 4)), ('NS', 'plus', F(5, 2))]
    report = dict(status='running', numerical_backend='native binary64', records=[])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.coefficients.parent.mkdir(parents=True, exist_ok=True)
    with args.coefficients.open('wb') as output:
        for sector, action, label in tasks:
            record = dict(sector=sector, action=action, label=str(label), parity=0)
            print(json.dumps(dict(**record, status='starting')), flush=True)
            tick = time.perf_counter()
            try:
                if sector == 'R':
                    terms = getattr(provider, action)(label, 0)
                    diagnostic = provider.diagnostics[action, label, 0]
                else:
                    module = CachedActionModule('NS', b, complex(F(11, 23)))
                    terms, diagnostic = solve_ns_l1(module, label)
                record.update(seconds=time.perf_counter()-tick, status='computed', terms=len(terms),
                    diagnostic={k:v for k,v in diagnostic.items() if k != 'coefficients'})
                pickle.dump(dict(record=record, coefficients=terms), output, protocol=5)
            except Exception as error:
                record.update(seconds=time.perf_counter()-tick, status='failed',
                              error_type=type(error).__name__, error=str(error))
            report['records'].append(record)
            args.json.write_text(json.dumps(report, indent=2, default=str)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k != 'diagnostic'}), flush=True)
    report['status'] = 'samples completed' if all(r['status']=='computed' for r in report['records']) else 'sample guard failure; see records'
    args.json.write_text(json.dumps(report, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
