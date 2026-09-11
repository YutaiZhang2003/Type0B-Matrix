"""Focused 40-digit regression against the original CCY recurrence.

Tests only the rearranged recurrence on symmetric/asymmetric plumbing
budgets, both Virasoro copies. It does not invoke physical PBW or branching.
"""

from fractions import Fraction as F
from pathlib import Path
import argparse
import json

import pipeline
from forward_ccy import ForwardPuncturedCCY
from punctured_ccy import PuncturedCCY
from series_algebra import diagonal_virasoro_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    pipeline.set_arithmetic(40)
    b = pipeline.numeric(F(7, 5))
    momenta = tuple(pipeline.numeric(F(x)) for x in ('11/23','13/29','17/31'))
    q = b+1/b
    central = (1+3*q*q/(1-b*b), 1+3*q*q/(1-1/(b*b)))
    external = (-(1+2*b*b)/(2*(1-b*b)), (b*b+2)/(2*(1-b*b)))
    report = dict(status='passed', digits=40, cases=[])
    for budget, labels in (((5,5),(F(0),-F(1,4),F(1,4),-F(1,4))),
                           ((4,3),(F(0),F(1,4),F(3,4),-F(1,4)))):
        weights = [pipeline.weights(b,p,n) for p,n in
                   zip((momenta[0],momenta[1],momenta[1],momenta[2]),labels)]
        for copy in (0,1):
            engines = [cls(central[copy], [h[copy] for h in weights], external[copy], dps=40)
                       for cls in (PuncturedCCY,ForwardPuncturedCCY)]
            old, new = [e.reduced_series(indices=diagonal_virasoro_indices(*budget)) for e in engines]
            error = max(abs(new[k]-value)/max(1,abs(value)) for k,value in old.items())
            if error >= pipeline.ARITHMETIC.mpf('1e-28'):
                raise AssertionError((budget,copy,error))
            report['cases'].append(dict(budget=budget, labels=list(map(str,labels)), copy=copy+1,
                coefficients=len(old), maximum_scaled_difference=str(error)))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
