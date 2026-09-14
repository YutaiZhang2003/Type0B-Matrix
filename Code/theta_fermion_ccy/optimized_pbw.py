"""Direct physical NS-R-R sewing with cached metadata and native FLINT contractions.

No enlarged block, branching coefficient, or auxiliary fermion is computed.
The Ward and mode-action equations come from the existing physical PBW oracle.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ramond_zero_mode_recovery'))
import complex_arithmetic as ca
from check_level10_complex import load_runner


def encode(value):
    z = value.x if isinstance(value, ca.F) else value
    if not z.is_finite():
        raise ArithmeticError('Nonfinite physical PBW coefficient')
    return {'real': z.real.mid().str(45, radius=False),
            'imag': z.imag.mid().str(45, radius=False)}


class OptimizedPBW:
    def __init__(self, runner, b, momenta, p=0, f=0, eta=1, opposite=False):
        self.runner = runner
        self.oracle = runner.Check(b, momenta)
        self.modules = self.oracle.direct[0].pbw_modules
        self.p, self.f = p, f
        self.etas = eta, -eta if opposite else eta
        self.forms = tuple(self.oracle.human_form(p, f, e) for e in self.etas)
        # Mode actions do not depend on eta. Each form keeps its own scalar values.
        self.forms[1].modules = self.forms[0].modules
        self.phases = [((-1+ca.I)/ca.SQRT2)**k for k in range(3)]
        self.times = dict(gram_entries=0., gram_inverse=0., metadata=0.,
                          vertices=0., contractions=0.)
        self.counts = dict(gram_entries=0, gram_matrices=0, vertex_entries=0,
                           reused_vertex_tensors=0, contractions=0)

    @lru_cache(None)
    def edge(self, slot, level, parity):
        tick = time.perf_counter()
        module = self.modules[slot]
        states = tuple(s for s in self.runner.mb.metadata(module.sector, level)
                       if module.parity(s) == parity)
        words = tuple(module.word(s) for s in states)
        grounds = tuple(module.ground(s) for s in states)
        self.times['metadata'] += time.perf_counter()-tick
        tick = time.perf_counter()
        n = len(states)
        matrix = ca.acb_mat(n, n)
        for i, a in enumerate(states):
            for j in range(i+1):
                z = module.inner(a, states[j]).x
                matrix[i,j] = z
                if i != j:
                    matrix[j,i] = z
                self.counts['gram_entries'] += 1
        self.times['gram_entries'] += time.perf_counter()-tick
        tick = time.perf_counter()
        inverse = matrix.inv()
        inverse = ca.acb_mat(n,n,[z.mid() for z in inverse.entries()])
        self.times['gram_inverse'] += time.perf_counter()-tick
        self.counts['gram_matrices'] += 1
        return words, grounds, inverse

    @lru_cache(None)
    def permutations(self, shape):
        a,b,c = shape
        # From (a,b,c) to (b,a,c), then (b,a,c) to (c,a,b).
        first = tuple((i*b+j)*c+k for j in range(b) for i in range(a) for k in range(c))
        second = tuple((j*a+i)*c+k for k in range(c) for i in range(a) for j in range(b))
        final = tuple((i*b+j)*c+k for k in range(c) for i in range(a) for j in range(b))
        return first, second, final

    def contract(self, left, kernels, right, shape):
        a,b,c = shape
        first, second, final = self.permutations(shape)
        value = kernels[0] * ca.acb_mat(a,b*c,right)
        values = value.entries()
        value = kernels[1] * ca.acb_mat(b,a*c,[values[i].mid() for i in first])
        values = value.entries()
        value = kernels[2] * ca.acb_mat(c,a*b,[values[i].mid() for i in second])
        values = value.entries()
        total = (ca.acb_mat(1,a*b*c,[left[i] for i in final]) *
                 ca.acb_mat(a*b*c,1,[z.mid() for z in values]))[0,0]
        return ca.F(total)

    def coefficient(self, levels):
        result = [ca.F(0)]*8
        for parity in (0,1):
            parities = levels[0]%2, parity, (self.f+levels[0]+parity)%2
            edges = [self.edge(slot, level if slot == 0 else level//2, pr)
                     for slot,(level,pr) in enumerate(zip(levels,parities))]
            shape = tuple(len(e[0]) for e in edges)
            if not all(shape):
                continue
            tick = time.perf_counter()
            tensors = []
            for vertex,form in enumerate(self.forms):
                if vertex and self.etas[0] == self.etas[1]:
                    tensors.append(tensors[0])
                    self.counts['reused_vertex_tensors'] += 1
                    continue
                values = []
                for w1 in edges[0][0]:
                    for w2,g2 in zip(edges[1][0],edges[1][1]):
                        for w3,g3 in zip(edges[2][0],edges[2][1]):
                            values.append((self.phases[g2+g3]*form.value(w1,w2,g2,w3,g3)).x)
                tensors.append(values)
                self.counts['vertex_entries'] += len(values)
            self.times['vertices'] += time.perf_counter()-tick
            tick = time.perf_counter()
            value = self.contract(tensors[0],[e[2] for e in edges],tensors[1],shape)
            self.times['contractions'] += time.perf_counter()-tick
            self.counts['contractions'] += 1
            index = ((parities[0]+self.p)%2) | parities[1]<<1 | parities[2]<<2
            result[index] = self.runner.theta_quadratic_sign(index)*value
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',type=int,default=5)
    parser.add_argument('--dps',type=int,default=40)
    parser.add_argument('--mode',choices=('ordinary','inserted'),required=True,
                        help='Physical eta*eta-prime = +1 or -1; neither mode inserts an operator in PBW.')
    parser.add_argument('--implementation',choices=('optimized','baseline'),default='optimized')
    parser.add_argument('--json',type=Path,required=True)
    args = parser.parse_args()
    if args.level < 0 or args.dps < 30:
        parser.error('Need nonnegative level and at least 30 digits')
    bits = (args.dps*3322+999)//1000+3
    ca.ctx.prec = bits
    # Recreate precision-dependent constants after selecting working precision.
    ca.I, ca.SQRT2 = ca.F(ca.acb(0,1)), ca.F(ca.acb(2).sqrt())
    start = time.perf_counter()
    runner = load_runner()
    if args.implementation == 'optimized':
        for name in ('word_level','word_parity','state_parity'):
            runner.mb.namespace[name] = lru_cache(None)(runner.mb.namespace[name])
    b = Fraction(7,5)
    momenta = tuple(map(Fraction,('11/23','13/29','17/31')))
    opposite = args.mode == 'inserted'
    if args.implementation == 'optimized':
        engine = OptimizedPBW(runner,b,momenta,opposite=opposite)
        coefficient = engine.coefficient
    else:
        engine = runner.Check(b,momenta)
        coefficient = lambda levels: engine.physical_coefficient(levels,0,0,(1,-1 if opposite else 1))
    setup = time.perf_counter()-start
    levels = sorted(((a,2*b,2*c) for a in range(2*args.level+1)
                     for b in range(args.level+1) for c in range(args.level+1)),
                    key=lambda x:(sum(x),x))
    raw = []
    for j,n in enumerate(levels):
        raw.append((n,coefficient(n)))
        if (j+1)%50 == 0:
            print(f'PBW {args.mode}: {j+1}/{len(levels)} monomials, {time.perf_counter()-start:.3f} s',flush=True)
    internal = time.perf_counter()-start
    rows = [{'exponents':[n[0],n[1]//2,n[1]//2,n[2]],'values':[encode(z) for z in values]}
            for n,values in raw]
    times = dict(setup=setup,total=internal)
    counts = {'monomials':len(rows),'parity_slots':8*len(rows)}
    if args.implementation == 'optimized':
        times.update(engine.times)
        counts.update(engine.counts)
        counts['ward_cache'] = engine.forms[0].value.cache_info()._asdict()
    report = dict(status='computed',implementation='Python/FLINT '+args.implementation+' physical PBW',
                  mode=args.mode,truncation='per-edge',q_level_cutoffs=[args.level]*3,
                  dps=args.dps,precision_bits=bits,b=str(b),momenta=list(map(str,momenta)),
                  p=0,f=0,etas=[1,-1 if opposite else 1],timing_seconds=times,counts=counts,
                  coefficient_convention='q1^(a/2) q2^l q3^(d/2), exponents [a,l,l,d]',
                  coefficients=rows)
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'timing_seconds':times,'counts':counts}),flush=True)


if __name__ == '__main__':
    main()
