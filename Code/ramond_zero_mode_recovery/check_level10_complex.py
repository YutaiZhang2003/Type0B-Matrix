"""High-precision complex specialization of the same optimized sewing code."""

from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys
import time
import types

import complex_arithmetic as ca

HERE=Path(__file__).resolve().parent


def load_backend():
    name='zero_mode_complex_backend'
    module=types.ModuleType(name)
    module.__file__=str(HERE/'modular_backend.py')
    sys.modules[name]=module
    source=(HERE/'modular_backend.py').read_text()
    source=source.replace('from modular_arithmetic import','from complex_arithmetic import')
    first=source.index('def sqrt(value):')
    last=source.index('\nnamespace = dict(',first)
    source=source[:first]+'''def sqrt(value):
    return F(F(value).x.sqrt())

EIGHTH_ROOT_TWO=F(2)**Fraction(1,8)

'''+source[last:]
    source=source.replace('F(int(coefficient))','F(coefficient)').replace('F(int(c))','F(c)')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    module.Weights.triple=module.lru_cache(None)(module.Weights.triple)
    return module


def load_runner():
    backend=load_backend()
    name='zero_mode_complex_runner'
    module=types.ModuleType(name)
    sys.modules[name]=module
    source=(HERE/'check_level10_modular.py').read_text()
    source=source.replace('import modular_backend as mb','import zero_mode_complex_backend as mb')
    source=source.replace('from modular_arithmetic import','from complex_arithmetic import')
    source=source.replace('import numpy as np','from complex_arithmetic import NumpyProxy\nnp=NumpyProxy()')
    for expression in ('norm','1/norm','c','F(Fraction(1,2))','mm(left.reshape(-1),value.reshape(-1))'):
        source=source.replace('int('+expression+')',expression)
    exec(compile(source,str(HERE/'check_level10_modular.py'),'exec'),module.__dict__)
    return module


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',type=int,default=10)
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--physical-reference-level',type=int,default=-1,
                        help='Optional direct PBW comparison cutoff (0 to 3); disabled by default.')
    parser.add_argument('--kernel',choices=('degenerate','oscillator'),default='degenerate')
    args=parser.parse_args()
    if not -1<=args.physical_reference_level<=3:
        parser.error('The direct physical reference is restricted to levels 0 through 3.')
    m=load_runner();cutoff=2*args.level
    start=time.perf_counter()
    from degenerate_kernel import make_check
    constructor=make_check(m) if args.kernel=='degenerate' else m.Check
    check=constructor(Fraction(7,5),tuple(map(Fraction,('11/23','13/29','17/31'))))
    auxiliary_start=time.perf_counter();a=m.auxiliary(cutoff,1)
    auxiliary_seconds=time.perf_counter()-auxiliary_start
    h={};numerator_start=time.perf_counter()
    for n in m.level_triples(cutoff):
        h[n]=check.enlarged_coefficient(n,0,0,(1,1),1)
        check.vertex.cache_clear()
        m.log(complex_completed_levels=n,seconds=time.perf_counter()-start)
    numerator_seconds=time.perf_counter()-numerator_start
    recovery_start=time.perf_counter();physical=m.recover(h,a,cutoff)
    recovery_seconds=time.perf_counter()-recovery_start
    timing=dict(auxiliary_seconds=auxiliary_seconds,numerator_seconds=numerator_seconds,
                recovery_seconds=recovery_seconds,total_seconds=time.perf_counter()-start)
    m.log(complex_block_timing=timing)
    args.json.with_suffix('.cold.json').write_text(json.dumps(dict(
        status='block computed; validation recorded separately',total_level=args.level,precision_bits=ca.ctx.prec,
        cold_block_timing=timing,
        coefficients=[dict(twice_levels=n,values=[str(x) for x in value]) for n,value in physical.items()]
    ),indent=2)+'\n')
    reference_start=time.perf_counter()
    cases=list(product((0,1),(0,1),(-1,1),(-1,1),(1,2))) if args.all_cases else [(0,0,1,1,1)]
    auxiliary={1:a}
    if args.all_cases:auxiliary[2]=m.auxiliary(cutoff,2)
    numerators={case:{} for case in cases};numerators[0,0,1,1,1]=h
    references={case[:4]:{} for case in cases}
    for n in m.level_triples(cutoff):
        for p,f,eta,eta_prime,cut in cases:
            key=p,f,eta,eta_prime
            if sum(n)<=2*args.physical_reference_level and n not in references[key]:
                references[key][n]=check.physical_coefficient(n,p,f,(eta,eta_prime))
            if n not in numerators[key+(cut,)]:numerators[key+(cut,)][n]=check.enlarged_coefficient(n,p,f,(eta,eta_prime),cut)
        check.vertex.cache_clear()
        m.log(complex_validation_completed_levels=n,seconds=time.perf_counter()-reference_start)
    records=[];recovered_cases={}
    for case in cases:
        recovered=m.recover(numerators[case],auxiliary[case[4]],cutoff)
        recovered_cases[case]=recovered
        reference=references[case[:4]]
        absolute=max((abs(x-y) for n in reference for x,y in zip(recovered[n],reference[n])),default=None)
        scaled=max((abs(x-y)/max(1,abs(y)) for n in reference for x,y in zip(recovered[n],reference[n])),default=None)
        records.append(dict(p=case[0],f=case[1],etas=list(case[2:4]),cut=case[4],
                            physical_reference_monomials=len(reference),maximum_absolute_error=absolute,maximum_scaled_error=scaled))
    cut_records=[]
    if args.all_cases:
        for key in references:
            left= recovered_cases[key+(1,)];right=recovered_cases[key+(2,)]
            absolute=max(abs(x-y) for n in left for x,y in zip(left[n],right[n]))
            scaled=max(abs(x-y)/max(1,abs(x),abs(y)) for n in left for x,y in zip(left[n],right[n]))
            cut_records.append(dict(p=key[0],f=key[1],etas=list(key[2:]),maximum_absolute_error=absolute,maximum_scaled_error=scaled))
    errors=[r for r in records+cut_records if r['maximum_scaled_error'] is not None]
    max_error=max((r['maximum_absolute_error'] for r in errors),default=None)
    max_scaled=max((r['maximum_scaled_error'] for r in errors),default=None)
    report=dict(total_level=args.level,precision_bits=ca.ctx.prec,kernel=args.kernel,arithmetic='FLINT complex midpoints; small-term cutoff 1e-80',
                cold_block_timing=timing,validation_after_cold_block_seconds=time.perf_counter()-reference_start,
                total_suite_seconds=time.perf_counter()-start,cases=records,cut_comparisons=cut_records,
                physical_reference_level=args.physical_reference_level,
                maximum_absolute_error=max_error,maximum_scaled_error=max_scaled,edges=check.edge_checks,
                coefficients=[dict(twice_levels=n,values=[dict(real=complex(x).real,imag=complex(x).imag) for x in value]) for n,value in physical.items()])
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    assert max_scaled is None or max_scaled<1e-25,report
    m.log(complex_result='passed' if errors else 'computed without independent comparison',maximum_scaled_error=max_scaled)


if __name__=='__main__':main()
