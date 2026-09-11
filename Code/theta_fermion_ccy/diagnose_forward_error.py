"""Targeted error attribution for two saved problematic CCY coefficients.

Profiles the existing forward routine to retain its final path weights;
does not change the production recurrence or run a physical-block pipeline.
The signed error decomposition uses 40-digit arithmetic throughout.
"""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

import mpmath as mp

import pipeline
from forward_ccy import ForwardPuncturedCCY, seed_splits
from punctured_ccy import PuncturedCCY, _vector_id


def make_engine(labels, digits):
    pipeline.set_arithmetic(digits)
    b = pipeline.numeric(F(7, 5))
    p = tuple(pipeline.numeric(F(x)) for x in ('11/23', '13/29', '17/31'))
    q = b+1/b
    central = 1+3*q*q/(1-1/(b*b))
    external = (b*b+2)/(2*(1-b*b))
    weights = [pipeline.weights(b, momentum, n)[1]
               for momentum, n in zip((p[0], p[1], p[1], p[2]), labels)]
    return ForwardPuncturedCCY(central, weights, external, dps=digits)


def capture(engine, target):
    captured = {}
    closest_pole = {}
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is ForwardPuncturedCCY.reduced_series.__code__:
            captured.update(frame.f_locals['totals'])
        if event == 'return' and frame.f_code is PuncturedCCY._factor_uncached.__code__:
            c, pole = frame.f_locals['c'], frame.f_locals['pole']
            separation = abs(c-pole)/max(1, abs(c), abs(pole))
            if not closest_pole or separation < closest_pole['relative_separation']:
                closest_pole.update(relative_separation=separation, parent_c=c, pole=pole,
                                    absolute_separation=abs(c-pole))
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        value = engine.reduced_series(indices=product(*(range(n+1) for n in target)))[target]
    finally:
        sys.setprofile(previous)
    terms = {}
    for shift, remainder in seed_splits(_vector_id(target)):
        if shift in captured:
            terms[shift] = (captured[shift], engine._global_indexed(shift, remainder))
    return value, terms, closest_pole


def main():
    started = perf_counter()
    mp.mp.dps = 40
    report = {'scope': 'Two individual copy-2 CCY coefficients; no branching, fermion, or PBW run',
              'digits': 40, 'records': []}
    labels = tuple(map(F, ('0', '-3/4', '-1/4', '3/4')))
    for target in ((0, 8, 8, 0), (5, 8, 8, 0)):
        native = make_engine(labels, 0)
        high = make_engine(labels, 40)
        rounded = ForwardPuncturedCCY(native.c, native.weights, native.external_weight, dps=40)
        nv, nt, np = capture(native, target)
        hv, ht, hp = capture(high, target)
        rv, rt, rp = capture(rounded, target)
        to_mp = mp.mp.mpc
        # Native K,G are evaluated at the same inputs as the rounded run.
        # Separate the path-weight and seed errors by a telescoping identity.
        path_error = mp.fsum((to_mp(nt[k][0])-rt[k][0])*rt[k][1] for k in rt)
        seed_error = mp.fsum(to_mp(nt[k][0])*(to_mp(nt[k][1])-rt[k][1]) for k in rt)
        exact_sum_native_terms = mp.fsum(to_mp(k)*to_mp(g) for k, g in nt.values())
        final_arithmetic_error = to_mp(nv)-exact_sum_native_terms
        input_error = rv-hv
        total_error = to_mp(nv)-hv
        term_abs_sum = mp.fsum(abs(k*g) for k, g in ht.values())
        parts = dict(rounded_input_error=input_error,
                     residue_path_weight_error=path_error,
                     global_seed_evaluation_error=seed_error,
                     final_product_and_sum_roundoff=final_arithmetic_error)
        record = dict(labels=list(map(str, labels)), copy=2, levels=target,
                      native_value=str(nv), reference40=str(hv),
                      signed_error=str(total_error),
                      relative_error=str(abs(total_error)/abs(hv)),
                      seed_terms=len(ht), sum_absolute_seed_terms=str(term_abs_sum),
                      largest_seed_term=str(max(abs(k*g) for k, g in ht.values())),
                      cancellation_ratio=str(term_abs_sum/abs(hv)),
                      decimal_digits_lost_to_final_cancellation=str(mp.log10(term_abs_sum/abs(hv))),
                      signed_error_components={k: str(v) for k, v in parts.items()},
                      closest_pole_native={k: str(v) for k, v in np.items()},
                      closest_pole_reference40={k: str(v) for k, v in hp.items()},
                      decomposition_residual=str(total_error-mp.fsum(parts.values())))
        report['records'].append(record)
        for engine in (native, high, rounded):
            engine.clear_caches()
    report['diagnostic_wall_seconds'] = perf_counter()-started
    path = Path(__file__).resolve().parent/'results'/'diagnosis_forward_ccy_error.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
