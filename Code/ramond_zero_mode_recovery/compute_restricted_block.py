"""Time one physical NSRR block using the restricted convolution inverse.

The enlarged numerator is the diagonal double-Virasoro branch sum. Each
ordinary Virasoro block is sewn using its own Ward forms and Gram matrices.
All arithmetic uses the same 384-bit backend as the zero-mode computation.
Only the sector and constant-term requirements of the inverse are checked;
there are no optional or automatic cross-checks against another calculation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import time


def labels_and_levels(level):
    ns = []
    for k in range(-2 * level - 2, 2 * level + 3):
        n = Fraction(k, 2)
        if 2 * n * n <= level:
            ns.append(n)
    ramond = []
    for k in range(-4 * level - 3, 4 * level + 4, 2):
        n = Fraction(k, 4)
        if 2 * n * n - Fraction(1, 8) <= level:
            ramond.append(n)
    result = []
    for labels in product(ns, ramond, ramond):
        base = tuple(4 * n * n - (Fraction(1, 4) if e else 0)
                     for e, n in enumerate(labels))
        if sum(base) <= 2 * level:
            result.append((labels, tuple(map(int, base))))
    return sorted(result, key=lambda item: (sum(item[1]), item[0]))


def integer_levels(level):
    for total in range(level + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                yield a, b, total - a - b


def ordinary_virasoro_series(m, weights, central_charge, cutoff):
    form = m.mb.VirasoroForm(weights, central_charge)
    grams = [[m.vir_gram(h, central_charge, n) for n in range(cutoff + 1)]
             for h in weights]
    answer = {}
    for levels in integer_levels(cutoff):
        edges = [grams[e][n] for e, n in enumerate(levels)]
        tensor = m.array([form.value(*states)
                          for states in product(*(edge[0] for edge in edges))])
        tensor = tensor.reshape(tuple(len(edge[0]) for edge in edges))
        answer[levels] = m.Check.contract(tensor, [edge[2] for edge in edges], tensor)
    return answer


def multiply_series(left, right, cutoff, zero):
    answer = {}
    for a, av in left.items():
        if not av:
            continue
        for b, bv in right.items():
            if not bv:
                continue
            n = tuple(x + y for x, y in zip(a, b))
            if sum(n) <= cutoff:
                answer[n] = answer.get(n, zero) + av * bv
    return answer


def project_sector(m, series, tolerance):
    """Enforce the I_+ domain required by the restricted inverse."""
    projected = {}
    largest_absolute = largest_scaled = 0.0
    for n, vector in series.items():
        values = tuple((v + (-1) ** (((i >> 1) & 1) + ((i >> 2) & 1))
                        * vector[i ^ 6]) / 2 for i, v in enumerate(vector))
        # Measure before small differences are clipped by the scalar wrapper.
        absolute = max(float(abs(a.x - b.x).upper()) for a, b in zip(vector, values))
        scaled = absolute / max(1.0, *(abs(v) for v in vector))
        largest_absolute = max(largest_absolute, absolute)
        largest_scaled = max(largest_scaled, scaled)
        projected[n] = values
    if largest_scaled > tolerance:
        raise ValueError(f'Input outside I_+: {largest_scaled:.6e} > {tolerance:.6e}')
    return projected, dict(maximum_absolute_distance=largest_absolute,
                           maximum_scaled_distance=largest_scaled)


def restricted_recovery(m, numerator, auxiliary, cutoff):
    result = {}
    positive = [(n, v) for n, v in auxiliary.items() if any(n) and any(v)]
    for n in m.level_triples(cutoff):
        value = numerator[n]
        for s, coefficient in positive:
            rest = tuple(a - b for a, b in zip(n, s))
            if min(rest) >= 0:
                term = m.star(coefficient, result[rest])
                value = tuple(a - b for a, b in zip(value, term))
        result[n] = tuple(v / 2 for v in value)
    return result


def encode_series(series):
    return [dict(twice_levels=n, values=[str(v) for v in values])
            for n, values in series.items()]


def main():
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--sector-tolerance', type=float, default=1e-30)
    args = parser.parse_args()
    if args.level < 0 or not 0 < args.sector_tolerance < 1:
        parser.error('Require nonnegative level and sector tolerance between 0 and 1.')

    import complex_arithmetic as ca
    from check_level10_complex import load_runner
    m = load_runner()
    import_seconds = time.perf_counter() - wall_start
    started = time.perf_counter()
    b = Fraction(7, 5)
    momenta = tuple(map(Fraction, ('11/23', '13/29', '17/31')))
    check = m.Check(b, momenta)
    entries = labels_and_levels(args.level)
    cutoff = 2 * args.level
    zero = m.F(0)
    numerator = {n: [zero] * 8 for n in m.level_triples(cutoff)}
    setup_seconds = time.perf_counter() - started

    stage = time.perf_counter()
    primary_data = {}
    for index, (labels, base) in enumerate(entries, 1):
        ns_parity = int(2 * labels[0]) % 2
        data = []
        for alpha2 in (0, 1):
            alpha3 = (ns_parity + alpha2) % 2
            raw = check.raw(labels, alpha2, alpha3, 0, 1)
            norm = m.mb.namespace['ns_norm_squared'](labels[0], check.b, check.momenta[0])
            for e, alpha in ((1, alpha2), (2, alpha3)):
                norm *= m.mb.namespace['ramond_norm_squared'](labels[e], alpha, check.b, check.momenta[e])
            component = ns_parity | (alpha2 << 1) | (alpha3 << 2)
            prefactor = (-1) ** int(2 * labels[0]) * m.theta_quadratic_sign(component) * raw * raw / norm
            data.append((component, prefactor))
        primary_data[labels] = data
        if index % 25 == 0 or index == len(entries):
            m.log(stage='branching primary coefficients',completed=index,total=len(entries),
                  elapsed_seconds=time.perf_counter()-stage)
    branching_seconds = time.perf_counter() - stage

    stage = time.perf_counter()
    ordinary_seconds = product_seconds = assembly_seconds = 0.0
    for index, (labels, base) in enumerate(entries, 1):
        remaining = (cutoff - sum(base)) // 2
        step = time.perf_counter()
        copies = [ordinary_virasoro_series(m, check.weights.triple(labels, a),
                                           check.weights.central_charges[a], remaining)
                  for a in (0, 1)]
        ordinary_seconds += time.perf_counter() - step
        step = time.perf_counter()
        paired = multiply_series(*copies, remaining, zero)
        product_seconds += time.perf_counter() - step
        step = time.perf_counter()
        for n, value in paired.items():
            exponent = tuple(s + 2 * k for s, k in zip(base, n))
            for component, prefactor in primary_data[labels]:
                numerator[exponent][component] += prefactor * value
        assembly_seconds += time.perf_counter() - step
        if index % 25 == 0 or index == len(entries):
            m.log(stage='ordinary Virasoro pairs',completed=index,total=len(entries),
                  elapsed_seconds=time.perf_counter()-stage)
    virasoro_stage_seconds = time.perf_counter() - stage
    enlarged_seconds = time.perf_counter() - started

    stage = time.perf_counter()
    auxiliary = m.auxiliary(cutoff, 1, insertion='identity')
    auxiliary_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    numerator_projected, numerator_sector = project_sector(m, numerator, args.sector_tolerance)
    auxiliary_projected, auxiliary_sector = project_sector(m, auxiliary, args.sector_tolerance)
    expected = tuple(m.F(i in (0, 6)) for i in range(8))
    if max(abs(a - b) for a, b in zip(auxiliary_projected[0, 0, 0], expected)) > args.sector_tolerance:
        raise ValueError('The auxiliary constant must be 1 + eta_2 eta_3.')
    sector_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    physical = restricted_recovery(m, numerator_projected, auxiliary_projected, cutoff)
    recovery_seconds = time.perf_counter() - stage
    computation_seconds = time.perf_counter() - started

    timing = dict(setup=setup_seconds,branching_primary_data=branching_seconds,
                  ordinary_virasoro_blocks=ordinary_seconds,virasoro_products=product_seconds,
                  branch_sum_assembly=assembly_seconds,virasoro_stage_total=virasoro_stage_seconds,
                  enlarged_block_total=enlarged_seconds,auxiliary=auxiliary_seconds,
                  restricted_inverse_domain_check=sector_seconds,recovery=recovery_seconds,
                  total_computation=computation_seconds,imports=import_seconds,
                  with_branching_data_available=computation_seconds-branching_seconds)
    common = dict(total_level=args.level,precision_bits=ca.ctx.prec,
                  parameters=dict(b=str(b),P=list(map(str,momenta)),fermion_parity=0,
                                  primary_parity=0,three_point_eta=[1,1]),
                  exponent_convention='twice_levels e denote q_i^(e_i/2)',
                  parity_index='epsilon_1 + 2 epsilon_2 + 4 epsilon_3; eta_i^2=1',
                  arithmetic='FLINT 384-bit complex midpoints; scalar terms below 1e-80 discarded')
    args.json.parent.mkdir(parents=True,exist_ok=True)
    enlarged_path = args.json.with_suffix('.enlarged.json')
    auxiliary_path = args.json.with_suffix('.auxiliary.json')
    primary_path = args.json.with_suffix('.branching.json')
    stage = time.perf_counter()
    enlarged_path.write_text(json.dumps(dict(common,block='widehat F_0^(+,+)',
                                              coefficients=encode_series(numerator)),indent=2)+'\n')
    auxiliary_path.write_text(json.dumps(dict(common,block='F_auxiliary',
                                              coefficients=encode_series(auxiliary)),indent=2)+'\n')
    primary_path.write_text(json.dumps(dict(common,description='Normalized squared branching factors including theta sewing signs',
        coefficients=[dict(labels=list(map(str,labels)),components=[dict(parity_index=i,value=str(v)) for i,v in values])
                      for labels,values in primary_data.items()]),indent=2)+'\n')
    report = dict(common,block='F_0^(+,+)',status='computed; restricted-inverse domain requirement passed',
                  method=dict(numerator='Diagonal double-Virasoro branch sum with ordinary Ward/Gram sewing',
                              primary_data='Explicit chi branch states and three-point forms',
                              recovery='Restricted star inverse in I_+',direct_physical_block_computed=False,
                              independent_crosschecks_run=False,c_recursion_used=False),
                  counts=dict(monomials=len(physical),parity_components=8*len(physical),
                              branch_label_triples=len(entries),ordinary_virasoro_blocks=2*len(entries)),
                  timing_seconds=timing,sector_requirement=dict(tolerance=args.sector_tolerance,
                      enlarged=numerator_sector,auxiliary=auxiliary_sector),
                  enlarged_file=str(enlarged_path),auxiliary_file=str(auxiliary_path),
                  branching_file=str(primary_path),coefficients=encode_series(physical))
    report['timing_seconds']['serialization_before_final_write'] = time.perf_counter()-stage
    report['timing_seconds']['wall_before_final_write'] = time.perf_counter()-wall_start
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    m.log(result=report['status'],timing_seconds=timing,sector_requirement=report['sector_requirement'],
          physical_file=str(args.json),wall_through_output_seconds=time.perf_counter()-wall_start)


if __name__ == '__main__':
    main()
