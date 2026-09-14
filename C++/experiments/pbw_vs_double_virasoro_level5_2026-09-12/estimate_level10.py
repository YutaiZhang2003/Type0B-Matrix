"""Count optimized PBW work and scale measured stages; evaluate no PBW data."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def character(cutoff, fermions, bosons):
    values = [1]+[0]*cutoff
    for k in bosons:
        for j in range(k,cutoff+1):
            values[j] += values[j-k]
    for k in fermions:
        for j in range(cutoff,k-1,-1):
            values[j] += values[j-k]
    return values


def counts(level):
    ns = character(2*level,range(1,2*level+1,2),range(2,2*level+1,2))
    rr = character(level,range(1,level+1),range(1,level+1))
    volume = sum(ns)*sum(rr)**2
    shapes = set((a,b,c) for a in ns for b in rr for c in rr)
    slots = 3*sum(a*b*c for a,b,c in shapes)
    large_ints = 3*sum(max(a*b*c-257,0) for a,b,c in shapes)
    return {
        'ns_dimensions':ns,'ramond_dimensions_per_parity':rr,
        'monomials':len(ns)*len(rr)**2,
        'vertex_entries_positive':2*volume,'vertex_entries_negative':4*volume,
        'gram_triangle_entries':sum(d*(d+1)//2 for d in ns)+4*sum(d*(d+1)//2 for d in rr),
        'gram_cube_proxy':sum(d**3 for d in ns)+4*sum(d**3 for d in rr),
        'contraction_products':sum(2*a*b*c*(a+b+c+1) for a in ns for b in rr for c in rr),
        'basis_metadata_entries':sum(ns)+4*sum(rr),
        'mean_total_level_per_vertex_entry':float(sum(Fraction(i+2*j+2*k,2)*a*b*c
            for i,a in enumerate(ns) for j,b in enumerate(rr) for k,c in enumerate(rr))/volume),
        'permutation_cache_shapes':len(shapes),'permutation_cache_tuple_slots':slots,
        'permutation_cache_estimated_bytes':3*len(shapes)*sys.getsizeof(())+
            slots*(sys.getsizeof((None,))-sys.getsizeof(()))+large_ints*sys.getsizeof(1000),
        'positive_top_level_ward_key_tuple_bytes':2*volume*sys.getsizeof((None,)*6),
        'negative_top_level_ward_key_tuple_bytes':4*volume*sys.getsizeof((None,)*6),
    }


def main():
    timing_path = HERE/'results/timings.json'
    measured = json.loads(timing_path.read_text())
    work = {n:counts(n) for n in (5,10)}
    ratio = {key:work[10][key]/work[5][key] for key in (
        'monomials','vertex_entries_positive','vertex_entries_negative',
        'gram_triangle_entries','gram_cube_proxy','contraction_products','basis_metadata_entries',
        'mean_total_level_per_vertex_entry')}
    forecasts = []
    for row in measured['runs']:
        if row['algorithm'] != 'pbw_optimized':
            continue
        mode = row['mode']
        sign = 'positive' if mode == 'ordinary' else 'negative'
        assert row['counts']['vertex_entries'] == work[5]['vertex_entries_'+sign]
        assert row['counts']['gram_entries'] == work[5]['gram_triangle_entries']
        t = row['timing_seconds']
        phases = {key:t[key]*ratio[feature] for key,feature in (
            ('vertices','vertex_entries_'+sign),('gram_entries','gram_triangle_entries'),
            ('gram_inverse','gram_cube_proxy'),('contractions','contraction_products'),
            ('metadata','basis_metadata_entries'))}
        phases['setup'] = t['setup']
        phases['other_internal'] = (t['total']-sum(t[k] for k in phases))*ratio['monomials']
        phases['process_overhead'] = (row['wall_seconds']-t['total'])*ratio['monomials']
        steady = sum(phases.values())
        growth = phases['vertices']*(ratio['mean_total_level_per_vertex_entry']-1)
        forecasts.append({'mode':mode,'measured_level5_wall_seconds':row['wall_seconds'],
            'constant_unit_cost_stage_seconds':phases,
            'constant_unit_cost_seconds':steady,
            'level_weighted_ward_sensitivity_seconds':steady+growth,
            'ward_mean_level_cost_multiplier':ratio['mean_total_level_per_vertex_entry'],
            'top_level_ward_entries_at_least':work[10]['vertex_entries_'+sign]})
    source = ROOT/'Code/theta_fermion_ccy/optimized_pbw.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == measured['pbw_source_sha256']
    report = {'precision_bits':136,'dps':40,'q_level_cutoffs':[10,10,10],
        'implementation':'current optimized Python/FLINT physical PBW',
        'new_physical_gram_ward_block_evaluations':0,'counts':work,'growth_ratios':ratio,
        'forecasts':forecasts,'python_version':sys.version,
        'ward_cache_key_tuple_bytes':sys.getsizeof((None,)*6),
        'assumptions':[
            'Baseline holds measured seconds per requested vertex entry and per contraction product fixed.',
            'Gram inverse scaling uses sum(d^3) as a proxy; Gram costs are negligible in the prediction.',
            'The sensitivity model increases Ward cost per requested entry by the exact ratio of vertex-weighted mean total levels. This is an assumption, not a measured Ward-operation count.',
            'Neither scenario is a confidence interval, runtime bound, or a forecast for a future compiled Ward implementation.',
            'Higher levels may change internal Ward-cache misses, recursion lengths, hash costs, matrix efficiency, and numerical cancellation.',
            'Both time scenarios assume sufficient RAM without paging. The present unbounded caches retain every top-level Ward key plus additional internal keys and values.',
            'Memory figures count only key tuples and cached permutation tables; scalar values, dictionary storage, words, action caches, temporary matrices, and allocator overhead are additional.',
            'Permutation memory estimates assume CPython integer results above 256 are individually allocated, as in the current generator expressions.',
        ],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),source,timing_path)}}
    (HERE/'results/optimized_pbw_level10_estimate.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Optimized PBW estimate: independent-edge level ten','',
        'This estimate uses the newly measured 40-digit / 136-bit independent-edge '
        'level-five stage times and exact basis/work counts. No new physical Gram matrix, '
        'Ward value, or PBW block was evaluated. The target is q1,q2,q3 <= 10 independently.','',
        '| Work | Level 5 | Level 10 | Growth |','|---|---:|---:|---:|']
    for key,label in (('vertex_entries_positive','Vertex entries, positive sector'),
                      ('vertex_entries_negative','Vertex entries, negative sector'),
                      ('contraction_products','Dense contraction multiplications'),
                      ('gram_triangle_entries','Gram triangle entries'),('gram_cube_proxy','Cubic Gram proxy')):
        lines.append(f'| {label} | {work[5][key]:,} | {work[10][key]:,} | {ratio[key]:.1f}x |')
    lines += ['', '| Sector | Constant unit costs | Longer-Ward sensitivity scenario |',
              '|---|---:|---:|']
    for f in forecasts:
        lines.append(f'| {f["mode"]} | {f["constant_unit_cost_seconds"]/3600:.1f} h | '
                     f'{f["level_weighted_ward_sensitivity_seconds"]/3600:.1f} h |')
    lines += ['', 'The first scenario keeps the measured cost per requested vertex entry '
        'and contraction product fixed. It predicts 34.7/70.9 hours of vertex work and '
        '4.5/4.7 hours of contractions for the positive/negative sectors. The remaining '
        'stages contribute little.','',
        f'The second scenario multiplies only the vertex cost by '
        f'{ratio["mean_total_level_per_vertex_entry"]:.3f}: the entry-weighted mean total '
        f'level grows from {work[5]["mean_total_level_per_vertex_entry"]:.2f} to '
        f'{work[10]["mean_total_level_per_vertex_entry"]:.2f}. This illustrates sensitivity '
        'to longer Ward recursions; it is not an actual count of their internal operations. '
        'The two scenarios are not upper/lower bounds or a confidence interval.','',
        '## Memory condition','',
        f'The unchanged code retains at least {work[10]["vertex_entries_positive"]:,} '
        f'and {work[10]["vertex_entries_negative"]:,} distinct top-level Ward keys in the '
        'two sectors. At the measured CPython six-argument tuple size, these keys alone '
        f'occupy approximately {work[10]["positive_top_level_ward_key_tuple_bytes"]/1e9:.1f} '
        f'and {work[10]["negative_top_level_ward_key_tuple_bytes"]/1e9:.1f} GB (decimal), '
        'excluding all values, dictionary storage, internal Ward states, and mode-action caches. '
        f'The persistent tensor-permutation tables add approximately '
        f'{work[10]["permutation_cache_estimated_bytes"]/1e9:.1f} GB. These are partial '
        'memory estimates, not total peak RSS.','',
        'Consequently the time estimates require sufficient RAM without paging. A bounded '
        'or more compact cache design would be needed on machines below that requirement; '
        'changing cache retention can also change the runtime.','',
        'The earlier double-Virasoro level-ten results are 6m57s for the ordinary production '
        'pipeline and 27m35s for the inserted production pipeline, with approximately 19 minutes '
        'projected for the two latest inserted CCY optimizations. Only the first two are '
        'measured full level-ten runs.','',
        'Reproduce using the benchmark Python environment:', '', '```sh',
        '/private/tmp/theta_fermion_ccy_env/bin/python C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/estimate_level10.py',
        '```','', 'Raw counts, phase estimates, assumptions, and source hashes are in '
        '`results/optimized_pbw_level10_estimate.json`.','']
    (HERE/'optimized_pbw_level10_estimate.md').write_text('\n'.join(lines))
    print(json.dumps({'growth_ratios':ratio,'forecasts':forecasts,
                     'memory_level10':{k:v for k,v in work[10].items() if 'bytes' in k}},indent=2))


if __name__ == '__main__':
    main()
