"""Compare COMPLETE two-momentum integrals, never blocks or partial grids."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from benchmark_so7e8_fivepoint_primary import pairs,unpairs


def compare(reports):
    reports=tuple(reports)
    if any(r['status']!='complete_unconverged' or r['completed_nodes']!=r['total_nodes'] for r in reports):
        raise ValueError('crossing requires complete grids; partial or failed sums cannot be compared')
    if {r['parameters']['channel'] for r in reports}!={'A','B','C'} or len(reports)!=3:
        raise ValueError('one complete result from each of A, B, C is required')
    by_channel={r['parameters']['channel']:r for r in reports}
    anchor=by_channel['A']['parameters']
    for r in reports:
        if r['parameters'].get('resummation','independent_nome_diagnostic')!=anchor.get('resummation','independent_nome_diagnostic'):
            raise ValueError('mismatched resummation: do not compare different truncation schemes')
        for key in ('momenta','points','maximum_twice_levels','convention'):
            if r['parameters'][key]!=anchor[key]:
                raise ValueError(f'mismatched observable or cutoff: {key}')
        # Orchestration files can differ (serial and process-pool drivers);
        # the mathematical generators, weights and transformer cannot.
        for key,value in anchor['sources'].items():
            if not key.startswith('benchmark_') and r['parameters']['sources'].get(key)!=value:
                raise ValueError(f'mismatched CFT implementation: {key}')
    cut=tuple(anchor['maximum_twice_levels'])
    largest=f'{cut[0]},{cut[1]}'
    values={ch:unpairs(r['results'][largest]['total']) for ch,r in by_channel.items()}
    refinements={}
    for ch,r in by_channel.items():
        items=[]
        for axis in (0,1):
            for shift in (4,2):
                low=list(cut)
                high=list(cut)
                low[axis]-=shift
                high[axis]-=shift-2
                lo,hi=','.join(map(str,low)),','.join(map(str,high))
                if lo not in r['results'] or hi not in r['results']:
                    continue
                a=unpairs(r['results'][lo]['total'])
                b=unpairs(r['results'][hi]['total'])
                delta=abs(a-b)/np.maximum(np.maximum(abs(a),abs(b)),1e-300)
                items.append(dict(axis=axis+1,from_twice_levels=low,to_twice_levels=high,
                                  relative_shifts=delta.tolist()))
        refinements[ch]=items
    ratios=[]
    for a,b in combinations('ABC',2):
        ratios.append(dict(channels=[a,b],conditional_ratio=pairs(values[a]/values[b]),
            ratio_minus_one=pairs(values[a]/values[b]-1),
            symmetric_relative_difference=(abs(values[a]-values[b])/np.maximum(abs(values[a]),abs(values[b]))).tolist()))
    blocks_pass=all(len(rows)==4 and all(max(row['relative_shifts'])<2e-5 for row in rows)
                    for rows in refinements.values())
    crossing_pass=all(max(row['symmetric_relative_difference'])<1e-4 for row in ratios)
    return dict(status='not_certified',momenta=anchor['momenta'],points=anchor['points'],
        integrated_values={ch:pairs(v) for ch,v in values.items()},ratios=ratios,
        independent_edge_cutoff_refinements=refinements,
        last_two_block_refinements_pass=blocks_pass,crossing_at_this_grid_passes=crossing_pass,
        momentum_refinement_performed=False,momentum_convergence_certified=False,
        interpretation='Finite-grid results only. Do not interpret crossing residuals as a physical discrepancy before both numerical refinements pass.',
        equal_metric_negative_control=dict(A_over_B=pairs(2*values['A']/values['B']),A_over_C=pairs(4*values['A']/values['C'])),
        physical_amplitude_authorized=False,physical_amplitude=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare([json.loads(p.read_text()) for p in args.reports])
    result['input_reports']=[str(p.resolve()) for p in args.reports]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('points','momenta','input_reports')},indent=2))
