"""Paired cutoff changes at the primary run's identical coarse sample prefix."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _z(value):
    return complex(value['real'],value['imag'])


def _encoded(value):
    return {'real':value.real,'imag':value.imag}


def _rows(root, radius=.01):
    rows = {}
    corner = None
    for path in sorted((root/'shards').glob('task_*.json')):
        data = json.loads(path.read_text())
        task = data['cluster_task']
        for row in data.get('results',[data]):
            if row['collar_radius'] != radius:
                continue
            if row['corner_contribution_computed']:
                if corner is not None:
                    raise ValueError('more than one deterministic corner')
                corner = _z(row['corner_contribution'])
            rows[int(task['shard_index'])] = (int(task['seed']),row)
    if corner is None:
        raise ValueError('missing corner')
    return rows,corner


def compare_refinements(run_root, names=('projection_radius','block_depth','momentum_order','momentum_tail')):
    run_root = Path(run_root)
    base_rows,base_corner = _rows(run_root)
    comparisons = []
    pending = []
    for name in names:
        root=run_root/'checks'/name
        if not (root/'summary.json').exists():
            pending.append(name)
            continue
        rows,corner = _rows(root)
        if set(rows) != set(base_rows):
            raise ValueError('refinement shard set mismatch')
        shifts=[]
        for shard,(seed,row) in sorted(rows.items()):
            base_seed,base=base_rows[shard]
            if seed != base_seed or row['replicates'] != base['replicates']:
                raise ValueError('refinement seeds/replicates are not paired')
            fixed_fields=['outgoing_energies','incoming_energy','face_sampling','radial_power',
                          'block_backend','momentum_refinement_shells',
                          'momentum_singularity_subtraction']
            if name!='projection_radius':fixed_fields.append('projection_radius')
            if name!='block_depth':fixed_fields.append('global_max_twice_levels')
            if name!='momentum_order':fixed_fields.append('momentum_orders')
            if name!='momentum_tail':fixed_fields.append('momentum_maximum')
            if any(row.get(key)!=base.get(key) for key in fixed_fields):
                raise ValueError('more than the declared numerical control changed')
            nb,nf=row['bulk_samples_per_replicate'],row['face_samples_per_replicate']
            prefix={p['replicate']:p for p in base['sampling_prefix_estimates']
                    if p['bulk_samples']==nb and p['face_samples']==nf}
            if len(prefix)!=row['replicates']:
                raise ValueError('matching primary sample prefix missing')
            for rep,(b,f) in enumerate(zip(row['bulk_estimates'],row['face_estimates'])):
                old=prefix[rep]
                shifts.append(_z(b)+_z(f)+corner-_z(old['bulk_estimate'])-_z(old['face_estimate'])-base_corner)
        shifts=np.asarray(shifts,dtype=complex)
        mean=complex(np.mean(shifts))
        se=complex(np.std(shifts.real,ddof=1),np.std(shifts.imag,ddof=1))/math.sqrt(len(shifts))
        comparisons.append({'control':name,'collar_radius':.01,'replicate_count':len(shifts),
                            'raw_paired_shift':_encoded(mean),
                            'raw_standard_error_real':se.real,'raw_standard_error_imag':se.imag,
                            'normalized_amplitude_shift':_encoded(1j*mean/math.pi**2),
                            'normalized_standard_error_real':se.imag/math.pi**2,
                            'normalized_standard_error_imag':se.real/math.pi**2})
    result={'schema':'type0b-fivepoint-paired-refinements-v1','comparisons':comparisons,
            'pending_controls':pending,'accuracy_target_established':False,
            'interpretation':'Each shift uses the same coarse sample prefix in both configurations. It is a cutoff diagnostic, not a correction silently applied to the primary result. Statistical agreement alone does not certify total accuracy.'}
    path=run_root/'refinement_comparison.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(path)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args()
    compare_refinements(args.run_root)
