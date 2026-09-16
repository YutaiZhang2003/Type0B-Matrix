"""Common physical integrands for the momentum-method comparison."""
from functools import lru_cache
import importlib.util
from pathlib import Path

import run_nsrr_nsnsns_offaxis_constant_scan as scan
from fast_constants import FastPositiveConstants

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'nsrr_double_virasoro_N7_L5_20260911'
spec=importlib.util.spec_from_file_location('validated_endpoint_recursion',BASE/'validation/audit_all_ns_depth.py')
depth=importlib.util.module_from_spec(spec);spec.loader.exec_module(depth)


class Evaluator:
    def __init__(self, point_ids=None, *, reference_constants=False):
        self.config=scan.load(BASE/'config.json')
        scan.validate_config(self.config)
        self.points=[p for p in self.config['points'] if point_ids is None or p['point_id'] in point_ids]
        self.constants=(scan.trial.GenericSuperLiouvilleConstants(self.config['b'],dps=35)
                        if reference_constants else FastPositiveConstants(self.config['b']))
        self.recursions={p['point_id']:depth.FixedEndpointRecursion(
            channel='theta',q_values=tuple(complex(q) for q in p['target']['q_values']),
            global_method='resummed',global_tolerance=2e-8,global_max_total_occupation=36,
            vacuum_word_length=7,vacuum_max_mode=50) for p in self.points}
        for r in self.recursions.values():r.endpoint_cap=8
        for name in ('osp_two_chain_kernel','osp_norm','theta_orientation_sign'):
            fn=getattr(depth.core,name)
            if not hasattr(fn,'cache_clear'):setattr(depth.core,name,lru_cache(maxsize=200000)(fn))

    def source(self, momenta, levels=(1,3)):
        b=self.config['b'];trial=scan.trial
        bry=self.constants.rr_ns_constants(momenta[1],momenta[0],momenta[2])
        components,checks=trial.block_components(b,momenta[::-1],max(levels))
        rows=[]
        for point in self.points:
            q=tuple(complex(v) for v in point['source']['q_values'])
            values={}
            for level in levels:
                amplitudes={}
                for lift in scan.SOURCE_FIXED_SPIN_LIFTS:
                    plumbing=trial.NSRRPlumbingInputs(q,lift,trial.GEOMETRY_SECTORS)
                    primary=plumbing.primary(b,momenta)
                    blocks=trial.evaluate_blocks(components,plumbing.q_slots,plumbing.lifts_slots,level)
                    amplitudes[lift]={c:primary*blocks[c] for c in scan.CHANNELS}
                projected=scan.project_source_fixed_spin(amplitudes)
                local=scan.contract_physical_blocks(projected,bry)
                values[str(level)]=scan.UNSCALED_M_OVER_LOCAL_KERNEL*local['total']
            rows.append({'point_id':point['point_id'],'values':values})
        return rows,checks

    def target(self, momenta, orders=(0,6)):
        rows=[]
        for point in self.points:
            r=self.recursions[point['point_id']]
            values={}
            for order in orders:
                sectors=scan.scan.all_ns_node(
                    b=self.config['b'],q_values=r.q_values,lifts=point['target']['lifts'],
                    recursion_order=order,momenta=momenta,measure=1.0,
                    constants=self.constants,recursion=r,block_method='collision_aware_mp',
                    block_working_precision=50)
                values[str(order)]=list(sectors)
            rows.append({'point_id':point['point_id'],'values':values})
            r.components.clear()
        depth.residue.cache_clear()
        for name in ('osp_two_chain_kernel','osp_norm'):
            getattr(depth.core,name).cache_clear()
        return rows,{'method':'collision-aware c recursion','endpoint_cap':8,
                     'vacuum_word_length':7,'vacuum_max_mode':50}
