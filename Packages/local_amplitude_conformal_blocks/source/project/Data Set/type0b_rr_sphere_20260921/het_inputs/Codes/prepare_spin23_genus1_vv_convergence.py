"""Bounded, resumable spectral refinement; never submits amplitude jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from spin23_genus1_vv_bank import prepare_node, reference_laguerre_rule, VVBank, merge_nodes


def provenance():
    import spin23_genus1_recursive_sewing as ns
    import spin23_genus1_branch_recursion as ramond
    import spin23_genus1_vv_bank as vv
    import spin23_genus1_ns_c_recursion as c
    import spin23_super_liouville_data as constants
    return {Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (ns,ramond,vv,c,constants)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','merge'))
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--order',type=int,default=16)
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--tasks',type=int,default=64)
    args=p.parse_args();root=args.directory;root.mkdir(parents=True,exist_ok=True)
    nodes,weights=reference_laguerre_rule(args.order)
    hashes=provenance()
    config=dict(energy=[0.,.25],order=args.order,cutoff=8,source_hashes=hashes)
    config['preparation_id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if args.stage=='prepare':
        if not 0<=args.task<args.tasks:
            raise ValueError('invalid shard')
        from spin23_genus1_banked_integral import _clear_node_caches
        for i in range(args.task,len(nodes),args.tasks):
            path=root/f'node_{i}.npz'
            if path.exists():
                VVBank.load(path,expected=dict(preparation_id=config['preparation_id'],node_index=i))
                continue
            start=time.perf_counter()
            bank=prepare_node(nodes[i],.25j,cutoff=8)
            if provenance()!=hashes:
                raise RuntimeError('source changed during coefficient preparation')
            bank.metadata.update(config,node_index=i)
            bank.save(path)
            print(json.dumps(dict(node=i,total=len(nodes),seconds=time.perf_counter()-start)),flush=True)
            _clear_node_caches()
    else:
        banks=[VVBank.load(root/f'node_{i}.npz',expected=dict(
            preparation_id=config['preparation_id'],node_index=i)) for i in range(len(nodes))]
        out=merge_nodes(banks,nodes,weights,dict(config,spectral_rule=dict(
            method='reference_laguerre',order=args.order,reference_scale=3.141592653589793)))
        out.save(root/f'bank_order{args.order}.npz')
        (root/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':main()
