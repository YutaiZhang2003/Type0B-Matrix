#!/usr/bin/env python3
"""Check saved L8 coefficients against current direct native channels.

Three existing panel-grid nodes span the head, bulk and suppressed tail.
All four equal-eta channels used by the r=-1 comparison are recomputed,
including f=1 directly (without the bank's parity reconstruction).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for directory in ('Code/genus_2','Code/full_ramond_block_runtime','Code/c_Recursion'):
    sys.path.insert(0,str(ROOT/directory))
import numpy as np
from nsrr_cpp_backend import NativeNSRR, implementation_hashes
from generic_super_liouville_structure_constants import GenericSuperLiouvilleConstants
import check_nsrr_factor4_convergence as study


def check(index):
    inputs=study.Inputs()
    config=inputs.read(study.BUNDLE/'config.json')
    spec=config['nodes'][index]
    with np.load(io.BytesIO(inputs.bytes(study.BUNDLE/spec['input'])),allow_pickle=False) as z:
        momenta=z['momenta'].tolist(); ex=z['exponents']; coeff=z['values']; constants=z['norms']
    native=NativeNSRR(1.4,momenta[::-1],8,dps=40)
    comparisons=0; error=0.; projected_error=0.
    geometry=inputs.read(study.BASE/'config.json')['geometry']
    for f in (0,1):
        for eta in (1,-1):
            channel=(f,eta,eta);i=study.CHANNELS.index(channel)
            fresh=native.physical_components(*channel)
            expected=np.array([fresh[tuple(e)] for e in ex])
            comparisons+=expected.size
            error=max(error,float(np.max(abs(expected-coeff[i])/np.maximum(1,abs(expected)))))
            for g in geometry:
                q=np.array(list(map(complex,g['q_source']))[::-1])
                monomials=np.exp(np.sum(ex*np.log(q)/2,axis=1))
                a=np.sqrt(2)*np.sum(expected[:,[0,1,4,5]],axis=1)
                b=np.sqrt(2)*np.sum(coeff[i,:,[0,1,4,5]],axis=0)
                av=np.sum(a*monomials);bv=np.sum(b*monomials)
                projected_error=max(projected_error,abs(av-bv)/max(1,abs(av)))
    # Fast saved constants must retain the same BRY normalization.
    engine=GenericSuperLiouvilleConstants(1.4,dps=40,mu=1,include_cosmological_prefactor=False)
    reference=np.array(engine.rr_ns_constants(momenta[1],momenta[0],momenta[2]),complex)
    constant_error=float(np.max(abs(constants/reference-1)))
    result=dict(index=index,momenta=momenta,scalar_coefficient_comparisons=comparisons,
        maximum_scaled_coefficient_error=error,maximum_scaled_projected_block_error=projected_error,
        maximum_relative_BRY_error=constant_error,native_diagnostics=native.diagnostics(),input_sha256=inputs.files)
    assert error<1e-8 and projected_error<1e-8 and constant_error<1e-9,result
    study.save(study.OUT/'bank_checks'/f'node-{index:05d}.json',result)
    return result


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(check,(0,628,1727)))
    result=dict(status='passed',scope='Three saved nodes, all four directly recomputed equal-eta channels through L8; not a full bank recomputation',
        nodes=rows,implementation_sha256=implementation_hashes(),
        maximum_scaled_coefficient_error=max(r['maximum_scaled_coefficient_error'] for r in rows),
        maximum_scaled_projected_block_error=max(r['maximum_scaled_projected_block_error'] for r in rows),
        maximum_relative_BRY_error=max(r['maximum_relative_BRY_error'] for r in rows))
    study.save(study.OUT/'bank_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('nodes','implementation_sha256')},indent=2))


if __name__=='__main__':main()
