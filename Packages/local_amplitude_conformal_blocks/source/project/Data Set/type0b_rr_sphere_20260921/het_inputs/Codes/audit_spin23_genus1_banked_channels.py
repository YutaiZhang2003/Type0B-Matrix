"""Compare spectrally integrated necklace/S charts after bank preparation.

This diagnostic does not authorize a new production chart. It records all
component residuals and both retained levels, rather than judging agreement
from a cancellation in the total spin sum.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from spin23_genus1_coefficient_bank import CoefficientBank
from spin23_genus1_banked_integral import load_run,write_json
from spin23_genus1_channel_atlas import overlap_diagnostic


def encode(value):
    if isinstance(value,dict): return {k:encode(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [encode(v) for v in value]
    if isinstance(value,np.ndarray): return encode(value.tolist())
    if isinstance(value,complex): return dict(real=value.real,imag=value.imag)
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def audit(run_directory):
    config,manifest=load_run(run_directory); root=Path(run_directory)
    complete=json.loads((root/'banks'/'complete.json').read_text())
    if complete['bank_id']!=manifest['bank_id']:raise ValueError('incompatible bank completion marker')
    banks={}
    for ei,energy in enumerate(config.orientations):
        path=root/'banks'/f'e{ei}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=complete['files'][path.name]:
            raise ValueError('changed spectral bank')
        banks[energy]=CoefficientBank.load(path,expected={'bank_id':manifest['bank_id'],'energy':list(energy)})
    fixtures=(
        ('previous_modular_fixture',.1+1.25j,(0j,.18+.36j,.39+.81j)),
        ('balanced',.2+1.1j,(0j,.34+.36j,.71+.74j)),
        ('short_original_edge',.1+1.25j,(0j,.35+.06j,.73+.75j)),
    )
    rows=[]
    for name,tau,points in fixtures:
        for energy in config.energies:
            try:
                result=overlap_diagnostic(tau,points,energy,banks,config.cutoffs)
                result['charts']=[asdict(c) for c in result['charts']]
                rows.append(dict(fixture=name,energy=energy,**result))
            except (ValueError,ArithmeticError) as exc:
                rows.append(dict(fixture=name,energy=energy,error_type=type(exc).__name__,error=str(exc)))
    return encode(dict(schema='spin23-svv-banked-channel-overlap-v1',run_id=manifest['run_id'],
        bank_id=manifest['bank_id'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_enabled=False,cutoffs=config.cutoffs,records=rows,
        interpretation='Differences include spectral quadrature and block truncation. No pair/comb block is supplied.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    write_json(args.output,audit(args.run_dir))
