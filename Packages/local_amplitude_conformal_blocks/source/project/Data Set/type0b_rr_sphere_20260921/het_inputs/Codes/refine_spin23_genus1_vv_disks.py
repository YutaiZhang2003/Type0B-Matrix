"""Add the derived angular Gaussian term to an existing matched integral.

This operation needs only the collision banks, not another bulk integration.
It retains the input report and records the additive correction separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from spin23_genus1_vv_matching import gauss_interval,load_cusp_collision
from spin23_genus1_vv_collision_radial import disk_correction,cusp_disk_correction
from spin23_genus1_vv_ope import OPEBank


def correction(ope,cusp,*,radius,start,order=16):
    value=0j
    for x,xw in zip(*gauss_interval(-.5,.5,order)):
        for y,yw in zip(*gauss_interval(np.sqrt(1-x*x),start,order)):
            value+=xw*yw*disk_correction(ope,x+1j*y,radius).sum()
    return value+cusp_disk_correction(cusp,start,radius).sum()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ope',type=Path,default=Path('data_exports/spin23_genus1_vv/completion/ope_24_16.npz'))
    p.add_argument('--cusp',type=Path,default=Path('data_exports/spin23_genus1_vv/completion/cusp_collision_40_32.npz'))
    args=p.parse_args();raw=json.loads(args.input.read_text());ope=OPEBank.load(args.ope)
    cusp=load_cusp_collision(args.cusp,ope.metadata['energy'])
    rows=[]
    for row in raw['rows']:
        controls=row['compact'];radius=controls['radius'];start=controls['start']
        d=correction(ope,cusp,radius=radius,start=start)
        old=np.array(row['total']['real'])+1j*np.array(row['total']['imag'])
        new=old+d
        rows.append(dict(controls={k:controls[k] for k in ('start','radius','order','cutoffs')},
            original_total=row['total'],angular_disk_correction=dict(real=d.real,imag=d.imag),
            total=dict(real=new.real.tolist(),imag=new.imag.tolist())))
    inputs=[args.input,args.ope,args.cusp]
    source=[Path(__file__),Path('Codes/spin23_genus1_vv_collision_radial.py')]
    result=dict(schema='spin23-vv-angularly-corrected-disks-v1',rows=rows,
        input_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source},
        euclidean_amplitude_certified=False,correction='fixed timelike zero-mode angular term, no fit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(row['controls'],row['total'],flush=True)


if __name__=='__main__':main()
