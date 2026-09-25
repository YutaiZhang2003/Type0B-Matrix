"""Fresh CFT/amplitude convergence audit of c-recursion -> truncated nome.

No proposed amplitude is an input. This is a finite set of complex-energy
tests, not certification of the real-axis continuation or every kinematic point.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import spin23_singlet_amplitudes as s
import ns_elliptic_conversion as e
from check_spin23_cft_repairs import COMPONENTS, POINTS, build, pair, legacy_value


ROOT=Path(__file__).resolve().parent.parent
CASES={"equal":(1/9+.2j,)*3+(1/3+.6j,),
       "unequal":(.11+.15j,.17+.16j,.23+.18j,.51+.49j)}


def cft_snapshot(atlas,order,chart="selected",representation="elliptic_nome"):
    z=np.asarray([value for _,value in POINTS])
    gs,gt=e.nome_geometry(z),e.nome_geometry(1-z)
    use_t=(abs(gt[0])<abs(gs[0])) if chart=="selected" else np.full(len(z),chart=="t")
    result={left+right:np.zeros(len(z),dtype=complex) for left,right in COMPONENTS}
    for crossed in (False,True):
        mask=use_t==crossed
        if not np.any(mask): continue
        data=atlas.original_t if crossed else atlas.original_s
        coord=1-z[mask] if crossed else z[mask]
        q,theta,_=tuple(a[mask] for a in (gt if crossed else gs))
        for left,right in COMPONENTS:
            terms=[]
            for k in data.kernels:
                terms.append(k.quadrature_weight*s._contract_grid(k,left,right,coord,coord.conj(),
                    q=q if representation=="elliptic_nome" else None,
                    qbar=q.conj() if representation=="elliptic_nome" else None,
                    theta3=theta if representation=="elliptic_nome" else None,
                    theta3bar=theta.conj() if representation=="elliptic_nome" else None,
                    order=order,series_parameter=representation))
            result[left+right][mask]=np.sum(terms,axis=0,dtype=np.clongdouble)
    return [{"point":name,"z":pair(z[i]),"values":{k:pair(v[i]) for k,v in result.items()}}
            for i,(name,_) in enumerate(POINTS)]


def changes(first,second,selected=None):
    result=[]
    for a,b in zip(first,second,strict=True):
        assert a['point']==b['point']
        if selected is not None and a['point'] not in selected: continue
        errors={}
        for name,value in a['values'].items():
            av,bv=complex(*value),complex(*b['values'][name])
            errors[name]={"absolute":abs(bv-av),"relative":abs(bv-av)/max(abs(av),abs(bv),1e-300)}
        result.append({"point":a['point'],"changes":errors})
    return {"points":result,"maximum_relative":max(v['relative'] for row in result for v in row['changes'].values())}


def amplitude(atlas,energies,order,**overrides):
    options=dict(epsilon0=.30,epsilon1=.24,theta_orders=(24,24,96),radial_order=36,
                 disk_total_order=28,crossed_disk_total_order=28,include_v_to_vss=True,
                 series_parameter="elliptic_nome")
    options.update(overrides)
    started=time.perf_counter()
    values=s._evaluate_at_order(atlas,energies,order=order,**options)
    result={"order":order,"settings":options,"seconds":time.perf_counter()-started,
            "values":{name:pair(getattr(values,name)) for name in ('ssvv_raw','ssss_raw','v_to_vss_raw')},
            "pieces":{process:{k:pair(v) for k,v in pieces.items()} for process,pieces in values.pieces.items()}}
    print(json.dumps({"amplitude":result['values'],"order":order,"seconds":result['seconds'],"settings":options}),flush=True)
    return result


def amplitude_change(a,b):
    return changes([{"point":"amplitude","values":a['values']}],[{"point":"amplitude","values":b['values']}])


def audit(label,do_amplitudes):
    energies=CASES[label]
    fine=build(energies,9,(16,96,32))
    sweeps={str(n):cft_snapshot(fine,n) for n in (5,7,9)}
    crossing={}
    for n in (5,7,9):
        direct,crossed=cft_snapshot(fine,n,"s"),cft_snapshot(fine,n,"t")
        crossing[str(n)]={"s":direct,"t":crossed,
            "central_overlap":changes(direct,crossed,{"overlap_left","overlap_right"})}
    coarse=build(energies,9,(12,64,24))
    old_sewing=cft_snapshot(fine,9,representation="sewing")
    report={"energies":list(map(pair,energies)),"momentum_quadrature":fine.momentum_quadrature,
            "cft_order_sweep":sweeps,"cft_order_changes":{f"{a}_to_{b}":changes(sweeps[str(a)],sweeps[str(b)]) for a,b in ((5,7),(7,9))},
            "crossing":crossing,"momentum_100_to_144":changes(cft_snapshot(coarse,9),sweeps['9']),
            "sewing9_to_nome9":changes(old_sewing,sweeps['9'])}
    print(json.dumps({"case":label,"cft_7_to_9":report['cft_order_changes']['7_to_9']['maximum_relative'],
                      "crossing9":crossing['9']['central_overlap']['maximum_relative'],
                      "momentum":report['momentum_100_to_144']['maximum_relative']}),flush=True)
    if do_amplitudes:
        amp={str(n):amplitude(fine,energies,n) for n in (5,7,9)}
        refined=amplitude(fine,energies,9,theta_orders=(36,36,144),radial_order=54)
        local=amplitude(fine,energies,9,disk_total_order=22,crossed_disk_total_order=22)
        patch=amplitude(fine,energies,9,epsilon0=.24,epsilon1=.18,theta_orders=(36,36,144),radial_order=54)
        mom=amplitude(coarse,energies,9)
        report['amplitudes']={"order_sweep":amp,"7_to_9":amplitude_change(amp['7'],amp['9']),
            "moduli_refined":refined,"moduli_change":amplitude_change(amp['9'],refined),
            "local_order22":local,"local_order22_to28":amplitude_change(local,amp['9']),
            "smaller_patches":patch,"patch_change":amplitude_change(refined,patch),
            "momentum100":mom,"momentum_change":amplitude_change(mom,amp['9'])}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=tuple(CASES),default=list(CASES))
    parser.add_argument('--skip-amplitudes',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'Data Set/validation/spin23_c_recursion_elliptic_production.json')
    args=parser.parse_args()
    files=['Codes/spin23_singlet_amplitudes.py','Codes/spin23_ns_c_recursion.py',
           'Codes/heterotic_so23_1to3_vvvv_fit_bundle/ns_elliptic_conversion.py',
           'Codes/heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py',
           'Codes/check_spin23_elliptic_production.py']
    report={**e.representation_metadata('elliptic_nome',9),"scope":__doc__,"cases":{},
            "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    for label in args.cases:
        report['cases'][label]=audit(label,not args.skip_amplitudes)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved '+str(args.output),flush=True)


if __name__=='__main__':
    main()
