"""Reproducible, component-resolved VV boundary overlap diagnostics.

These compare approximations at a fixed spectral rule. They do not certify
momentum convergence, BRST boundary terms, or a physical reflection amplitude.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

from spin23_genus1_vv_bank import VVBank, VVGeometry, evaluate_density
from spin23_genus1_vv_tail import leading_cusp_density, integrated_cusp_density


def encoded(value):
    value = np.asarray(value, complex)
    return dict(real=value.real.tolist(), imag=value.imag.tolist())


def relative(a, b):
    return (abs(a-b)/np.maximum(abs(a), 1e-300)).tolist()


def provenance():
    names = ('Codes/spin23_genus1_vv_bank.py', 'Codes/spin23_genus1_vv_tail.py',
             'Codes/spin23_genus1_vv_boundaries.py', 'Codes/spin23_genus1_vv_tail_bank.py',
             'Codes/audit_spin23_genus1_vv_boundaries.py','Codes/spin23_genus1_recursive_sewing.py',
             'Codes/spin23_genus1_ns_c_recursion.py','Codes/spin23_type0b_reference.py',
             'Codes/spin23_genus1_vv_conventions.py',
             'Codes/spin23_super_liouville_data.py','Codes/spin23_singlet_amplitudes.py')
    return {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in names}


def cusp_audit(bank):
    angles = (np.arange(64)+.5)/64-.5
    points = (.23+.65j, .13+.8j, .35+.05j, -.25+.3j, .04+.12j)
    overlap = []; matching = []
    for z in points:
        for height in (2., 3., 4.):
            geometry = VVGeometry(angles+1j*height, np.tile([0, z], (len(angles), 1)))
            bulk = evaluate_density(bank, geometry, (6, 8)).sum(axis=2).mean(axis=0)
            for i, cutoff in enumerate((6, 8)):
                tail = leading_cusp_density(bank, z, height, cutoff)
                overlap.append(dict(z=encoded(z), height=height, twice_cutoff=cutoff,
                    bulk=encoded(bulk[i]), tail=encoded(tail),
                    relative_pco_error=relative(bulk[i], tail)))
        exact, error = quad_vec(lambda y: leading_cusp_density(bank, z, y, 8),
                               4., np.inf, epsabs=1e-12, epsrel=1e-10)
        analytic = integrated_cusp_density(bank, z, 4., 8)
        # A genuine bulk shell, rather than integrating the tail twice.
        def shell(y):
            geometry = VVGeometry(angles+1j*y, np.tile([0, z], (len(angles), 1)))
            return evaluate_density(bank, geometry, (8,))[:, 0].sum(axis=1).mean(axis=0)
        shell_value, shell_error = quad_vec(shell, 3., 4., epsabs=1e-11, epsrel=1e-8)
        tail_difference = integrated_cusp_density(bank,z,3.,8)-analytic
        matching.append(dict(z=encoded(z), analytic_integral=encoded(analytic),
            numerical_integral=encoded(exact), quadrature_error=float(error),
            relative_integral_error=relative(exact, analytic),
            bulk_shell=encoded(shell_value), tail_difference=encoded(tail_difference),
            shell_quadrature_error=float(shell_error),
            relative_height_matching_error=relative(shell_value,tail_difference)))
    return dict(overlap=overlap, height_matching=matching)


def elliptic_audit(bank, target):
    import mpmath as mp
    from spin23_genus1_vv_tail_bank import prepare_tail_bank, elliptic_geometry, TailBank
    started=time.perf_counter()
    if target.exists():
        tail=TailBank.load(target)
        if not np.array_equal(tail.momenta,bank.momenta) or tail.metadata['source_bank_id'] != bank.metadata.get('bank_id'):
            raise ValueError('elliptic bank does not belong to the requested necklace')
    else:
        tail=prepare_tail_bank(bank,8);tail.save(target)
    preparation=time.perf_counter()-started
    points=(.23+.65j,.13+.8j,.35+.05j,-.35+.05j,.04+.12j,.015+.03j)
    rows=[]
    for z in points:
        x=np.exp(2j*np.pi*z);nome,theta=elliptic_geometry(x)
        with mp.workdps(40):
            exact_nome=complex(mp.exp(-mp.pi*mp.ellipk(1-x)/mp.ellipk(x)))
            exact_theta=complex(mp.jtheta(3,0,exact_nome))
        values=[leading_cusp_density(tail,z,4.,k) for k in (4,6,8)]
        polynomial=leading_cusp_density(bank,z,4.,8)
        rows.append(dict(z=encoded(z), nome=encoded(nome),
            relative_nome_error=float(abs(nome/exact_nome-1)),
            relative_theta_error=float(abs(theta/exact_theta-1)),
            elliptic=encoded(values), polynomial=encoded(polynomial),
            relative_q6_q8_difference=relative(values[-1],values[-2]),
            relative_polynomial_difference=relative(values[-1],polynomial)))
    return dict(bank=str(target),preparation_seconds=preparation,points=rows)


def spectral_audit(bank,cache_dir,endpoint=False):
    from spin23_genus1_vv_bank import reference_laguerre_rule
    from spin23_genus1_vv_tail_bank import prepare_tail_spectral,tail_endpoint_rule,TailBank
    omega=complex(*bank.metadata['energy']);rows=[]
    cache_dir.mkdir(parents=True,exist_ok=True)
    rules=([(12,16,2.),(20,24,2.),(28,32,2.),(28,40,3.)] if endpoint
           else [(n,n+1,None) for n in (8,12,20,28,40)])
    for order,long_order,long_max in rules:
        momenta,weights=(tail_endpoint_rule(order,long_order,long_max) if endpoint
                        else reference_laguerre_rule(order))
        key=dict(energy=bank.metadata['energy'],short_order=order,long_order=long_order,
            long_max=long_max,q_order=8,source_hashes=provenance())
        identity=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        cache=cache_dir/f'{identity}.npz'
        if cache.exists():
            tail=TailBank.load(cache)
            if tail.metadata.get('spectral_audit_key')!=key:
                raise ValueError('cached spectral bank has incompatible provenance')
        else:
            tail=prepare_tail_spectral(omega,momenta,weights)
            tail.metadata['spectral_audit_key']=key;tail.save(cache)
        for z in (.23+.65j,.35+.05j,.04+.12j):
            rows.append(dict(order=order,nodes=len(momenta),z=encoded(z),
                long_order=long_order,long_max=long_max,cache=str(cache),
                density=encoded([leading_cusp_density(tail,z,y,8) for y in (4.,8.,16.)]),
                integrated=encoded(integrated_cusp_density(tail,z,4.,8))))
        print(json.dumps(dict(spectral_order_complete=order)),flush=True)
    return dict(heights=[4.,8.,16.],integrated_start=4.,points=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('cusp','elliptic','spectral','endpoint'))
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tail-bank',type=Path)
    parser.add_argument('--cache-dir',type=Path,default=Path('data_exports/spin23_genus1_vv/tail_spectral_cache'))
    args=parser.parse_args();started=time.perf_counter()
    initial_sources=provenance()
    bank=VVBank.load(args.bank)
    if args.mode=='elliptic' and args.tail_bank is None:
        parser.error('--tail-bank is required for elliptic checks')
    if args.mode=='cusp':result=cusp_audit(bank)
    elif args.mode=='elliptic':result=elliptic_audit(bank,args.tail_bank)
    else:result=spectral_audit(bank,args.cache_dir,args.mode=='endpoint')
    if provenance()!=initial_sources:
        raise RuntimeError('audit source files changed during execution; rerun before using this result')
    report=dict(schema='spin23-vv-boundary-overlap-audit-v1',mode=args.mode,
        energy=bank.metadata['energy'],source_bank=str(args.bank),
        source_bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        spectral_nodes=len(bank.momenta),source_hashes=initial_sources,
        physical_amplitude_certified=False,results=result,
        seconds=time.perf_counter()-started)
    from spin23_type0b_reference import REFERENCE_COMMIT
    from spin23_genus1_vv_conventions import CONVENTION
    report['reference_commit']=REFERENCE_COMMIT
    report['physical_component_convention']=CONVENTION
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
