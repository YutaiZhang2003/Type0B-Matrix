"""Separate momentum convergence from elliptic-H truncation in mixed channels.

All block coefficients still come from double Virasoro / ordinary c-recursion.
Each expensive highest-order kernel is evaluated once and truncated to lower
H orders. A source- and parameter-keyed JSON cache permits quadrature-node
reuse and interrupted-run recovery without reusing stale CFT values.
No physical sewing coefficient is fitted or changed here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from benchmark_so7e8_four_ramond_convergence import MOMENTA
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from ramond_sphere_uniformization import EllipticPrefactoredRamondSphereSeries
from so7e8_human_conventions import transport_kernel_to_human_basis
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_ramond_sphere_integrand import build_fixed_p_two_ramond_kernel
from so7e8_two_ramond_liouville_integral import _uniformize_kernel
from spin23_singlet_amplitudes import _stable_complex_fsum
from so7e8_sphere_branches import branch_metadata


POINTS = (.35+.1j,.5+.1j,.65+.1j)


def pairs(values):
    values=np.asarray(values,complex)
    return np.stack((values.real,values.imag),axis=-1).tolist()


def unpairs(values):
    data=np.asarray(values,float)
    return data[...,0]+1j*data[...,1]


def source_hashes():
    root=Path(__file__).parent
    names=("benchmark_so7e8_momentum_refinement.py","so7e8_human_conventions.py",
           "so7e8_ramond_sphere_integrand.py","so7e8_ramond_cocycles.py",
           "so7e8_ramond_fourpoint.py","so7e8_mixed_double_virasoro.py",
           "so7e8_internal_ramond_double_virasoro.py","virasoro_sphere_c_recursion.py",
           "ramond_sphere_uniformization.py","sphere_block_uniformization.py","so7e8_sphere_branches.py","spin23_super_liouville_data.py",
           "spin23_two_virasoro_ramond.py","spin23_ns_torus_two_virasoro.py",
           "spin23_genus1_recursion.py","so7e8_two_ramond_liouville_integral.py",
           "heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py")
    files=[root/name for name in names]
    files += sorted((root/"ns_algebra").glob("*.py"))+sorted((root/"ramond_algebra").glob("*.py"))
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def truncate_kernel(kernel, order):
    """Truncate H, not z; also retain consistent lower-order OPE/source data."""
    if not isinstance(order,int) or order<0:
        raise ValueError("order must be a nonnegative twice-level cutoff")
    order=order-order%2 if kernel.channel=="direct" else order
    cache={}
    def truncate(series):
        if id(series) in cache:
            return cache[id(series)]
        if not isinstance(series,EllipticPrefactoredRamondSphereSeries):
            raise TypeError("the refinement requires double-Virasoro elliptic-H series")
        if order>series.source.maximum_twice_level:
            raise ValueError("cannot extend an H series by truncating it")
        # An odd NS series requested through even N has its last allowed
        # level at N-1. Preserve that known cutoff, without manufacturing
        # an unknown coefficient or rejecting the complete kernel at N.
        known=min(order,series.known_through_twice_level)
        source=replace(series.source,
            coefficients={n:x for n,x in series.source.coefficients.items() if n<=order},
            maximum_twice_level=order)
        result=replace(series,source=source,h_coefficients=series.h_coefficients[:known+1],
                       known_through_twice_level=known)
        cache[id(series)]=result
        return result
    return replace(kernel,terms=tuple(replace(t,holomorphic_series=truncate(t.holomorphic_series),
                     antiholomorphic_series=truncate(t.antiholomorphic_series)) for t in kernel.terms))


class DensityCache:
    """Only complete finite values are cached; malformed/stale entries fail."""
    def __init__(self,directory,*,channel,orders,momenta=MOMENTA,points=POINTS,picture="z",
                 limit_options=None):
        if channel not in ("crossed","direct") or picture not in ("z","one"):
            raise ValueError("invalid channel or picture")
        orders=tuple(sorted(set(orders)))
        if not orders or any(not isinstance(n,int) or n<2 for n in orders):
            raise ValueError("block orders must be at least two")
        self.channel=channel
        self.orders=tuple(sorted(set(n-n%2 if channel=="direct" else n for n in orders)))
        self.momenta=tuple(map(complex,momenta))
        require_conservative_real_p_chamber(self.momenta[:3],self.momenta[3])
        self.points=np.asarray(points,complex)
        if self.points.ndim!=1 or not len(self.points) or not np.all(np.isfinite(self.points)) or np.any((self.points==0)|(self.points==1)):
            raise ValueError("finite nonpuncture moduli required")
        self.picture=picture
        self.limit_options=dict(radius=.04,check_radius=.05,samples=24,tolerance=1e-8)
        if set(limit_options or {})-set(self.limit_options):
            raise ValueError("unknown double-Virasoro finite-part option")
        self.limit_options.update(limit_options or {})
        self.sources=source_hashes()
        self.identity=dict(schema=1,sources=self.sources,channel=channel,maximum_order=max(self.orders),
                           momenta=pairs(self.momenta),points=pairs(self.points),picture=picture,
                           species="VV",normalization="complete human convention transport",
                           finite_part_options=self.limit_options,coordinate_branch=branch_metadata())
        self.signature=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()
        self.directory=Path(directory)/self.signature
        self.directory.mkdir(parents=True,exist_ok=True)
        self.hits=0
        self.misses=0

    def __call__(self,momentum):
        p=float(momentum)
        if not np.isfinite(p) or p<=0:
            raise ValueError("positive finite P required; the endpoint is a limit")
        path=self.directory/(hashlib.sha256(p.hex().encode()).hexdigest()+".json")
        if path.exists():
            record=json.loads(path.read_text())
            if record["signature"]!=self.signature or record["P_hex"]!=p.hex():
                raise ValueError("cache identity mismatch")
            self.hits+=1
        else:
            start=time.perf_counter()
            k=build_fixed_p_two_ramond_kernel(p,external_liouville_momenta=self.momenta,
                time_momenta=(*self.momenta[:3],-self.momenta[3]),ns_at_z="V",ns_at_one="V",
                zero_ramond_family="Psi_tilde",infinity_ramond_family="Psi",channel=self.channel,
                picture_zero_at=self.picture,maximum_twice_level=max(self.orders),
                crossed_block_backend="double_virasoro" if self.channel=="crossed" else "inverse_gram",
                direct_block_backend="double_virasoro",allow_uncertified_direct_projector=self.channel=="direct",
                double_virasoro_limit_options=self.limit_options)
            k=transport_kernel_to_human_basis(k)
            # Store every lower cutoff for reuse in subsequent diagnostic queries.
            values={str(n):{name:pairs(v) for name,v in _uniformize_kernel(
                truncate_kernel(k,n),elliptic_prefactor=True).evaluate_many(self.points).items()}
                for n in range(2,max(self.orders)+1,2 if self.channel=="direct" else 1)}
            record=dict(signature=self.signature,P=p,P_hex=p.hex(),values=values,
                        seconds=time.perf_counter()-start)
            self._validate(record)
            path.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
            self.misses+=1
        self._validate(record)
        return {n:{name:unpairs(v) for name,v in record["values"][str(n)].items()} for n in self.orders}

    def _validate(self,record):
        for order in self.orders:
            tensors=record["values"][str(order)]
            if set(tensors)!={"F0","F2"}:
                raise ValueError("cached VV tensor components are incomplete")
            for values in tensors.values():
                data=np.asarray(values,float)
                if data.shape!=(len(self.points),2) or not np.all(np.isfinite(data)):
                    raise ArithmeticError("invalid/nonfinite cached CFT value")


def integrate_rule(evaluate,rule,orders,*,progress=False):
    """Integrate the unchanged density with dP/pi, retaining regional sums."""
    samples={order:{"F0":[],"F2":[]} for order in orders}
    start=time.perf_counter()
    for index,p in enumerate(rule.momenta):
        values=evaluate(p)
        for order in orders:
            for name,value in values[order].items():
                if not np.all(np.isfinite(value)):
                    raise ArithmeticError(f"nonfinite CFT density at P={p}, N={order}")
                samples[order][name].append(value)
        if progress:
            print(f"P {index+1}/{len(rule.momenta)} = {p:.7g}; {time.perf_counter()-start:.1f}s",flush=True)
    result={}
    for order,tensors in samples.items():
        result[order]={}
        for name,values in tensors.items():
            weighted=np.asarray(values)*(rule.weights/np.pi)[:,None]
            total=np.array([_stable_complex_fsum(col) for col in weighted.T])
            regions={label:np.array([_stable_complex_fsum(col) for col in weighted[rule.regions==r].T])
                     for r,label in enumerate(("endpoint","bulk","tail"))}
            result[order][name]=dict(value=pairs(total),regions={k:pairs(v) for k,v in regions.items()},
                absolute_weighted_sum=np.sum(np.abs(weighted),axis=0).tolist())
    return dict(momentum_rule=rule.metadata(),blocks=result,seconds=time.perf_counter()-start)


def relative_change(new,old):
    a,b=unpairs(new),unpairs(old)
    return (np.abs(a-b)/np.maximum(np.maximum(np.abs(a),np.abs(b)),1e-300)).tolist()


def run(*,channel,orders,nodes,cache_dir,output,options=None,progress=True,
        probe_momenta=(),picture="z",limit_options=None):
    options=dict(options or {})
    options.setdefault("beta",0. if channel=="direct" else 2.)
    evaluator=DensityCache(cache_dir,channel=channel,orders=orders,picture=picture,limit_options=limit_options)
    report=dict(schema="mixed-momentum-refinement-v1",channel=channel,
        parameters=evaluator.identity,cache_signature=evaluator.signature,threshold_options=options,
        physical_sewing_certified=False,full_pbw_used=False,rows=[],endpoint_probe=[],
        requested_nodes=list(nodes),status="running")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report["cache_hits"],report["cache_misses"]=evaluator.hits,evaluator.misses
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    for p in probe_momenta:
        values=evaluator(p)
        report["endpoint_probe"].append(dict(P=p,values={str(n):{k:pairs(v/np.pi) for k,v in values[n].items()}
                                                               for n in evaluator.orders}))
        save()
    for count in nodes:
        rule=threshold_weighted_rule(count,options=options)
        row=integrate_rule(evaluator,rule,evaluator.orders,progress=progress)
        row["relative_previous_momentum_change"]={}
        row["relative_previous_block_change"]={}
        previous=report["rows"][-1] if report["rows"] else None
        for i,n in enumerate(evaluator.orders):
            row["relative_previous_momentum_change"][n]={name:(None if previous is None else
                relative_change(row["blocks"][n][name]["value"],previous["blocks"][n][name]["value"]))
                for name in ("F0","F2")}
            row["relative_previous_block_change"][n]={name:(None if i==0 else
                relative_change(row["blocks"][n][name]["value"],row["blocks"][evaluator.orders[i-1]][name]["value"]))
                for name in ("F0","F2")}
        report["rows"].append(row)
        save()
        if progress:
            print(json.dumps(dict(completed_nodes=count,
                momentum_changes=row["relative_previous_momentum_change"],
                block_changes=row["relative_previous_block_change"])),flush=True)
    report["status"]="complete"
    save()
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel",choices=("crossed","direct"),required=True)
    parser.add_argument("--orders",nargs="+",type=int,default=[4])
    parser.add_argument("--nodes",nargs="+",type=int,default=[24,48,96])
    parser.add_argument("--threshold-options",type=json.loads)
    parser.add_argument("--limit-options",type=json.loads,
                        help="Explicit two-radius finite-part extraction controls")
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--probe-momenta",nargs="*",type=float,default=[])
    parser.add_argument("--picture",choices=("z","one"),default="z")
    args=parser.parse_args()
    run(channel=args.channel,orders=args.orders,nodes=args.nodes,cache_dir=args.cache_dir,
        output=args.output,options=args.threshold_options,probe_momenta=args.probe_momenta,picture=args.picture,
        limit_options=args.limit_options)
