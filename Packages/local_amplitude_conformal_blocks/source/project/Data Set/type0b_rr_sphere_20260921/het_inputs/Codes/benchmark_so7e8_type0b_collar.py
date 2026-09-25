"""Bounded mixed-primary checks of the reused Type0B collar layer.

No momentum-integrated crossing or heterotic amplitude is inferred from
these fixed-momentum diagnostics. JSON distinguishes reference-layer tests,
algebraic coefficient checks, and finite-depth local moduli integrals.
"""
import argparse
import cmath
from itertools import product
import json
from pathlib import Path
import time

import mpmath as mp

from so7e8_fivepoint_blocks import EDGES
from so7e8_fivepoint_collar import primary_collar_engine,integrate_primary_patch
from so7e8_fivepoint_primary import primary_labels,primary_sewing_weights,REAL_MOMENTA
from so7e8_fivepoint_uniformization import fivepoint_b1_bank
from so7e8_type0b_boundary import boundary_layers,source_manifest
from so7e8_ramond_metric import CONDITIONAL_RAMOND_METRIC,inverse_pairing_factor


def encode(value):
    if isinstance(value,dict):
        return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [encode(v) for v in value]
    if isinstance(value,(complex,mp.mpc)):
        return [float(value.real),float(value.imag)]
    if isinstance(value,mp.mpf):
        return float(value)
    return value


def run(output,*,integrate_channel='C',radial_order=6,angular_order=6):
    start=time.perf_counter()
    reference=boundary_layers().integration.complete_forest_oracle()
    report=dict(status='running',reference=source_manifest(),reference_oracle=reference,
        physical_amplitude=False,global_moduli_integral=False,crossing_certified=False,
        momentum_convergence_certified=False,channels={},local_integrals=[])
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report['seconds']=time.perf_counter()-start
        output.write_text(json.dumps(encode(report),indent=2,allow_nan=False)+'\n')
    save()
    try:
        for channel in 'ABC':
            momenta=(.61,.83)
            bank,defect=fivepoint_b1_bank(primary_labels(channel),channel=channel,
                internal_momenta=momenta,external_momenta=REAL_MOMENTA,maximum_twice_levels=(4,4),
                radius=.18,check_radius=.22,samples=32,contour_symmetry='real_primary')
            weights=primary_sewing_weights(channel,momenta,REAL_MOMENTA)
            metric=inverse_pairing_factor(EDGES[channel],convention=CONDITIONAL_RAMOND_METRIC)
            engine=primary_collar_engine(bank,weights,metric_multiplier=metric)
            corner=engine.coefficients((0,1),4)
            expected={key:sum(metric*weights[label]*source.coefficients.get(key,0)**2
                              for label,source in bank.items()) for key in product(range(5),repeat=2)}
            error=max(float(abs(corner[key]-value)/max(1,abs(value))) for key,value in expected.items())
            face_rows=[]
            for edge in (0,1):
                r,t=.004,.25+.06j
                coefficients=engine.coefficients((edge,),4,t)
                values=[]
                for j in range(16):
                    xy=[t,t]
                    xy[edge]=r*cmath.exp(2j*mp.pi*(j+.125)/16)
                    values.append(engine.value(*xy))
                actual=sum(values)/16/mp.exp(engine.betas[edge]*mp.log(r))
                predicted=sum(c*r**key[edge] for key,c in coefficients.items())
                face_rows.append(dict(edge=edge,relative_error=float(abs(actual/predicted-1))))
            report['channels'][channel]=dict(internal_momenta=momenta,external_momenta=REAL_MOMENTA,
                metric_multiplier=metric,b1_radius_defect=defect,radial_exponents=engine.betas,
                corner_coefficient_scaled_defect=error,face_checks=face_rows)
            save()
            print(channel,'coefficient_defect',error,'face_errors',[x['relative_error'] for x in face_rows],flush=True)
            if channel==integrate_channel:
                for radius in (.03,.05):
                    for depth in (2,4):
                        result=integrate_primary_patch(engine,radius=.3,collar=radius,maximum_increment=depth,
                            radial_order=radial_order,angular_order=angular_order)
                        report['local_integrals'].append(dict(result,channel=channel,collar=radius,
                            patch_radius=.3,radial_order=radial_order,angular_order=angular_order))
                        save()
                        print('local',channel,'rho',radius,'depth',depth,'total',complex(result['total']),flush=True)
        report['status']='adapter_checks_passed' if reference['passed'] and all(
            row['corner_coefficient_scaled_defect']<1e-9 and max(x['relative_error'] for x in row['face_checks'])<1e-7
            for row in report['channels'].values()) else 'adapter_checks_failed'
    except BaseException as exc:
        report['status']='failed'
        report['failure']=dict(type=type(exc).__name__,message=str(exc))
        save()
        raise
    save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--integrate-channel',choices=('A','B','C','none'),default='C')
    parser.add_argument('--radial-order',type=int,default=6)
    parser.add_argument('--angular-order',type=int,default=6)
    args=parser.parse_args()
    run(args.output,integrate_channel=args.integrate_channel,radial_order=args.radial_order,angular_order=args.angular_order)
