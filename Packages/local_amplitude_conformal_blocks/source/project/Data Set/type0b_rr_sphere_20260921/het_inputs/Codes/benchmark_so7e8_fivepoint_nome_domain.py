"""Mixed-collision domain test for the proposed independent nome conversion.

This is geometry plus an EXACT rational test function, not a string amplitude
or a numerical crossing test. Formal roundtrip correctness is insufficient
for convergence at finite moduli in more than one variable.
"""
import argparse
import json
from pathlib import Path

import mpmath as mp
from scipy.optimize import brentq

from so7e8_fivepoint_blocks import MixedFivePointSeries
from so7e8_fivepoint_primary import POINTS,sewing_coordinates
from so7e8_fivepoint_uniformization import to_nome_series
from sphere_block_uniformization import elliptic_nome


def lambda_positive(nome):
    with mp.workdps(40):
        return float(mp.re((mp.jtheta(2,0,nome)/mp.jtheta(3,0,nome))**4))


def mixed_collision_test(q1,q2):
    radii=tuple(float(abs(elliptic_nome(q))) for q in (q1,q2))
    lambda_sum=sum(map(lambda_positive,radii))
    collision=None
    if lambda_sum>1:
        t=brentq(lambda t:sum(lambda_positive(t*r) for r in radii)-1,1e-8,1.)
        a,b=(lambda_positive(t*r) for r in radii)
        qstar=(-a/(1-a),-b/(1-b))
        collision=dict(common_radius_fraction=t,negative_nome_point=[-t*r for r in radii],
                       sewing_point=list(qstar),sewing_product=qstar[0]*qstar[1])
    return dict(nome_radii=radii,lambda_sum=lambda_sum,
        mixed_collision_inside_target_polydisk=collision is not None,collision=collision,
        full_convergence_certified=False)


def collision_toy_source(order):
    return MixedFivePointSeries({(2*k,2*k):1. for k in range(order+1)},'C',1.,(0,0),
        (0,)*5,(0,)*5,(0,0),(1,1,1),(2*order,2*order),'exact toy 1/(1-q1*q2)')


def run():
    geometry=[]
    for channel in 'ABC':
        for j,point in enumerate(POINTS,1):
            geometry.append(dict(channel=channel,point=j,**mixed_collision_test(*sewing_coordinates(channel,*point))))
    toy=[]
    for order in (2,4,8,12,16,20):
        source=collision_toy_source(order)
        transformed=to_nome_series(source)
        for channel in 'ABC':
            q=sewing_coordinates(channel,*POINTS[0])
            exact=1/(1-q[0]*q[1])
            toy.append(dict(channel=channel,point=1,integer_order_per_edge=order,
                raw_sewing_relative_error=float(abs(source.value(*q)/exact-1)),
                independent_nome_relative_error=float(abs(transformed.value(*q)/exact-1))))
    return dict(status='independent_nome_domain_preflight_failed',geometry=geometry,toy=toy,
        exact_test_function='1/(1-q1*q2); isolates the z2=1 collision',
        primary_CFT_crossing_result=False,physical_amplitude=False,
        conclusion='The independent lambda substitutions do not preserve the useful five-point sewing convergence domain. A/C require a different joint continuation or resummation before an amplitude claim.',
        mathematical_identity='Lambda(-r)=Lambda(r)/(Lambda(r)-1), so a mixed collision is inside the target polydisk when Lambda(r1)+Lambda(r2)>1.',
        source='https://dlmf.nist.gov/23.18')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'])
