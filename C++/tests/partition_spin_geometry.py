#!/usr/bin/env python3
"""Check the paper's literal PBW block against marked-spin bosonization.

The NS vacuum amplitudes are built from finite Majorana Wick state sums.
The independent answer is the bosonized Majorana partition function.
Both plumbing charts are checked against their marked periods and spins.
"""
import argparse
import cmath
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
for directory in ('Code','Code/genus_2','Code/genus_2_cross_channel'):
    sys.path.insert(0,str(ROOT/directory))
from free_majorana_pair_of_pants import majorana_three_point, ns_fermion_states_at_twice_level
from fixed_spin_free_plumbing import charged_frame, charge_lattice_sum
from spin_structure import SpinCharacteristic
from plumbing_algorithms import solve_theta_collocation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skip-literal-mismatch-audit',action='store_true',
                        help='Only certify periods and spin transport; retain literal diagnostic values')
    args=parser.parse_args();start=time.monotonic()
    config=json.loads(args.config.read_text())
    for source,target in zip(config['source_marked_spins'],config['target_marked_spins']):
        spin=SpinCharacteristic.from_pairs(source).transport(config['source_to_target'])
        assert spin.pairs==tuple(map(tuple,target))
    q_source=tuple(complex(z) for z in config['point']['q_source'])
    q=tuple(complex(z) for z in config['point']['q_target'])
    source_geometry=charged_frame(q_source[::-1],max_mode=20)
    geometry=charged_frame(q[::-1],max_mode=20)
    source_period=source_geometry.omega_charge-np.asarray(config['source_period_branch'])
    target_period=geometry.omega_charge-np.asarray(config['target_period_branch'])
    source_collocation=solve_theta_collocation(*q_source[::-1],basis_order=32,samples_per_seam=160).omega
    target_collocation=solve_theta_collocation(*q[::-1],basis_order=32,samples_per_seam=160).omega
    collocation_period_residual=max(float(np.max(abs(source_collocation-source_period))),
                                    float(np.max(abs(target_collocation-target_period))))
    if collocation_period_residual>2e-9:raise AssertionError(('independent marked period',collocation_period_residual))
    symplectic=np.asarray(config['source_to_target'])
    A,B,C,D=symplectic[:2,:2],symplectic[:2,2:],symplectic[2:,:2],symplectic[2:,2:]
    mapped=(A@source_period+B)@np.linalg.inv(C@source_period+D)
    period_residual=float(np.max(abs(mapped-target_period)))
    mapped_direct=(A@source_collocation+B)@np.linalg.inv(C@source_collocation+D)
    direct_period_residual=float(np.max(abs(mapped_direct-target_collocation)))
    if period_residual>2e-9:raise AssertionError(('source/target periods differ',period_residual))
    if direct_period_residual>2e-9:raise AssertionError(('independent surfaces differ',direct_period_residual))
    cutoff=8
    states=[ns_fermion_states_at_twice_level(n) for n in range(2*cutoff+1)]
    coefficients={}
    for n1 in range(2*cutoff+1):
        for n2 in range(2*cutoff+1-n1):
            for n3 in range(2*cutoff+1-n1-n2):
                value=sum(majorana_three_point(*triple)**2 for triple in itertools.product(states[n1],states[n2],states[n3]))
                if value: coefficients[n1,n2,n3]=value
    signs=((1,1,1),(1,-1,1),(1,1,-1),(1,-1,-1))
    F=[0j]*4
    for n,value in coefficients.items():
        epsilon=tuple(x%2 for x in n)
        K=sum(epsilon[i]*epsilon[j] for i in range(3) for j in range(i+1,3))%2
        monomial=cmath.exp(sum(x*cmath.log(z)/2 for x,z in zip(n,q)))
        for i,lift in enumerate(signs):
            F[i]+=(-1)**K*value*monomial*math.prod(s**e for s,e in zip(lift,epsilon))
    rows=[]
    branch=np.asarray(config['target_period_branch'])
    for spin,eta in enumerate(signs):
        # The bosonization routine stores Omega_marked + branch. Its beta
        # argument is computed from the same marked characteristic.
        beta=(int(eta[2]<0),int(eta[1]<0))
        marked_beta=tuple(((np.asarray(beta)+np.diag(branch))%2).tolist())
        actual=abs(F[spin])**2
        expected=abs(geometry.boson_chiral*charge_lattice_sum(geometry.omega_charge,((0,0),beta),cutoff=5))
        relative=abs(actual/expected-1)
        if relative<0.01 and not args.skip_literal_mismatch_audit:
            raise AssertionError(('expected mismatch under the current theta map',beta,relative))
        rows.append(dict(marked_characteristic=[[0,0],list(marked_beta)],eta_e=eta,
                         relative_mismatch=relative,literal_diagonal_trial=actual,bosonization_Z=expected))
    sign_obstruction=all(math.prod(lift[i]*lift[j] for i,j in ((0,1),(0,2),(1,2)))==1
                         for lift in itertools.product((1,-1),repeat=3))
    if not sign_obstruction:raise AssertionError('linear theta signs unexpectedly reverse all three loops')
    report=dict(level=cutoff,seconds=time.monotonic()-start,
                spin_transport_cases=2,period_residual=period_residual,
                independent_collocation_period_residual=direct_period_residual,
                collocation_vs_marked_period_residual=collocation_period_residual,
                linear_theta_map_sign_obstruction=sign_obstruction,
                Majorana_Wick_monomials=len(coefficients),rows=rows,
                minimum_relative_mismatch=min(row['relative_mismatch'] for row in rows))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
