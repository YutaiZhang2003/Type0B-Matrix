#!/usr/bin/env python3
"""Test the paper's BPZ sewing against free-fermion bosonization.

Both NS and NSRR use actual sphere Ward/Wick vertices. No partition sewing
matrix, reflected amplitude, Liouville constant or fitted coefficient enters
the geometric state sums. The R ground multiplicity is normalized by 1/sqrt(2).
"""
import argparse
import cmath
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for directory in ('Code','Code/genus_2','Code/genus_2_cross_channel'):
    sys.path.insert(0,str(ROOT/directory))
from free_majorana_pair_of_pants import majorana_three_point, ns_fermion_states_at_twice_level
from fixed_spin_free_plumbing import charged_frame, charge_lattice_sum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--NS-relative-tolerance',type=float,default=3e-8,
                        help='Finite-level target Majorana error tolerance (default: 3e-8)')
    parser.add_argument('--R-relative-tolerance',type=float,default=1e-7,
                        help='Finite-level source Majorana error tolerance (default: 1e-7)')
    args=parser.parse_args();start=time.monotonic()
    config=json.loads(args.config.read_text())
    output=subprocess.check_output([str(ROOT/'C++/bin/partition_bpz')],text=True)
    native=[json.loads(line) for line in output.splitlines()]
    cutoff=8
    states=[ns_fermion_states_at_twice_level(n) for n in range(2*cutoff+1)]
    coefficients={}
    for n1 in range(2*cutoff+1):
        for n2 in range(2*cutoff+1-n1):
            for n3 in range(2*cutoff+1-n1-n2):
                value=sum(majorana_three_point(*triple)**2 for triple in itertools.product(states[n1],states[n2],states[n3]))
                if value: coefficients[n1,n2,n3]=value
    rows=[]
    q=tuple(complex(z) for z in config['point']['q_target'])
    geometry=charged_frame(q[::-1],max_mode=20)
    target_branch=config['target_period_branch']
    if config['source_eta_e'] != [[1,-1,1],[-1,-1,1]]:
        raise ValueError('source fixed-spin lift dictionary changed')
    for spin,marked in enumerate(config['target_marked_spins']):
        # The branch shift converts the marked characteristic to the charge
        # frame. The raw tube lifts include eta_1=-1; the infinity BPZ
        # factor is applied coefficientwise below, not hidden in the signs.
        beta=tuple((marked[1][i]-target_branch[i][i])%2 for i in range(2))
        signs=tuple(config['target_eta_e'][spin])
        if signs != (-1,(-1)**beta[1],(-1)**beta[0]):
            raise ValueError('target fixed-spin lift does not match the marked characteristic')
        F=0j
        for n,value in coefficients.items():
            e=tuple(x%2 for x in n)
            K=sum(e[i]*e[j] for i in range(3) for j in range(i+1,3))%2
            monomial=cmath.exp(sum(x*cmath.log(z)/2 for x,z in zip(n,q)))
            # Free-fermion Li-Z dual and both infinity-slot vertices.
            phase=(1j)**sum(e)*(-1)**e[0]
            F+=(-1)**K*phase*value*monomial*math.prod(s**p for s,p in zip(signs,e))
        expected=geometry.boson_chiral*charge_lattice_sum(geometry.omega_charge,((0,0),beta),cutoff=5)
        error=abs(F**2/expected-1)
        rows.append(dict(sector='NS',marked_characteristic=marked,eta_e=signs,
                         complex_squared_relative_error=error))
        if error>args.NS_relative_tolerance:raise AssertionError(rows[-1])
    q=tuple(complex(z) for z in config['point']['q_source'])
    geometry=charged_frame(q[::-1],max_mode=20)
    primary=cmath.exp((cmath.log(q[1])+cmath.log(q[2]))/16)/math.sqrt(2)
    for spin,beta in enumerate(((0,0),(1,1))):
        source_signs=tuple(config['source_eta_e'][spin])
        if source_signs != ((-1)**spin,-1,1):
            raise ValueError('source fixed-spin lift does not match the free-fermion reference')
        def evaluate(level):
            return primary*sum(float(Fraction(row[3+spin]))*cmath.exp(
                row[0]*cmath.log(q[0])/2+row[1]*cmath.log(q[1])+row[2]*cmath.log(q[2]))
                for row in native[-1]['coefficients'] if row[0]/2+row[1]+row[2]<=level)
        value=evaluate(cutoff)
        expected=geometry.boson_chiral*charge_lattice_sum(geometry.omega_charge,((1,1),beta),cutoff=5)
        error=abs(value**2/expected-1)
        previous_errors=[abs(evaluate(level)**2/expected-1) for level in (6,7)]
        rows.append(dict(sector='NSRR',marked_characteristic=[[1,1],list(beta)],eta_e=source_signs,complex_squared_relative_error=error,
                         previous_level_errors=dict(zip((6,7),previous_errors))))
        if not error<previous_errors[1]<previous_errors[0] or error>args.R_relative_tolerance:raise AssertionError(rows[-1])
    report=dict(level=cutoff,seconds=time.monotonic()-start,native_BPZ_tests=native[:2],
                NS_Wick_monomials=len(coefficients),R_Ward_monomials=len(native[-1]['coefficients']),
                rows=rows,maximum_complex_squared_relative_error=max(r['complex_squared_relative_error'] for r in rows),
                reference='Tuite-Zuevsky 1007.5203, Eqs. (10), (23), (26), (29), (78)',
                partition_matrix_used=False,structure_constants_used=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
