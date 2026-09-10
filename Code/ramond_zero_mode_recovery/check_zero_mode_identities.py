"""Exact zero-mode and full-algebra inverse checks through total level 10."""

import json
from pathlib import Path
import time

from check_level10_modular import auxiliary, convolution, recover, star, ZERO, F, mb
from nsrr_genus2_block import level_triples, ramond_fermion_states


def main():
    start=time.perf_counter();cutoff=20
    module=mb.FreeFieldModule('R',F(2),F(1))
    def theta(state):
        modes,g=state
        return (modes,1-g),F((-1)**(len(modes)+g+1))
    count=0;torus=[]
    for level in range(11):
        trace=0
        for modes,g in ramond_fermion_states(level):
            state=tuple(map(int,modes)),g
            target,coefficient=module.apply_auxiliary(0,state)
            final,phase=theta(target)
            assert final==state and mb.SQRT2*coefficient*phase==F((-1)**g)
            middle,phase1=theta(state);final,phase2=theta(middle)
            assert final==state and phase1*phase2==F(-1)
            for mode in range(-10,11):
                first,c1=module.apply_auxiliary(mode,state)
                left=None if not c1 else theta(first)[0]
                lc=F(0) if not c1 else c1*theta(first)[1]
                middle,c2=theta(state)
                right,c3=module.apply_auxiliary(mode,middle)
                rc=c2*c3
                assert (not lc and not rc) or (left==right and lc==-rc)
                count+=1
            trace+=(-1)**len(modes)
        assert trace%2==0
        torus.append(trace//2)
    eta=[1]+[0]*10
    for n in range(1,11):
        for k in range(10,n-1,-1):eta[k]-=eta[k-n]
    assert torus==eta
    x=(F(0),)*6+(F(1),F(0));unit=(F(1),)+ZERO[1:]
    records=[]
    for cut in (1,2):
        a=auxiliary(cutoff,cut)
        ordinary=auxiliary(cutoff,cut,'identity')
        saturated=auxiliary(cutoff,cut,'saturated')
        for n in a:
            assert tuple((x+y)/2 for x,y in zip(ordinary[n],saturated[n]))==a[n]
            assert star(x,ordinary[n])==ordinary[n]
            assert star(x,saturated[n])==tuple(-v for v in saturated[n])
        one={n:(unit if n==(0,0,0) else ZERO) for n in level_triples(cutoff)}
        inv=recover(one,a,cutoff)
        assert convolution(a,inv,cutoff)==one
        records.append(dict(cut=cut,monomials=len(a),inverse_residual_nonzero=0))
    b=F(7)/5;q=b+1/b
    vacuum=mb.FreeFieldModule('NS',b,-q/2);psi={((1,),(),()):F(1)}
    null_checks=[]
    for copy,t in ((1,2*b*b/(1-b*b)),(2,2/(b*b-1))):
        first=vacuum.apply_embedded(copy,-1,vacuum.apply_embedded(copy,-1,psi))
        second=vacuum.apply_embedded(copy,-2,psi)
        assert all(first.get(k,F(0))+t*second.get(k,F(0))==0 for k in set(first)|set(second))
        assert not vacuum.apply_embedded(copy,1,psi) and not vacuum.apply_embedded(copy,2,psi)
        null_checks.append(dict(copy=copy,null_residual_nonzero=0))
    report=dict(total_level=10,prime=mb.PRIME,zero_mode_anticommutators=count,torus_eta_coefficients=torus,
                auxiliary_checks=records,degenerate_null_checks=null_checks,elapsed_seconds=time.perf_counter()-start)
    path=Path(__file__).with_name('zero_mode_identities_level10.json')
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
