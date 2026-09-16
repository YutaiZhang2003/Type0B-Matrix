"""Directed all-NS checks of the CURRENT human-note vertex convention.

The historical generic-parity audit used a different provisional crossing
(-1)^[(p2+B) delta3]. It is deliberately not used here.
"""
from pathlib import Path
from itertools import product
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
CODE=REPO/'Code'
for sub in ['c_Recursion','genus_2_cross_channel','double_virasoro/all_ns',
            'unused_human_note_computation/pbw_c_recursion_double_virasoro_crosscheck']:
    sys.path.insert(0,str(CODE/sub))
import sympy as sp
import check_second_virasoro_primary as algebra
import compare_ns_genus2_double_virasoro as numerical

def current_vertex(form,vectors):
    total=sp.S.Zero
    p=form.primary_parities
    for components in product(*(tuple(vector.items()) for vector in vectors)):
        states=tuple(c[0] for c in components)
        fermions=tuple(state[0] for state in states)
        physical=tuple(state[1] for state in states)
        A,B,C=(algebra.state_parity(state) for state in physical)
        delta3=len(fermions[2])%2
        phase=(-1)**(B*delta3+p[0]*A+p[1]*C)
        total+=sp.prod(c[1] for c in components)*phase*algebra.current_fermion_three_point(fermions)*form.value(*physical)
    return sp.cancel(total)

def lower(module,b,copy,vector):
    out={}
    for state,value in vector.items():
        for target,coefficient in algebra.double_virasoro_action(module,b,copy,-1,state).items():
            out[target]=out.get(target,sp.S.Zero)+value*coefficient
    return {k:sp.cancel(v) for k,v in out.items() if v!=0}

def run():
    start=time.perf_counter()
    b=sp.Rational(7,5);q=b+1/b
    momenta=(sp.Rational(11,23),sp.Rational(13,29),sp.Rational(17,31))
    c=sp.Rational(3,2)+3*q*q
    h=tuple(q*q/8-p*p/2 for p in momenta)
    modules=[algebra.ExactNSVermaModule(c=c,weight=x) for x in h]
    vectors=[{0:algebra.v0(),1:algebra.vhalf(q,p)} for p in momenta]
    even=algebra.ExactNSDescendantThreeForm(c=c,weights=h)
    local=0;ward=0;failures=[]
    for parities in product(range(2),repeat=3):
        form=algebra.ExactNSDescendantThreeForm(c=c,weights=h,primary_parities=parities)
        for labels in product(range(2),repeat=3):
            chosen=tuple(vectors[i][labels[i]] for i in range(3))
            ground=current_vertex(form,chosen)
            residual=sp.cancel(ground-current_vertex(even,chosen));local+=1
            if residual: failures.append(dict(type='primary-parity independence',parities=parities,labels=labels,residual=str(residual)))
            for copy in [1,2]:
                weights=[]
                for j in range(3):
                    first=algebra.first_copy_weight(b,momenta[j],sp.Rational(labels[j],2))
                    weights.append(first if copy==1 else h[j]+sp.Rational(labels[j]**2,2)-first)
                expected=[weights[0]+weights[1]-weights[2],weights[0]-weights[1]-weights[2],weights[1]+weights[2]-weights[0]]
                for slot in range(3):
                    descendant=list(chosen);descendant[slot]=lower(modules[slot],b,copy,chosen[slot])
                    residual=sp.cancel(current_vertex(form,tuple(descendant))-expected[slot]*ground);ward+=1
                    if residual:failures.append(dict(type='Virasoro Ward',parities=parities,labels=labels,copy=copy,slot=slot,residual=str(residual)))
    exact_seconds=time.perf_counter()-start
    # Legacy all-NS backend has complex-double series arithmetic. Its branching
    # polynomials are evaluated with 40 decimal digits before conversion.
    start=time.perf_counter();bf=float(b);pf=tuple(float(x) for x in momenta);cutoff=4
    numerator=numerical.double_virasoro_enlarged_series(b=bf,momenta=pf,cutoff=cutoff,precision=40)
    fermion=numerical.auxiliary_majorana_series(cutoff=cutoff)
    recovered=numerical.divide_theta_star_series(numerator,fermion,cutoff=cutoff)
    direct=numerical.direct_ns_series(c=complex(c),weights=tuple(complex(x) for x in h),cutoff=cutoff)
    errors=[abs(recovered[k]-direct[k])/max(1,abs(recovered[k]),abs(direct[k])) for k in direct]
    samples=[dict(twice_levels=k,DV=[recovered[k].real,recovered[k].imag],PBW=[complex(v).real,complex(v).imag]) for k,v in direct.items()]
    spin_checks=0;spin_worst=0.
    for lifts in product([-1,1],repeat=3):
        for a in [0,1]:
            args=dict(q_values=(.013,.017,.011),lifts=lifts,sector=a)
            x=numerical.evaluated_sector(recovered,**args);y=numerical.evaluated_sector(direct,**args)
            spin_worst=max(spin_worst,abs(x-y)/max(1,abs(x),abs(y)));spin_checks+=1
    sources={str(Path(m.__file__).relative_to(REPO)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
             for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(getattr(m,'__file__')).startswith(str(CODE))}
    sources[str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(scope='current all-NS local vertex, all 8 intrinsic-primary parities; branch labels n_i in {0,1/2}; L_-1 Ward in both Virasoro copies/all slots. Closed all-NS theta block p_i=0 total<=2 numerical check.',
                exact_local_parity_comparisons=local,exact_local_Virasoro_Ward_checks=ward,exact_failures=failures,
                exact_seconds=exact_seconds,numerical_series_arithmetic='complex double; 40-digit branching polynomial evaluation',
                numerical_coefficients=len(samples),numerical_max_scaled=max(errors),numerical_tolerance=1e-10,
                evaluated_spin_sector_cases=spin_checks,evaluated_spin_sector_max_scaled=spin_worst,
                numerical_seconds=time.perf_counter()-start,coefficients=samples,source_sha256=sources,
                passed=not failures and max(errors)<1e-10)
    (ROOT/'theta_all_ns_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['coefficients','source_sha256','exact_failures']})
    if not result['passed']:raise SystemExit('all-NS validation failed')

if __name__=='__main__':run()
