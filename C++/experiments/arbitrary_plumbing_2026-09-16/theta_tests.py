"""Fresh directed theta graph checks; every numerator is computed by sewing.

Run after the initial notes have been written:
    python3 theta_tests.py
The enlarged PBW numerator is never defined by convolution of the physical result.
"""
from decimal import Decimal as D, localcontext
from itertools import product
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CPP = ROOT.parents[1]
ZERO = (D(0), D(0))
TOLERANCE = D("1e-20")

def add(a,b): return a[0]+b[0],a[1]+b[1]
def sub(a,b): return a[0]-b[0],a[1]-b[1]
def mul(a,b): return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def scale(a,b): return a[0]*b,a[1]*b
def norm(a): return (a[0]*a[0]+a[1]*a[1]).sqrt()
def unpack(rows):
    return {tuple(row['exponents']):[(D(x['real']),D(x['imag'])) for x in row['values']] for row in rows}

def parity(index): return [(index>>j)&1 for j in range(3)]

def graph_sign(index):
    # Vertex order (1,2,3;1',2',3') -> edge order (1,1';2,2';3,3').
    target=[0,3,1,4,2,5]
    position={halfedge:i for i,halfedge in enumerate(target)}
    p=parity(index)*2
    inversions=sum(p[i]*p[j] for i in range(6) for j in range(i+1,6)
                   if position[i]>position[j])
    return (-1)**inversions

def kernel(a,b): return graph_sign(a)*graph_sign(b)*graph_sign(a^b)

def convolution(a,b,level):
    answer={}
    for ka,va in a.items():
        for kb,vb in b.items():
            key=tuple(x+y for x,y in zip(ka,kb))
            if sum(key)>2*level: continue
            row=answer.setdefault(key,[ZERO]*8)
            for i,x in enumerate(va):
                for j,y in enumerate(vb):
                    row[i^j]=add(row[i^j],scale(mul(x,y),kernel(i,j)))
    return answer

def compare(name,a,b):
    checks=[]
    for key in sorted(set(a)|set(b),key=lambda k:(sum(k),k)):
        for index,(x,y) in enumerate(zip(a.get(key,[ZERO]*8),b.get(key,[ZERO]*8))):
            absolute=norm(sub(x,y));relative=absolute/max(D(1),norm(x),norm(y))
            checks.append(dict(exponents=key,parity_index=index,absolute=str(absolute),scaled=str(relative)))
    failures=[x for x in checks if D(x['scaled'])>TOLERANCE]
    return dict(name=name,components=len(checks),failed_components=len(failures),
                max_scaled=str(max(D(x['scaled']) for x in checks)),
                max_absolute=str(max(D(x['absolute']) for x in checks)),
                first_failure=failures[0] if failures else None)

def evaluate_spins(series):
    answer={}
    for key,row in series.items():
        values=[]
        for spin in range(8):
            value=ZERO
            for epsilon,coefficient in enumerate(row):
                value=add(value,scale(coefficient,(-1)**((epsilon&spin).bit_count())))
            values.append(value)
        answer[key]=values
    return answer

def sector(series,sigma):
    return {key:[scale(row[index^6],sigma*kernel(6,index^6)) for index in range(8)]
            for key,row in series.items()}

def combinatorial_checks():
    q=lambda i: sum(parity(i)[a]*parity(i)[b] for a,b in [(0,1),(0,2),(1,2)])%2
    assert all(graph_sign(i)==(-1)**q(i) for i in range(8))
    for a,b,c in product(range(8),repeat=3):
        assert kernel(a,b)*kernel(a^b,c)==kernel(b,c)*kernel(a,b^c)
    for a,b in product(range(8),repeat=2):
        p,r=parity(a),parity(b)
        explicit=sum(p[i]*r[j]+p[j]*r[i] for i,j in [(0,1),(0,2),(1,2)])
        assert kernel(a,b)==(-1)**explicit
    return dict(halfedge_permutations=8,cocycle_associativity_cases=512,
                explicit_convolution_kernel_cases=64,passed=True)

def main():
    level=3;records=[];checks=[]
    for primary,f,eta,mode in product(range(2),range(2),[1,-1],['ordinary','inserted']):
        name=f'theta_p{primary}_{mode}_f{f}_eta{eta}_L{level}'
        path=ROOT/f'{name}.json'
        command=[str(ROOT/'theta_validate'),str(level),str(f),str(eta),mode,str(path),str(primary)]
        started=time.perf_counter();run=subprocess.run(command,capture_output=True,text=True)
        record=dict(name=name,command=command,exit_code=run.returncode,wall_seconds=time.perf_counter()-started)
        (ROOT/f'{name}.log').write_text(run.stderr);records.append(record)
        if run.returncode: raise RuntimeError(run.stderr)
        data=json.loads(path.read_text())
        physical=unpack(data['direct_physical_pbw']);dv=unpack(data['coefficients'])
        raw=unpack(data['direct_enlarged_pbw']);numerator=unpack(data['dv_full_numerator'])
        fermion=unpack(data['full_auxiliary']);recovered=unpack(data['direct_enlarged_recovery'])
        local=[compare('DV recovery vs independent physical PBW',dv,physical),
               compare('full DV numerator vs direct enlarged PBW',numerator,raw),
               compare('direct enlarged recovery vs physical PBW',recovered,physical),
               compare('explicit graph convolution vs direct enlarged PBW',convolution(physical,fermion,level),raw),
               compare('all spin evaluations: DV vs physical PBW',evaluate_spins(dv),evaluate_spins(physical)),
               compare('recoverable sector identity',physical,sector(physical,-1 if mode=='inserted' else 1))]
        if mode=='inserted':
            split=unpack(data['full_split_auxiliary']);split_raw=unpack(data['direct_split_enlarged_pbw'])
            diagonal=lambda x:{k:v for k,v in x.items() if k[1]==k[2]}
            local += [compare('split fermion diagonal = zero-mode factor',diagonal(split),fermion),
                      compare('split enlarged diagonal = zero-mode enlarged PBW',diagonal(split_raw),raw),
                      compare('full split DV numerator vs direct split enlarged PBW',
                              unpack(data['full_split_dv_numerator']),split_raw),
                      compare('full split graph convolution vs direct split enlarged PBW',convolution(physical,split,level),split_raw),
                      compare('opposite eta signs: uninserted enlarged PBW vanishes',
                              unpack(data['opposite_uninserted_enlarged_pbw']),{})]
        for check in local: check['case']=name
        checks.extend(local)
        print(name, 'checks', len(local), 'failed',sum(x['failed_components'] for x in local),flush=True)
    sources=list((CPP/'include/ramond').glob('*.hpp'))
    sources+=list((ROOT.parent/'unphased_vertex_2026-09-15/include/ramond').glob('*.hpp'))
    sources += [ROOT/name for name in ['theta_validate.cpp','theta_literal_split.hpp','theta_full_split_dv.hpp','theta_tests.py','theta_makefile']]
    hashes={str(path.relative_to(CPP)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    summary=dict(scope='theta NS-R-R, NS primary parity p=0,1, total level <=3; f=0,1, both eta signs and both eta*eta-prime signs; all 8 parity slots and all 8 plumbing spin evaluations',
                 dps=40,tolerance=str(TOLERANCE),graph_combinatorics=combinatorial_checks(),
                 runs=records,comparisons=checks,source_sha256=hashes,
                 passed=all(not x['failed_components'] for x in checks),
                 limitation='These are genus-two theta low-level checks at generic momenta, not a proof of all-genus identities or degenerate-weight limits.')
    (ROOT/'theta_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['passed']: raise SystemExit('theta validation failed; see theta_summary.json')

if __name__=='__main__':
    with localcontext() as context:
        context.prec=65
        main()
