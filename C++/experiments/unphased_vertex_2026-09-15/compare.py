"""Compare the directed low-level runs without evaluating any blocks."""
import json
from pathlib import Path
from decimal import Decimal as D, localcontext
ROOT=Path(__file__).resolve().parent

def unpack(v): return D(v['real']),D(v['imag'])
def norm(v): return (v[0]*v[0]+v[1]*v[1]).sqrt()
def mapping(path):
    return {tuple(row['exponents']):[unpack(v) for v in row['values']] for row in json.loads(path.read_text())['coefficients']}
def compare(candidate,reference,level=3):
    a,b=mapping(candidate),mapping(reference)
    slots=[]
    for k in sorted(set(a)|set(b),key=lambda k:(sum(k),k)):
        if sum(k)>2*level: continue
        assert k in a and k in b,(candidate,k)
        for i,(x,y) in enumerate(zip(a[k],b[k])):
            error=norm((x[0]-y[0],x[1]-y[1])); scaled=error/max(D(1),norm(x),norm(y))
            slots.append(dict(exponents=k,parity_index=i,candidate=list(map(str,x)),reference=list(map(str,y)),absolute=str(error),scaled=str(scaled)))
    failed=[x for x in slots if D(x['scaled'])>D('1e-20')]
    return dict(candidate=candidate.name,reference=reference.name,total_level=level,components=len(slots),
                max_absolute=str(max(D(x['absolute']) for x in slots)),max_scaled=str(max(D(x['scaled']) for x in slots)),
                failed_components=len(failed),first_failure=failed[0] if failed else None,
                worst=max(slots,key=lambda x:D(x['scaled'])))

if __name__=='__main__':
    with localcontext() as ctx:
        ctx.prec=65
        results=[]
        for mode in ['ordinary','inserted']:
            ref=ROOT/'results'/f'{mode}_physical_pbw_L3.json'
            for method in ['production_keep','unphased_remove','unphased_keep','literal_pbw']:
                candidate=ROOT/'results'/f'{mode}_{method}_L3.json'
                result=compare(candidate,ref);results.append(result)
                print(candidate.name,'max_scaled',result['max_scaled'],'failures',result['failed_components'],'first',result['first_failure'])
        (ROOT/'results/comparison.json').write_text(json.dumps(results,indent=2)+'\n')
