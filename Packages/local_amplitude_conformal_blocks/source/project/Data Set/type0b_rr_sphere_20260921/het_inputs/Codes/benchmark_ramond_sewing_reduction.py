"""Low-order human-frame theta reduction; NOT a physical crossing test.

PBW is used only through R level one. All beta fixtures are real algebraic
beta, so conjugating evaluated oracle tensors reverses convention phases
without changing parameters. The general contraction helper never conjugates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--type0b-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'Type0B-matrix')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
ward_directory = args.type0b_root / 'Code/double_virasoro/nsrr'
sys.path.insert(0, str(ward_directory))
from ramond_sewing_reduction import theta_reduction_terms, contract_theta_reduction
import numpy as np
import sympy as sp
from itertools import product
from nsrr_genus2_block import HumanNSRRThetaOracle
from ramond_pbw_generalized_ward import word_parity

SIGNS=(1,-1)
BASE=dict(central_charge=sp.Rational(27,2), h_ns=sp.Rational(5,9),
          beta_r1=sp.Rational(2,7), beta_r2=sp.Rational(3,8), primary_parity=0)
ORACLES={(f,s,t):HumanNSRRThetaOracle(form_parity=f,etas=(s,t),**BASE)
         for f,s,t in product((0,1),SIGNS,SIGNS)}

def quadratic(p):
    return (-1)**(p[0]*p[1]+p[0]*p[2]+p[1]*p[2])

def chiral_data(oracle,level,edge):
    if edge==0:
        states=oracle.ns_module.basis(level)
        states=[oracle._ns_word(x) for x in states]
        gram=np.linalg.inv(oracle.ns_basis_inverse(level)[1])
        parities=[word_parity(x) for x in states]
        ch=None
    else:
        module=oracle.r_modules[edge-1]
        states=module.basis(level)
        gram=np.array([[complex(module.inner_product(x,y)) for y in states] for x in states])
        parities=[x.parity for x in states]
        ch=np.zeros_like(gram)
        for k,state in enumerate(states):
            j=next(j for j,x in enumerate(states) if x.word==state.word and x.ground!=state.ground)
            ch[j,k]=1j if state.ground else 1
    return states,parities,gram,ch

def metrics(oracle,level,barlevel,edge):
    hs,hp,hg,hc=chiral_data(oracle,level,edge)
    bs,bp,bg,bc=chiral_data(oracle,barlevel,edge)
    dimension=len(hs)*len(bs)
    fullparities=[(a+b)%2 for a,b in product(hp,bp)]
    sign=np.diag([(-1)**(a*b) for a,b in product(hp,bp)])
    metric=sign@np.kron(hg,bg.conjugate())
    involution=None if edge==0 else 1j*np.kron(hc,bc.conjugate()@np.diag([(-1)**p for p in bp]))
    projector=np.eye(dimension) if edge==0 else (np.eye(dimension)-involution)/2
    return hs,bs,hp,bp,fullparities,projector@np.linalg.inv(metric)

def check(level,CL,CR,lifts,barlevel=None):
    barlevel=level if barlevel is None else barlevel
    oracle=ORACLES[0,1,1]
    data=[metrics(oracle,n,m,i) for i,(n,m) in enumerate(zip(level,barlevel))]
    states=[x[0] for x in data];bstates=[x[1] for x in data]
    parities=[x[2] for x in data];bparities=[x[3] for x in data]
    shape=tuple(len(x)*len(y) for x,y in zip(states,bstates))
    tensors=[]
    for coupling in (CL,CR):
        tensor=np.zeros(shape,complex)
        for I in product(*(range(n) for n in shape)):
            hol=[i//len(st) for i,st in zip(I,bstates)]
            anti=[i%len(st) for i,st in zip(I,bstates)]
            p=[pa[h] for pa,h in zip(parities,hol)]
            b=[pa[a] for pa,a in zip(bparities,anti)]
            f=sum(p)%2
            if sum(b)%2!=f:continue
            sep=(-1)**(b[0]*(p[1]+p[2])+b[1]*p[2])
            for s,t in product(SIGNS,repeat=2):
                fh=ORACLES[f,s,s].forms[0];fa=ORACLES[f,t,t].forms[0]
                x,y,z=[st[h] for st,h in zip(states,hol)]
                X,Y,Z=[st[a] for st,a in zip(bstates,anti)]
                hv=complex(fh.value(x,y.word,y.ground,z.word,z.ground))
                av=complex(fa.value(X,Y.word,Y.ground,Z.word,Z.ground)).conjugate()
                tensor[I]+=sep*coupling[f,s,t]*hv*av
        tensors.append(tensor)
    for I in product(*(range(n) for n in shape)):
        total=[d[4][i] for d,i in zip(data,I)]
        tensors[0][I]*=quadratic(total)*np.prod([x**p for x,p in zip(lifts,total)])
    direct=np.einsum('abc,ad,be,cf,def->',tensors[0],data[0][5],data[1][5],data[2][5],tensors[1])
    def block(f,s,t,lifts,anti=False):
        coeff=ORACLES[f,s,t].coefficient_components(*(barlevel if anti else level))
        value=sum(c*np.prod([lam**((p>>i)&1) for i,lam in enumerate(lifts)]) for p,c in enumerate(coeff))
        return value.conjugate() if anti else value
    shifted=(lifts[0],lifts[1],-lifts[2])
    hol={(s,t,flip):block(0,s,t,shifted if flip else lifts)
         for s,t,flip in product(SIGNS,SIGNS,(False,True))}
    anti={(s,t,flip):block(0,s,t,shifted if flip else lifts,True)
          for s,t,flip in product(SIGNS,SIGNS,(False,True))}
    reduced=contract_theta_reduction(theta_reduction_terms(CL,CR),hol,anti)
    return direct,reduced


def ward_audit():
    from ramond_pbw_generalized_ward import GeneralizedNRRWard
    opts=dict(p_phi=0,h_ns=BASE['h_ns'],central_charge=BASE['central_charge'],
              beta_second=BASE['beta_r1'],beta_third=BASE['beta_r2'],
              h_second=BASE['central_charge']/24-BASE['beta_r1']**2,
              h_third=BASE['central_charge']/24-BASE['beta_r2']**2)
    forms={(f,s):GeneralizedNRRWard(form_parity=f,eta=s,**opts)
           for f,s in product((0,1),SIGNS)}
    nswords=((),(('G',-sp.Rational(1,2)),),(('L',-1),))
    rwords=((),(('G',-1),),(('L',-1),))
    count=0
    for f,s,x,y,z,a,b in product((0,1),SIGNS,nswords,rwords,rwords,(0,1),(0,1)):
        original=forms[1-f,s].value(x,y,a,z,b)
        lhs2=(sp.I if a else 1)*forms[f,s].value(x,y,1-a,z,b)
        lhs3=(-1)**(word_parity(x)+word_parity(y)+a)*(sp.I if b else 1)*forms[f,s].value(x,y,a,z,1-b)
        assert sp.simplify(lhs2-sp.I**f*s*original)==0
        assert sp.simplify(lhs3-sp.I**(1-f)*original)==0
        count+=2
    return count


def main():
    levels=((0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1))
    pairs=[(x,x) for x in levels]+[((1,0,0),(0,0,0)),((0,0,0),(1,0,0)),((0,1,0),(1,0,0))]
    CL={x:complex(j+1,.1*j) for j,x in enumerate(product((0,1),SIGNS,SIGNS))}
    CR={x:complex(9-j,.2*j) for j,x in enumerate(product((0,1),SIGNS,SIGNS))}
    rows=[]
    for level,barlevel in pairs:
        for lifts in product(SIGNS,repeat=3):
            direct,reduced=check(level,CL,CR,lifts,barlevel)
            residual=abs(direct-reduced)
            assert residual < 2e-11
            rows.append(dict(levels=list(level),antiholomorphic_levels=list(barlevel),
                lifts=list(lifts),direct=[direct.real,direct.imag],
                reduced=[reduced.real,reduced.imag],absolute_error=residual))
    block_count=0
    block_error=0.
    for s,t,level in product(SIGNS,SIGNS,levels):
        even=ORACLES[0,s,t].coefficient_components(*level)
        odd=ORACLES[1,s,t].coefficient_components(*level)
        for p in range(8):
            block_error=max(block_error,abs(odd[p]-1j*(-1)**((p>>2)&1)*even[p^4]))
            block_count+=1
    assert block_error<1e-12
    # A failed ground-only dictionary trial is deliberately NOT installed.
    e,o=2.,.6
    trial={x:0j for x in CL}
    trial.update({(0,1,1):e/4,(0,-1,-1):o/4,
                  (1,1,1):1j*o/4,(1,-1,-1):1j*e/4})
    trial_rows=[]
    for level in ((0,0,0),(1,0,0),(0,1,0)):
        unrestricted=0j
        for f,s,t,u,v in product((0,1),SIGNS,SIGNS,SIGNS,SIGNS):
            h=sum(ORACLES[f,s,t].coefficient_components(*level))
            a=sum(ORACLES[f,u,v].coefficient_components(*level)).conjugate()
            unrestricted+=(-1)**f*trial[f,s,u]*CR[f,t,v]*h*a
        projected=check(level,trial,CR,(1,1,1))[0]
        trial_rows.append(dict(levels=list(level),
            projection_covariance_residual=abs(unrestricted-projected)))
    assert trial_rows[0]['projection_covariance_residual']<1e-12
    assert trial_rows[1]['projection_covariance_residual']>1e-3
    sources=(args.type0b_root/'Human Notes/SCblock.tex',
             ward_directory/'ramond_pbw_generalized_ward.py',
             ward_directory/'nsrr_genus2_block.py')
    report=dict(schema='human-nsrr-restricted-sewing-reduction-v1',
        status='algebraic_reduction_checked_physical_vertex_dictionary_not_certified',
        parameters={k:str(v) for k,v in BASE.items()},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        exact_ward_identities=ward_audit(),chiral_lift_comparisons=block_count,
        chiral_lift_max_absolute_error=block_error,
        nonchiral_state_sum_comparisons=len(rows),
        nonchiral_state_sum_max_absolute_error=max(x['absolute_error'] for x in rows),
        nonchiral_state_sum_rows=rows,
        rejected_ground_only_dictionary_trial=trial_rows,
        Liouville_integrated=False,physical_cross_channel_comparison=False,
        production_changed=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='nonchiral_state_sum_rows'},indent=2))


if __name__=='__main__':
    main()

