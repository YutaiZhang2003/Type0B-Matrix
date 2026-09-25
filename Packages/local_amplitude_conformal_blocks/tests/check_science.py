"""Independent identities and comparisons with actual production data."""
from pathlib import Path
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.engine import enable, pairs, PROJECT, source_files

def sphere_banks():
    enable('type0b_rr_sphere')
    import frozen
    import literature_precision_blocks
    from literature_self_dual_blocks import ExtrapolatedLiteratureBlocks
    nodes=json.loads((ROOT/'provenance/production_nodes.json').read_text())
    errors={}; count=0
    for family,item in nodes.items():
        task=item['task']; record=json.loads((frozen.INPUTS/f"native/banks/{item['index']:04d}.json").read_text())
        assert frozen.digest(record['payload'])==record['payload_sha256']
        expected=frozen.unpairs(record['payload']['coefficients'])
        p=tuple(frozen.unpairs(task['momenta'])); err=0.
        for chi,pp in enumerate((p,tuple(x.conjugate() for x in p))):
            b=ExtrapolatedLiteratureBlocks(family,pp,task['P'],2)
            from so7e8_literature_campaign import coefficient_table
            actual=coefficient_table(b); target=expected[chi,...,:3]
            err=max(err,float(np.max(abs(actual-target)/np.maximum(1,abs(target)))))
            count+=actual.size
        assert err<2e-11,(family,err)
        errors[family]=err
    from normalization import ns_structure_constant,rr_ns_structure_constants
    identity=[]
    for p in (.23,.67):
        c=ns_structure_constant(p,p,1j*(1-1e-8),60)
        e,o=rr_ns_structure_constants(p,p,1j*(1-1e-8),60)
        ratio=(e+o)/(2*c)
        assert abs(ratio-.5)<2e-8
        identity.append(dict(P=p,half_sum_R_over_NS=pairs(ratio)))
    return dict(fresh_coefficients_compared=count,maximum_scaled_errors=errors,
                production_order=5,fresh_twice_level=2,identity_residues=identity)

def ns_crossing():
    enable('h_recursion');enable('c_Recursion')
    from sphere_four_point import BRYNSFourPointCorrelator
    p=[.5,1/3,.25,.6]
    kwargs=dict(bry_q_order=8,central_charge_shift=0,structure_precision=35,block_working_precision=60)
    a=BRYNSFourPointCorrelator(**dict(zip(('p1','p2','p3','p4'),p)),**kwargs)
    b=BRYNSFourPointCorrelator(p1=p[2],p2=p[1],p3=p[0],p4=p[3],**kwargs)
    rows=[]
    for z in (.37,.37+.11j):
        s=a.evaluate_g(z,p_max=5,quadrature_order=36)
        t=b.evaluate_g(1-z,p_max=5,quadrature_order=36)
        err=abs(s-t)/max(abs(s),abs(t))
        assert err<1e-9,(z,err)
        rows.append(dict(z=pairs(z),direct=pairs(s),crossed=pairs(t),relative_error=err))
    return dict(bry_q_order=8,momentum_nodes=36,Pmax=5,c=13.5,rows=rows)

def torus_ns():
    enable('type0b_genus1')
    from prepare import node
    from even import ChiralBank,theta,eta
    from runtime.engine import torus_ns as adapter
    p=(.6,.7);omega=.2+.25j;tau=.13+1.2j;z=.04+.18j;cut=2
    n=node(p,omega,cut,precision=40)
    arrays={k:np.array([n[k]]) for k in ('ns','r','ns_weights','r_weights')}
    arrays.update(ns_levels=n['ns_levels'],r_levels=n['r_levels'],momenta=np.array([p]),weights=np.array([1/math.pi**2]))
    bank=ChiralBank(dict(energy=[omega.real,omega.imag],cutoff=cut),arrays,Path('.'),'fresh')
    production=bank.components(tau,z,cutoffs=(cut,),factored=False)[0,0]
    result=adapter(dict(momenta=p,omega=omega,tau=tau,z=z,maximum_twice_level=cut))
    et=complex(eta(tau));prime=complex(theta(.5,.5,-z,tau)/theta(.5,.5,0,tau,1))
    green=-2*np.log(abs(prime))+2*math.pi*z.imag*z.imag/tau.imag
    common0=np.exp(2*(1+omega**2)*math.log(2*math.pi)-omega**2*green)*abs(et)**2/(64*np.sqrt(8*math.pi**2*tau.imag))
    err=0.
    for i,(spin,aa,bb) in enumerate((('NS',0,0),('NS_tilde',0,.5),('R',.5,0))):
        th=complex(theta(aa,bb,0,tau));sz=complex(theta(aa,bb,-z,tau))/(th*prime)
        left=[2j*math.pi,-omega**2*sz];right=[-2j*math.pi,-omega**2*sz.conjugate()]
        v=np.array(result['sectors'][spin]['cylinder_component_dP0dP1_integrands']);v=v[:,0]+1j*v[:,1]
        expected=np.array([v[k]*left[a]*right[b]*common0*abs(et/th) for k,(a,b) in enumerate(((0,0),(0,1),(1,0),(1,1)))])
        err=max(err,float(np.max(abs(expected-production[i])/np.maximum(1,abs(production[i])))))
    assert err<2e-12,err
    return dict(maximum_scaled_difference_from_production_even_assembly=err,
                components_checked=12,omega=pairs(omega),complex_energy=True)

def torus_r():
    enable('type0b_rr_genus1')
    from internal_normalization import CorrelatorLocalRROPE,CorrelatorRadialTrace,CORRELATOR
    from local_density import LocalRROPE
    from radial_trace import RadialTrace
    from mixed_blocks import MixedNSRamondPlumbingBlock,RamondState
    errors=[];change=[]
    for omega in (.25j,.2+.25j):
        with LocalRROPE(.6,.7,omega,bridge_order=1,ns_twice_cutoff=2,r_cutoff=1) as raw, CorrelatorLocalRROPE(.6,.7,omega,bridge_order=1,ns_twice_cutoff=2,r_cutoff=1) as normalized:
            for d in (1,2,3,4):
                x=raw.liouville(.13+1.2j,.025+.018j,d)
                y=normalized.liouville(.13+1.2j,.025+.018j,d)
                errors.append(float(np.max(abs(y-CORRELATOR.ope_factor(d)*x))))
                change.append(float(np.max(abs(raw.chiral_values(.13+1.2j,.025+.018j,d,1,1,None)-normalized.chiral_values(.13+1.2j,.025+.018j,d,1,1,None)))))
        with RadialTrace(.6,.7,omega) as raw,CorrelatorRadialTrace(.6,.7,omega) as norm:
            for d in (1,2,3,4):
                kwargs=dict(ns_cutoff=2,r_cutoff=1)
                b=norm.bare(.13+1.2j,.04+.18j,d,**kwargs)
                errors.append(float(np.max(abs(b-2*raw.bare(.13+1.2j,.04+.18j,d,**kwargs)))))
                zero=norm.raised(.13+1.2j,.04+.18j,d,**kwargs)[:,0,0]
                errors.append(float(np.max(abs(zero-b*np.array([-1j,1j])*omega**2/2))))
    b=MixedNSRamondPlumbingBlock(p_ns=.6,p_r=.7,omega=.2+.25j)
    state=RamondState((),0)
    actual=b.coefficient(2,0,state,state)
    expected=complex((b.h_ns+b.h_r-b.h_ext)**2/(2*b.h_ns))
    errors.append(abs(actual-expected))
    assert max(errors)<2e-11 and max(change)==0
    return dict(maximum_normalization_or_G0_or_global_residual=max(errors),
                maximum_change_to_chiral_blocks=max(change),
                radial_factor=2,ope_factors=[2,2,1,1])

if __name__=='__main__':
    functions=dict(sphere_banks=sphere_banks,ns_crossing=ns_crossing,torus_ns=torus_ns,torus_r=torus_r)
    result=functions[sys.argv[1]]()
    result['loaded_package_sources']=source_files()
    print(json.dumps(result,allow_nan=False))
