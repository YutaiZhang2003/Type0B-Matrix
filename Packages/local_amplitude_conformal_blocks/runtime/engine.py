"""Expose blocks and their actual local nonchiral assembly, with explicit metadata.

The legacy source trees have overlapping module names. Use one case type per
Python process; run.py enforces this for the example suite.
"""
from pathlib import Path
from itertools import product
import json
import math
import sys
import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE/'source/project'
CONVENTION = 'local_amplitude_production_2026-09-24'

def enable(name):
    path = str(PROJECT/'Code'/name)
    if path not in sys.path: sys.path.insert(0,path)

def number(x):
    z = complex(*x) if isinstance(x,list) else complex(x)
    if not np.isfinite(z): raise ValueError('finite complex input required')
    return z

def pairs(x):
    a = np.asarray(x,dtype=complex)
    if not np.isfinite(a).all(): raise ArithmeticError('nonfinite output')
    return np.stack((a.real,a.imag),axis=-1).tolist()

def integer(config, name, default, minimum=0):
    value = config.get(name,default)
    if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return value

def source_files():
    return sorted({str(Path(m.__file__).resolve().relative_to(BUNDLE))
                   for m in tuple(sys.modules.values())
                   if getattr(m,'__file__',None)
                   and Path(m.__file__).resolve().is_relative_to(BUNDLE)})

def sphere_ns(config):
    enable('h_recursion'); enable('c_Recursion')
    from sphere_four_point import BRYNSFourPointCorrelator
    p = list(map(number,config['momenta']))
    if len(p)!=4: raise ValueError('four ordered momenta required')
    z = number(config['z']); momentum = float(config['P'])
    if momentum <= 0: raise ValueError('P must be positive')
    order = integer(config,'order',4,1)
    block = BRYNSFourPointCorrelator(**dict(zip(('p1','p2','p3','p4'),p)),
        block_order=order, block_backend='c',
        central_charge_shift=float(config.get('central_charge_shift',1e-5)),
        block_working_precision=integer(config,'precision',60,30),structure_precision=50)
    primary,starred = block._blocks(momentum,z)
    h = [block._block_value(b,z,k) for b in (primary,starred) for k in ('even','odd')]
    a = [block._block_value(b,z.conjugate(),k) for b in (primary,starred) for k in ('even','odd')]
    density = block.momentum_integrands(momentum,z)
    out = dict(block_order=order,block_central_charge=block.block_central_charge,
        block_labels=['F_even','F_odd','F_even_starstar','F_odd_starstar'],holomorphic=pairs(h),
        antiholomorphic=pairs(a),bry_block_labels=['P_even','P_odd','D_even','D_odd'],
        bry_holomorphic=pairs(np.array(h)*[1,-1,1,-1]),
        bry_antiholomorphic=pairs(np.array(a)*[1,-1,1,-1]),
        structure_products=pairs(block._structure_products(momentum)),
        dP_integrand={k:pairs(getattr(density,k)) for k in ('G','H','J')},
        measure='1/pi already included in dP_integrand',
        coefficient_labels='holomorphic/antiholomorphic are stored fixed-parity blocks; bry_* convert both odd signs once')
    if 'quadrature' in config:
        q=config['quadrature']; v=block.evaluate(z,p_max=float(q['Pmax']),quadrature_order=int(q['nodes']))
        out['finite_spectral_integral']={k:pairs(getattr(v,k)) for k in ('G','H','J')}
        out['quadrature']=q
    return out

def sphere_r(config):
    enable('type0b_rr_sphere')
    import frozen
    # Same precision bootstrap as the frozen production bank preparation.
    import literature_precision_blocks
    from literature_self_dual_correlator import AnalyticSewing
    from literature_self_dual_blocks import ExtrapolationOptions
    from literature_component_blocks import SECTORS
    from normalization import external_factor
    family=config['family']
    if family not in SECTORS: raise ValueError('family must be mixed_ns, mixed_r or rrrr')
    p=tuple(map(number,config['momenta'])); z=number(config['z']); P=float(config['P'])
    if len(p)!=4 or P<=0: raise ValueError('four momenta and positive internal P required')
    cut=integer(config,'maximum_twice_level',2)
    external=tuple(tuple(x) if isinstance(x,list) else x for x in config['external'])
    sewn=AnalyticSewing(family,p,P,cut,options=ExtrapolationOptions(float(config.get('epsilon',.01))))
    factor=external_factor(p,SECTORS[family],60)
    value=sewn.value(z,external)
    e=tuple(config.get('chiral_external',[0,0,0,0]))
    labels=frozen.sign_pairs(family)
    chiral=[sewn.blocks.value(z,e,k,l,r) for k in (0,1) for l,r in labels]
    anti=[np.conj(sewn.dual.value(z,e,k,l,r)) for k in (0,1) for l,r in labels]
    return dict(sectors=list(SECTORS[family]),maximum_twice_level=cut,
        chiral_external=list(e),block_labels=[dict(parity=k,left=l,right=r) for k in (0,1) for l,r in labels],
        holomorphic=pairs(chiral),antiholomorphic=pairs(anti),
        stripped_sewn_density=pairs(value),external_leg_factor=pairs(factor),
        dP_integrand=pairs(factor*value/math.pi),
        measure='1/pi included exactly once in dP_integrand',
        physical_external=external,metadata=sewn.metadata(),
        normalization='unchanged sphere scattering half-sum fields; effective D_R/D_NS=1/2')

def torus_ns(config):
    enable('type0b_genus1')
    from prepare import node
    from even import pairing_phase
    from ramond_descendants import chiral_table
    p=tuple(float(x) for x in config['momenta']); omega=number(config['omega'])
    tau,z=number(config['tau']),number(config['z'])
    if len(p)!=2 or min(p)<=0 or not 0<z.imag<tau.imag:
        raise ValueError('two positive momenta and 0<Im(z)<Im(tau) required')
    cutoff=integer(config,'maximum_twice_level',2)
    if cutoff%2: raise ValueError('torus NS/R shared cutoff must be even')
    data=node(p,omega,cutoff,precision=40)
    logq=2j*math.pi*np.array([z,tau-z]); forms=((0,0),(1,1))
    rows={}
    # Exactly the PP/PP coefficient and GG form conversion in even.components.
    for spin in ('NS','NS_tilde','R','R_tilde'):
        sector='r' if spin.startswith('R') else 'ns'
        levels=data[sector+'_levels']; coeff=data[sector]
        if spin=='R_tilde':
            coeff=np.stack([chiral_table(p,omega,w,cutoff=cutoff,supertrace=True)[1]
                            for w in ((1,1),(0,0))])
        powers=np.exp(levels@logq/2)
        if spin=='NS_tilde': powers*=(-1.)**levels[:,-1]
        hol=coeff@powers; anti=coeff@powers.conjugate()
        sign=np.array([-1.,1.]).reshape((2,)+(1,)*(hol.ndim-2))
        hol[0]*=sign; anti[0]*=sign
        coupling=data[sector+'_weights'][1]
        components=np.array([np.sum(coupling*hol[a]*anti[b]) for a,b in ((0,0),(0,1),(1,0),(1,1))])
        weights=np.array([(1+x*x)/2 if sector=='ns' else 9/16+x*x/2 for x in p])
        primary=np.exp(np.sum((weights-13.5/24)*(logq+logq.conjugate())))
        rows[spin]=dict(levels=levels.tolist(),coefficients=pairs(coeff),
            holomorphic=pairs(hol),antiholomorphic=pairs(anti),
            PP_pairing_coefficients=pairs(coupling),primary_product=pairs(primary),
            cylinder_component_dP0dP1_integrands=pairs(primary*components/math.pi**2))
    return dict(components=['GG/GG','GG/PP','PP/GG','PP/PP'],forms=forms,
        signs=list(product((1,-1),repeat=2)),sectors=rows,
        measure='1/pi^2 included; internal primary propagation included',
        frame='production cylinder component convention: GG multiplied by -(-1)^f; external (2pi) and PCO factors excluded',
        R_tilde_scope='Liouville supertrace only; not the completed odd-spin string density',
        maximum_twice_level=cutoff)

def torus_r(config):
    enable('type0b_rr_genus1')
    from internal_normalization import CORRELATOR,CorrelatorLocalRROPE,CorrelatorRadialTrace
    tau,z=number(config['tau']),number(config['z']); omega=number(config['omega'])
    p,q=map(float,config['momenta'])
    if min(p,q)<=0: raise ValueError('positive internal momenta required')
    out=dict(normalization=CORRELATOR.metadata(),
             measure='returned Liouville values exclude dP0 dP1/pi^2',
             common_string_amplitude_constant=None)
    if config['channel']=='ope':
        with CorrelatorLocalRROPE(p,q,omega,bridge_order=integer(config,'bridge_order',2),
             ns_twice_cutoff=integer(config,'ns_twice_cutoff',2),r_cutoff=integer(config,'r_cutoff',1)) as block:
            out['couplings']={n:pairs(getattr(block.couplings,n)) for n in ('sphere','ns_handle','r_handle')}
            out['spins']={}
            for d in (1,2,3,4):
                out['spins'][str(d)]=dict(bare=pairs(block.liouville(tau,z,d)),
                    chiral_even_eta_ground=pairs(block.chiral_values(tau,z,d,1,1,None)),
                    G0_G0=pairs(block.liouville(tau,z,d,modes=(0,0))))
    elif config['channel']=='radial':
        with CorrelatorRadialTrace(p,q,omega) as block:
            out['couplings']=pairs(block.couplings)
            cut=dict(ns_cutoff=integer(config,'ns_twice_cutoff',2),r_cutoff=integer(config,'r_cutoff',1))
            out['spins']={str(d):dict(bare=pairs(block.bare(tau,z,d,**cut)),
                G0_G0=pairs(block.raised(tau,z,d,**cut)[:,0,0])) for d in (1,2,3,4)}
    else: raise ValueError('RR torus channel must be ope or radial')
    return out

def evaluate(config):
    functions={'sphere_ns':sphere_ns,'sphere_r':sphere_r,'torus_ns':torus_ns,'torus_r':torus_r}
    kind=config['kind']
    if kind not in functions: raise ValueError('unsupported evaluator kind')
    result=functions[kind](config)
    return dict(schema='local-conformal-block-example-v1',kind=kind,convention=CONVENTION,
        configuration=config,result=result,loaded_package_sources=source_files(),
        moduli_integral_computed=False)

def cross_check():
    enable('type0b_rr_sphere')
    from compute import overlap
    raw=overlap('t0250',5)
    decode=lambda x: np.array(x)[...,0]+1j*np.array(x)[...,1]
    relative=lambda x,y: float(np.max(abs(x-y)/np.maximum(np.maximum(abs(x),abs(y)),1e-300)))
    summary=dict(four_R_t=raw['four_r']['relative_t'],four_R_u=raw['four_r']['relative_u'])
    for ch in ('s','t','u'):
        a=decode(raw['mixed'][ch]['one']['sum']);b=decode(raw['mixed'][ch]['infinity']['sum'])
        summary['mixed_picture_'+ch]=relative(a,b)
        if ch!='s':summary['mixed_s_'+ch]=relative(decode(raw['mixed']['s']['one']['sum']),a)
    tolerance=5e-4
    if not all(np.isfinite(x) and x<tolerance for x in summary.values()):
        raise AssertionError('Retained-bank crossing regression failed: '+str(summary))
    return dict(kind='cross_channel',passed=True,regression_tolerance=tolerance,
        scope='fresh spectral assembly from frozen order-five t0250 production banks; no new bank generation or moduli integral; regression tolerance is not an absolute error bound',
        summary=summary,raw=raw,loaded_package_sources=source_files())
