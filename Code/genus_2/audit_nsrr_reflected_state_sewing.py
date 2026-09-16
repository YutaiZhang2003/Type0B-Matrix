#!/usr/bin/env python3
"""Audit a reflection-frame NSRR matrix, then recombine saved complex blocks.

The high-order scalar data cannot support this new matrix. Only complete
saved datasets containing individual complex blocks are recombined.
"""
from __future__ import annotations
import argparse
import cmath
from datetime import datetime,timezone
import itertools
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for path in ('Code','Code/genus_2','Code/double_virasoro/nsrr','Code/c_Recursion','Code/genus_2_cross_channel'):
    sys.path.insert(0,str(ROOT/path))
import sympy as s
from nsrr_reflected_state_sewing import ReflectedNSRRStateSewing,reflection_candidate,E0
from ramond_pbw_generalized_ward import GeneralizedNRRWard,word_parity
from nsrr_genus2_block import HumanNSRRThetaOracle,level_triples
from physical_nsrr_sewing import CHANNELS,SOURCE_FIXED_SPIN_LIFTS,contract_physical_blocks
from nsrr_plumbing_adapter import NSRRPlumbingInputs,GEOMETRY_SECTORS
from recombine_saved_genus2_coefficient_ledger import Inputs,object_digest,decode,encode,csum,relative,write_csv

DATA=ROOT/'Data Set'
OUTPUT=DATA/'nsrr_reflected_state_sewing_20260915'


def exact_half_level():
    h,c,p2,p3=s.symbols('h c p_1 p_0',positive=True)
    el,ol,er,or_=s.symbols('E_L O_L E_R O_R',real=True)
    # Direct physical-family Ward amplitudes before identifying the pants.
    def vertex(e,o):
        dp,dm=(e+o)/2,(e-o)/2
        u=(1-s.I)/s.sqrt(2)
        beta2,beta3=s.I*p2/s.sqrt(2),s.I*p3/s.sqrt(2)
        return (-s.I*u*(beta3*dp-beta2*dm),u*(beta3*dm-beta2*dp))
    left,right=vertex(el,ol),vertex(er,or_)
    half=s.simplify(sum(a*s.conjugate(b) for a,b in zip(left,right))/(2*h))
    expected=(el*er*(p3-p2)**2+ol*or_*(p3+p2)**2)/(8*h)
    assert s.simplify(half-expected)==0
    # With the two-lift descendant blocks F0=sqrt(2)(1+alpha*x),
    # F1=-i sqrt(2)(1-alpha*x), determine each 2x2 form-parity matrix.
    a,d,re,im,alpha=s.symbols('a d r y alpha',real=True)
    matrix=s.Matrix([[a,re+s.I*im],[re-s.I*im,d]])/16
    v=s.sqrt(2)*s.Matrix([1,-s.I]);w=s.sqrt(2)*alpha*s.Matrix([1,s.I])
    ground=s.simplify((v.T*matrix*s.conjugate(v))[0])
    cross=s.simplify((w.T*matrix*s.conjugate(v))[0])
    mixed=s.simplify((w.T*matrix*s.conjugate(w))[0])
    equal=s.solve([ground-s.Rational(1,2),s.re(cross)-alpha/2,s.im(cross),mixed-alpha**2/2],(a,d,re,im),dict=True)
    opposite=s.solve([ground,s.re(cross),s.im(cross),mixed],(a,d,re,im),dict=True)
    assert equal==[{a:4,d:0,re:0,im:0}] and opposite==[{a:0,d:0,re:0,im:0}]
    legacy=s.Matrix([[1,-s.I],[s.I,1]])/16
    assert s.simplify((w.T*legacy*s.conjugate(v))[0])==0
    # Both NS descendants, from the independent two-algebra Ward reduction.
    expected_mixed=(el*er*(p3-p2)**4+ol*or_*(p3+p2)**4)/(32*h*h)
    independent=ReflectedNSRRStateSewing(h=h,c=c,beta2=s.I*p2/s.sqrt(2),beta3=s.I*p3/s.sqrt(2))
    words=((('G',-s.Rational(1,2)),),(),())
    amplitudes={t:{(a,b):independent.vertex(words,words,a,b,t) for a,b in itertools.product((0,1),repeat=2)} for t in (1,-1)}
    mixed_direct=s.simplify(sum(
        (el*amplitudes[1][key]+ol*amplitudes[-1][key])/2
        *s.conjugate((er*amplitudes[1][key]+or_*amplitudes[-1][key])/2)
        for key in amplitudes[1])/(2*h)**2)
    assert s.simplify(mixed_direct-expected_mixed)==0
    return {'ground':str((el*er+ol*or_)/2),'q_NS_half':str(s.factor(half)),
            'q_NS_half_qbar_NS_half':str(expected_mixed),
            'legacy_q_NS_half':'0',
            'matrix_equal_eta':[['1/4','0'],['0','0']],
            'matrix_opposite_eta':[['0','0'],['0','0']],
            'matrix_uniqueness_scope':'Within matrices diagonal in the eta,eta-prime pair, with an arbitrary Hermitian 2x2 form-parity matrix per pair; not a proof excluding all possible eta-mixing matrices.',
            'propagation':'Common primary |P|^2 stripped; x=q_NS^(1/2) is a descendant power.'}


def clifford_form_checks():
    """Check the zero-mode intertwiner relations beyond ground states."""
    h,c=s.Rational(7,10),s.Rational(81,5)
    b2,b3=s.I*s.Rational(4,10),s.I*s.Rational(7,10)
    forms={(f,t):GeneralizedNRRWard(p_phi=0,form_parity=f,eta=t,h_ns=h,
        h_second=c/24-b2*b2,h_third=c/24-b3*b3,beta_second=b2,beta_third=b3,central_charge=c)
        for f,t in itertools.product((0,1),(1,-1))}
    ns=[(),(('G',-s.Rational(1,2)),),(('L',-1),),(('G',-s.Rational(3,2)),),
        (('L',-1),('G',-s.Rational(1,2)))]
    r=[(),(('L',-1),),(('G',-1),)]
    u=(1-s.I)/s.sqrt(2);j=s.Matrix([[0,u],[s.conjugate(u),0]])
    counts={'J_R1':0,'J_R0':0,'outgoing_bra_phase':0}
    for f,t,a,x,y,b,g in itertools.product((0,1),(1,-1),ns,r,r,(0,1),(0,1)):
        value=forms[f,t].value(a,x,b,y,g)
        other=forms[1-f,t].value(a,x,b,y,g)
        p2=(word_parity(x)+b)%2;p3=(word_parity(y)+g)%2
        lhs2=(-1)**word_parity(x)*j[1-b,b]*forms[f,t].value(a,x,1-b,y,g)
        lhs3=(-1)**word_parity(y)*j[1-g,g]*forms[f,t].value(a,x,b,y,1-g)
        assert s.simplify(lhs2-s.I**f*t*s.conjugate(u)*(-1)**p2*other)==0
        assert s.simplify(lhs3-s.I**f*u*other)==0
        assert s.simplify(s.I**(b+g)*s.conjugate(value)-(-s.I)**f*(-1)**p3*value)==0
        for key in counts:counts[key]+=1
    tables={};trace_count=0
    for p,pa in itertools.product((0,1),repeat=2):
        matrices={eta:{(0,0):s.diag(1,eta),(1,0):s.Matrix([[0,-s.I],[eta,0]]),
                      (0,1):s.Matrix([[0,s.I],[eta,0]]),(1,1):s.diag(-s.I,s.I*eta)}[p,pa]
                  for eta in (1,-1)}
        for eta,etap in itertools.product((1,-1),repeat=2):
            assert s.trace(matrices[eta]*matrices[etap].conjugate().T)==1+eta*etap
            trace_count+=1
        tables[str((p,pa))]={str(t):str(m) for t,m in matrices.items()}
    return dict(exact_Ward_checks=counts,physical_family_matrices=tables,
        family_trace_identity='Tr(K_eta K_eta_prime^dagger)=1+eta*eta_prime=2 delta_eta,eta_prime',
        exact_family_trace_checks=trace_count,
        conditions='Normalized two-family small modules, J-adapted bases, reflected duals, no inserted parity defect. The family table and its reduction do not specify a general modular transport.')


def bra_and_clifford():
    i=s.I;u=(1-i)/s.sqrt(2)
    b=s.diag(1,i);bt=s.diag(1,-i)
    d=s.diag(1,-i,i,-1)
    conjugation=s.diag(1,i,-i,-1)
    assert conjugation*d==s.eye(4)
    dual=conjugation*s.conjugate(E0)
    assert s.simplify(dual.T*d*E0)==s.eye(2)
    j=s.Matrix([[0,u],[s.conjugate(u),0]]);jb=s.conjugate(j);parity=s.diag(1,-1)
    gamma=s.simplify(i*s.kronecker_product(j*parity,jb))
    assert gamma*gamma==s.eye(4) and gamma==gamma.conjugate().T
    assert s.simplify(gamma*E0+E0)==s.zeros(4,2)
    assert s.simplify((s.eye(4)-gamma)/2-E0*E0.conjugate().T)==s.zeros(4)
    return {'graded_bilinear_gram':str(d),'induced_bra_embedding':str(dual),
            'restricted_ground_gram':'I_2','clifford_gamma':str(gamma),
            'projector':'(I-Gamma)/2',
            'scope':'Radial Hermitian reflection at real h,c and imaginary beta. Transport to other plumbing orientations remains explicit.'}


def coefficient_checks():
    low=[(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,0,1)]
    fixtures=[(ReflectedNSRRStateSewing(),list(itertools.product(low,repeat=2))),
              (ReflectedNSRRStateSewing(h=s.Rational(11,8),c=s.Rational(143,10),
                 beta2=s.I*s.Rational(3,5),beta3=s.I*s.Rational(9,10)),
               list(itertools.product([(1,1,0),(1,0,1),(0,1,1)],[(0,0,0),(1,0,0),(0,1,0)])))]
    rows=[]
    for index,(oracle,pairs) in enumerate(fixtures):
        triples=set(n for pair in pairs for n in pair)
        components={}
        for eta in (1,-1):
            chiral=HumanNSRRThetaOracle(central_charge=oracle.c,h_ns=oracle.h,
                beta_r1=oracle.betas[0],beta_r2=oracle.betas[1],form_parity=0,primary_parity=0,etas=(eta,eta))
            components[eta]={n:math.sqrt(2)*sum(z for parity,z in enumerate(chiral.coefficient_components(*n))
                                               if not parity&2) for n in triples}
        for n,m in pairs:
            direct=oracle.coefficient(n,m)
            for eta,etap in itertools.product((1,-1),repeat=2):
                expected=components[eta][n]*components[eta][m].conjugate() if eta==etap else 0j
                actual=direct[eta,etap]
                error=abs(actual-expected)/max(1,abs(actual),abs(expected))
                assert error<2e-12,(index,n,m,eta,etap,actual,expected)
                rows.append(dict(fixture=index,levels=n,anti_levels=m,eta_left=eta,eta_right=etap,
                    direct_real=actual.real,direct_imag=actual.imag,reduced_real=expected.real,
                    reduced_imag=expected.imag,scaled_error=error))
    return rows


def free_checks(inputs):
    from fixed_spin_free_plumbing import charged_frame,fixed_spin_chiral_partition
    from physical_free_plumbing_resummation import theta_charged_boson_resummation
    import numpy as np
    cfg=inputs.read(DATA/'nsrr_double_virasoro_N7_L5_20260911/config.json')
    original=tuple(map(complex,next(p for p in cfg['points'] if p['point_id']=='generic_04')['source']['q_values']))
    p0,p1=s.Rational(31,100),s.Rational(47,100)
    rows=[]
    for eta in (1,-1):
        charge0,charge1=p0,-eta*p1
        h=(charge0+charge1)**2/2
        oracle=HumanNSRRThetaOracle(central_charge=s.Rational(3,2),h_ns=h,
            beta_r1=s.I*p1/s.sqrt(2),beta_r2=s.I*p0/s.sqrt(2),form_parity=0,primary_parity=0,etas=(eta,eta))
        components={e:oracle.coefficient_components(e[0],e[1]//2,e[2]//2) for e in level_triples(6)}
        for scale in (.02,.1):
            q=tuple(scale*v for v in original);qs=q[::-1]
            weights=(h,s.Rational(1,16)+p1*p1/2,s.Rational(1,16)+p0*p0/2)
            primary=cmath.exp(sum(float(w)*cmath.log(z) for w,z in zip(weights,qs)))
            f=math.sqrt(2)*sum(complex(z)*math.prod(qs[k]**(e[k]/2) for k in range(3))
                for e,vector in components.items() for parity,z in enumerate(vector) if not parity&2)
            frame=charged_frame(q,max_mode=32)
            fermion=fixed_spin_chiral_partition(q,frame.omega_charge,((1,1),(0,0)),
                                               period_branch=np.zeros((2,2),dtype=int),max_mode=32)
            boson=theta_charged_boson_resummation(q,alpha_zero=float(charge0),alpha_one=float(charge1),max_mode=32)
            reference=complex(boson.chiral_value)*complex(fermion['majorana_chiral'])
            ratio=primary*f/reference
            # Total order three, with the next omitted total degree >=7/2.
            assert abs(ratio-1)<5e-7,(eta,scale,ratio)
            rows.append(dict(eta=eta,q_scale=scale,total_descendant_order=3,
                scalar_charges_geometry=[float(charge0),float(charge1),-float(charge0+charge1)],
                primary=encode(primary),descendant_block=encode(f),chiral_value=encode(primary*f),
                bosonized_chiral=encode(reference),norm_ratio=abs(ratio),
                phase_difference_rad=cmath.phase(ratio),complex_relative_error=abs(ratio-1)))
        print(f'Completed independent one-Majorana free check for eta={eta:+d}.',flush=True)
    return rows


def saved_low_level_checks(inputs):
    """Compare current PBW blocks with saved complex values before sewing."""
    rows=[]
    fixtures=(('nsrr_factorized_sign_trial_L3_N5_20260830',91),
              ('nsrr_factorized_sign_trial_L3_N5_20260830',153),
              ('nsrr_trial_L5_N3_local_20260830',13))
    for dataset,index in fixtures:
        directory=DATA/dataset
        config=inputs.read(directory/'summary.json')['config']
        paths=sorted((directory/'shards').glob('node-*.json'))
        shard=inputs.read(paths[index])
        b=s.Rational(str(config['b']))
        p0,p1,pns=map(lambda v:s.Rational(str(v)),shard['momenta_geometry'])
        c=s.Rational(3,2)+3*(b+1/b)**2
        h=(b+1/b)**2/8+pns**2/2
        coefficients={}
        for f,eta,etap in CHANNELS:
            oracle=HumanNSRRThetaOracle(central_charge=c,h_ns=h,
                beta_r1=s.I*p1/s.sqrt(2),beta_r2=s.I*p0/s.sqrt(2),
                form_parity=f,primary_parity=0,etas=(eta,etap))
            coefficients[f,eta,etap]={e:oracle.coefficient_components(e[0],e[1]//2,e[2]//2)
                                      for e in level_triples(2)}
        points={p['t']:p for p in config['points']}
        for saved in shard['rows']:
            level=float(saved['level'])
            if level>1:continue
            q=tuple(map(complex,points[saved['t']]['q_geometry']))[::-1]
            lifts=tuple(saved['lifts_geometry'])[::-1]
            for channel,z in zip(CHANNELS,saved['blocks']):
                expected=csum(complex(value)*math.prod(
                    q[k]**(e[k]/2)*lifts[k]**((parity>>k)&1) for k in range(3))
                    for e,vector in coefficients[channel].items() if sum(e)<=round(2*level)
                    for parity,value in enumerate(vector))
                actual=decode(z)
                error=abs(actual-expected)/max(1,abs(actual),abs(expected))
                assert error<2e-11,(dataset,index,channel,level,actual,expected)
                rows.append(dict(dataset=dataset,node_index=index,t=saved['t'],level=level,
                    lifts_geometry=saved['lifts_geometry'],channel=channel,
                    saved_real=actual.real,saved_imag=actual.imag,
                    PBW_real=expected.real,PBW_imag=expected.imag,scaled_error=error))
        print(f'Current PBW convention matches {dataset}, node {index}, through level 1.',flush=True)
    return rows


def saved_comparison(inputs):
    free={p['t']:p for p in inputs.read(DATA/'fixed_spin_free_NSrr_20260830/summary.json')['points']}
    target={r['t']:r for r in inputs.read(DATA/'nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.json')['rows']
            if r['quadrature_order']==5 and r['recursion_order']==16}
    output=[];primary_error=0.0;node_count=0
    for name in ('nsrr_factorized_sign_trial_L3_N5_20260830','nsrr_trial_L5_N3_local_20260830'):
        directory=DATA/name;config=inputs.read(directory/'summary.json')['config'];digest=object_digest(config)
        assert config['channels']==[list(v) for v in CHANNELS]
        tasks=[(n,i) for n in config['quadrature_orders'] for i in range(n**3)]
        paths=sorted((directory/'shards').glob('node-*.json'));assert len(paths)==len(tasks)
        sums={}
        for index,(path,task) in enumerate(zip(paths,tasks)):
            shard=inputs.read(path);node_count+=1
            assert (shard['index'],shard['quadrature_order'],shard['node'],shard['config_digest'])==(index,*task,digest)
            constants=tuple(decode(v) for v in shard['C_BRY'])
            by_row={(r['t'],float(r['level']),tuple(r['lifts_geometry'])):r for r in shard['rows']}
            assert len(by_row)==len(shard['rows'])==len(config['points'])*len(config['levels'])*len(config['lifts_geometry'])
            for point in config['points']:
                t=point['t'];assert point['q_geometry']==free[t]['source_NSrr']['q_values']
                expected_primary=NSRRPlumbingInputs(tuple(map(complex,point['q_geometry'])),(1,1,1),GEOMETRY_SECTORS).primary(config['b'],shard['momenta_geometry'])
                for level in config['levels']:
                    rows=[by_row[t,float(level),lift] for lift in SOURCE_FIXED_SPIN_LIFTS]
                    primary=decode(rows[0]['primary'])
                    assert primary==decode(rows[1]['primary'])
                    primary_error=max(primary_error,relative(primary,expected_primary))
                    f={channel:csum(decode(r['blocks'][a]) for r in rows)/math.sqrt(2) for a,channel in enumerate(CHANNELS)}
                    new=reflection_candidate(f,constants,constants,primary)
                    legacy=contract_physical_blocks({a:primary*z for a,z in f.items()},constants)['total']
                    key=(name,task[0],float(level),t)
                    terms=sums.setdefault(key,{k:[] for k in ('EE','OO','legacy')})
                    for k in ('EE','OO'):terms[k].append(shard['measure']*new['terms'][k])
                    terms['legacy'].append(shard['measure']*complex(legacy))
        kappa=1+2*(config['b']+1/config['b'])**2
        for (dataset,n,level,t),parts in sorted(sums.items()):
            ee,oo,old=[csum(parts[k]) for k in ('EE','OO','legacy')];z=ee+oo
            sq=z/free[t]['source_NSrr']['Z_free']**kappa
            tq=target[t]['target_Z']/free[t]['target_NSnsns']['Z_free']**kappa
            output.append(dict(dataset=dataset,t=t,source_order=level,momentum_order=n,
                source_Z_real=z.real,source_Z_imag=z.imag,EE_Z_real=ee.real,OO_Z_real=oo.real,
                source_Q_real=sq.real,source_Q_imag=sq.imag,target_Z=target[t]['target_Z'],target_Q=tq,
                ratio_norm=abs(sq/tq),ratio_phase_rad=cmath.phase(sq/tq),
                legacy_local_Z=old.real,relative_change_vs_legacy_local=(z/old).real-1))
    assert primary_error<2e-13
    return output,dict(input_nodes=node_count,maximum_primary_relative_error=primary_error)


def target_convention_control(inputs):
    """Record the independently saved obstruction to a label-only spin map."""
    path=DATA/'nsrr_spin_quadrature_t060_20260830/spin_basis_fivepoint.json'
    saved=inputs.read(path)
    target_config=inputs.read(DATA/'nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.json')['config']['baseline_config']
    target_points={p['t']:p['charts']['target_nsnsns'] for p in target_config['points']}
    free_points={p['t']:p['target_NSnsns'] for p in inputs.read(DATA/'fixed_spin_free_NSrr_20260830/summary.json')['points']}
    matrix=s.ones(4)/2-s.eye(4)
    assert matrix*matrix==s.eye(4)
    rows=[]
    for point in saved['points']:
        assert s.Matrix(point['free_even_sector_U'])==matrix.evalf()
        target=point['charts']['target']
        assert target['F_equals_U_D_max_absolute_error']<2e-12
        assert target_points[point['t']]['lifts']==target['numerator_lifts']
        assert relative(target['Z_free_desired'],free_points[point['t']]['Z_free'])<2e-12
        rows.append(dict(t=point['t'],numerator_lifts=target['numerator_lifts'],
            fixed_spin_unfiltered_lifts=target['fixed_spin_unfiltered_lifts'],
            filtered_over_fixed_spin_free=target['selected_filtered_over_desired_free'],
            saved_basis_identity_error=target['F_equals_U_D_max_absolute_error']))
    return dict(source=str(path.relative_to(ROOT)),rows=rows,
        free_basis_matrix='U=ones(4,4)/2-I; U^2=I',
        conclusion='The selected literal Human-Note lift has free control F=U D, not one raw fixed-spin D. This free identity does not by itself derive the interacting bra or transport matrix.',
        applied_to_interacting_target=False)


def report(output,result):
    """Write the numerical report directly from the audited results."""
    source3={r['t']:r for r in result['saved_comparison']
             if (r['source_order'],r['momentum_order'])==(3,5)}
    source5={r['t']:r for r in result['saved_comparison']
             if (r['source_order'],r['momentum_order'])==(5,3)}
    allrows={(r['dataset'],r['momentum_order'],r['source_order'],r['t']):r
             for r in result['saved_comparison']}
    convergence=[]
    for t in sorted(source3):
        name3=source3[t]['dataset'];name5=source5[t]['dataset']
        block0=allrows[name5,3,3,t]['source_Z_real']
        block1=allrows[name5,3,5,t]['source_Z_real']
        momentum0=allrows[name3,4,3,t]['source_Z_real']
        momentum1=allrows[name3,5,3,t]['source_Z_real']
        convergence.append(dict(t=t,L3_to_L5_at_N3=abs(block1/block0-1),
                                N4_to_N5_at_L3=abs(momentum1/momentum0-1)))
    write_csv(output/'convergence.csv',convergence)
    lines=[
        '# NSRR sewing with reflected physical states',
        '',
        'Date: 2026-09-15. This supersedes the legacy matrix **within the declared radial reflection frame**. The global interacting spin transport is still unresolved.',
        '',
        '## Result',
        '',
        'An independent state sum from the supplied three-point coefficients selects the even-three-form channels (0,+,+) and (0,−,−), with coefficients E_L E_R/4 and O_L O_R/4 in the saved two-lift block basis. The legacy assembly cancels the first NS half-level coefficient, although the physical Ward identities give a nonzero answer. A common normalization change cannot repair this.',
        '',
        'The corrected local assembly passes the direct state checks and the complex free scalar plus one-Majorana checks. Its ratio to the saved all-NS target is about 0.25. These are conditional cross-channel comparisons; they do not establish agreement of the same fixed-spin partition function.',
        '',
        '## Block and primary conventions',
        '',
        r'Write $\widehat F_f^{\eta\eta^\prime}=(F_f^{\eta\eta^\prime}(+++)+F_f^{\eta\eta^\prime}(+-+))/\sqrt2$, with lifts in geometry order $(0,1,\infty)$. Each $F$ is descendant-only, as defined in Human Notes/SCblock.tex. The even three-form $\widehat F_0$ includes NS half-integer descendant levels.',
        '',
        r'For $A=(f,\eta,\eta^\prime)$ and the contraction $\widehat F^T M\overline{\widehat F}$, the two nonzero entries are',
        '',
        r'$$M_{(0,+,+),(0,+,+)}=\frac{E_LE_R}{4},\qquad M_{(0,-,-),(0,-,-)}=\frac{O_LO_R}{4}.$$',
        '',
        r'The physical coefficients are $d_\pm=(E\pm O)/2$, and the chiral coefficients are $c_+=E/2$, $c_-=O/2$. The left and right constants are independent analytic products; their phases are not replaced by absolute values.',
        '',
        r'$$Z_s^{\rm refl}=\int_{\mathbb R_+^3}\frac{d^3p}{\pi^3}\,|P_s|^2\left[\frac{E_LE_R}{4}|\widehat F_0^{++}|^2+\frac{O_LO_R}{4}|\widehat F_0^{--}|^2\right].$$',
        '',
        r'$$P_s=\exp\!\left[h_R(p_0)\Log q_0+h_R(p_1)\Log q_1+h_{\rm NS}(p_\infty)\Log q_\infty\right],\quad h_{\rm NS}(p)=\frac{Q^2}{8}+\frac{p^2}{2},\quad h_R=h_{\rm NS}+\frac1{16},\quad Q=b+b^{-1}.$$',
        '',
        r'The target uses its own momenta and $q$ values, with all three weights $h_{\rm NS}$. No $q^h$ or $q^{-c/24}$ is absorbed into $F$ or $M$. The saved source primary factors agree exactly with the adapter at every recombined node.',
        '',
        '## Independent norm and phase checks',
        '',
        f"- {len(result['coefficient_checks'])} full physical-state coefficient comparisons: maximum scaled error {max(r['scaled_error'] for r in result['coefficient_checks']):.3e}.",
        f"- {len(result['saved_low_level_checks'])} saved complex-block values independently reproduced by the current PBW calculation through level 1, across all eight channels and all four saved lifts at three momentum nodes: maximum scaled error {max(r['scaled_error'] for r in result['saved_low_level_checks']):.3e}.",
        '- 2,160 exact Ward checks of the two Ramond Clifford intertwiners and the outgoing-bra phase; 16 exact physical-family trace checks.',
        '- The physical Ramond Gram matrix is restricted to the two-family module before inversion. The antiholomorphic NS descendant changes the intrinsic parity in the holomorphic Ward reduction.',
        '',
        r'The table compares the complex chiral value $P\widehat F_0^{\eta\eta}$ with an independent charged-Heisenberg scalar times the bosonized single-Majorana partition function. The Majorana branch is continued from the positive Ramond degeneration. The series has total descendant cutoff 3.',
        '',
        '| η | q scale | Norm ratio | Phase difference (rad) | Complex relative error |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in result['free_checks']:
        lines.append(f"| {row['eta']:+d} | {row['q_scale']:.2f} | {row['norm_ratio']:.13f} | {row['phase_difference_rad']:.3e} | {row['complex_relative_error']:.3e} |")
    lines += [
        '',
        'The small error decreases with the plumbing scale, consistently with the omitted descendants. This tests a chiral phase directly. The near-zero phase of an integrated nonchiral partition function would only test its reality.',
        '',
        '## Recomputed genus-two values',
        '',
        r'Use $\mathcal Q_\gamma=Z_\gamma/(Z_{{\rm free},\gamma})^\kappa$, with $b=1.4$, $\kappa=1+2Q^2=9.940408163265307$, and the saved corrected fixed-spin free denominator for each chart.',
        '',
        'The source table uses total descendant cutoff L=3 and N=5 momentum quadrature per edge. The target is the saved N=5, recursion-order-16 numerator, divided by the corrected free factor; its obsolete saved normalized Q is not used.',
        '',
        '| t | Source Z, L3/N5 | Source Q | Saved target Q | Q source / Q target |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for t,row in sorted(source3.items()):
        lines.append(f"| {t:.2f} | {row['source_Z_real']:.10e} | {row['source_Q_real']:.10e} | {row['target_Q']:.10e} | {row['ratio_norm']:.9f} |")
    lines += [
        '',
        'The second available complex dataset reaches L=5 at N=3. Its target remains N=5, so this is not a comparison at equal quadrature order.',
        '',
        '| t | Source Z, L5/N3 | Source Q | Q source / saved Q target |',
        '| ---: | ---: | ---: | ---: |',
    ]
    for t,row in sorted(source5.items()):
        lines.append(f"| {t:.2f} | {row['source_Z_real']:.10e} | {row['source_Q_real']:.10e} | {row['ratio_norm']:.9f} |")
    lines += [
        '',
        'Changing one cutoff at a time gives the following relative changes. They are convergence diagnostics, not error bounds.',
        '',
        '| t | L3 → L5, fixed N3 | N4 → N5, fixed L3 |',
        '| ---: | ---: | ---: |',
    ]
    for row in convergence:
        lines.append(f"| {row['t']:.2f} | {row['L3_to_L5_at_N3']:.3e} | {row['N4_to_N5_at_L3']:.3e} |")
    lines += [
        '',
        '## Remaining channel-convention issue',
        '',
        'The saved all-NS numerator uses a literal Human-Note lift. Its independent free-fermion control obeys F=U D, where U=ones(4,4)/2−I and D contains four raw fixed-spin chiral functions. Thus identifying that numerator with one spin structure requires a bra and spin-basis conversion. This free identity alone does not justify applying U to interacting blocks.',
        '',
        'At t=0.60 the selected filtered free partition function is 0.9527553293 times the desired fixed-spin free partition function; the numerical identity F=U D is accurate to 2.3e−16 there. This is a nonchiral partition-function ratio, not a chiral norm ratio. The selected numerator lift is (+,−,+), while the desired raw target spin uses (−,+,+), in geometry order. The individual chiral functions must be matched.',
        '',
        'BRY use πδ(P−P′) two-point normalization for both NS primaries and both Ramond families, equations (3.1) and (3.6). That convention does not supply a factor two per Ramond edge. Their presentation also distinguishes the local Ramond field from the defect Ramond field; the closed-surface defect prescription still has to be tracked. See [BRY, §3.1](https://arxiv.org/html/2201.05621#S3.SS1).',
        '',
        'The high-order scalar files have already contracted the legacy matrix and do not retain the individual complex blocks. They cannot be recombined with this new matrix. The new tables therefore use the complete L3/N5 and L5/N3 complex datasets. No factor four is fitted, and no production kernel is changed.',
        '',
        '## Derivation, files, and reproduction',
        '',
        f'- [Derivation of the reflected matrix and signs](<{ROOT}/Machine Notes/Genus 2/NSRR_REFLECTED_STATE_SEWING_2026-09-15.md>)',
        f'- [Two nonzero matrix entries](<{output}/matrix.csv>)',
        f'- [Independent state coefficients](<{output}/coefficient_checks.csv>)',
        f'- [Current PBW versus saved complex blocks](<{output}/saved_low_level_checks.csv>)',
        f'- [Complex free-field comparisons](<{output}/free_checks.csv>)',
        f'- [All saved-node comparisons, including EE and OO separately](<{output}/saved_comparison.csv>)',
        f'- [Convergence at fixed companion cutoff](<{output}/convergence.csv>)',
        f'- [Full audit](<{output}/summary.json>)',
        f'- [Input SHA-256 manifest](<{output}/provenance.json>)',
        '',
        'Run from the repository root with Python providing SymPy, NumPy and SciPy:',
        '',
        chr(96)*3+'sh',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\',
        '  python Code/genus_2/audit_nsrr_reflected_state_sewing.py',
        chr(96)*3,
        '',
    ]
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();inputs=Inputs()
    for path in (Path(__file__),ROOT/'Code/genus_2/nsrr_reflected_state_sewing.py',
                 ROOT/'Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py',
                 ROOT/'Code/double_virasoro/nsrr/nsrr_genus2_block.py',ROOT/'Code/genus_2/physical_nsrr_sewing.py',
                 ROOT/'Code/genus_2/fixed_spin_free_plumbing.py',ROOT/'Code/genus_2/physical_free_plumbing_resummation.py',
                 ROOT/'Code/genus_2/nsrr_plumbing_adapter.py',ROOT/'Human Notes/SCblock.tex',
                 ROOT/'Code/genus_2/recombine_saved_genus2_coefficient_ledger.py',
                 ROOT/'Code/genus_2/audit_nsrr_comparison_spin_basis.py',
                 ROOT/'Machine Notes/Genus 2/NSRR_REFLECTED_STATE_SEWING_2026-09-15.md'):
        inputs.bytes(path)
    result=dict(schema='nsrr-reflected-state-sewing-v1',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        status='reflection_frame_matrix_checked; global_interacting_spin_transport_not_certified',
        primary_definition='F descendant-only; P=exp(sum h_i Log q_i) outside F and M',
        sources=['https://arxiv.org/pdf/1012.2974','https://arxiv.org/pdf/0810.1203','https://arxiv.org/html/2201.05621'],
        exact=exact_half_level(),bra=bra_and_clifford(),clifford=clifford_form_checks(),coefficient_checks=coefficient_checks())
    print(f'Passed {len(result["coefficient_checks"])} independent physical-state coefficient comparisons.',flush=True)
    result['free_checks']=free_checks(inputs)
    result['saved_low_level_checks']=saved_low_level_checks(inputs)
    result['saved_comparison'],result['saved_checks']=saved_comparison(inputs)
    result['target_convention_control']=target_convention_control(inputs)
    result['production_kernel_modified']=False;result['fitted_normalization']=False
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for name in ('coefficient_checks','free_checks','saved_low_level_checks','saved_comparison'):write_csv(output/(name+'.csv'),result[name])
    write_csv(output/'matrix.csv',[dict(A=str(a),B=str(a),constant_product=k,coefficient='1/4',includes_primary=False)
        for a,k in (((0,1,1),'E_L*E_R'),((0,-1,-1),'O_L*O_R'))])
    report(output,result)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(sha256=dict(sorted(inputs.files.items()))),indent=2)+'\n')
    selected=[r for r in result['saved_comparison'] if (r['source_order'],r['momentum_order'])==(3,5)]
    print(json.dumps(dict(output=str(output),max_coefficient_error=max(r['scaled_error'] for r in result['coefficient_checks']),
        free_checks=[{k:r[k] for k in ('eta','q_scale','norm_ratio','phase_difference_rad','complex_relative_error')} for r in result['free_checks']],
        comparison=selected),indent=2))

if __name__=='__main__':main()
