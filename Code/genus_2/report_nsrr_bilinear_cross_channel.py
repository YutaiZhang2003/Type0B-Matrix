#!/usr/bin/env python3
"""Summarize the complete cross-channel experiment without fitting a scale."""
from pathlib import Path
from datetime import datetime, timezone
import cmath
import hashlib
import json
import math
import sys
import numpy as np
from scipy.special import roots_genlaguerre

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Code/genus_2'))
from recombine_saved_genus2_coefficient_ledger import Inputs,decode,encode,write_csv

OUT=ROOT/'Data Set/nsrr_bilinear_cross_channel_20260915'


def matrix(row):
    return np.array([[decode(z) for z in r] for r in row['density']])


def quadrature_checks(inputs):
    records=[]
    for name in ('nsrr_factorized_sign_trial_L3_N5_20260830','nsrr_trial_L5_N3_local_20260830'):
        directory=ROOT/'Data Set'/name
        config=inputs.read(directory/'summary.json')['config']
        for path in sorted((directory/'shards').glob('node-*.json')):
            row=inputs.read(path)
            records.append((config['q_envelope'],row['quadrature_order'],row['node'],row['momenta_geometry'],row['measure']))
    config=inputs.read(OUT/'config.json')
    for path in sorted((OUT/'target').rglob('node-*.json')):
        row=inputs.read(path)
        records.append((config['target_envelope'],row['N'],row['index'],row['momenta'],row['measure']))
    p_error=0.;w_error=0.
    for envelope,n,index,momenta,measure in records:
        x,w=roots_genlaguerre(n,-.5)
        ids=(index//n**2,(index//n)%n,index%n)
        computed=[];weight=1.
        for q,k in zip(envelope,ids):
            a=-math.log(abs(q));computed.append(math.sqrt(x[k]/a))
            weight*=w[k]*math.exp(x[k])/(2*math.pi*math.sqrt(a))
        p_error=max(p_error,max(abs(a-b) for a,b in zip(momenta,computed)))
        w_error=max(w_error,abs(weight/measure-1))
    assert p_error<1e-13 and w_error<1e-12
    return dict(nodes=len(records),maximum_momentum_absolute_error=p_error,
                maximum_measure_relative_error=w_error,measure='d^3p/pi^3 in both channels')


def main():
    inputs=Inputs()
    config=inputs.read(OUT/'config.json')
    source=inputs.read(OUT/'source.json');targets=inputs.read(OUT/'target_summary.json')
    initial=inputs.read(OUT/'comparison.json')
    assert initial['target_nodes']==91 and initial['target_global_failures']==0
    target={(r['N'],r['t'],r['order']):matrix(r) for r in targets}
    src={}
    repeated_error=0.
    for row in source:
        key=(row['N'],row['t'],row['L'])
        value=matrix(row)
        if key in src:
            repeated_error=max(repeated_error,float(np.max(abs(src[key]-value))/np.max(abs(value))))
        src[key]=value
    assert repeated_error<1e-10
    kappa=1+2*(config['b']+1/config['b'])**2
    geometry={p['t']:p for p in config['geometry']}
    def frame(t):
        g=geometry[t]
        return (g['source_free']/g['target_free'])**kappa
    def ratio(n,t,L,R,vector=None):
        s=src[n,t,L];a=target[n,t,R]
        if vector is None: zs,zt=np.trace(s),np.trace(a)
        else: zs,zt=vector@s@vector.conjugate(),vector@a@vector.conjugate()
        return complex(zs/zt/frame(t)),complex(zs),complex(zt)
    rows=[];maximum_imag=0.
    for n,t,L,R in [(3,t,5.,12) for t in geometry]+[(3,.6,3.,12),(3,.6,5.,8),(4,.6,3.,8),(4,.6,3.,12)]:
        for label,v in [('s=+1',np.array([1,-1j])/math.sqrt(2)),
                        ('s=-1',np.array([1,1j])/math.sqrt(2)),
                        ('resolved_00',np.array([1,0])),('resolved_11',np.array([0,1])),('spin_sum',None)]:
            r,zs,zt=ratio(n,t,L,R,v)
            maximum_imag=max(maximum_imag,abs(r.imag))
            rows.append(dict(t=t,N=n,source_L=L,target_R=R,comparison=label,
                source_Z=zs.real,target_Z=zt.real,free_frame_power=frame(t),
                ratio=r.real,norm_ratio=abs(r),phase_difference_rad=cmath.phase(r),
                relative_disagreement=abs(r-1)))
    write_csv(OUT/'summary.csv',rows)
    controls=[]
    for label,v in [('s=+1',np.array([1,-1j])/math.sqrt(2)),
                    ('s=-1',np.array([1,1j])/math.sqrt(2)),('spin_sum',None)]:
        base=ratio(3,.6,3.,12,v)[0]
        refined=ratio(3,.6,5.,12,v)[0]
        higher_n=ratio(4,.6,3.,12,v)[0]
        low_r=ratio(3,.6,5.,8,v)[0]
        controls.append(dict(comparison=label,source_L3_to_L5_relative_change=(refined/base-1).real,
            target_R8_to_R12_ratio_relative_change=(refined/low_r-1).real,
            joint_N3_to_N4_at_L3_relative_change=(higher_n/base-1).real))
    # Independent checks of basis normalization and positivity, plus the
    # identity Z_+ + Z_- = Tr K, which removes cross-spin interference.
    trace_residual=0.;hermitian_residual=0.
    for m in list(src.values())+list(target.values()):
        scale=np.max(abs(m));hermitian_residual=max(hermitian_residual,float(np.max(abs(m-m.conjugate().T))/scale))
        assert np.min(np.linalg.eigvalsh((m+m.conjugate().T)/2)) > -1e-12*scale
        v=np.array([1,-1j])/math.sqrt(2);w=v.conjugate()
        trace_residual=max(trace_residual,abs(v@m@v.conjugate()+w@m@w.conjugate()-np.trace(m))/scale)
    fresh=inputs.read(OUT/'fresh_source_checks.json')
    def source_scalar(row):
        v=np.array([1,-1j*row['s']])/math.sqrt(2)
        return abs(v@src[3,row['t'],row['L']]@v.conjugate())
    impact=max(row['absolute_change']/source_scalar(row) for row in fresh['weighted_changes'])
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='fails_cross_channel_agreement_at_tested_cutoffs',
        b=config['b'],kappa=kappa,target_nodes=initial['target_nodes'],
        five_point_tests=[r for r in rows if r['N']==3 and r['source_L']==5 and r['target_R']==12],
        controls=controls,quadrature_check=quadrature_checks(inputs),
        source_dataset_overlap_relative_error=repeated_error,
        maximum_density_hermiticity_error=hermitian_residual,
        maximum_spin_sum_identity_error=float(trace_residual),
        maximum_partition_ratio_imaginary_part=maximum_imag,
        maximum_relative_spin_transport_phase_error=max(abs(decode(g['measured_relative_spin_phase'])-1) for g in geometry.values()),
        source_checks=dict(complex_block_comparisons=fresh['checks'],
            maximum_scaled_block_difference=fresh['max_block_error'],
            maximum_single_checked_node_change_relative_to_integral=impact,
            scope=fresh['scope']),
        source_primary_max_relative_error=config['source_primary_max_relative_error'],
        interpretation='The local matrix checks do not establish a crossing-invariant partition function. Both the coherent combinations and their phase-independent sum fail the tested cross-channel comparison. No overall normalization is fitted or changed.',
        limits=['Finite cutoff differences are convergence observations, not rigorous integration error bounds.',
            'The off-diagonal interacting spin/fusion transport is tested using the minimal continuation of the independently verified free spin dictionary; it is not independently proved by that dictionary.',
            'The spin-sum test eliminates this relative-phase interference, but still tests the proposed global identification of the local sewing data.',
            'A failure of nonchiral assembly does not by itself falsify the independently checked chiral double-Virasoro construction.'])
    save=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    save(OUT/'result.json',result)
    original=inputs.read(OUT/'provenance.json')
    target_manifest=inputs.read(OUT/'target_provenance.json')
    for name,digest in {**original,**target_manifest}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name in ('Code/genus_2/report_nsrr_bilinear_cross_channel.py','Code/genus_2/check_cross_channel_source_blocks.py'):
        inputs.bytes(ROOT/name)
    save(OUT/'report_provenance.json',inputs.files)
    lines=['# New NSRR pairing: cross-channel test fails','',
        'The new local NSRR matrix does not yield cross-channel agreement in this experiment. '
        'The complex all-NS lift vector was recomputed and both even RR components were transported separately. '
        'The ratios remain close to one quarter. No fitted multiplier was applied.','',
        '## Comparison definition','',
        r'$\mathcal R=(Z_{\rm NSRR}/Z_{\rm NSNSNS})(Z_{{\rm free},t}/Z_{{\rm free},s})^\kappa$, '
        r'with $b=1.4$ and $\kappa=1+2(b+b^{-1})^2=9.940408163265307$. Agreement requires $\mathcal R=1$.','',
        'The source uses the proposed `B_L B_R/8 * [[1,s],[s,1]]` on the `r=-1` channels. '
        'In the raw spin basis this is the normalized combination `(D_00-i*s*D_11)/sqrt(2)`. '
        'The all-NS target is assembled from all four complex Human-Note lift blocks, with its exact sector-dependent '
        'quadratic-parity basis transform and the matching spin combination. Each pant retains its supplied BRY constants. '
        'The bilinear anti data specialize to complex conjugates only on the real physical slice.','',
        'The marked source spins `[11|00]`, `[11|11]` transport to target `[00|00]`, `[00|10]`. '
        'In the target charge marking these are raw lifts `(-,+,+)` and `(+,+,+)`. '
        'The independently continued single-Majorana roots give relative transport phase +1 across all five surfaces, '
        f'with maximum complex discrepancy `{result["maximum_relative_spin_transport_phase_error"]:.3e}`. '
        'The two matched free-field frame ratios agree. The other two RR characteristics map to a target with a Ramond edge and are not compared to an all-NS target.','',
        'Primary factors `exp(sum h_i Log q_i)` are outside blocks and matrices. '
        'The source R weights include 1/16; the target has three NS weights. '
        'The source and target measures are independently checked as `d^3p/pi^3`.','',
        '## Five complete, matched-quadrature comparisons','',
        'Source: total descendant cutoff L=5; target: recursion twice-level R=12 with resummed global blocks. '
        'Both channels use N=3 nodes per momentum direction.','',
        '| t | R(s=+1) | R(s=-1) | R(spin sum) |','|---:|---:|---:|---:|']
    for t in geometry:
        chosen={r['comparison']:r for r in rows if r['t']==t and r['N']==3 and r['source_L']==5 and r['target_R']==12}
        lines.append(f'| {t:.2f} | {chosen["s=+1"]["ratio"]:.9f} | {chosen["s=-1"]["ratio"]:.9f} | {chosen["spin_sum"]["ratio"]:.9f} |')
    lines+=['','The spin sum is the trace of the two-by-two integrated spin pairing: '
        '`Z_+ + Z_- = Z_00 + Z_11`. It contains no cross-spin interference. '
        'Its mismatch therefore cannot be repaired by changing only the relative Majorana phase.','',
        '## Cutoff controls at t=0.60','',
        '| Comparison | Source L3→L5 at N3 | Target R8→R12 at N3 | Joint N3→N4 at L3 |',
        '|---|---:|---:|---:|']
    for c in controls:
        lines.append(f'| {c["comparison"]} | {c["source_L3_to_L5_relative_change"]:.3e} | '
                     f'{c["target_R8_to_R12_ratio_relative_change"]:.3e} | {c["joint_N3_to_N4_at_L3_relative_change"]:.3e} |')
    central={r['comparison']:r for r in rows if r['N']==4 and r['target_R']==12}
    lines+=['',f'The N4, L3/R12 spin-sum ratio is `{central["spin_sum"]["ratio"]:.9f}`. '
        'These variations are much smaller than the roughly 75% discrepancy. They are convergence diagnostics, not rigorous remainder bounds.','',
        '## Independent numerical checks','',
        '- 91 complete fresh target nodes: 27 nodes at five surfaces, plus 64 nodes at the central surface. '
        'Every point retains both three-form sectors, four complex lifts, and both target recursion cutoffs. No partial-grid ratios are reported.',
        f'- {result["quadrature_check"]["nodes"]} source/target node measures reconstructed independently; maximum relative measure discrepancy '
        f'`{result["quadrature_check"]["maximum_measure_relative_error"]:.3e}`.',
        f'- 120 source-block spot comparisons against the current native double-Virasoro engine through L5. '
        f'The extreme tail reaches a scaled block difference of `{fresh["max_block_error"]:.3e}`; '
        f'the largest single checked node correction divided by the full integral is only `{impact:.3e}`. '
        'This is a three-node check, not a complete native recomputation.',
        '- Source primary-prefactor comparison is exact at saved precision; target primaries, weights and logarithms are recorded separately.',
        '- All target global sums converge. The spin-pairing matrices are Hermitian and positive within numerical precision.',
        '', '## What this does and does not establish','',
        '**The proposal fails this cross-channel test.** Its local BPZ and descendant checks remain local checks; '
        'they are insufficient to certify the full genus-two decomposition. The global pairing/defect prescription and relative channel normalization need to be revisited. '
        'The result does not isolate an error in the chiral double-Virasoro procedure, which was not modified.','',
        'The off-diagonal interacting transport uses the minimal extension of the checked free spin dictionary. '
        'That extension is a tested proposal, not an independent proof of the interacting fusion kernel. '
        'The separately reported spin-sum discrepancy survives elimination of the relative-phase interference.','',
        'BRY equations (3.6) and (3.8) retain the common `pi delta` two-point normalization and '
        '`C^±=(E±O)/2`; no change to these inputs is inferred from the numerical gap. '
        '[BRY §3.1](https://arxiv.org/html/2201.05621#S3.SS1).','',
        '## Artifacts and reproduction','',
        '- `result.json`, `summary.csv`: conclusions, five-point table and cutoff controls.',
        '- `comparison.json`, `comparison.csv`: full source/target combinations at the tested cutoffs.',
        '- `config.json`: geometry, spin transport, primary convention and implementation hashes.',
        '- `target/`: complete fresh complex all-NS block vectors, primaries and constants.',
        '- `source.json`: recombined saved complex NSRR data.',
        '- `fresh_source_checks.json`: native source spot checks and weighted impact.',
        '- `provenance.json`, `target_provenance.json`, `report_provenance.json`: input SHA-256 records.','',
        '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\',
        '  python Code/genus_2/test_nsrr_bilinear_cross_channel.py --workers 3',
        'python Code/genus_2/check_cross_channel_source_blocks.py',
        'python Code/genus_2/report_nsrr_bilinear_cross_channel.py','```','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],controls=controls,source_check_integral_impact=impact),indent=2))


if __name__=='__main__': main()
