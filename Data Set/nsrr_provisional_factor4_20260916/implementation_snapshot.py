#!/usr/bin/env python3
"""Test the explicit factor-four assumption using saved complex genus-two data.

Recombine complete q-independent L8 banks with the NEW bilinear matrix.
Keep the old quadrature study immutable. A fresh R16 all-NS control uses
the same central N3 grid as the saved R8/R12 blocks.
"""
from __future__ import annotations

import argparse
import cmath
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('Code', 'Code/genus_2', 'Code/c_Recursion',
                  'Code/full_ramond_block_runtime', 'Code/genus_2_cross_channel',
                  'Data Set/nsrr_momentum_integration_20260911'):
    sys.path.insert(0, str(ROOT / directory))
import numpy as np
from fast_constants import FastPositiveConstants
from nsrr_bilinear_sewing import CHANNELS
from nsrr_normalization import PROVISIONAL_FOUR, normalization_metadata, contract_normalized_nsrr
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
from recombine_saved_genus2_coefficient_ledger import Inputs, decode, encode, csum, object_digest, write_csv
import refine_nsrr_bilinear_quadrature as previous

DATA = ROOT / 'Data Set'
OUT = DATA / 'nsrr_provisional_factor4_20260916'
BASE = DATA / 'nsrr_bilinear_cross_channel_20260915'
QUAD = DATA / 'nsrr_bilinear_quadrature_20260915'
BANK = DATA / 'nsrr_moduli_extension_20260913/coefficient_bank/N7'
BUNDLE = DATA / 'nsrr_per_edge8_generic04_20260914/bundle'
LEVELS = (3, 4, 5, 6, 7, 8)
POLICY = normalization_metadata(PROVISIONAL_FOUR)
VECTORS = previous.VECTORS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def matrix(value):
    return np.array([[decode(z) for z in row] for row in value], complex)


def encoded(value):
    return [[encode(z) for z in row] for row in value]


def add(matrices):
    return np.array([[csum(m[i, j] for m in matrices) for j in range(2)] for i in range(2)])


def scalars(m):
    return {label: previous.scalar(m, v) for label, v in VECTORS.items()}


def density(blocks, constants, primary, measure):
    """Full spin matrix; use the normalized bilinear API as an independent check."""
    k = np.zeros((2, 2), complex)
    for j, eta in enumerate((1, -1)):
        d = np.array([blocks[0, eta, eta], 1j * blocks[1, eta, eta]])
        k += constants[j] ** 2 / 4 * np.outer(d, d.conjugate())
    return POLICY['normalization_factor'] * measure * abs(primary) ** 2 * k


def check_contraction(k, blocks, constants, primary, measure):
    for sign in (1, -1):
        result = contract_normalized_nsrr(descendant_blocks=blocks,
            antiholomorphic_blocks={key: z.conjugate() for key, z in blocks.items()},
            left_bry=constants, right_bry=constants, physical_lifts_slots=(sign, -1, 1),
            primary=primary, antiholomorphic_primary=primary.conjugate())
        v = VECTORS[f's={sign:+d}']
        assert abs(measure * result['total'] - v @ k @ v.conjugate()) < 5e-13 * np.linalg.norm(k)
        assert result['normalization_status'] == 'assumption'


def bank_convergence():
    inputs = Inputs()
    config = inputs.read(BASE / 'config.json')
    manifest = inputs.read(BANK / 'manifest.json')
    frozen = inputs.read(BUNDLE / 'config.json')
    assert config['b'] == frozen['b'] == 1.4
    assert config['parameters']['include_cosmological_prefactor'] is False
    assert manifest['level'] == 8 and manifest['requested_native_dps'] == 40
    assert len(frozen['nodes']) == 12 ** 3
    axes = manifest['nodes_and_weights']
    expected = {(a, b, c) for a in range(17) for b in range(0, 17-a, 2)
                for c in range(0, 17-a-b, 2)}
    constants_engine = FastPositiveConstants(config['b'])
    all_rows = []; checks = []; examples = []
    for grid, n in (('adapted_N7', 7), ('panel_N12', 12)):
        sums = {(g['t'], level): [] for g in config['geometry'] for level in LEVELS}
        absolute_steps = {(g['t'], level, label): [] for g in config['geometry']
                          for level in LEVELS[1:] for label in VECTORS}
        for index in range(n ** 3):
            if grid == 'adapted_N7':
                path = BANK / f'node-{index:05d}.npz'
                with np.load(io.BytesIO(inputs.bytes(path)), allow_pickle=False) as z:
                    ex, coefficients = z['exponents'], z['values']
                    meta = json.loads(str(z['metadata']))
                assert meta['manifest_digest'] == object_digest(manifest)
                assert meta['index'] == index and meta['level'] == 8
                assert meta['channel_order'] == list(map(list, CHANNELS))
                ids = np.unravel_index(index, (n,) * 3)
                momenta = [axes[e]['nodes'][j] for e, j in enumerate(ids)]
                measure = math.prod(axes[e]['weights'][j] for e, j in enumerate(ids))
                assert meta['momenta'] == momenta
                constants = tuple(map(complex, constants_engine.rr_ns_constants(momenta[1], momenta[0], momenta[2])))
            else:
                spec = frozen['nodes'][index]
                path = BUNDLE / spec['input']
                content = inputs.bytes(path)
                assert hashlib.sha256(content).hexdigest() == spec['sha256']
                with np.load(io.BytesIO(content), allow_pickle=False) as z:
                    ex, coefficients = z['exponents'], z['values']
                    momenta, constants = z['momenta'].tolist(), tuple(map(complex, z['norms']))
                assert spec['index'] == index and spec['momenta'] == momenta
                measure = spec['measure']
            assert set(map(tuple, ex)) == expected and coefficients.shape == (8, 285, 8)
            assert np.all(np.isfinite(coefficients)) and measure > 0
            projected = math.sqrt(2) * coefficients[:, :, [0, 1, 4, 5]].sum(axis=2)
            for g in config['geometry']:
                q = tuple(map(complex, g['q_source']))
                primary = NSRRPlumbingInputs(q, (1, 1, 1), GEOMETRY_SECTORS).primary(config['b'], momenta)
                # q powers use precisely the saved principal branch, in NS,R1,R0 order.
                monomials = np.exp(np.sum(ex * (np.log(np.asarray(q[::-1])) / 2), axis=1))
                last = None
                for level in LEVELS:
                    mask = ex.sum(axis=1) <= 2 * level
                    f = dict(zip(CHANNELS, np.sum(projected[:, mask] * monomials[mask], axis=1)))
                    k = density(f, constants, primary, measure)
                    sums[g['t'], level].append(k)
                    if last is not None:
                        for label, v in VECTORS.items():
                            delta = np.trace(k-last) if v is None else v @ (k-last) @ v.conjugate()
                            absolute_steps[g['t'], level, label].append(abs(delta))
                    last = k
                    if index in (0, n ** 3 // 2, n ** 3 - 1) and g['t'] == .6:
                        check_contraction(k, f, constants, primary, measure)
                        examples.append(dict(grid=grid, index=index, level=level, primary=encode(primary),
                            blocks=list(map(encode, f.values())), weighted_density=encoded(k)))
            if index % 256 == 0:
                print(json.dumps(dict(stage='saved_source_bank', grid=grid, done=index+1, total=n**3)), flush=True)
        for g in config['geometry']:
            last = None
            for level in LEVELS:
                k = add(sums[g['t'], level]); values = scalars(k)
                row = dict(grid=grid, nodes=n**3, t=g['t'], L=level, density=encoded(k), values=values,
                    off_diagonal_phase_rad=cmath.phase(k[0, 1]), maximum_relative_step=None,
                    relative_matrix_step=None, off_diagonal_phase_step_rad=None, absolute_nodal_relative_steps=None)
                if last is not None:
                    row.update(maximum_relative_step=max(abs(values[label]/last['values'][label]-1) for label in values),
                        relative_matrix_step=float(np.linalg.norm(k-matrix(last['density']))/np.linalg.norm(k)),
                        off_diagonal_phase_step_rad=cmath.phase(k[0, 1]/matrix(last['density'])[0, 1]),
                        absolute_nodal_relative_steps={label: math.fsum(absolute_steps[g['t'], level, label])/values[label]
                                                      for label in values})
                all_rows.append(row); last = row
        checks.append(dict(grid=grid, complete_nodes=n**3, coefficients_per_channel=285, channels=8))
    save(OUT / 'source_banks.json', dict(**POLICY, rows=all_rows, complete_grids=checks, examples=examples))
    save(OUT / 'source_provenance.json', inputs.files)


def target_control(index):
    config = previous.read(QUAD / 'config.json')
    config['target_order'] = 16
    path = OUT / 'target_R16_N3' / f'node-{index:04d}.json'
    design = dict(config=config, implementation_sha256={str(p.relative_to(ROOT)): previous.sha(p)
        for p in (Path(__file__), ROOT/'Code/genus_2/refine_nsrr_bilinear_quadrature.py',
                  ROOT/'Code/c_Recursion/ns_genus2_partition.py', ROOT/'Code/genus_2/all_ns_reflected_sewing.py')})
    digest = object_digest(design)
    if path.exists():
        assert previous.read(path)['design_digest'] == digest
        return
    value = previous.target_node(config, 3, index)
    save(path, dict(index=index, N=3, R=16, design_digest=digest, complete=True, **value))


def run_target(workers):
    def one(index):
        command = [sys.executable, str(Path(__file__)), '--target-index', str(index)]
        result = subprocess.run(command, text=True, capture_output=True, env={**os.environ,
            'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'PYTHONDONTWRITEBYTECODE':'1'})
        if result.returncode:
            raise RuntimeError(result.stderr)
        return index
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(one, i) for i in range(27)]), 1):
            print(json.dumps(dict(stage='target_R16_N3', done=done, total=27, index=future.result())), flush=True)


def report():
    inputs = Inputs()
    # Verify the historical convergence evidence, including all original node hashes.
    manifest = inputs.read(QUAD / 'final_provenance.json')
    historical_hashes = manifest.get('input_sha256', manifest)
    # The manifest is a flat path-to-hash mapping in the saved study.
    for name, digest in historical_hashes.items():
        assert hashlib.sha256(inputs.bytes(ROOT/name)).hexdigest() == digest, name
    for name, digest in inputs.read(OUT/'source_provenance.json').items():
        assert hashlib.sha256(inputs.bytes(ROOT/name)).hexdigest() == digest, name
    banks = inputs.read(OUT/'source_banks.json')
    final = inputs.read(QUAD/'final_result.json')
    last = inputs.read(QUAD/'target-refinement-N10.json')
    target = matrix(last['target_density']); frame = last['free_frame_power']
    history = []
    for n in range(3, 11):
        row = inputs.read(QUAD/(f'order-N{n}.json' if n <= 7 else f'target-refinement-N{n}.json'))
        zs = scalars(POLICY['normalization_factor'] * matrix(row['source_density']))
        zt = scalars(matrix(row['target_density']))
        for label in zs:
            history.append(dict(source_N=min(n, 7), target_N=n, source_L=5, target_R=12,
                comparison=label, source_Z=zs[label], target_Z=zt[label], ratio=zs[label]/zt[label]/frame))
    write_csv(OUT/'quadrature.csv', history)
    block_rows = []
    for row in banks['rows']:
        if row['t'] != .6:
            continue
        for label, value in row['values'].items():
            zt = scalars(target)[label]
            block_rows.append(dict(source_grid=row['grid'], source_L=row['L'], target_N=10,
                target_R=12, comparison=label, source_Z=value, target_Z=zt, ratio=value/zt/frame,
                maximum_relative_level_step=row['maximum_relative_step'],
                absolute_nodal_relative_step=(row['absolute_nodal_relative_steps'] or {}).get(label)))
    write_csv(OUT/'block_orders.csv', block_rows)
    # Complete N3 all-NS R16 control, compared with the saved complex R8/R12 matrices.
    nodes = [inputs.read(OUT/'target_R16_N3'/f'node-{i:04d}.json') for i in range(27)]
    assert all(r['index'] == i and r['complete'] and r['global_nonconverged_calls'] == 0 for i, r in enumerate(nodes))
    assert len({r['design_digest'] for r in nodes}) == 1
    target_orders = {r['order']: matrix(r['density']) for r in inputs.read(BASE/'target_summary.json')
                     if r['N'] == 3 and r['t'] == .6}
    target_orders[16] = add([matrix(r['density']) for r in nodes])
    target_steps = []
    for low, high in ((8, 12), (12, 16)):
        a, b = target_orders[low], target_orders[high]
        target_steps.append(dict(N=3, low_R=low, high_R=high,
            scalar_relative_changes={label: scalars(b)[label]/scalars(a)[label]-1 for label in VECTORS},
            relative_matrix_change=float(np.linalg.norm(b-a)/np.linalg.norm(b)),
            phase_change_rad=cmath.phase(b[0,1]/a[0,1])))
    best = next(r for r in banks['rows'] if r['t'] == .6 and r['L'] == 8 and r['grid'] == 'panel_N12')
    source = matrix(best['density'])
    results = {label: dict(source_Z=value, target_Z=scalars(target)[label],
                          ratio=value/scalars(target)[label]/frame,
                          residual=value/scalars(target)[label]/frame-1)
               for label, value in best['values'].items()}
    baseline = inputs.read(QUAD/'order-N7.json')
    panel5 = next(r for r in banks['rows'] if r['t'] == .6 and r['L'] == 5 and r['grid'] == 'panel_N12')
    adapted5 = next(r for r in banks['rows'] if r['t'] == .6 and r['L'] == 5 and r['grid'] == 'adapted_N7')
    independent_source_grid_changes = {
        'panel_N12_vs_Laguerre_N7_at_L5': {label: panel5['values'][label]/(4*baseline['values'][label]['source_Z'])-1 for label in VECTORS},
        'panel_N12_vs_adapted_N7_at_L5': {label: panel5['values'][label]/adapted5['values'][label]-1 for label in VECTORS}}
    result = dict(status='complete_conditional_normalization_test', **POLICY, b=1.4, t=.6,
        definition='R = (Z_NSRR / Z_allNS) (Z_free,target / Z_free,source)^kappa',
        primary_policy='Channel-dependent q^h outside descendant blocks and M; unchanged principal logarithms',
        source_grid='saved complete panel_N12, 1728 nodes', source_L=8, target_grid='Laguerre N10, 1000 nodes', target_R=12,
        values=results, quadrature_history=history, source_block_orders=block_rows, target_block_order_controls=target_steps,
        source_independent_grid_controls=independent_source_grid_changes,
        source_last_level_steps=[r for r in banks['rows'] if r['grid']=='panel_N12' and r['t']==.6 and r['L'] in (7,8)],
        original_L5_quadrature_status=final['status'], last_source_quadrature_steps=final['last_source_steps'],
        last_target_quadrature_steps=final['last_target_steps'],
        off_diagonal_phase_difference_rad=cmath.phase(source[0,1]/target[0,1]),
        trace_normalized_matrix_difference_frobenius=float(np.linalg.norm(source/np.trace(source)-target/np.trace(target))),
        phase_scope='arg(K_01) of the integrated spin matrix; positive factor four changes no phase. The scalar physical-slice partition functions are real and positive.',
        limitations=['Factor four is a user-requested assumption, not a derivation of the global sewing normalization.',
            'Interacting spin transport still uses the minimal extension of the checked free-spin dictionary.',
            'Target R12-to-R16 control is at N3; it is not an R16 recomputation of the N10 integral.',
            'Other four surfaces have higher source block data but only coarse N3 target quadrature.',
            'Observed changes are finite convergence diagnostics, not rigorous remainder bounds.'])
    save(OUT/'result.json', result)
    for path in (Path(__file__), ROOT/'Code/genus_2/nsrr_normalization.py', ROOT/'Code/genus_2/nsrr_bilinear_sewing.py',
                 ROOT/'Code/genus_2/nsrr_plumbing_adapter.py', ROOT/'Code/genus_2/nsrr_resummed_sewing.py'):
        inputs.bytes(path)
    save(OUT/'provenance.json', inputs.files)
    write_report(result, banks)
    print(json.dumps({k: result[k] for k in ('status','normalization_status','values','target_block_order_controls',
        'source_independent_grid_controls','off_diagonal_phase_difference_rad')}, indent=2), flush=True)


def write_report(result, banks):
    lines = ['# Provisional factor four: genus-two convergence test', '',
        '**Assumption:** `M_provisional = 4 M_local`. This changes the coefficient `B_L B_R/8` to '
        '`B_L B_R/2` in the new Human-Note bilinear NSRR matrix. It is applied once to the entire matrix, '
        'including interference. BRY constants, descendant blocks, spin phases, measure and channel-specific primary powers are unchanged.', '',
        'The local BPZ kernel remains the reference. No derivation of the extra global factor is claimed. '
        'The old interference matrix is not used.', '',
        r'Comparison: $R=(Z_{NSRR}/Z_{NSNSNS})(Z_{free,t}/Z_{free,s})^\kappa$, with $b=1.4$, '
        r'$\kappa=9.940408163265307$; agreement means $R=1$.', '',
        '## Quadrature at fixed source L5 / target R12', '',
        'Both source and target are complete grids. The source is held at N7 after its two successive '
        'changes fall below 1e-4. The target is continued to N10. These are the saved complex matrices, newly scaled by the explicit assumption.', '',
        '| Source N | Target N | R(s=+1) | R(s=-1) | R(spin sum) |', '|---:|---:|---:|---:|---:|']
    for n in range(3,11):
        rows={r['comparison']:r for r in result['quadrature_history'] if r['target_N']==n}
        lines.append(f'| {min(n,7)} | {n} | {rows["s=+1"]["ratio"]:.10f} | {rows["s=-1"]["ratio"]:.10f} | {rows["spin_sum"]["ratio"]:.10f} |')
    lines += ['', '## Saved block-order convergence', '',
        'All eight complex channel/parity coefficient tables are retained through total level 8 on both '
        'the adapted N7 grid (343 nodes) and panel N12 grid (1728 nodes). They are independent of q and '
        'were evaluated on the same five saved surfaces with the new bilinear pairing. The table fixes '
        'the central surface t=0.60 and the all-NS N10/R12 target, using panel N12 for the source.', '',
        '| Source L | R(s=+1) | R(s=-1) | R(spin sum) | Largest absolute nodal order change / Z |',
        '|---:|---:|---:|---:|---:|']
    for level in LEVELS:
        rows={r['comparison']:r for r in result['source_block_orders'] if r['source_grid']=='panel_N12' and r['source_L']==level}
        delta=max((r['absolute_nodal_relative_step'] or 0) for r in rows.values())
        lines.append(f'| {level} | {rows["s=+1"]["ratio"]:.10f} | {rows["s=-1"]["ratio"]:.10f} | {rows["spin_sum"]["ratio"]:.10f} | {delta:.3e} |')
    lines += ['', 'The absolute nodal diagnostic sums absolute changes before momentum integration, '
        'so cancellations between momentum nodes cannot conceal that measured finite-order difference.', '',
        '### Target order control', '', '| Target R step (N3) | Largest scalar relative change | Matrix relative change |',
        '|---|---:|---:|']
    for row in result['target_block_order_controls']:
        lines.append(f'| {row["low_R"]} to {row["high_R"]} | {max(map(abs,row["scalar_relative_changes"].values())):.3e} | {row["relative_matrix_change"]:.3e} |')
    lines += ['', '### Independent source quadrature control at L5', '']
    for label, values in result['source_independent_grid_controls'].items():
        lines.append(f'- `{label}`: largest scalar change `{max(map(abs,values.values())):.3e}`.')
    lines += ['', '## Result and phase', '', '| Comparison | Ratio at source L8 / target R12 | Residual (%) |', '|---|---:|---:|']
    for label, row in result['values'].items():
        lines.append(f'| {label} | {row["ratio"]:.10f} | {100*row["residual"]:+.6f} |')
    lines += ['', f'The off-diagonal integrated-spin-matrix phase difference is '
        f'`{result["off_diagonal_phase_difference_rad"]:.10g}` rad; the trace-normalized matrix distance is '
        f'`{result["trace_normalized_matrix_difference_frobenius"]:.6g}`. Multiplication by positive four leaves phases unchanged. '
        'These matrix phases are distinct from an overall chiral Pfaffian phase. The tested real-slice scalar partition functions are positive.', '',
        '**The factor removes the gross normalization discrepancy, but residual differences remain.** '
        'Block and quadrature convergence must not be identified with a proof of the normalization or the interacting spin transport.', '',
        '## Scope and reproduction', '']
    lines += ['- '+s for s in result['limitations']]
    lines += ['', '`result.json`, `quadrature.csv`, `block_orders.csv`, `source_banks.json` and `provenance.json` '
        'retain numerical results and input hashes. `target_R16_N3/` contains the complete fresh target control. '
        'The historical factor-one studies remain intact.', '',
        '```sh', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\',
        '  python3 Code/genus_2/check_nsrr_factor4_convergence.py --stage all --workers 4', '```', '',
        'For the pointwise resummed worker select `--sewing-convention human-bilinear '
        '--normalization provisional-times-four --physical-lifts-slots 1 -1 1` (or -1 -1 1 for the other tested sign). '
        'That legacy frozen worker retains its supplied target; use the spin-matrix comparison in this report for crossing.', '']
    (OUT/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('source','target','report','all'), default='all')
    parser.add_argument('--target-index', type=int, choices=range(27))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.target_index is not None:
        target_control(args.target_index)
        return
    if args.stage in ('source','all'): bank_convergence()
    if args.stage in ('target','all'): run_target(args.workers)
    if args.stage in ('report','all'): report()


if __name__ == '__main__':
    main()
