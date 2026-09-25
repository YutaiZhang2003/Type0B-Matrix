"""Resumable imaginary-energy VV scan using the completed matched integrand.

Prepare all recursive tables before integrating. The reducer imports no MQM
prediction and retains numerical controls alongside every amplitude.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from audit_spin23_genus1_vv_completion import encode, integral_worker, modular
from refine_spin23_genus1_vv_disks import correction
from spin23_genus1_coefficient_bank import digest
from spin23_genus1_vv_bank import VVBank, merge_nodes, prepare_node, reference_laguerre_rule
from spin23_genus1_vv_matching import load_cusp_collision, puncture_tail
from spin23_genus1_vv_ope import OPEBank
from spin23_genus1_vv_tail_bank import TailBank

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(encode(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_files():
    files = set((ROOT/'Codes').glob('spin23*.py'))
    files.update((ROOT/'Codes').glob('audit_spin23_genus1_vv*.py'))
    files.update((ROOT/'Codes').glob('prepare_spin23_genus1_vv*.py'))
    files.add(ROOT/'Codes/refine_spin23_genus1_vv_disks.py')
    files.update((ROOT/'Codes').glob('*.py'))
    for package in ('Codes/ns_algebra', 'Codes/ramond_algebra', 'spin23_generated'):
        files.update((ROOT/package).rglob('*.py'))
    return sorted(files)


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files()}


def initialize(directory, config):
    directory = Path(directory)
    energies = config['kappas']
    if (not energies or len(set(energies)) != len(energies)
            or any(not np.isfinite(k) or not 0 < k < .5 for k in energies)):
        raise ValueError('distinct imaginary energies with 0<kappa<1/2 required')
    if config['workers'] != 200 or config['spectral_order'] != 16 or config['cutoff'] != 8:
        raise ValueError('this production profile uses 200 workers, order 16, and level four')
    manifest = dict(schema='spin23-vv-energy-scan-v1', config=config,
                    source_hashes=source_hashes(), target='genus-one vector two-point',
                    prescription='Euclidean matched pilot including angular disk correction',
                    prediction_used_in_integration=False)
    manifest['run_id'] = digest(manifest)
    path = directory/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('run directory belongs to another configuration or source snapshot')
    else:
        write_json(path, manifest)
    for ei in range(len(energies)):
        (directory/f'e{ei:02d}'/'nodes').mkdir(parents=True, exist_ok=True)
    return manifest


def load_run(directory):
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    saved = manifest.copy()
    run_id = saved.pop('run_id')
    if digest(saved) != run_id or source_hashes() != manifest['source_hashes']:
        raise ValueError('run provenance mismatch')
    return manifest


def assignments(task, workers, energy_count, node_count):
    if not 0 <= task < workers:
        raise ValueError('invalid worker index')
    for flat in range(task, energy_count*node_count, workers):
        ni, ei = divmod(flat, energy_count)
        yield ei, ni


def prepare(directory, task):
    from spin23_genus1_banked_integral import _clear_node_caches
    directory = Path(directory)
    manifest = load_run(directory); config = manifest['config']
    momenta, _ = reference_laguerre_rule(config['spectral_order'])
    for ei, ni in assignments(task, config['workers'], len(config['kappas']), len(momenta)):
        kappa = config['kappas'][ei]
        expected = dict(run_id=manifest['run_id'], energy=[0., kappa], node_index=ni)
        path = directory/f'e{ei:02d}'/'nodes'/f'p{ni:03d}.npz'
        if path.exists():
            bank = VVBank.load(path, expected=expected)
            if not np.array_equal(bank.momenta, [momenta[ni]]):
                raise ValueError('cached node momentum mismatch')
            continue
        started = time.perf_counter()
        try:
            bank = prepare_node(momenta[ni], 1j*kappa, cutoff=config['cutoff'])
            bank.metadata.update(expected)
            bank.save(path)
        except Exception as exc:
            write_json(directory/'failures'/f'e{ei:02d}_p{ni:03d}.json',
                       dict(expected, error_type=type(exc).__name__, error=str(exc)))
            raise
        finally:
            _clear_node_caches()
        print(json.dumps(dict(energy_index=ei, kappa=kappa, node=ni,
                              seconds=time.perf_counter()-started)), flush=True)
    load_run(directory)


def matched_row(bank, tail, ope, cusp, *, start, radius, order):
    """Same assembly as the completed pilot, including its fixed disk term."""
    row = integral_worker((bank, tail, ope, cusp, start, radius, order))
    op = OPEBank.load(ope)
    cp = load_cusp_collision(cusp, op.metadata['energy'])
    delta = correction(op, cp, radius=radius, start=start)
    row['total_before_angular_correction'] = row['total'].copy()
    row['angular_disk_correction'] = delta
    row['total'] = row['total']+delta
    return row


def fractional_difference(value, reference):
    return float(abs(value-reference)/max(abs(reference), 1e-300))


def finish(directory, ei):
    directory = Path(directory)
    manifest = load_run(directory); config = manifest['config']
    if not 0 <= ei < len(config['kappas']):
        raise ValueError('invalid energy index')
    root = directory/f'e{ei:02d}'; kappa = config['kappas'][ei]
    output = root/'amplitude.json'
    if output.exists():
        if json.loads(output.read_text())['run_id'] != manifest['run_id']:
            raise ValueError('completed energy belongs to another run')
        print('already integrated', ei, flush=True)
        return
    started = time.perf_counter()
    momenta, weights = reference_laguerre_rule(config['spectral_order'])
    bank_path = root/'bulk.npz'
    expected = dict(run_id=manifest['run_id'], energy=[0., kappa])
    if bank_path.exists():
        bank = VVBank.load(bank_path, expected=expected)
    else:
        banks = [VVBank.load(root/'nodes'/f'p{ni:03d}.npz',
                            expected=dict(expected, node_index=ni)) for ni in range(len(momenta))]
        bank = merge_nodes(banks, momenta, weights, dict(expected,
            spectral_rule=dict(method='reference_laguerre', order=config['spectral_order'],
                               reference_scale=float(np.pi)),
            aggregate_preparation_seconds=sum(b.metadata['preparation_seconds'] for b in banks)))
        bank.save(bank_path)
        del banks
    paths = {kind: root/f'{kind}.npz' for kind in ('ope', 'tail', 'cusp')}
    for kind, path in paths.items():
        # The existing offline preparer verifies energy, source, and quadrature
        # identity on resume. One worker respects this job's one-core allocation.
        subprocess.run([sys.executable, '-u', str(ROOT/'Codes/prepare_spin23_genus1_vv_completion.py'),
                        kind, '--omega', str(1j*kappa), '--workers', '1', '--output', str(path)],
                       check=True)
    for kind in ('ope', 'tail'):
        cls = OPEBank if kind == 'ope' else TailBank
        if cls.load(paths[kind]).metadata['energy'] != [0., kappa]:
            raise ValueError('auxiliary bank energy mismatch')
    controls = [(3., .08, 10), (3., .12, 10)]
    if ei in config['refinement_indices']:
        controls += [(3., .08, 14), (4., .08, 10)]
    rows = []
    for start, radius, order in controls:
        checkpoint = root/f'integral_Y{start:g}_r{radius:g}_n{order}.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['run_id'] != manifest['run_id']:
                raise ValueError('integral checkpoint belongs to another run')
            rows.append(saved['row'])
        else:
            row = matched_row(bank_path, paths['tail'], paths['ope'], paths['cusp'],
                              start=start, radius=radius, order=order)
            saved = dict(run_id=manifest['run_id'], row=encode(row))
            write_json(checkpoint, saved); rows.append(saved['row'])
        print(json.dumps(dict(energy_index=ei, controls=[start, radius, order],
                              total=rows[-1]['total'])), flush=True)
    tail = TailBank.load(paths['tail'])
    cusp = load_cusp_collision(paths['cusp'], [0., kappa])
    extended = puncture_tail(tail, cusp, start=1.+1e-12, radius=.08, order=16, s_max=88.)
    def total(row):
        return np.array(row['total']['real'])+1j*np.array(row['total']['imag'])
    base = total(rows[0]); value = base[-1]
    tail64 = rows[0]['resummed_strip_and_tail']['total']
    tail64 = complex(tail64['real'], tail64['imag'])
    diagnostics = dict(level_3_to_4=fractional_difference(base[0], value),
        radius_008_to_012=fractional_difference(total(rows[1])[-1], value),
        puncture_64_to_88=float(abs(extended['total']-tail64)/max(abs(value), 1e-300)))
    if len(rows) > 2:
        diagnostics['geometry_10_to_14'] = fractional_difference(total(rows[2])[-1], value)
        diagnostics['height_3_to_4'] = fractional_difference(total(rows[3])[-1], value)
    modular_rows = modular(bank)
    diagnostics['modular_max_component_residual'] = max(
        float(np.max(r['residual'][-1])) for r in modular_rows if r['method'] == 'factored')
    tolerance = config['diagnostic_relative_tolerance']
    flagged = {key: val for key, val in diagnostics.items() if val > tolerance}
    load_run(directory)
    write_json(output, dict(schema='spin23-vv-energy-amplitude-v1',
        run_id=manifest['run_id'], energy_index=ei, energy=[0., kappa],
        amplitude=value, rows=rows, extended_puncture_tail=extended,
        relative_controls=diagnostics, controls_over_tolerance=flagged,
        diagnostic_tolerance=tolerance, numerical_controls_passed=not flagged,
        modular_checks=modular_rows, seconds=time.perf_counter()-started,
        aggregate_bulk_preparation_seconds=bank.metadata.get('aggregate_preparation_seconds'),
        input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [bank_path, *paths.values()]},
        euclidean_amplitude_certified=False, prediction_used_in_integration=False))


def reduce(directory):
    directory = Path(directory); manifest = load_run(directory)
    rows = []
    for ei, kappa in enumerate(manifest['config']['kappas']):
        row = json.loads((directory/f'e{ei:02d}'/'amplitude.json').read_text())
        if row['run_id'] != manifest['run_id'] or row['energy'] != [0., kappa]:
            raise ValueError('amplitude provenance mismatch')
        rows.append(row)
    write_json(directory/'summary.json', dict(run_id=manifest['run_id'], energies=len(rows),
        target='V to V', rows=[{k: row[k] for k in ('energy', 'amplitude', 'relative_controls',
                                                   'controls_over_tolerance')} for row in rows],
        all_numerical_controls_passed=all(r['numerical_controls_passed'] for r in rows),
        euclidean_amplitude_certified=False, prediction_used_in_integration=False))
    temporary = directory/'amplitudes.pending.csv'
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['kappa', 'amplitude_real', 'amplitude_imag', 'level_relative',
                         'radius_relative', 'puncture_cutoff_relative', 'numerical_controls_passed'])
        for row in rows:
            d = row['relative_controls']
            writer.writerow([row['energy'][1], row['amplitude']['real'], row['amplitude']['imag'],
                             d['level_3_to_4'], d['radius_008_to_012'], d['puncture_64_to_88'],
                             row['numerical_controls_passed']])
    temporary.replace(directory/'amplitudes.csv')
    print('saved', directory/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('init', 'prepare', 'finish', 'reduce'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'init':
        if args.config is None: parser.error('init requires --config')
        initialize(args.directory, json.loads(args.config.read_text()))
    elif args.stage == 'prepare': prepare(args.directory, args.task)
    elif args.stage == 'finish': finish(args.directory, args.task)
    else: reduce(args.directory)


if __name__ == '__main__': main()
