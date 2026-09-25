"""Reproducible bounded SO(23) five-point pilot; generated JSON is not a fit."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from spin23_fivepoint_density import SToFourVectors
from spin23_fivepoint_integration import (spectral_density, integrate_patch_qmc, PatchSettings,
    global_importance_pilot, PilotBudgetExceeded)
from spin23_fivepoint_type0b_reference import source_manifest


ENERGIES = (.02+.08j, .03+.09j, .04+.10j, .05+.11j)
X, Y = .17+.08j, .52-.11j


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def local_manifest():
    names = ('spin23_fivepoint_type0b_reference.py', 'spin23_fivepoint_c_elliptic.py',
        'spin23_fivepoint_density.py', 'spin23_fivepoint_integration.py', 'benchmark_spin23_1to4.py',
        'spin23_sphere_fivepoint.py', 'spin23_super_liouville_data.py',
        'heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py')
    return {n: hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('density', 'patch', 'global'))
    parser.add_argument('--counts', default='12,13')
    parser.add_argument('--cut', default='4,4', help='maximum twice-levels, not physical levels')
    parser.add_argument('--precision', type=int, default=40)
    parser.add_argument('--sample-power', type=int, default=4)
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--collar', type=float, default=.08)
    parser.add_argument('--ope-depth', type=int, default=4)
    parser.add_argument('--wall-seconds', type=float, default=180)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new artifact name')
    counts, cut = (tuple(map(int, s.split(','))) for s in (args.counts, args.cut))
    started = perf_counter()
    result = dict(schema='hetso23_1to4_exploratory_v1', theory='HetSO(23)', process='S_to_VVVV',
        target_relative_error=.03, amplitude_qualified=False, energies=ENERGIES,
        source_manifest=source_manifest(), local_sources=local_manifest())
    if args.mode == 'density':
        rows = []
        for order, x, y, jacobian in ((tuple(range(5)), X, Y, 1.),
                ((2, 3, 4, 1, 0), 1-Y, X/(X-1), abs(1/(X-1)**2)**2)):
            def progress(row):
                if row['completed'] % 80 == 0:
                    print(json.dumps(dict(ordering=order, **row)), flush=True)
            row = spectral_density(ENERGIES, x, y, ordering=order, counts=counts,
                maximum_twice_levels=cut, precision=args.precision, progress=progress)
            row['area_to_base'] = jacobian
            row['base_values'] = row['values']*jacobian
            rows.append(row)
        a, b = (r['base_values'] for r in rows)
        result.update(observable='momentum_integrated_density_NOT_moduli_integrated_amplitude',
            rows=rows, channel_ratio=b/a, channel_gap_absolute=np.abs(b-a),
            channel_gap_relative=np.abs(b-a)/np.maximum(np.abs(a), np.abs(b)),
            convergence_certified=False)
    elif args.mode == 'patch':
        momenta = (.37, .61)
        bank = SToFourVectors(ENERGIES, momenta, maximum_twice_levels=cut, precision=args.precision)
        settings = PatchSettings(sample_power=args.sample_power, replicates=args.replicates,
            collar=args.collar, maximum_increment=args.ope_depth)
        arrays = []
        for j, engine in enumerate(bank.densities):
            arrays.append(integrate_patch_qmc(engine, settings))
            print(json.dumps(dict(tensor=j, elapsed_seconds=perf_counter()-started)), flush=True)
        parts = np.stack(arrays, axis=1)
        values = parts.sum(axis=-1)
        se = np.sqrt(np.sum(np.abs(values-values.mean(axis=0))**2, axis=0)/(args.replicates*(args.replicates-1)))
        result.update(observable='single_chart_fixed_momenta_collar_integral_NOT_amplitude',
            complete_global_integral=False, momenta=momenta, ordering=tuple(range(5)),
            maximum_twice_levels=cut, settings=asdict(settings), replicate_parts=parts,
            chart_subtotal=values.mean(axis=0), chart_subtotal_standard_error=se,
            beta=[[complex(b) for b in e.betas] for e in bank.densities],
            estimated_reference_global_chart_nodes=120*counts[0]*counts[1])
    else:
        settings = PatchSettings(sample_power=args.sample_power, replicates=args.replicates,
            collar=args.collar, maximum_increment=args.ope_depth)
        def progress(row):
            if row['completed_draws'] % 8 == 0:
                print(json.dumps(row), flush=True)
        try:
            result.update(global_importance_pilot(ENERGIES, counts=counts, maximum_twice_levels=cut,
                precision=args.precision, settings=settings, wall_seconds=args.wall_seconds, progress=progress))
        except PilotBudgetExceeded as exc:
            result.update(exc.partial)
    result['elapsed_seconds'] = perf_counter()-started
    # Generated numerical artifact, not a hand-maintained source edit.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(json_value(result), stream, indent=2, allow_nan=False)
        stream.write('\n')
    summary = {k: result[k] for k in ('observable', 'elapsed_seconds', 'complete', 'values', 'standard_errors',
        'channel_ratio', 'channel_gap_relative', 'accuracy') if k in result}
    print('RESULT '+json.dumps(json_value(summary)), flush=True)


if __name__ == '__main__':
    main()
