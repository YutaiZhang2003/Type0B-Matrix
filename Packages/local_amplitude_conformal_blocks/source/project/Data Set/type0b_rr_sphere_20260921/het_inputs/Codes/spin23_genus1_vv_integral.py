"""Resumable recursive coefficient preparation and four-variable VV integration.

Outputs retain spin and PCO components, correlated level differences, spectral
grid identity, and explicit cusp/necklace regulators. They are not certified
on-shell amplitudes until boundary completion and analytic continuation pass.
"""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from spin23_genus1_vv_bank import VVBank, VVGeometry, spectral_rule, reference_laguerre_rule, merge_nodes, evaluate_density
from spin23_genus1_vv_conventions import CONVENTION
from spin23_genus1_coefficient_bank import digest
from spin23_genus1_banked_integral import write_json, source_hashes, statistics, complex_json


@dataclass(frozen=True)
class Config:
    energies: tuple = (.25, .5, .75, 1.)
    imaginary_energies: bool = False
    cutoffs: tuple = (4, 6, 8)
    spectral_order: int = 4
    momentum_rule: str = 'legendre'
    reference_scale: float = math.pi
    p_max: float = 4.
    spectral_power: float = 1.25
    precision: int = 24
    replicates: int = 4
    samples: int = 4096
    seed: int = 19024
    batch_size: int = 64
    # Decreasing gap tests missing necklace strips; increasing Y tests the cusp.
    profiles: tuple = ((2., .12), (2., .06), (2., .03), (4., .12), (4., .06))

    def __post_init__(self):
        if not self.energies or any(not math.isfinite(e) or e <= 0 for e in self.energies):
            raise ValueError('positive external energies required')
        if type(self.imaginary_energies) is not bool:
            raise ValueError('imaginary_energies must be Boolean')
        if self.imaginary_energies and any(e >= .5 for e in self.energies):
            raise ValueError('this Euclidean pilot stays within 0<Im(omega)<1/2')
        if not self.cutoffs or any(type(c) is not int or c < 0 or c > 8 or c % 2 for c in self.cutoffs):
            raise ValueError('integer total levels through four required')
        if tuple(sorted(set(self.cutoffs))) != tuple(self.cutoffs):
            raise ValueError('increasing distinct cutoffs required')
        if type(self.samples) is not int or self.samples < 2 or self.samples & (self.samples-1):
            raise ValueError('power-of-two samples required')
        if type(self.replicates) is not int or self.replicates < 2:
            raise ValueError('two or more independent scrambled replicates required')
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('positive batch size required')
        if not self.profiles or any(not math.isfinite(y) or y <= 1 or not 0 <= gap < .5 for y, gap in self.profiles):
            raise ValueError('finite cusp cutoffs >1 and gaps in [0,.5) required')
        grid(self)

    @property
    def frequencies(self):
        return tuple(complex(0, e) if self.imaginary_energies else complex(e) for e in self.energies)


def load_config(raw):
    raw = dict(raw)
    for k in ('energies', 'cutoffs'): 
        if k in raw: raw[k] = tuple(raw[k])
    if 'profiles' in raw: raw['profiles'] = tuple(tuple(p) for p in raw['profiles'])
    return Config(**raw)


def grid(config):
    if config.momentum_rule == 'legendre':
        return spectral_rule(config.spectral_order, config.p_max, config.spectral_power)
    if config.momentum_rule == 'reference_laguerre':
        return reference_laguerre_rule(config.spectral_order, config.reference_scale)
    raise ValueError('unknown momentum rule')


def bank_config(config):
    return dict(energies=config.energies, imaginary_energies=config.imaginary_energies, cutoff=max(config.cutoffs),
        spectral_order=config.spectral_order, momentum_rule=config.momentum_rule,
        reference_scale=config.reference_scale, p_max=config.p_max if config.momentum_rule == 'legendre' else None,
        spectral_power=config.spectral_power, precision=config.precision)


def initialize(directory, config):
    directory = Path(directory)
    if (directory/'manifest.json').exists():
        old, _ = load_run(directory)
        if old != config: raise ValueError('existing run has different configuration')
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('nodes', 'banks', 'replicates', 'failures'): (directory/name).mkdir(exist_ok=True)
    hashes = source_hashes()
    manifest = dict(schema='spin23-vv-run-v1', config=asdict(config), source_sha256=hashes,
        bank_id=digest(dict(config=bank_config(config), source_sha256=hashes)))
    manifest['run_id'] = digest(manifest)
    write_json(directory/'manifest.json', manifest)


def load_run(directory):
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    if manifest['schema'] != 'spin23-vv-run-v1': raise ValueError('unsupported run')
    if manifest['source_sha256'] != source_hashes(): raise ValueError('source changed since initialization')
    return load_config(manifest['config']), manifest


def prepare_shard(directory, task_index=0, task_count=1):
    from spin23_genus1_vv_bank import prepare_node
    from spin23_genus1_banked_integral import _clear_node_caches
    config, manifest = load_run(directory); directory = Path(directory)
    if not 0 <= task_index < task_count: raise ValueError('invalid task partition')
    momenta, weights = grid(config)
    for flat in range(task_index, len(momenta)*len(config.energies), task_count):
        ei, ni = divmod(flat, len(momenta)); path = directory/'nodes'/f'e{ei}_p{ni}.npz'
        omega = config.frequencies[ei]
        expected = dict(bank_id=manifest['bank_id'], node_index=ni,
                        energy=[omega.real, omega.imag])
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.exists():
                bank = VVBank.load(path, expected=expected)
                if not np.array_equal(bank.momenta, [momenta[ni]]): raise ValueError('wrong momentum checkpoint')
                continue
            try:
                bank = prepare_node(momenta[ni], omega, cutoff=max(config.cutoffs), precision=config.precision)
                bank.weights[0] = weights[ni]; bank.metadata.update(expected); bank.save(path)
                print(f'prepared {path.name} in {bank.metadata["preparation_seconds"]:.2f}s', flush=True)
            except Exception as exc:
                write_json(directory/'failures'/f'e{ei}_p{ni}.json', dict(expected,
                    momentum=momenta[ni].tolist(), error_type=type(exc).__name__, error=str(exc)))
                raise
            finally: _clear_node_caches()


def merge_banks(directory):
    config, manifest = load_run(directory); directory = Path(directory)
    momenta, weights = grid(config)
    for ei, energy in enumerate(config.frequencies):
        banks = [VVBank.load(directory/'nodes'/f'e{ei}_p{ni}.npz', expected=dict(
            bank_id=manifest['bank_id'], node_index=ni, energy=[energy.real, energy.imag])) for ni in range(len(momenta))]
        merged = merge_nodes(banks, momenta, weights, dict(bank_id=manifest['bank_id'],
            spectral_rule=dict(order=config.spectral_order, method=config.momentum_rule,
                reference_scale=config.reference_scale,
                p_max=config.p_max if config.momentum_rule == 'legendre' else None, power=config.spectral_power),
            all_node_checks=[b.metadata['ramond_checks'] for b in banks],
            preparation_seconds=sum(b.metadata['preparation_seconds'] for b in banks)))
        merged.save(directory/'banks'/f'e{ei}.npz')


def map_geometry(unit, profile):
    """Pull back d^2tau d^2z, including every rejected sample in the denominator."""
    u = np.asarray(unit, float); upper, gap = profile
    if u.ndim != 2 or u.shape[1] != 4 or not np.isfinite(u).all() or np.any((u <= 0) | (u >= 1)):
        raise ValueError('four open-unit coordinates required')
    norm = math.pi/3-1/upper
    x = np.sin(-math.pi/6+u[:, 0]*math.pi/3)
    for _ in range(12):
        residual = np.arcsin(x)+math.pi/6-(x+.5)/upper-u[:, 0]*norm
        x = np.clip(x-residual/(1/np.sqrt(1-x*x)-1/upper), -.5, .5)
    if np.max(abs(np.arcsin(x)+math.pi/6-(x+.5)/upper-u[:, 0]*norm)) > 2e-14:
        raise ArithmeticError('fundamental-domain inverse CDF failed')
    y = 1/(1/np.sqrt(1-x*x)-u[:, 1]*(1/np.sqrt(1-x*x)-1/upper))
    tau = x+1j*y; points = np.column_stack((np.zeros(len(u)), u[:, 2]+u[:, 3]*tau))
    accepted = (u[:, 3] >= gap) & (u[:, 3] <= 1-gap)
    return tau, points, norm*y**3, accepted


def integrate_replicate(directory, replicate):
    config, manifest = load_run(directory); directory = Path(directory)
    if not 0 <= replicate < config.replicates: raise ValueError('invalid replicate')
    output = directory/'replicates'/f'r{replicate}.json'
    if output.exists():
        if json.loads(output.read_text())['run_id'] != manifest['run_id']: raise ValueError('wrong existing replicate')
        return
    banks = [VVBank.load(directory/'banks'/f'e{ei}.npz', expected=dict(bank_id=manifest['bank_id'], energy=[e.real, e.imag]))
             for ei, e in enumerate(config.frequencies)]
    unit = qmc.Sobol(d=4, scramble=True, seed=config.seed+replicate).random_base2(int(math.log2(config.samples)))
    total = np.zeros((len(config.profiles), len(banks), len(config.cutoffs), 3, 2), complex)
    accepted_counts = np.zeros(len(config.profiles), int)
    started = time.perf_counter()
    for first in range(0, len(unit), config.batch_size):
        chunk = unit[first:first+config.batch_size]
        for pi, profile in enumerate(config.profiles):
            tau, points, weight, accepted = map_geometry(chunk, profile)
            accepted_counts[pi] += accepted.sum()
            if not accepted.any(): continue
            geom = VVGeometry(tau[accepted], points[accepted])
            for ei, bank in enumerate(banks):
                values = evaluate_density(bank, geom, config.cutoffs)
                total[pi, ei] += np.einsum('glsc,g->lsc', values, weight[accepted])
    write_json(output, dict(run_id=manifest['run_id'], replicate=replicate,
        physical_component_convention=CONVENTION,
        samples=config.samples, accepted_counts=accepted_counts.tolist(),
        mean=complex_json(total/config.samples), seconds=time.perf_counter()-started))


def decode_complex(value):
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def reduce_run(directory):
    config, manifest = load_run(directory); directory = Path(directory)
    records = [json.loads((directory/'replicates'/f'r{i}.json').read_text()) for i in range(config.replicates)]
    for i, r in enumerate(records):
        if r['run_id'] != manifest['run_id'] or r['replicate'] != i or r['samples'] != config.samples:
            raise ValueError('incompatible replicate')
    values = np.stack([decode_complex(r['mean']) for r in records])
    total = values.sum(axis=(-1, -2))
    result = dict(schema='spin23-vv-regulated-integral-v1', run_id=manifest['run_id'], config=asdict(config),
        physical_component_convention=CONVENTION,
        scope='Regulated necklace integrals; boundary completion and physical analytic continuation remain unverified',
        axes=['profile', 'energy', 'total_twice_level', 'spin', 'PCO'],
        spin_labels=['NS', 'NS_tilde', 'R'], PCO_labels=['GG/PP', 'PP/PP'],
        components=statistics(values), total=statistics(total),
        adjacent_level_difference=statistics(np.diff(total, axis=-1)),
        profile_difference_from_first=statistics(total-total[:, :1]),
        accepted_counts=[r['accepted_counts'] for r in records],
        integration_seconds=sum(r['seconds'] for r in records),
        physical_amplitude_certified=False)
    write_json(directory/'results.json', result)
    return result


def modular_audit(directory):
    """Independent complete-spectral-sum overlap; does not enable a new chart."""
    config, manifest = load_run(directory); directory = Path(directory)
    tests = ((1j, .249+.5j), (.1+1.25j, .23+.55j))
    records = []
    for tau, z in tests:
        ts = -1/tau; zs = z/tau
        zs -= math.floor(zs.imag/ts.imag)*ts; zs -= math.floor(zs.real)
        g0 = VVGeometry([tau], [[0, z]]); g1 = VVGeometry([ts], [[0, zs]])
        for ei, e in enumerate(config.frequencies):
            bank = VVBank.load(directory/'banks'/f'e{ei}.npz', expected=dict(bank_id=manifest['bank_id'], energy=[e.real, e.imag]))
            a = evaluate_density(bank, g0, config.cutoffs)[0]
            b = evaluate_density(bank, g1, config.cutoffs)[0][:, (0, 2, 1)]*abs(tau)**-6
            scale = np.maximum(abs(a), abs(b))
            relative = abs(a-b)/np.maximum(scale, 1e-300)
            records.append(dict(tau=[tau.real, tau.imag], z=[z.real, z.imag], energy=[e.real, e.imag],
                original=complex_json(a), pulled_back=complex_json(b),
                relative_component_difference=relative.tolist(),
                relative_total_difference=(abs(a.sum(axis=(-1,-2))-b.sum(axis=(-1,-2)))/
                    np.maximum(abs(a.sum(axis=(-1,-2))), 1e-300)).tolist()))
    write_json(directory/'modular_audit.json', dict(run_id=manifest['run_id'],
        physical_component_convention=CONVENTION,
        production_chart_enabled=False, records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('init', 'prepare', 'merge', 'integrate', 'reduce', 'modular'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--task-count', type=int, default=1)
    args = parser.parse_args()
    if args.stage == 'init': initialize(args.run_dir, load_config(json.loads(args.config.read_text())) if args.config else Config())
    elif args.stage == 'prepare': prepare_shard(args.run_dir, args.task_index, args.task_count)
    elif args.stage == 'merge': merge_banks(args.run_dir)
    elif args.stage == 'integrate': integrate_replicate(args.run_dir, args.task_index)
    elif args.stage == 'reduce': reduce_run(args.run_dir)
    elif args.stage == 'modular': modular_audit(args.run_dir)


if __name__ == '__main__': main()
