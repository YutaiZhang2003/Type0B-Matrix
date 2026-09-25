"""Prepare -> merge -> six-dimensional integrate -> reduce, with resumable banks.

All coefficients use recursion. Geometry workers load complete arrays and never
import coefficient builders. The object is a regulated necklace integral, with
independent spectral-order/tail and boundary checks still required.
"""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy.stats import qmc

from spin23_genus1_coefficient_bank import CoefficientBank, digest, merge_nodes, spectral_rule
from spin23_genus1_banked_geometry import GeometryBatch, evaluate_density


@dataclass(frozen=True)
class BankedConfig:
    energies: tuple = ((.25, .35), (.3, .3), (1/3, 2/3), (.5, .5))
    cutoffs: tuple = (6, 8)
    spectral_order: int = 3
    p_max: float = 4.
    spectral_power: float = 1.25
    precision: int = 24
    replicates: int = 4
    samples: int = 4096
    seed: int = 19023
    tau2_max: float = 2.
    gap: float = .12
    distance: float = .05
    batch_size: int = 64

    def __post_init__(self):
        if not self.energies or any(len(e)!=2 or any(not math.isfinite(x) or x<=0 for x in e) for e in self.energies):
            raise ValueError('positive outgoing energies required')
        if not self.cutoffs or tuple(sorted(set(self.cutoffs)))!=tuple(self.cutoffs) or any(
                type(c) is not int or c<0 or c>8 or c%2 for c in self.cutoffs):
            raise ValueError('increasing integer total levels, at most four, required')
        if type(self.samples) is not int or self.samples<2 or self.samples&(self.samples-1):
            raise ValueError('sample count must be a power of two >=2')
        if type(self.replicates) is not int or self.replicates<2:
            raise ValueError('at least two scrambled replicates required')
        if not math.isfinite(self.tau2_max) or self.tau2_max<=1 or not 0<=self.gap<1/3 or not math.isfinite(self.distance) or self.distance<0:
            raise ValueError('invalid regulators')
        if type(self.batch_size) is not int or self.batch_size<1:
            raise ValueError('positive batch size required')
        spectral_rule(self.spectral_order, self.p_max, self.spectral_power)

    @property
    def orientations(self):
        return tuple(dict.fromkeys(tuple(e) for pair in self.energies for e in (pair, pair[::-1])))


def source_hashes():
    """Verify a deployed snapshot once per worker; hash local dependencies otherwise."""
    root=Path(__file__).resolve().parents[1]
    manifest=root/'source_manifest.json'
    if manifest.exists():
        hashes=json.loads(manifest.read_text())
    else:
        names=list((root/'Codes').glob('spin23*.py'))+list((root/'Codes').glob('audit_spin23_genus1*.py'))
        names+=list((root/'Codes').rglob('*.py'))
        hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in names}
    for name, expected in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'source changed: {name}')
    return hashes


def write_json(path, value):
    path=Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary=Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True); raise
    os.replace(temporary, path)


def initialize(directory, config):
    directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
    manifest=dict(schema='spin23-svv-banked-run-v1', config=asdict(config),
                  source_sha256=source_hashes(), channel='necklace', block_backend='recursion')
    # Geometry sampling, regulators, batching, and the requested lower prefixes
    # do not change the coefficient bank. Preserve reuse across such campaigns.
    manifest['bank_id']=digest(dict(energies=config.orientations,cutoff=max(config.cutoffs),
        spectral_order=config.spectral_order,p_max=config.p_max,spectral_power=config.spectral_power,
        precision=config.precision,source_sha256=manifest['source_sha256'],channel='necklace'))
    manifest['run_id']=digest(manifest)
    path=directory/'run.json'
    if path.exists():
        if json.loads(path.read_text())!=json.loads(json.dumps(manifest)):
            raise ValueError('existing run has different configuration or sources; use a new directory')
    else:
        for child in ('nodes', 'banks', 'geometry', 'failures'):
            (directory/child).mkdir(exist_ok=True)
        write_json(path, manifest)
    return manifest


def load_run(directory):
    manifest=json.loads((Path(directory)/'run.json').read_text())
    payload={k:v for k,v in manifest.items() if k!='run_id'}
    if digest(payload)!=manifest['run_id'] or source_hashes()!=manifest['source_sha256']:
        raise ValueError('run manifest integrity or source mismatch')
    options=dict(manifest['config']); options['energies']=tuple(map(tuple, options['energies']))
    options['cutoffs']=tuple(options['cutoffs'])
    return BankedConfig(**options), manifest


def _clear_node_caches():
    # External branch data depend only on energy and can usefully survive.
    from spin23_genus1_recursive_sewing import ns_coefficients, ramond_batch, ramond_coefficients
    from spin23_genus1_branch_recursion import ordinary_coefficients, vertex
    for function in (ns_coefficients, ramond_batch, ramond_coefficients, ordinary_coefficients, vertex):
        function.cache_clear()


def prepare_shard(directory, task_index, task_count):
    from spin23_genus1_coefficient_bank import prepare_node
    config, manifest=load_run(directory); directory=Path(directory)
    momenta, weights=spectral_rule(config.spectral_order, config.p_max, config.spectral_power)
    if not 0<=task_index<task_count: raise ValueError('invalid worker partition')
    n=len(momenta)
    for flat in range(task_index, n*len(config.orientations), task_count):
        ei, ni=divmod(flat, n); path=directory/'nodes'/f'e{ei}_p{ni}.npz'
        expected=dict(bank_id=manifest['bank_id'], energy=list(config.orientations[ei]), node_index=ni)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if path.exists():
                bank=CoefficientBank.load(path, expected=expected)
                if not np.array_equal(bank.momenta, [momenta[ni]]): raise ValueError('wrong momentum checkpoint')
                continue
            started=time.perf_counter()
            try:
                bank=prepare_node(momenta[ni], config.orientations[ei], cutoff=max(config.cutoffs), precision=config.precision)
                bank.metadata.update(expected)
                bank.weights[0]=weights[ni]
                bank.save(path)
            except Exception as exc:
                write_json(directory/'failures'/f'e{ei}_p{ni}.json', dict(expected,
                    momentum=momenta[ni].tolist(), error_type=type(exc).__name__, error=str(exc)))
                raise
            finally:
                _clear_node_caches()
            print(f'prepared {path.name} in {time.perf_counter()-started:.2f}s', flush=True)


def merge_banks(directory):
    config, manifest=load_run(directory); directory=Path(directory)
    momenta, weights=spectral_rule(config.spectral_order, config.p_max, config.spectral_power)
    for ei, energy in enumerate(config.orientations):
        banks=[CoefficientBank.load(directory/'nodes'/f'e{ei}_p{ni}.npz', expected=dict(
            bank_id=manifest['bank_id'], energy=list(energy), node_index=ni)) for ni in range(len(momenta))]
        metadata=dict(bank_id=manifest['bank_id'], spectral_rule=dict(order=config.spectral_order,
            p_max=config.p_max, power=config.spectral_power, measure='product(dP/pi)'),
            all_node_checks=[b.metadata['ramond_checks'] for b in banks],
            preparation_seconds=sum(b.metadata['preparation_seconds'] for b in banks))
        merged=merge_nodes(banks, momenta, weights, metadata)
        merged.save(directory/'banks'/f'e{ei}.npz')
    write_json(directory/'banks'/'complete.json', dict(bank_id=manifest['bank_id'],
        files={f'e{ei}.npz':hashlib.sha256((directory/'banks'/f'e{ei}.npz').read_bytes()).hexdigest()
               for ei in range(len(config.orientations))}))


def reuse_banks(directory, previous):
    """Reuse complete coefficients with new geometry samples/regulators."""
    config,manifest=load_run(directory); _,old=load_run(previous)
    if manifest['bank_id']!=old['bank_id']:
        raise ValueError('the requested coefficients differ from the reusable bank')
    previous=Path(previous); directory=Path(directory)
    complete=json.loads((previous/'banks'/'complete.json').read_text())
    if complete['bank_id']!=manifest['bank_id']: raise ValueError('bank identity mismatch')
    for name,expected in complete['files'].items():
        if Path(name).name!=name: raise ValueError('invalid bank artifact name')
        source=previous/'banks'/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected: raise ValueError('reusable bank changed')
        destination=directory/'banks'/name
        if not destination.exists(): shutil.copyfile(source,destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=expected: raise ValueError('incompatible destination bank')
    write_json(directory/'banks'/'complete.json',complete)


def map_geometry(unit, config):
    """Vectorized six-dimensional version of the previous nine-variable map."""
    u=np.asarray(unit, float)
    if u.ndim!=2 or u.shape[1]!=6 or not np.isfinite(u).all() or np.any((u<=0)|(u>=1)):
        raise ValueError('six coordinates strictly inside the unit cube required')
    upper=config.tau2_max; norm=math.pi/3-1/upper
    x=np.sin(-math.pi/6+u[:, 0]*math.pi/3)
    for _ in range(12):
        residual=np.arcsin(x)+math.pi/6-(x+.5)/upper-u[:, 0]*norm
        x=np.clip(x-residual/(1/np.sqrt(1-x*x)-1/upper), -.5, .5)
    if np.max(abs(np.arcsin(x)+math.pi/6-(x+.5)/upper-u[:, 0]*norm))>2e-14:
        raise ArithmeticError('fundamental-domain inverse CDF failed')
    lower=np.sqrt(1-x*x); y=1/(1/lower-u[:, 1]*(1/lower-1/upper)); tau=x+1j*y
    swapped=u[:, 3]>u[:, 5]
    first=u[:, 2]+u[:, 3]*tau; second=u[:, 4]+u[:, 5]*tau
    z=np.column_stack((np.zeros(len(u)), np.where(swapped, second, first), np.where(swapped, first, second)))
    low=np.minimum(u[:, 3], u[:, 5]); high=np.maximum(u[:, 3], u[:, 5])
    gaps=np.minimum(np.minimum(low, high-low), 1-high)
    distance=np.full(len(u), np.inf)
    for i,j in ((0,1), (0,2), (1,2)):
        delta=z[:, j]-z[:, i]; base=np.floor(delta.imag/y)
        for offset in (-1, 0, 1, 2):
            shifted=delta-(base+offset)*tau
            distance=np.minimum(distance, abs(shifted-np.rint(shifted.real)))
    accepted=(gaps>=config.gap)&(distance>=config.distance)&(gaps>1e-12)
    return tau, z, swapped, norm*y**4, accepted, gaps, distance


def evaluate_geometry_chunk(unit, config, banks):
    tau,z,swapped,weight,accepted,gaps,distance=map_geometry(unit, config)
    result=np.zeros((len(unit), len(config.energies), len(config.cutoffs), 3, 4), complex)
    for reverse in (False, True):
        indices=np.flatnonzero(accepted&(swapped==reverse))
        if not len(indices): continue
        geometry=GeometryBatch.build(tau[indices], z[indices])
        for ei, energy in enumerate(config.energies):
            orientation=tuple(energy[::-1] if reverse else energy)
            result[indices, ei]=evaluate_density(banks[orientation], geometry, config.cutoffs)*weight[indices, None, None, None]
    return result, accepted


def _save_chunk(path, metadata, values, accepted):
    metadata=dict(metadata, values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                  accepted_sha256=hashlib.sha256(accepted.tobytes()).hexdigest())
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary=Path(stream.name)
        np.savez(stream, metadata=json.dumps(metadata, sort_keys=True), values=values, accepted=accepted)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_chunk(path, expected):
    with np.load(path, allow_pickle=False) as data:
        meta=json.loads(str(data['metadata'])); values=data['values']; accepted=data['accepted']
    if any(meta.get(k)!=v for k,v in expected.items()): raise ValueError('geometry checkpoint mismatch')
    if hashlib.sha256(values.tobytes()).hexdigest()!=meta['values_sha256'] or hashlib.sha256(accepted.tobytes()).hexdigest()!=meta['accepted_sha256']:
        raise ValueError('corrupt geometry checkpoint')
    if not np.isfinite(values).all() or values.shape[0]!=len(accepted): raise ValueError('invalid geometry values')
    return values, accepted, meta


def integrate_replicate(directory, replicate):
    config, manifest=load_run(directory); directory=Path(directory)
    if not 0<=replicate<config.replicates: raise ValueError('invalid replicate')
    complete=json.loads((directory/'banks'/'complete.json').read_text())
    if complete['bank_id']!=manifest['bank_id']: raise ValueError('bank completion mismatch')
    banks={}
    for ei, energy in enumerate(config.orientations):
        path=directory/'banks'/f'e{ei}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=complete['files'][path.name]:
            raise ValueError('merged bank changed')
        banks[energy]=CoefficientBank.load(path, expected=dict(bank_id=manifest['bank_id'], energy=list(energy)))
    grid=qmc.Sobol(6, scramble=True, seed=config.seed+replicate).random_base2(config.samples.bit_length()-1)
    started=time.perf_counter()
    for start in range(0, config.samples, config.batch_size):
        unit=grid[start:start+config.batch_size]
        path=directory/'geometry'/f'r{replicate}_n{start}.npz'
        meta=dict(run_id=manifest['run_id'], replicate=replicate, start=start, count=len(unit),
                  unit_sha256=hashlib.sha256(unit.tobytes()).hexdigest())
        if path.exists():
            _load_chunk(path, meta); continue
        before=time.perf_counter(); values, accepted=evaluate_geometry_chunk(unit, config, banks)
        meta['evaluation_seconds']=time.perf_counter()-before
        _save_chunk(path, meta, values, accepted)
        print(f'replicate {replicate}: {start+len(unit)}/{config.samples}, elapsed {time.perf_counter()-started:.2f}s', flush=True)


def complex_json(values):
    a=np.asarray(values)
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def statistics(values):
    values=np.asarray(values); n=len(values); mean=values.mean(axis=0); delta=values-mean
    return dict(mean=complex_json(mean), stderr_real=np.sqrt((delta.real**2).sum(axis=0)/(n*(n-1))).tolist(),
                stderr_imag=np.sqrt((delta.imag**2).sum(axis=0)/(n*(n-1))).tolist(),
                covariance_real_imag=(np.sum(delta.real*delta.imag, axis=0)/(n*(n-1))).tolist())


def reduce_run(directory):
    config, manifest=load_run(directory); directory=Path(directory)
    averages=[]; counts=[]; seconds=[]
    for rep in range(config.replicates):
        grid=qmc.Sobol(6, scramble=True, seed=config.seed+rep).random_base2(config.samples.bit_length()-1)
        total=np.zeros((len(config.energies),len(config.cutoffs),3,4), complex); count=0
        for start in range(0, config.samples, config.batch_size):
            unit=grid[start:start+config.batch_size]
            meta=dict(run_id=manifest['run_id'], replicate=rep, start=start, count=len(unit),
                      unit_sha256=hashlib.sha256(unit.tobytes()).hexdigest())
            values, accepted, stored=_load_chunk(directory/'geometry'/f'r{rep}_n{start}.npz', meta)
            if values.shape!=(len(unit),)+total.shape or not np.array_equal(accepted,map_geometry(unit,config)[4]):
                raise ValueError('geometry checkpoint grid or tensor mismatch')
            if np.any(values[~accepted]!=0): raise ValueError('rejected points must contribute zero')
            total+=values.sum(axis=0); count+=int(accepted.sum()); seconds.append(stored['evaluation_seconds'])
        averages.append(total/config.samples); counts.append(count)
    raw=np.array(averages); amplitude=raw.sum(axis=(-1,-2))
    tree_factor=np.array([(p+q)*p*q for p,q in config.energies]); reduced=amplitude/tree_factor[None,:,None]
    result=dict(schema='spin23-svv-banked-regulated-integral-v1', run_id=manifest['run_id'], config=asdict(config),
        status='regulated necklace integral; spectral and boundary convergence not certified',
        geometry_dimension=6, deterministic_spectral_dimension=3, block_backend='recursion',
        accepted=counts, amplitude=statistics(amplitude), tree_reduced=statistics(reduced),
        cutoff_differences=statistics(np.diff(amplitude,axis=2)),
        tree_reduced_energy_differences=statistics(reduced[:,1:]-reduced[:,:1]),
        replicate_components=complex_json(raw), summed_geometry_cpu_seconds=sum(seconds),
        errors='scrambled-replicate statistics only; spectral, level, and regulator errors are separate',
        remaining=['spectral order and momentum tail convergence', 'necklace level convergence',
                   'pair/comb OPE charts, collision and cusp completion', 'physical continuation'])
    write_json(directory/'result.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init','prepare','merge','integrate','reduce'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reuse-banks-from',type=Path)
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--task-count', type=int, default=1)
    args=parser.parse_args()
    if args.command=='init':
        options=json.loads(args.config.read_text()) if args.config else {}
        if 'energies' in options: options['energies']=tuple(map(tuple, options['energies']))
        if 'cutoffs' in options: options['cutoffs']=tuple(options['cutoffs'])
        print(initialize(args.run_dir, BankedConfig(**options))['run_id'])
        if args.reuse_banks_from: reuse_banks(args.run_dir,args.reuse_banks_from)
    elif args.command=='prepare': prepare_shard(args.run_dir,args.task_index,args.task_count)
    elif args.command=='merge': merge_banks(args.run_dir)
    elif args.command=='integrate': integrate_replicate(args.run_dir,args.task_index)
    else: reduce_run(args.run_dir)


if __name__=='__main__': main()
