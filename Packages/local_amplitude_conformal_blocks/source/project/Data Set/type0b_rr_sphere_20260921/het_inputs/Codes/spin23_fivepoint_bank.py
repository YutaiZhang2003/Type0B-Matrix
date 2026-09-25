"""Immutable, precision-checked all-NS five-point coefficient banks.

The cutoff is TOTAL physical descendant level, including NS half-levels.
Only missing banks are built. Evaluation never calls coefficient recursion.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import mpmath as mp

from spin23_fivepoint_density import SToFourVectors
from spin23_fivepoint_type0b_reference import layers


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    """Atomic generated result; never overwrite a different accepted result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'preserve existing result: {path}')
        return
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)  # fail if a second writer already published
    finally:
        os.unlink(name)


def complex_rows(table, digits):
    return [[list(k), mp.nstr(v.real, digits), mp.nstr(v.imag, digits)]
        for k, v in sorted(table.items())]


def read_rows(rows):
    result = {tuple(k): mp.mpc(re, im) for k, re, im in rows}
    if len(result) != len(rows) or not all(mp.isfinite(x) for x in result.values()):
        raise ValueError('duplicate or nonfinite bank coefficients')
    return result


def coefficient_discrepancy(left, right, *, sparse=False):
    """Sparse transformed polynomials omit exact zeros, not computed data.

At equal weights, one precision may cancel a coefficient exactly while
another retains roundoff. Compare the union with explicit zero defaults.
The plane tables remain complete and must have identical support.
"""
    if not sparse and left.keys() != right.keys():
        raise ArithmeticError('precision evaluations have different plane coefficient support')
    return max((abs(left.get(k,0)-right.get(k,0))/max(mp.mpf(1),abs(right.get(k,0)))
        for k in left.keys() | right.keys()),default=mp.mpf(0))


def make_density(spec, precision=None, coefficient_store=None):
    order = spec['order']
    return SToFourVectors(tuple(complex(*z) for z in spec['energies']), spec['momenta'],
        ordering=spec['ordering'], maximum_twice_levels=(2*order, 2*order),
        maximum_total_twice_level=2*order, precision=precision or spec['precision'],
        coefficient_store=coefficient_store)


def build_bank(path, spec):
    path = Path(path)
    if path.exists():
        load_bank(path, spec)
        return dict(reused=True, path=str(path))
    started = perf_counter()
    precision, check = spec['precision'], spec['verification_precision']
    if check <= precision:
        raise ValueError('independent higher-precision verification required')
    store = layers().CoefficientStore(path.with_suffix('.sqlite'))
    try:
        low = make_density(spec, coefficient_store=store)
        high = make_density(spec, precision=check, coefficient_store=store)
        records = []
        max_error = mp.mpf(0)
        for key, block in low.blocks.items():
            reference = high.blocks[key]
            plane, ref_plane = block.coefficients(), reference.coefficients()
            table, ref_table = block.table(), reference.table()
            with mp.workdps(check):
                for left, right, sparse in ((plane, ref_plane, False), (table, ref_table, True)):
                    error = coefficient_discrepancy(left,right,sparse=sparse)
                    max_error = max(max_error,error)
                    if error > mp.mpf('1e-11'):
                        raise ArithmeticError(f'coefficient precision failure at {key}: {error}')
                records.append(dict(stars=list(key[0]), sectors=list(key[1]),
                    plane=complex_rows(plane, precision+12),
                    elliptic=complex_rows(table, precision+12)))
            print(json.dumps(dict(event='block_saved_in_cache', stars=key[0], sectors=key[1],
                elapsed_seconds=perf_counter()-started)), flush=True)
        payload = dict(schema='hetso23-fivepoint-bank-v1', spec=spec, spec_sha256=digest(spec),
            blocks=records, maximum_scaled_precision_error=float(max_error),
            precision_pair=[precision, check], elapsed_seconds=perf_counter()-started)
        payload['content_sha256'] = digest(payload)
        atomic_json(path, payload)
        return dict(reused=False, path=str(path), elapsed_seconds=payload['elapsed_seconds'])
    finally:
        store.close()


def load_bank(path, spec):
    payload = json.loads(Path(path).read_text())
    content_hash = payload.pop('content_sha256')
    if digest(payload) != content_hash or payload['spec'] != spec or payload['spec_sha256'] != digest(spec):
        raise ValueError('bank source/physical identity or checksum mismatch')
    bank = make_density(spec)
    seen = set()
    with mp.workdps(spec['precision']):
        for row in payload['blocks']:
            key = tuple(row['stars']), tuple(row['sectors'])
            if key in seen or key not in bank.blocks:
                raise ValueError('unexpected/duplicate block in bank')
            seen.add(key)
            block = bank.blocks[key]
            plane = read_rows(row['plane'])
            expected = {(i, j) for i in range(block.parities[0], 2*spec['order']+1, 2)
                for j in range(block.parities[1], 2*spec['order']+1, 2) if i+j <= 2*spec['order']}
            if set(plane) != expected:
                raise ValueError('incomplete physical total-level bank')
            block.raw.final_coefficients = plane
            block._saved_table = read_rows(row['elliptic'])
            if any(k not in expected for k in block._saved_table):
                raise ValueError('elliptic bank has unsupported levels')
            def forbidden(*args, **kwargs):
                raise RuntimeError('coefficient recursion forbidden in bank evaluation')
            block.raw._coefficient = forbidden
    if seen != set(bank.blocks):
        raise ValueError('missing external-descendant block')
    return bank
