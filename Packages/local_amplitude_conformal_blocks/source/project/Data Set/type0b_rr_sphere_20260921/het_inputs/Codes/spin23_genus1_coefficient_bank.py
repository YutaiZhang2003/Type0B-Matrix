"""Geometry-independent recursive coefficients for the SVV necklace.

Preparation is expensive. Evaluation imports no recursion or structure-constant
code, and never fills a missing bank on demand. Spectral weights include dP/pi;
the physical exp(sum(P_e**2 log|q_e|)) stays in the geometry evaluation.
"""
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

WORDS = ((1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0))
FORMS = tuple(f for f in product((0, 1), repeat=3) if sum(f) % 2)
SIGNS = tuple(product((1, -1), repeat=3))
SCHEMA = 'spin23-svv-recursive-bank-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def spectral_rule(order=3, p_max=4., power=1.25):
    """Fixed, distinct power-Legendre rules; order means n,n+1,n+2.

    Different edge rules avoid exact confluent-weight loci. This changes only
    quadrature nodes, not the theory. Neither this order nor p_max is a certified
    spectral accuracy: compare separate banks at higher order and larger p_max.
    """
    if type(order) is not int or order < 1 or not np.isfinite(p_max) or p_max <= 0:
        raise ValueError('positive order and finite momentum cutoff required')
    if not np.isfinite(power) or power < 1:
        raise ValueError('finite endpoint power >= 1 required')
    edges = []
    for edge in range(3):
        x, w = np.polynomial.legendre.leggauss(order + edge)
        t = (x + 1) / 2
        exponent = power + edge * 1e-6
        edges.append(tuple(zip(p_max*t**exponent,
                              w/2*p_max*exponent*t**(exponent-1)/math.pi)))
    triples = tuple(product(*edges))
    return (np.array([[p for p, w in row] for row in triples]),
            np.array([math.prod(w for p, w in row) for row in triples]))


def _phase(forms, left, ramond=False):
    right = WORDS[3]
    lp = tuple((f+a) % 2 for f, a in zip(forms, left))
    rp = tuple((f+b) % 2 for f, b in zip(forms, right))
    exponent = sum(f*b for f, b in zip(forms, right))
    exponent += sum(rp[i]*lp[j] for i in range(3) for j in range(i+1, 3))
    return (-1j)**(sum(right)+(0 if ramond else sum(forms))) * (-1)**exponent


def _levels(cutoff, step):
    return np.array([k for k in product(range(0, cutoff+1, step), repeat=3)
                     if sum(k) <= cutoff], dtype=np.int16)


def _align(levels, values, target):
    mapping = {tuple(k): i for i, k in enumerate(target)}
    out = np.zeros(values.shape[:-1] + (len(target),), complex)
    for i, k in enumerate(levels):
        out[..., mapping[tuple(k)]] = values[..., i]
    return out


def prepare_node(momentum, energy, *, cutoff=8, precision=24, continuation=None):
    """Build all four words and all forms/signs ONCE at the largest cutoff."""
    from spin23_genus1_recursive_sewing import ns_coefficients
    from spin23_genus1_branch_recursion import checked_ramond_tables
    from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants
    if type(cutoff) is not int or cutoff < 0 or cutoff > 8 or cutoff % 2:
        raise ValueError('an integer total level at most four is required')
    p = tuple(map(float, momentum)); e = tuple(map(float, energy))
    if len(p) != 3 or len(e) != 2 or any(not math.isfinite(x) or x <= 0 for x in p+e):
        raise ValueError('three positive internal momenta and two outgoing energies required')
    external = (complex(sum(e)), complex(e[0]), complex(e[1]))
    started = time.perf_counter()
    nl, rl = _levels(cutoff, 1), _levels(cutoff, 2)
    ns = np.zeros((4, 4, len(nl)), complex)
    ramond = np.zeros((4, 4, 8, len(rl)), complex)
    nw = np.zeros((4, 4), complex); rw = np.zeros((4, 4, 8), complex)
    nc = [ns_structure_constants(p[v-1], external[v], p[v], precision=precision) for v in range(3)]
    rc = [rr_ns_structure_constants(p[v-1], p[v], external[v], precision=precision) for v in range(3)]
    checks = []
    for wi, word in enumerate(WORDS):
        for fi, forms in enumerate(FORMS):
            levels, values = ns_coefficients(p, external, word, forms, (cutoff,)*3, cutoff)
            ns[wi, fi] = _align(levels, values, nl)
            nw[wi, fi] = _phase(forms, word)*math.prod(c[f] for c, f in zip(nc, forms))
        levels, values, forms, signs, diagnostics = checked_ramond_tables(
            internal_momenta=p, external_momenta=external, external_descendants=word,
            maximum_twice_levels=cutoff, maximum_total_twice_level=cutoff,
            **(continuation or {}))
        checks.append(diagnostics)
        for fi, f in enumerate(FORMS):
            for si, s in enumerate(SIGNS):
                ramond[wi, fi, si] = _align(levels, values[forms.index(f), signs.index(s)], rl)
                rw[wi, fi, si] = _phase(f, word, True)*math.prod(
                    c[0 if sign == 1 else 1]/2 for c, sign in zip(rc, s))
    arrays = dict(momenta=np.array([p]), weights=np.array([1.]), ns_levels=nl, r_levels=rl,
                  ns=ns[None], r=ramond[None], ns_weights=nw[None], r_weights=rw[None])
    metadata = dict(schema=SCHEMA, channel='necklace', energy=list(e), cutoff=cutoff,
                    precision=precision, ramond_checks=checks,
                    preparation_seconds=time.perf_counter()-started)
    bank = CoefficientBank(metadata, **arrays)
    bank.validate()
    return bank


@dataclass
class CoefficientBank:
    metadata: dict
    momenta: np.ndarray
    weights: np.ndarray
    ns_levels: np.ndarray
    r_levels: np.ndarray
    ns: np.ndarray
    r: np.ndarray
    ns_weights: np.ndarray
    r_weights: np.ndarray

    def validate(self):
        if self.metadata.get('schema') != SCHEMA or self.metadata.get('channel') != 'necklace':
            raise ValueError('unsupported coefficient bank')
        n = len(self.momenta)
        shapes = dict(momenta=(n, 3), weights=(n,), ns=(n, 4, 4, len(self.ns_levels)),
                      r=(n, 4, 4, 8, len(self.r_levels)), ns_weights=(n, 4, 4),
                      r_weights=(n, 4, 4, 8))
        for key, shape in shapes.items():
            value = getattr(self, key)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'invalid {key} array')
        if n == 0 or np.any(self.momenta <= 0) or np.any(self.weights <= 0):
            raise ValueError('positive spectral nodes and weights required')
        for name, step in (('ns_levels', 1), ('r_levels', 2)):
            if not np.array_equal(getattr(self, name), _levels(self.metadata['cutoff'], step)):
                raise ValueError(f'incomplete or misordered {name}')

    def save(self, path):
        """Atomic publication, no pickle; a partial write is never a valid bank."""
        self.validate()
        path = Path(path)
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != 'metadata'}
        metadata = dict(self.metadata)
        metadata['array_sha256'] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in payload.items()}
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez(stream, metadata=json.dumps(metadata, sort_keys=True, allow_nan=False), **payload)
                stream.flush(); os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, *, expected=None):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            arrays = {name: data[name].copy() for name in cls.__dataclass_fields__ if name != 'metadata'}
        if expected and any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError('coefficient-bank provenance or configuration mismatch')
        for k, value in arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != meta['array_sha256'][k]:
                raise ValueError(f'corrupt bank array: {k}')
        bank = cls(meta, **arrays); bank.validate()
        return bank

    def evaluate(self, plumbing, lifts, cutoffs=(6, 8), *, momentum_chunk=128):
        """Return [geometry, cutoff, spin, PCO] spectral integrals in batches.

        Both chiralities use the SAME coefficient functions of the momenta.
        Only q is conjugated on the right. Temporal parity is at edge two.
        Lower levels are slices of this bank, with no new coefficient calls.
        """
        q = np.asarray(plumbing, complex); lifts = np.asarray(lifts, int)
        if q.ndim != 2 or q.shape[1] != 3 or lifts.shape != q.shape:
            raise ValueError('geometry-by-three plumbing and lift arrays required')
        if not np.isfinite(q).all() or np.any(abs(q) <= 0) or np.any(abs(q) >= 1):
            raise ValueError('necklace powers must be strictly convergent')
        if not np.isin(lifts, (-1, 1)).all() or momentum_chunk < 1:
            raise ValueError('invalid lifts or momentum chunk')
        if not cutoffs or any(type(c) is not int or c < 0 or c > self.metadata['cutoff'] for c in cutoffs):
            raise ValueError('requested level is absent from this bank')
        logs = np.log(q).T
        result = np.zeros((len(q), len(cutoffs), 3, 4), complex)
        ns_primary = np.exp(-np.log(abs(q)).sum(axis=1)/8)
        for li, cutoff in enumerate(cutoffs):
            for spin in range(3):
                is_r = spin == 2
                levels = self.r_levels if is_r else self.ns_levels
                mask = levels.sum(axis=1) <= cutoff
                powers = np.exp(levels[mask] @ logs/2)
                if not is_r:
                    edge_lifts = lifts.copy()
                    if spin == 1: edge_lifts[:, 2] *= -1
                    powers *= np.prod(edge_lifts[None, :, :]**levels[mask, None, :], axis=-1)
                coefficients = self.r if is_r else self.ns
                coupling = self.r_weights if is_r else self.ns_weights
                for start in range(0, len(self.momenta), momentum_chunk):
                    sl = slice(start, start+momentum_chunk)
                    c = coefficients[sl][..., mask]
                    left = c @ powers
                    right = c[:, 3:4] @ powers.conj()
                    products = coupling[sl][..., None]*left*right
                    # Sum homogeneous forms and, in R, structure signs.
                    products = products.sum(axis=tuple(range(2, products.ndim-1)))
                    gaussian = np.exp(self.momenta[sl]**2 @ np.log(abs(q)).T)
                    integrated = np.einsum('mwg,mg,m->gw', products, gaussian, self.weights[sl])
                    result[:, li, spin] += integrated
                if not is_r: result[:, li, spin] *= ns_primary[:, None]
        if not np.isfinite(result).all():
            raise ArithmeticError('non-finite bank evaluation')
        return result


def merge_nodes(banks, momenta, weights, metadata):
    """Merge an exact, ordered grid; callers cannot silently omit a failure."""
    if not banks or len(banks) != len(momenta) or len(weights) != len(momenta):
        raise ValueError('incomplete spectral bank')
    first = banks[0]
    for bank, p in zip(banks, momenta):
        bank.validate()
        if not np.array_equal(bank.momenta, [p]) or any(
            bank.metadata[k] != first.metadata[k] for k in ('energy', 'cutoff', 'precision', 'channel', 'schema')
        ):
            raise ValueError('incompatible or misordered spectral nodes')
    arrays = {k: np.concatenate([getattr(b, k) for b in banks])
              for k in ('ns', 'r', 'ns_weights', 'r_weights')}
    return CoefficientBank(dict(first.metadata, **metadata), np.asarray(momenta), np.asarray(weights),
                           first.ns_levels, first.r_levels, **arrays)
