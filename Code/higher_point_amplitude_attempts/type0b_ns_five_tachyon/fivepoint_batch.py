"""Reusable momentum tensors for the all-NS c-series integrand.

Compilation uses the original arbitrary-precision recursion and disk store.
Only evaluation uses complex128, as does the public scalar density. Coherent
log lifts, endpoint phases, PCO cocycles and leading-state masks are retained.
Ill-conditioned rows fall back to the scalar evaluator. This module contains
no change to the integration prescription or truncation order.
"""
from collections import OrderedDict
from itertools import product
import cmath
import math

import mpmath
import numpy as np

from ns_global_osp_block import osp_norm, osp_sector_vertex


class TensorCache:
    """Bound array storage by bytes, independently of recursion scratch objects."""

    def __init__(self, maximum_bytes=512 * 1024**2):
        if maximum_bytes < 1:
            raise ValueError("tensor cache must have a positive byte budget")
        self.maximum_bytes = int(maximum_bytes)
        self.data = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.fallback_rows = self.evaluated_rows = 0

    def get(self, key):
        if key not in self.data:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        size = value.nbytes
        if key in self.data:
            self.bytes -= self.data.pop(key).nbytes
        if size > self.maximum_bytes:
            return
        while self.bytes + size > self.maximum_bytes:
            self.bytes -= self.data.popitem(last=False)[1].nbytes
            self.evictions += 1
        self.data[key] = value
        self.bytes += size


def _block(kernel, key, weights, descendants, internal, sectors):
    block = kernel._block_cache.get(key)
    if block is None:
        block = kernel._c_block_type(
            **kernel._recursion_charge_arguments("c"), external_weights=weights,
            external_descendants=descendants, internal_weights=internal,
            vertex_sectors=sectors, working_precision=kernel.block_working_precision,
            pole_tolerance=kernel.pole_tolerance,
        )
        kernel._block_cache[key] = block
    return block


def _polynomial(kernel, weights, descendants, momenta, sectors, leading):
    """Compile the same reduced polynomial as _chiral_block (including faces)."""
    internal = tuple(kernel.block_weight(p) for p in momenta)
    key = ("primary-seed" if len(leading) == 2 else "c",
           weights, momenta, sectors, descendants)
    block = _block(kernel, key, weights, descendants, internal, sectors)
    parities = block.compatible_level_parities()
    with mpmath.workdps(kernel.block_working_precision):
        if len(leading) == 2:
            return parities, {parities: block.global_coefficient(parities)}
        if len(leading) == 1 and kernel.factorize_single_primary:
            edge = leading[0]
            remaining = 1 - edge
            parity = parities[edge]
            if edge == 0:
                vertex = osp_sector_vertex(
                    sector=sectors[0], n1=0, n2=0, n3=0, epsilon1=parity,
                    epsilon2=descendants[1], epsilon3=descendants[0],
                    d1=mpmath.mpc(internal[0]), d2=mpmath.mpc(weights[1]), d3=mpmath.mpc(weights[0]))
                face_weights = (internal[0], *weights[2:])
                face_descendants = (parity, *descendants[2:])
                face_sectors = sectors[1:]
            else:
                vertex = osp_sector_vertex(
                    sector=sectors[2], n1=0, n2=0, n3=0,
                    epsilon1=descendants[4], epsilon2=descendants[3],
                    epsilon3=parity, d1=mpmath.mpc(weights[4]), d2=mpmath.mpc(weights[3]), d3=mpmath.mpc(internal[1]))
                face_weights = (*weights[:3], internal[1])
                face_descendants = (*descendants[:3], parity)
                face_sectors = sectors[:2]
            prefactor = vertex / osp_norm(mpmath.mpc(internal[edge]), 0, parity)
            face_internal = (internal[remaining],)
            face_key = ("factorized-face", "c", face_weights, face_internal,
                        face_sectors, face_descendants)
            face = _block(kernel, face_key, face_weights, face_descendants,
                          face_internal, face_sectors)
            if face.compatible_level_parities() != (parities[remaining],):
                raise ArithmeticError("factorized face parity changed")
            total = kernel.global_max_total_twice_level
            levels = [(n,) for n in range(parities[remaining],
                        kernel.global_max_twice_levels[remaining] + 1, 2)
                      if total is None or n <= total]
            face._prepare(levels)
            coefficients = {}
            for (n,) in levels:
                level = (parity, n) if edge == 0 else (n, parity)
                coefficients[level] = prefactor * face.final_coefficients[(n,)]
            return parities, coefficients
        maxima = tuple(parities[i] if i in leading else n
                       for i, n in enumerate(kernel.global_max_twice_levels))
        total = kernel.global_max_total_twice_level
        levels = [level for level in product(*(range(p, n + 1, 2)
                  for p, n in zip(parities, maxima)))
                  if total is None or sum(level) <= total]
        block._prepare(levels)
        return parities, {level: block.final_coefficients[level] for level in levels}


class MomentumTensor:
    def __init__(self, kernel, ordering, momenta, leading, sectors, terms):
        weights = tuple(kernel.external_weights[label] for label in ordering)
        descendants = [tuple(term.liouville_descendants[label] for label in ordering)
                       for term in terms]
        # Include both NS parities on a selected primary edge, even at cutoff 0,
        # matching the scalar primary projection's cutoff-independent definition.
        maxima = tuple(1 if i in leading else n
                       for i, n in enumerate(kernel.global_max_twice_levels))
        levels = tuple(product(*(range(n + 1) for n in maxima)))
        lookup = {level: i for i, level in enumerate(levels)}
        self.levels = np.asarray(levels, dtype=float)
        shape = (len(momenta), len(sectors), len(terms), len(levels))
        self.coefficients = np.zeros(shape, dtype=complex)
        self.parities = np.zeros((len(sectors), len(terms), 2), dtype=int)
        self.structure = np.zeros(shape[:2], dtype=complex)
        self.internal = np.asarray([tuple(kernel.block_weight(p) for p in pair)
                                    for pair in momenta], dtype=complex)
        self.external_exponents = np.asarray([
            (sum(weights[i] + .5 * desc[i] for i in range(2)),
             sum(weights[i] + .5 * desc[i] for i in range(3)))
            for desc in descendants], dtype=complex)
        external_momenta = tuple(kernel.external_momenta[label] for label in ordering)
        for row, pair in enumerate(momenta):
            for s, sector in enumerate(sectors):
                self.structure[row, s] = kernel._structure_product(external_momenta, pair, sector)
                for t, desc in enumerate(descendants):
                    parities, coefficients = _polynomial(kernel, weights, desc, pair, sector, leading)
                    self.parities[s, t] = parities
                    phase = (-1) ** sum(parities)
                    for level, coefficient in coefficients.items():
                        self.coefficients[row, s, t, lookup[level]] = complex(phase * coefficient)
        if not all(np.all(np.isfinite(array)) for array in
                   (self.coefficients, self.structure, self.internal)):
            raise ArithmeticError("nonfinite compiled momentum tensor")
        self.nbytes = sum(array.nbytes for array in (self.levels, self.coefficients,
                           self.structure, self.internal, self.external_exponents, self.parities))

    def evaluate(self, q_logs, external_h, external_a, cocycle, remainder_edges=()):
        logs = np.asarray(q_logs, dtype=complex)
        def half(lift, external, mask=None):
            monomials = np.exp(self.levels @ (lift / 2))
            products = self.coefficients * monomials
            if mask is not None:
                products *= mask
            polynomial = np.sum(products, axis=-1)
            bound = np.sum(np.abs(products), axis=-1)
            leading = np.exp((self.internal[:, None, :] - self.external_exponents) @ lift)
            factors = leading * external
            return polynomial * factors[:, None, :], bound * np.abs(factors[:, None, :])
        if remainder_edges:
            # A forest remainder retains products with at least one excited
            # chiral state on EACH subtracted edge. Sum those disjoint pieces
            # directly instead of cancelling F-P1-P2+P12 numerically.
            excited = self.levels[None, None, :, :] > self.parities[:, :, None, :]
            categories = tuple(product((False, True), repeat=len(remainder_edges)))
            masks = [np.all(excited[..., remainder_edges] == category, axis=-1)
                     for category in categories]
            holomorphic = [half(logs, external_h, mask) for mask in masks]
            antiholomorphic = [half(logs.conjugate(), external_a, mask) for mask in masks]
            pairs = [(h, a) for i, h in enumerate(holomorphic)
                     for j, a in enumerate(antiholomorphic)
                     if all(x or y for x, y in zip(categories[i], categories[j]))]
        else:
            # Analytic continuation keeps the coefficients/weights unchanged.
            pairs = [(half(logs, external_h), half(logs.conjugate(), external_a))]
        values = np.zeros(len(self.internal), dtype=complex)
        bounds = np.zeros(len(self.internal))
        for (hol, hol_bound), (anti, anti_bound) in pairs:
            products = (self.structure[:, :, None, None] * hol[:, :, :, None]
                        * anti[:, :, None, :] * cocycle)
            values += np.sum(products, axis=(1, 2, 3))
            bounds += np.sum(np.abs(self.structure) * np.sum(hol_bound, axis=2)
                             * np.sum(anti_bound, axis=2), axis=1)
        # Conservative rounding indicator, including polynomial and PCO sums.
        error = 128 * np.finfo(float).eps * bounds
        unsafe = (~np.isfinite(values) | ~np.isfinite(bounds)
                  | (error > 1e-11 * np.abs(values)))
        return values, unsafe


def momentum_densities(kernel, positions, channel, momenta, leading=(), remainder_edges=()):
    """Matter densities / pi^2 for a fixed grid; geometry is built only once."""
    from type0b_ns_five_tachyon import (
        ODD_SECTOR_ASSIGNMENTS, pco_chiral_terms,
        STANDARD_ZERO_DESCENDANT_PHASE, STANDARD_INFINITY_DESCENDANT_PHASE,
    )
    momenta = tuple(tuple(complex(p) for p in pair) for pair in momenta)
    leading = tuple(sorted(set(leading)))
    terms_h = pco_chiral_terms(positions=positions, signed_energies=kernel.signed_energies,
                              operator_order=channel.ordering)
    terms_a = pco_chiral_terms(
        positions=tuple(None if z is None else z.conjugate() for z in positions),
        signed_energies=kernel.signed_energies, operator_order=channel.ordering)
    # All mutable numerical controls belong in the identity; tests and audits
    # sometimes raise the truncation on an existing kernel instance.
    key = (channel.ordering, momenta, leading, kernel.external_momenta,
           kernel.block_central_charge, kernel.global_max_twice_levels,
           kernel.global_max_total_twice_level, kernel.factorize_single_primary,
           kernel.structure_precision, kernel.block_working_precision, kernel.pole_tolerance)
    tensor = kernel._momentum_tensor_cache.get(key)
    if tensor is None:
        tensor = MomentumTensor(kernel, channel.ordering, momenta, leading,
                                ODD_SECTOR_ASSIGNMENTS, terms_h)
        kernel._momentum_tensor_cache.put(key, tensor)
    weights = tuple(kernel.external_weights[label] for label in channel.ordering)
    def external(terms, anti):
        values = []
        with mpmath.workdps(kernel.block_working_precision):
            for term in terms:
                desc = tuple(term.liouville_descendants[label] for label in channel.ordering)
                phase = STANDARD_ZERO_DESCENDANT_PHASE ** desc[0]
                phase *= (STANDARD_INFINITY_DESCENDANT_PHASE.conjugate() if anti
                          else STANDARD_INFINITY_DESCENDANT_PHASE) ** desc[-1]
                values.append(complex(term.coefficient * phase * kernel._component_covariance(
                    channel, positions, weights, desc, antiholomorphic=anti)))
        return np.asarray(values)
    cocycle = np.asarray([[(-1) ** len(set(h.timelike_labels) & set(a.timelike_labels))
                           for a in terms_a] for h in terms_h])
    values, unsafe = tensor.evaluate(tuple(cmath.log(q) for q in (channel.q1, channel.q2)),
                                     external(terms_h, False), external(terms_a, True), cocycle,
                                     remainder_edges=remainder_edges)
    values *= complex(kernel._timelike_boson_factor(positions, kernel.signed_energies)) / math.pi**2
    unsafe |= ~np.isfinite(values)
    cache = kernel._momentum_tensor_cache
    cache.evaluated_rows += len(momenta)
    for row in np.flatnonzero(unsafe):
        cache.fallback_rows += 1
        if remainder_edges:
            # Retain arbitrary precision through the nonchiral subtraction.
            # This fallback is rare; it is independent of the tensor algebra.
            external_momenta = tuple(kernel.external_momenta[label] for label in channel.ordering)
            with mpmath.workdps(kernel.block_working_precision):
                value = mpmath.mpc(0)
                for sector in ODD_SECTOR_ASSIGNMENTS:
                    component = kernel._sector_component_kernel(positions, momenta[row], sector, channel)
                    for edge in remainder_edges:
                        component -= kernel._sector_component_kernel_boundary_primary(
                            positions, momenta[row], sector, channel, boundary_edge=(edge,))
                    if len(remainder_edges) == 2:
                        component += kernel._sector_component_kernel_boundary_primary(
                            positions, momenta[row], sector, channel, boundary_edge=(0, 1))
                    value += kernel._structure_product(external_momenta, momenta[row], sector) * component
                value /= math.pi**2
        elif leading:
            external_momenta = tuple(kernel.external_momenta[label] for label in channel.ordering)
            value = sum(kernel._structure_product(external_momenta, momenta[row], sector)
                        * kernel._sector_component_kernel_boundary_primary(
                            positions, momenta[row], sector, channel, boundary_edge=leading)
                        for sector in ODD_SECTOR_ASSIGNMENTS) / math.pi**2
        else:
            value = kernel.momentum_integrand(positions, momenta[row], channel=channel)
        values[row] = complex(value)
    return values
