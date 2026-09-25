"""Recursive primary and superdescendant torus one-point coefficients.

NS superdescendants use the verified necklace c-recursion with an identity
puncture and an explicitly checked diagonal collapse.  Ramond uses the same
two-Virasoro branching as the VV bank, but closes a single handle and uses
ordinary torus one-point h-recursion.  No inverse-Gram production fallback.
"""
from functools import lru_cache
import importlib.util
import math
import sys

import numpy as np


@lru_cache(maxsize=1)
def _virasoro_class():
    from spin23_type0b_reference import reference_root
    path = reference_root()/'Code/genus_2_cross_channel/virasoro_blocks.py'
    name = '_spin23_pinned_onepoint_virasoro'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.TorusOnePointVirasoroBlock


def generic_ramond_onepoint(b, momentum, external, component, cutoff):
    from spin23_genus1_branch_recursion import external_expansion, vertex, auxiliary_character
    from spin23_two_virasoro_ramond import embedded_branch_state, _branch_twice_grade
    from spin23_ramond_torus_recursion import ramond_branch_numbers
    ordinary = _virasoro_class()
    total = np.zeros((2, cutoff+1), complex)
    for label in ramond_branch_numbers(cutoff//2):
        for parity in (0, 1):
            edge = embedded_branch_state(b=b, sector='R', physical_momentum=momentum,
                                         branch_number=label, parity=parity)
            onset = _branch_twice_grade('R', label)
            order = (cutoff-onset)//2
            for ext, scale in external_expansion(b, external, component):
                weights = np.array([scale*vertex(edge, ext, edge, component, sign)/edge.norm
                                    for sign in (1, -1)])
                if not np.any(weights):
                    continue
                parts = []
                for index in (1, 2):
                    obj = ordinary(getattr(edge.parameters, f'c_{index}'),
                                   getattr(edge.parameters, f'h_{index}'),
                                   getattr(ext.parameters, f'h_{index}'))
                    parts.append(obj.descendant_coefficients(order))
                    obj._h_coefficient.cache_clear()
                values = np.convolve(*parts)[:order+1]
                total[:, onset:cutoff+1:2] += weights[:, None]*values
    character = auxiliary_character('R', cutoff)
    result = np.zeros_like(total)
    for k in range(cutoff+1):
        result[:, k] = (total[:, k]-sum(character[j]*result[:, k-j]
                         for j in range(1, k+1)))/character[0]
    return result[:, ::2]


def ramond_onepoint(momentum, external, cutoff=8, tolerance=2e-7):
    """Both external components and both HJS signs with checked finite parts."""
    from spin23_genus1_recursion import dictionary_finite_part
    keys = tuple((word, sign, k) for word in (0, 1) for sign in (0, 1)
                 for k in range(cutoff//2+1))
    previous = None; records = []
    for samples in (32, 64, 128, 256):
        def evaluator(b):
            values = [generic_ramond_onepoint(b, momentum, external, w, cutoff) for w in (0, 1)]
            return {key: values[key[0]][key[1], key[2]] for key in keys}
        values, checks = dictionary_finite_part(evaluator, keys=keys, radius=.60,
            check_radius=.70, samples=samples, inversion_symmetric=True)
        array = np.array([values[key] for key in keys]).reshape(2, 2, cutoff//2+1)
        scale = max(1., np.max(abs(array)))
        error = max(abs(d.value-d.check_value) for d in checks.values())/scale
        cross = None if previous is None else float(np.max(abs(array-previous))/scale)
        good = np.isfinite(array).all() and np.isfinite(error) and error <= tolerance
        records.append(dict(samples=samples, scaled_contour_error=float(error),
                            resolution_difference=cross, contour_pass=bool(good)))
        if good and (samples == 32 or previous is not None and cross <= tolerance):
            return array, records
        previous = array if good else None
    raise ArithmeticError(f'one-point Ramond finite part failed: {records}')


def ns_onepoint(momentum, external, cutoff=8, tolerance=2e-7):
    from spin23_type0b_reference import enable_reference_imports
    enable_reference_imports()
    from superconformal_torus_blocks import SelfDualNSTorusOnePointBlock
    from spin23_genus1_recursive_sewing import ns_coefficients
    primary = SelfDualNSTorusOnePointBlock(internal_momentum=momentum,
        external_momentum=external, radius=.15, check_radius=.20, samples=32)
    raw = primary.raw_coefficients(cutoff)
    scale = max(1., max(abs(v) for v in raw.values()))
    error = max(abs(d.value-d.check_value) for d in primary.coefficient_diagnostics(cutoff).values())/scale
    if not math.isfinite(error) or error > tolerance:
        raise ArithmeticError('NS primary one-point contour check failed')
    k, c = ns_coefficients((momentum, momentum), (external, 1j), (1, 0), (1, 0),
                           (cutoff, cutoff), 2*cutoff)
    diagonal = k[:, 0] == k[:, 1]
    off = float(np.max(abs(c[~diagonal]), initial=0.)/max(1., np.max(abs(c))))
    if off > tolerance:
        raise ArithmeticError('identity insertion did not collapse the NS handle')
    out = np.zeros((2, cutoff+1), complex)
    out[0] = [raw[j] for j in range(cutoff+1)]
    out[1, k[diagonal, 0]] = c[diagonal]
    return out, dict(primary_contour_error=float(error), identity_off_diagonal_error=off)
