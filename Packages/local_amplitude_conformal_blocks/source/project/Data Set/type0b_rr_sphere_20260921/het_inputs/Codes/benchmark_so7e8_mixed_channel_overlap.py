"""Compare P-integrated mixed NS/R channels using double Virasoro only.

Different internal channels must be compared AFTER the Liouville integral,
not at a common fixed P. This is an overlap diagnostic, not a sphere amplitude.
The direct nonchiral projector remains explicitly uncertified. No projector
phase or normalization is fitted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from benchmark_so7e8_four_ramond_convergence import MOMENTA
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import (
    threshold_weighted_rule,
)
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_ramond_sphere_integrand import build_fixed_p_two_ramond_kernel
from so7e8_two_ramond_liouville_integral import _uniformize_kernel
from spin23_singlet_amplitudes import _stable_complex_fsum


def _pairs(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def run(*, orders=(2, 4), nodes=12, species="VV", picture="z",
        momenta=MOMENTA, points=(.35+.1j, .5+.1j, .65+.1j),
        threshold_options=None, progress=False, sewing_convention="legacy"):
    momenta = tuple(map(complex, momenta))
    if len(momenta) != 4:
        raise ValueError("expected four external Liouville momenta")
    require_conservative_real_p_chamber(momenta[:3], momenta[3])
    if species not in ("SS", "SV", "VS", "VV"):
        raise ValueError("species must be SS, SV, VS, or VV")
    if picture not in ("z", "one"):
        raise ValueError("picture must be z or one")
    if sewing_convention not in ("legacy", "human"):
        raise ValueError("sewing_convention must be legacy or human")
    orders = tuple(orders)
    if not orders or any(not isinstance(order, int) or order < 1 for order in orders):
        raise ValueError("orders must contain positive twice-level cutoffs")
    points = np.asarray(points, dtype=complex)
    if points.ndim != 1 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("expected a nonempty finite one-dimensional modulus grid")
    if np.any((points == 0) | (points == 1)):
        raise ValueError("overlap points cannot be punctures")
    rule = threshold_weighted_rule(nodes, options=threshold_options)
    rows, previous, value_cache, transport_error_cache = [], {}, {}, {}
    for order in orders:
        started = time.perf_counter()
        effective_orders = {"crossed": order, "direct": order-order % 2}
        samples = {channel: {} for channel in ("crossed", "direct")}
        transport_defect = {channel: 0. for channel in samples}
        for index, internal in enumerate(rule.momenta):
            for channel in samples:
                cache_key = (channel, effective_orders[channel], float(internal))
                if cache_key not in value_cache:
                    kernel = build_fixed_p_two_ramond_kernel(
                        float(internal), external_liouville_momenta=momenta,
                        time_momenta=(*momenta[:3], -momenta[3]),
                        ns_at_z=species[0], ns_at_one=species[1],
                        zero_ramond_family="Psi_tilde", infinity_ramond_family="Psi",
                        channel=channel, picture_zero_at=picture,
                        maximum_twice_level=effective_orders[channel],
                        crossed_block_backend=("double_virasoro" if channel == "crossed"
                                               else "inverse_gram"),
                        direct_block_backend="double_virasoro",
                        allow_uncertified_direct_projector=(channel == "direct"),
                    )
                    old_values = _uniformize_kernel(
                        kernel, elliptic_prefactor=True).evaluate_many(points)
                    if sewing_convention == "human":
                        from so7e8_human_conventions import transport_kernel_to_human_basis
                        kernel = transport_kernel_to_human_basis(kernel)
                        new_values = _uniformize_kernel(
                            kernel, elliptic_prefactor=True).evaluate_many(points)
                        if any(not np.all(np.isfinite(v)) for v in (*old_values.values(),*new_values.values())):
                            raise ArithmeticError(f"nonfinite {channel} convention transport at P={internal}")
                        defect = max(float(np.max(np.abs(new_values[name]-value)
                                     /np.maximum(np.maximum(np.abs(value),np.abs(new_values[name])),1e-300)))
                                     for name,value in old_values.items())
                        transport_error_cache[cache_key] = defect
                        value_cache[cache_key] = new_values
                    else:
                        value_cache[cache_key] = old_values
                transport_defect[channel] = max(transport_defect[channel],transport_error_cache.get(cache_key,0.))
                # R has integer levels: e.g. N=4 -> 5 changes only NS.
                # Reuse the evaluated R kernel, not another coefficient run.
                values = value_cache[cache_key]
                for name, value in values.items():
                    if not np.all(np.isfinite(value)):
                        raise ArithmeticError(f"nonfinite {channel}/{name} at P={internal}")
                    samples[channel].setdefault(name, []).append(value)
            if progress:
                print(f"order {order}: P node {index+1}/{len(rule.momenta)} "
                      f"({internal:.6g}), elapsed {time.perf_counter()-started:.1f}s",
                      file=sys.stderr, flush=True)
        integrated, regions = {}, {}
        for channel, tensors in samples.items():
            integrated[channel], regions[channel] = {}, {}
            for name, values in tensors.items():
                weighted = np.asarray(values) * (rule.weights / np.pi)[:, None]
                integrated[channel][name] = np.asarray([
                    _stable_complex_fsum(column) for column in weighted.T])
                regions[channel][name] = {
                    label: _pairs([_stable_complex_fsum(column)
                                   for column in weighted[rule.regions == region].T])
                    for region, label in enumerate(("endpoint", "bulk", "tail"))}
        defects = {}
        for name, ns in integrated["crossed"].items():
            r = integrated["direct"][name]
            defects[name] = {
                "absolute": np.abs(ns-r).tolist(),
                "relative_to_larger_channel": (
                    np.abs(ns-r)/np.maximum(np.maximum(np.abs(ns), np.abs(r)), 1e-300)).tolist()}
        changes = {}
        for channel, tensors in integrated.items():
            changes[channel] = {
                name: (None if (channel, name) not in previous else
                       (np.abs(value-previous[channel, name]) /
                        np.maximum(np.abs(value), 1e-300)).tolist())
                for name, value in tensors.items()}
            previous.update({(channel, name): value for name, value in tensors.items()})
        rows.append(dict(maximum_twice_level=order,
                         channel_twice_level_cutoffs=effective_orders,
                         densities={channel: {name: _pairs(value) for name, value in tensors.items()}
                                    for channel, tensors in integrated.items()},
                         spectral_regions=regions, overlap_defect=defects,
                         maximum_fixed_p_basis_transport_defect=transport_defect,
                         relative_previous_order_change=changes,
                         seconds=time.perf_counter()-started))
    sources = ("so7e8_ramond_sphere_integrand.py", "so7e8_ramond_cocycles.py",
               "so7e8_mixed_double_virasoro.py", "so7e8_internal_ramond_double_virasoro.py",
               "virasoro_sphere_c_recursion.py", "ramond_sphere_uniformization.py",
               "spin23_super_liouville_data.py", "so7e8_two_ramond_liouville_integral.py",
               "benchmark_so7e8_mixed_channel_overlap.py",
               "so7e8_human_conventions.py",
               "heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py")
    return dict(backend="double_virasoro_in_both_channels", full_pbw_block_used=False,
                measure="integral_0^infinity dP/pi; no modulus integral",
                nonchiral_projector_certified=False, production_certified=False,
                sewing_convention=sewing_convention,
                convention_scope=("complete basis transport of the existing physical sewing tensor; "
                                  "not a newly derived physical projector"),
                species=species, picture_zero_at=picture, momenta=_pairs(momenta),
                moduli=_pairs(points), momentum_rule=rule.metadata(),
                source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in sources}, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--species", choices=("SS", "SV", "VS", "VV"), default="VV")
    parser.add_argument("--picture", choices=("z", "one"), default="z")
    parser.add_argument("--threshold-options", type=json.loads, default=None)
    parser.add_argument("--sewing-convention", choices=("legacy","human"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(orders=args.orders, nodes=args.nodes, species=args.species,
                 picture=args.picture, threshold_options=args.threshold_options, progress=True,
                 sewing_convention=args.sewing_convention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
