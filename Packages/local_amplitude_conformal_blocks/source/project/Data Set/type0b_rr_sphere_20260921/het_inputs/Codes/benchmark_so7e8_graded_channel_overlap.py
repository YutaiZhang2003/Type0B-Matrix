"""Test the derived sphere sign on the existing mixed-channel tensors.

Three direct-channel assemblies share EXACTLY the same chiral blocks:
legacy, graded_times_legacy (insert the derived sign while retaining the
old physical/basis factors), and graded_full_product (replace the legacy
projector phase by the derived sign). The last is an unprojected product-
module control, not a proposed heterotic GSO projection. No phase is fitted.

These diagnostics test whether the explicit sign suffices with the current
RRNS/external data. A failure is not a rejection of the completeness identity.
Both sign prescriptions remain outside the production amplitude builder.
"""

from dataclasses import replace
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from benchmark_so7e8_four_ramond_convergence import MOMENTA
from graded_sphere_sewing import sphere_parity_routes
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_ramond_sphere_integrand import build_fixed_p_two_ramond_kernel, ns_descendant_words
from so7e8_two_ramond_liouville_integral import _uniformize_kernel
from spin23_singlet_amplitudes import _stable_complex_fsum


ASSEMBLIES = ("legacy", "graded_times_legacy", "graded_full_product")


def graded_direct_kernel(kernel, *, assembly="graded_times_legacy"):
    """Apply explicit trinion/Gram signs to an audit-only direct kernel.

    Terms forbidden by parity-preserving chiral Grams are zero and removed.
    The full-product control divides out only the documented legacy phase;
    no structure constant, reflection label, PCO factor or block is changed.
    """
    if kernel.channel != "direct":
        raise ValueError("the derived sign adapter requires the direct R channel")
    if assembly not in ASSEMBLIES:
        raise ValueError("unknown direct assembly")
    if assembly == "legacy":
        return kernel
    terms = []
    for term in kernel.terms:
        if "graded_sewing_sign" in term.labels:
            raise ValueError("the derived sewing sign was already applied")
        external = term.labels["external"]
        hw, aw = ns_descendant_words(kernel.ns_at_z, kernel.ns_at_one,
                                      term.pco_branch, kernel.picture_zero_at)
        hol = (external.zero_hjs_ground_parity,
               sum(mode.kind == "G" for mode in hw[0]) % 2,
               sum(mode.kind == "G" for mode in hw[1]) % 2,
               external.infinity_hjs_ground_parity)
        anti = (external.zero_antiholomorphic_ground_parity,
                sum(mode.kind == "G" for mode in aw[0]) % 2,
                sum(mode.kind == "G" for mode in aw[1]) % 2,
                external.infinity_antiholomorphic_ground_parity)
        hf = term.labels["holomorphic_form_parities"]
        af = term.labels["antiholomorphic_form_parities"]
        routes = [r for r in sphere_parity_routes(hol, anti)
                  if r.holomorphic_form_parities == tuple(hf)
                  and r.antiholomorphic_form_parities == tuple(af)]
        if not routes:
            # The unchanged block generator gives zero for this mismatch.
            continue
        if len(routes) != 1:
            raise AssertionError("a homogeneous term must have one internal parity pair")
        route, legacy_phase = routes[0], 1
        if assembly == "graded_full_product":
            p, pb = route.internal_parities
            field = term.labels["internal_local_field"]
            if field == "R_plus" and p == pb:
                legacy_phase = (-1j) ** (2 * p)
            elif field == "R_minus" and p != pb:
                sl, sr = term.labels["signs"]
                legacy_phase = sl * sr
            else:
                raise ValueError("legacy field label and internal parities disagree")
        labels = dict(term.labels, graded_sewing_sign=route.sign,
                      graded_internal_parities=route.internal_parities,
                      graded_sign_exponents=(route.left_separation_exponent,
                                             route.right_separation_exponent,
                                             route.inverse_gram_exponent),
                      diagnostic_assembly=assembly)
        terms.append(replace(term, coefficient=term.coefficient * route.sign / legacy_phase,
                             labels=labels))
    return replace(kernel, terms=tuple(terms))


def _pairs(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def _relative(left, right):
    return np.abs(left - right) / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)


def _build(p, order, momenta, species, picture, channel):
    return build_fixed_p_two_ramond_kernel(
        p, external_liouville_momenta=momenta,
        time_momenta=(*momenta[:3], -momenta[3]),
        ns_at_z=species[0], ns_at_one=species[1],
        zero_ramond_family="Psi_tilde", infinity_ramond_family="Psi",
        channel=channel, picture_zero_at=picture, maximum_twice_level=order,
        crossed_block_backend="double_virasoro" if channel == "crossed" else "inverse_gram",
        direct_block_backend="double_virasoro", allow_uncertified_direct_projector=channel == "direct")


def _evaluate(kernel, points):
    values = _uniformize_kernel(kernel, elliptic_prefactor=True).evaluate_many(points)
    if any(not np.all(np.isfinite(v)) for v in values.values()):
        raise ArithmeticError("nonfinite kernel: node must not be dropped")
    return values


def run(*, orders=(2, 4), nodes=12, species="VV", picture="z", momenta=MOMENTA,
        points=(.35+.1j, .5+.1j, .65+.1j), threshold_options=None, progress=False):
    orders, momenta, points = tuple(orders), tuple(map(complex, momenta)), np.asarray(points, dtype=complex)
    if len(momenta) != 4:
        raise ValueError("expected four momenta")
    require_conservative_real_p_chamber(momenta[:3], momenta[3])
    if not orders or any(not isinstance(n, int) or n < 2 for n in orders):
        raise ValueError("orders must contain twice-level cutoffs >= 2")
    if species not in ("SS", "SV", "VS", "VV") or picture not in ("z", "one"):
        raise ValueError("invalid species or picture")
    if (points.ndim != 1 or len(points) == 0 or not np.all(np.isfinite(points))
            or np.any((points == 0) | (points == 1))):
        raise ValueError("points must be a finite nonempty grid away from punctures")
    rule = threshold_weighted_rule(nodes, options=threshold_options)
    rows, cache = [], {}
    for order in orders:
        started = time.perf_counter()
        effective = {"crossed": order, "direct": order - order % 2}
        local = {}
        for assembly in ("NS_reference",) + ASSEMBLIES:
            channel = "crossed" if assembly == "NS_reference" else "direct"
            values = []
            for position in ("z", "one"):
                key = (channel, effective[channel], .37, position)
                if key not in cache:
                    cache[key] = _build(.37, effective[channel], momenta, species, position, channel)
                kernel = cache[key]
                if channel == "direct":
                    kernel = graded_direct_kernel(kernel, assembly=assembly)
                values.append(_evaluate(kernel, points))
            local[assembly] = {name: dict(
                picture_z=_pairs(values[0][name]), picture_one=_pairs(values[1][name]),
                relative_defect=_relative(values[0][name], values[1][name]).tolist(),
            ) for name in values[0]}
        if progress:
            print(f"N={order} local picture defects: " + ", ".join(
                f"{name}={max(data['F0' if species=='VV' else next(iter(data))]['relative_defect']):.6g}"
                for name, data in local.items()), file=sys.stderr, flush=True)
        samples = {name: {} for name in ("NS_reference",) + ASSEMBLIES}
        for index, p in enumerate(rule.momenta):
            for channel in ("crossed", "direct"):
                key = (channel, effective[channel], float(p), picture)
                if key not in cache:
                    cache[key] = _build(float(p), effective[channel], momenta, species, picture, channel)
                original = cache[key]
                names = ("NS_reference",) if channel == "crossed" else ASSEMBLIES
                for name in names:
                    kernel = original if channel == "crossed" else graded_direct_kernel(original, assembly=name)
                    for tensor, values in _evaluate(kernel, points).items():
                        samples[name].setdefault(tensor, []).append(values)
            if progress:
                print(f"N={order}, node {index+1}/{len(rule.momenta)}, P={p:.6g}, "
                      f"elapsed={time.perf_counter()-started:.1f}s", file=sys.stderr, flush=True)
        integrated, regions = {}, {}
        for name, tensors in samples.items():
            integrated[name], regions[name] = {}, {}
            for tensor, values in tensors.items():
                weighted = np.asarray(values) * (rule.weights / np.pi)[:, None]
                integrated[name][tensor] = np.asarray([_stable_complex_fsum(v) for v in weighted.T])
                regions[name][tensor] = {label: _pairs([
                    _stable_complex_fsum(v) for v in weighted[rule.regions == region].T])
                    for region, label in enumerate(("endpoint", "bulk", "tail"))}
        overlap = {name: {tensor: dict(
            absolute=np.abs(value - integrated["NS_reference"][tensor]).tolist(),
            relative_to_larger=_relative(value, integrated["NS_reference"][tensor]).tolist(),
        ) for tensor, value in integrated[name].items()} for name in ASSEMBLIES}
        rows.append(dict(maximum_twice_level=order, channel_cutoffs=effective,
                         local_picture_at_P037=local,
                         densities={n: {t: _pairs(v) for t, v in d.items()} for n, d in integrated.items()},
                         spectral_regions=regions, overlap_defect=overlap,
                         seconds=time.perf_counter()-started))
        if progress:
            print(f"N={order} overlap defects: " + ", ".join(
                f"{name}={data['F0' if species=='VV' else next(iter(data))]['relative_to_larger']}"
                for name, data in overlap.items()), file=sys.stderr, flush=True)
    sources = (Path(__file__),) + tuple(Path(__file__).with_name(n) for n in (
        "graded_sphere_sewing.py", "so7e8_ramond_sphere_integrand.py", "so7e8_ramond_cocycles.py",
        "so7e8_internal_ramond_double_virasoro.py", "so7e8_mixed_double_virasoro.py",
        "spin23_super_liouville_data.py", "so7e8_two_ramond_liouville_integral.py"))
    return dict(backend="double_virasoro_then_elliptic_H", full_pbw_block_used=False,
                production_certified=False, phase_fitted=False,
                interpretation="sign-insertion diagnostic with unchanged RRNS/external data; full-product variant is not a GSO prescription",
                species=species, picture=picture, momenta=_pairs(momenta), moduli=_pairs(points),
                measure="integral_0^infinity dP/pi; no modulus integral",
                momentum_rule=rule.metadata(), rows=rows,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--species", choices=("SS", "SV", "VS", "VV"), default="VV")
    parser.add_argument("--picture", choices=("z", "one"), default="z")
    parser.add_argument("--threshold-options", type=json.loads)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(orders=args.orders, nodes=args.nodes, species=args.species,
                 picture=args.picture, threshold_options=args.threshold_options, progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")
