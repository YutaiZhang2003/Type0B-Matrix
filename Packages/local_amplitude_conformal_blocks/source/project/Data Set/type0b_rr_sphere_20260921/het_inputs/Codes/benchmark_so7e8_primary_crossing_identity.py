"""Primary RRNN correlator audit, with the normalization conflict exposed.

This is NOT the GSO/PCO heterotic amplitude. It keeps the stored BRY
structure functions unchanged and reports two different inverse metrics:
the documented D_R=D_NS and the identity-limit-compatible D_R=D_NS/2.
The latter is a diagnostic prediction, never a crossing-fitted multiplier.
No production kernel is changed or enabled by this driver.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from so7e8_mixed_double_virasoro import two_ramond_internal_ns_sld_double_virasoro_h_series as ns_block
from so7e8_internal_ramond_double_virasoro import ramond_sld_double_virasoro_h_series as r_block
from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants
from so7e8_sphere_branches import branch_metadata, sphere_channel_coordinates


MOMENTA = (.02+.22j, .03+.23j, .04+.24j, .09+.69j)
POINTS = (.35+.1j, .5+.1j, .65+.1j)
CONTOURS = (
    dict(radius=.10, check_radius=.12, samples=32, tolerance=1e-8),
    dict(radius=.10, check_radius=.12, samples=48, tolerance=1e-8),
    dict(radius=.08, check_radius=.10, samples=48, tolerance=1e-8),
)


def pairs(x):
    a = np.asarray(x, complex)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def unpairs(x):
    a = np.asarray(x, float)
    return a[..., 0] + 1j*a[..., 1]


def truncate_h(series, n):
    if n > series.source.maximum_twice_level:
        raise ValueError("cannot manufacture higher-order H coefficients")
    k = min(n, series.known_through_twice_level)
    return replace(series, h_coefficients=series.h_coefficients[:k+1],
                   known_through_twice_level=k)


def identity_limit_audit():
    """Same identity insertion and regulator in NS and R three-point data.

    The common leg normalization cancels. For P3=i(1-epsilon), P1=P2=P,
    Upsilon shift identities give E/C -> 1 and O/C -> 0. Hence
    ((E +/- O)/2)/C -> 1/2, in conflict with equal recorded two-point norms.
    No block or four-point correlator is used here.
    """
    rows = []
    for p in (.21, .7, 1.1):
        for epsilon in (.01, .001, .0001, .00001):
            identity_momentum = 1j*(1-epsilon)
            c = ns_structure_constants(p, p, identity_momentum, precision=70)[0]
            e, o = rr_ns_structure_constants(p, p, identity_momentum, precision=70)
            rows.append(dict(P=p, epsilon=epsilon, E_over_C=pairs(e/c),
                             O_over_C=pairs(o/c), Rplus_over_NS=pairs((e+o)/(2*c)),
                             Rminus_over_NS=pairs((e-o)/(2*c))))
    return dict(predicted_R_over_NS_norm=.5, documented_R_over_NS_norm=1.,
                fitted_to_crossing=False, rows=rows)


def checked_block(builder, **kwargs):
    attempts = []
    previous = None
    for index, options in enumerate(CONTOURS):
        try:
            result = builder(**kwargs, **options)
        except ArithmeticError as exc:
            if "radius check failed" not in str(exc):
                raise
            attempts.append(dict(options=options, error=str(exc)))
            continue
        attempts.append(dict(options=options, success=True))
        if index == 0:
            return result, attempts
        if previous is None:
            previous = result
            continue
        # A retried node must pass two independently sampled contours.
        x, y = np.asarray(previous.h_coefficients), np.asarray(result.h_coefficients)
        if not np.allclose(x, y, rtol=1e-7, atol=1e-9):
            raise ArithmeticError("independent successful H contours disagree")
        return previous, attempts
    raise ArithmeticError(f"no verified finite-part contour: {attempts}")


def source_hashes():
    root = Path(__file__).parent
    names = [Path(__file__).name, "so7e8_mixed_double_virasoro.py",
             "so7e8_internal_ramond_double_virasoro.py", "spin23_super_liouville_data.py",
             "spin23_two_virasoro_ramond.py", "spin23_ns_torus_two_virasoro.py",
             "spin23_genus1_recursion.py", "virasoro_sphere_c_recursion.py",
             "so7e8_ramond_fourpoint.py", "ramond_sphere_uniformization.py",
             "sphere_block_uniformization.py", "so7e8_sphere_branches.py", "so7e8_human_conventions.py",
             "heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py"]
    files = [root/name for name in names]
    files += sorted((root/"ns_algebra").glob("*.py")) + sorted((root/"ramond_algebra").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def density(p, channel, maximum, momenta, points, *, cut_side="upper"):
    """All sign sums, for the same external R+ V V R+ scalar observable.

    Primary-specific anti-chiral identities of the current block interface:
    R_a=R_h, NS_e,a=NS_e,h, NS_o,a=-i NS_o,h (momenta unchanged).
    These are exact convention relations, not real-momentum conjugation.
    The NS odd product therefore needs a minus sign to recover the analytic
    continuation of the positive-line absolute square. The even/odd primary
    coefficients are then C or Ctilde times (E,O)/2. This assumption is
    recorded separately from the unsettled absolute Ramond normalization.
    """
    p1, p2, p3, p4 = momenta
    z = np.asarray(points, complex)
    frame = sphere_channel_coordinates(z, channel="crossed" if channel == "NS" else "direct", cut_side=cut_side)
    orders = range(2, maximum+1, 2)
    values = {n: np.zeros((2 if channel == "NS" else 4, len(z)), complex) for n in orders}
    retries = []
    if channel == "NS":
        c = ns_structure_constants(p3, p2, float(p))
        constants = dict(zip((1, -1), rr_ns_structure_constants(p4, p1, float(p))))
        for component, sign in product(("even", "odd"), (-1, 1)):
            h, attempts = checked_block(ns_block, internal_momentum=float(p), external_momenta=momenta,
                maximum_twice_level=maximum, component=component, rr_structure_sign=sign,
                chirality="holomorphic")
            if len(attempts) > 1:
                retries.append(dict(component=component, sign=sign, attempts=attempts))
            parity = int(component == "odd")
            # For real P, F_o^*=+i F_o but current anti-interface gives -i F_o.
            # Thus the convention-correct squared odd block is +i F_h F_h(bar z).
            scalar = c[parity]*constants[sign]/2
            phase = 1j if parity else 1
            for n in orders:
                series = truncate_h(h, n)
                values[n][parity] += scalar*phase*series.value(frame.local, cut_side=frame.holomorphic_cut_side)*series.value(frame.local_bar, cut_side=frame.antiholomorphic_cut_side)
    else:
        left = dict(zip((1, -1), rr_ns_structure_constants(p4, float(p), p3)))
        right = dict(zip((1, -1), rr_ns_structure_constants(-float(p), p1, p2)))
        for j, (sl, sr) in enumerate(product((-1, 1), repeat=2)):
            h, attempts = checked_block(r_block, internal_momentum=float(p), external_momenta=momenta,
                maximum_twice_level=maximum, left_structure_sign=sl, right_structure_sign=-sr,
                chirality="holomorphic")
            if len(attempts) > 1:
                retries.append(dict(signs=(sl, sr), attempts=attempts))
            scalar = left[sl]*right[sr]/4
            for n in orders:
                series = truncate_h(h, n)
                values[n][j] += scalar*series.value(frame.local, cut_side=frame.holomorphic_cut_side)*series.value(frame.local_bar, cut_side=frame.antiholomorphic_cut_side)
    return {str(n): pairs(v) for n, v in values.items()}, retries


def run(channel, nodes, maximum, output, cache_dir, *, momenta=MOMENTA, points=POINTS, options=None):
    started = time.perf_counter()
    identity = dict(schema=1, channel=channel, maximum=maximum, momenta=pairs(momenta),
                    points=pairs(points), sources=source_hashes(), contours=CONTOURS,
                    coordinate_branch=branch_metadata())
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/signature
    cache.mkdir(parents=True, exist_ok=True)
    rule = threshold_weighted_rule(nodes, options=dict(beta=2 if channel == "NS" else 0, **(options or {})))
    totals = {n: np.zeros((3, 2 if channel == "NS" else 4, len(points)), complex)
              for n in range(2, maximum+1, 2)}
    report = dict(status="running", parameters=identity, signature=signature,
                  rule=rule.metadata(), identity_limit=identity_limit_audit(), completed_nodes=0,
                  physical_heterotic_amplitude=False, production_modified=False,
                  absolute_normalization_resolved=False, retries=[])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["seconds"] = time.perf_counter()-started
        report["results"] = {n: dict(total=pairs(v.sum(axis=(0, 1))), parts=pairs(v.sum(axis=0)),
                                      region_totals=pairs(v.sum(axis=1))) for n, v in totals.items()}
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    try:
        for i, (p, w, region) in enumerate(zip(rule.momenta, rule.weights, rule.regions)):
            name = cache/(hashlib.sha256(float(p).hex().encode()).hexdigest()+".json")
            if name.exists():
                entry = json.loads(name.read_text())
                if entry["signature"] != signature or entry["P_hex"] != float(p).hex():
                    raise ValueError("cache identity mismatch")
            else:
                values, retries = density(p, channel, maximum, momenta, points)
                entry = dict(signature=signature, P_hex=float(p).hex(), values=values, retries=retries)
                name.write_text(json.dumps(entry, indent=2, allow_nan=False)+"\n")
            report["retries"].extend(dict(P=float(p), **r) for r in entry["retries"])
            for n in totals:
                v = unpairs(entry["values"][str(n)])
                if v.shape != totals[n].shape[1:] or not np.all(np.isfinite(v)):
                    raise ArithmeticError("invalid primary correlator density")
                totals[n][int(region)] += w/np.pi*v
            report["completed_nodes"] = i+1
            save()
            print(channel, i+1, len(rule.momenta), "P", float(p), "seconds", report["seconds"], flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    report["status"] = "complete"
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("NS", "R"), required=True)
    parser.add_argument("--nodes", type=int, default=60)
    parser.add_argument("--maximum", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--real-momenta", action="store_true")
    args = parser.parse_args()
    run(args.channel, args.nodes, args.maximum, args.output, args.cache_dir,
        momenta=(.21, .32, .43, .54) if args.real_momenta else MOMENTA)
