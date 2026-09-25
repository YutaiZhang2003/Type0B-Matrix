"""Pointwise CFT audit, before free fields, PCO sums and moduli integration.

The legacy nome adapter is a diagnostic-only copy of the algebraic conversion
in spin23_singlet_amplitudes.py at 303cdf4. Production remains ordinary sewing.
No proposed amplitude formula is imported, fitted or used as a reference.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import spin23_singlet_amplitudes as s


COMPONENTS = tuple((left, right) for right in ("P", "O", "A", "M") for left in ("P", "M"))
PROCESSES = {"vvvv": "P", "ssvv": "O", "ssss": "A", "v_to_vss": "M"}
POINTS = (
    ("near_s_OPE", .06 + .02j),
    ("s_bulk", .25 + .10j),
    ("overlap_left", .42 + .13j),
    ("overlap_right", .53 + .10j),
    ("near_t_OPE", .94 + .02j),
    ("negative_bulk", -.80 + .10j),
    ("upper_bulk", .45 + .75j),
    ("upper_boundary", .49 + .84j),
    ("negative_boundary", -.95 + .05j),
)


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


@lru_cache(maxsize=None)
def modular_data(length):
    return s.ref._modular_series(length)


def legacy_coefficients(block, order):
    """Use exactly the available sewing coefficients, with no new CFT input."""
    if not 0 <= order <= block.q_order:
        raise ValueError("legacy comparison cannot use unavailable coefficients")
    length = 2 * order + 3
    z_series, u_series, theta3_series = modular_data(length)
    _, h2, h3, _ = block.effective_weights
    one_minus_z = -z_series.copy()
    one_minus_z[0] += 1
    theta_exponent = 6 - 4 * sum(block.effective_weights)
    prefactor = s.ref._series_multiply(
        s.ref._series_power(u_series, block.h_internal - .5, length),
        s.ref._series_power(one_minus_z, h2 + h3 - .5, length), length,
    )
    prefactor = s.ref._series_multiply(
        prefactor, s.ref._series_power(theta3_series, -theta_exponent, length), length,
    )
    even = s.ref._series_multiply(
        prefactor, s.ref._series_compose(block.even_z[:order+1], z_series, length), length,
    )
    even[2*order+1:] = 0
    sqrt_z = np.zeros(length, dtype=complex)
    sqrt_z[1:] = 4 * s.ref._series_power(u_series, .5, length)[:-1]
    odd = s.ref._series_multiply(prefactor, s.ref._series_multiply(
        sqrt_z, s.ref._series_compose(block.odd_z[:order+1], z_series, length), length,
    ), length)
    odd[2*order+2:] = 0
    return even, odd


def legacy_value(block, parity, coordinate, order, *, coefficients=None, geometry=None):
    """Reconstruct the old PLANE block, not a pillow correlator or measure."""
    if coefficients is None:
        coefficients = legacy_coefficients(block, order)
    qhat, theta3 = s.ref._q_grid(coordinate) if geometry is None else geometry
    h1, h2, h3, h4 = block.effective_weights
    prefactor = np.exp(
        (block.h_internal - .5)*np.log(16*qhat)
        + (.5-h1-h2)*np.log(coordinate)
        + (.5-h2-h3)*np.log(1-coordinate)
        + (6-4*(h1+h2+h3+h4))*np.log(theta3)
    )
    return prefactor * np.polynomial.polynomial.polyval(np.sqrt(qhat), coefficients[parity])


def cft_components(data, coordinate, order, representation="sewing"):
    """Integrate each physical SL component, with unchanged complex weights.

    Anti-chiral blocks are evaluated at zbar with the SAME external weights;
    at complex energies they are not the complex conjugates of chiral blocks.
    Return sum-of-absolute-summands scales to expose spectral cancellation.
    """
    if representation not in ("sewing", "legacy_nome"):
        raise ValueError("unknown representation")
    coordinate = np.asarray(coordinate, dtype=complex)
    if coordinate.ndim != 1 or not 0 <= order <= data.kernels[0].blocks["P"].q_order:
        raise ValueError("one-dimensional coordinates and available order required")
    geometry = None
    if representation == "legacy_nome":
        geometry = (s.ref._q_grid(coordinate), s.ref._q_grid(coordinate.conj()))
    values = {left+right: [] for left, right in COMPONENTS}
    scales = {name: [] for name in values}
    for kernel in data.kernels:
        evaluated = {}
        for name, block in kernel.blocks.items():
            coefficients = legacy_coefficients(block, order) if geometry is not None else None
            for side, z in enumerate((coordinate, coordinate.conj())):
                for parity in (0, 1):
                    evaluated[name, side, parity] = (
                        block.direct_value(parity, z, order) if geometry is None else
                        legacy_value(block, parity, z, order, coefficients=coefficients, geometry=geometry[side])
                    )
        for left, right in COMPONENTS:
            terms = []
            for parity, structure in ((0, kernel.even_structure), (1, kernel.odd_structure)):
                lp = s.internal_parity(kernel.blocks[left].words, parity)
                rp = s.internal_parity(kernel.blocks[right].words, parity)
                terms.append(kernel.quadrature_weight * structure * evaluated[left, 0, lp] * evaluated[right, 1, rp])
            values[left+right].append(terms[0]+terms[1])
            scales[left+right].append(abs(terms[0])+abs(terms[1]))
    return (
        {name: np.asarray(np.sum(terms, axis=0, dtype=np.clongdouble), dtype=complex) for name, terms in values.items()},
        {name: np.asarray(np.sum(terms, axis=0, dtype=np.longdouble), dtype=float) for name, terms in scales.items()},
    )


def snapshot(atlas, order, representation="sewing", chart="selected", ordering="original"):
    z = np.asarray([value for _, value in POINTS])
    use_t = abs(1-z) < abs(z) if chart == "selected" else np.full(z.size, chart == "t")
    if chart not in ("selected", "s", "t"):
        raise ValueError("unknown chart")
    values = {left+right: np.empty_like(z) for left, right in COMPONENTS}
    scales = {name: np.empty(z.size) for name in values}
    for crossed in (False, True):
        mask = use_t == crossed
        if not np.any(mask):
            continue
        data = getattr(atlas, ordering + ("_t" if crossed else "_s"))
        coordinate = 1-z[mask] if crossed else z[mask]
        result, norms = cft_components(data, coordinate, order, representation)
        for name in result:
            values[name][mask], scales[name][mask] = result[name], norms[name]
    energies = getattr(atlas, ordering+"_s").energies
    lam = energies[1]*energies[2]/(1-z)
    rows = []
    for i, (label, point) in enumerate(POINTS):
        raw = {"C_"+name: value[i] for name, value in values.items()}
        pco_scales = {}
        for process, right in PROCESSES.items():
            primary, descendant = values["P"+right][i], values["M"+right][i]
            raw["B_"+process] = descendant + lam[i]*primary
            pco_scales["B_"+process] = abs(descendant)+abs(lam[i]*primary)
        rows.append({
            "point": label, "z": pair(point), "chart": "t" if use_t[i] else "s",
            "abs_sewing_coordinate": float(abs(1-point) if use_t[i] else abs(point)),
            "values": {name: pair(value) for name, value in raw.items()},
            "spectral_cancellation": {"C_"+name: float(scales[name][i]/max(abs(value[i]), 1e-300)) for name, value in values.items()},
            "pco_cancellation": {name: float(scale/max(abs(raw[name]), 1e-300)) for name, scale in pco_scales.items()},
        })
    return {"order": order, "representation": representation, "chart": chart, "ordering": ordering, "points": rows}


def compare(reference, candidate):
    """Movement relative to the reference; retain absolute differences too."""
    rows = []
    for old, new in zip(reference["points"], candidate["points"], strict=True):
        if old["point"] != new["point"]:
            raise ValueError("point grids differ")
        changes = {}
        for name, value in old["values"].items():
            a, b = complex(*value), complex(*new["values"][name])
            delta = abs(b-a)
            changes[name] = {"absolute": delta, "relative_to_reference": delta/max(abs(a), 1e-300),
                             "symmetric_relative": delta/max(abs(a), abs(b), 1e-300)}
        rows.append({"point": old["point"], "z": old["z"], "changes": changes})
    return {"points": rows, "max_CFT_relative": max(v["relative_to_reference"] for row in rows
               for name, v in row["changes"].items() if name.startswith("C_")),
            "max_PCO_relative": max(v["relative_to_reference"] for row in rows
               for name, v in row["changes"].items() if name.startswith("B_"))}


def build(energies, order, nodes, scheme="threshold_weighted", backend="c_recursion"):
    started = time.perf_counter()
    atlas = s._build_atlas(energies, q_order=order, p_nodes=nodes, p_max=4.0, p_cut=.03,
                           momentum_scheme=scheme, infinite_gauss_scale=.5,
                           gram_condition_limit=1e13, block_backend=backend)
    print(json.dumps({"built_order": order, "scheme": scheme, "backend": backend,
                      "nodes": len(atlas.original_s.kernels), "seconds": time.perf_counter()-started}), flush=True)
    return atlas


def audit_case(label, energies, orders):
    maximum = max(orders)
    fine = build(energies, maximum, (16, 96, 32))
    orderings = ("original",) if energies[0] == energies[1] == energies[2] else ("original", "swapped")
    sweeps = {ordering: {str(order): snapshot(fine, order, ordering=ordering) for order in orders} for ordering in orderings}
    crossing = {}
    for order in orders:
        # Both full spectral components, with crossed external patterns and
        # phases supplied by the atlas; NEVER compare individual blocks.
        direct = snapshot(fine, order, chart="s")
        crossed = snapshot(fine, order, chart="t")
        comparison = compare(direct, crossed)
        comparison["points"] = [row for row in comparison["points"]
            if abs(complex(*row["z"])) < .99 and abs(1-complex(*row["z"])) < .99]
        # Exclude non-overlap points from maxima as well as the displayed rows.
        for prefix, key in (("C_", "max_CFT_relative"), ("B_", "max_PCO_relative")):
            comparison[key] = max(change["relative_to_reference"] for row in comparison["points"]
                                  for name, change in row["changes"].items() if name.startswith(prefix))
        crossing[str(order)] = {"s": direct, "t": crossed, "comparison_in_overlap": comparison}
    print(json.dumps({"completed": label, "stage": "sewing_sweep_and_crossing"}), flush=True)

    comparison_order = 7 if maximum >= 7 else maximum
    baseline = snapshot(fine, comparison_order)
    momentum = []
    for name, counts, scheme in (
        ("threshold_100", (12, 64, 24), "threshold_weighted"),
        ("logarithmic_128", 128, "infinite_gauss"),
        ("historical_segmented_Pmax4", "segmented", "cutoff"),
    ):
        other = build(energies, comparison_order, counts, scheme)
        values = snapshot(other, comparison_order)
        momentum.append({"label": name, "quadrature": other.momentum_quadrature, "values": values,
                         "change_from_threshold_144": compare(baseline, values)})

    legacy_orders = sorted(set((comparison_order, min(maximum, comparison_order+2))))
    legacy = {str(order): snapshot(fine, order, representation="legacy_nome", chart="s") for order in legacy_orders}
    same_chart_legacy = snapshot(fine, comparison_order, representation="legacy_nome")
    # Same coefficients, same spectral quadrature, only the finite-series
    # representation changes. The old implementation used the s chart in bulk.
    conversion = {
        "same_selected_chart_reference_legacy": compare(same_chart_legacy, baseline),
        "legacy_selected_chart": same_chart_legacy,
        "historical_s_chart_reference_legacy": compare(legacy[str(comparison_order)], baseline),
        "legacy_s_order_sweep": legacy,
        "legacy_order_change": compare(legacy[str(legacy_orders[0])], legacy[str(legacy_orders[-1])]),
    }
    print(json.dumps({"completed": label, "stage": "quadrature_and_legacy_comparison"}), flush=True)

    # A matched-node, low-level complete-CFT comparison to the expensive Gram
    # oracle. Higher-level coefficient checks live in the separate c-recursion
    # validation report; this check does not pretend to test Gram at order 11.
    oracle_order = 3
    recursive = build(energies, oracle_order, (8, 32, 16))
    oracle = build(energies, oracle_order, (8, 32, 16), backend="inverse_gram")
    gram, recur = snapshot(oracle, oracle_order), snapshot(recursive, oracle_order)
    backend_comparison = {"order": oracle_order, "nodes": len(oracle.original_s.kernels),
                          "gram": gram, "c_recursion": recur, "change_from_gram": compare(gram, recur)}
    return {"label": label, "energies": [pair(value) for value in energies],
            "quadrature": fine.momentum_quadrature,
            "sewing_order_sweeps": sweeps,
            "sewing_order_changes": {ordering: {f"{lo}_to_{hi}": compare(sweeps[ordering][str(lo)], sweeps[ordering][str(hi)])
                                      for lo, hi in zip(orders[:-1], orders[1:])} for ordering in orderings},
            "crossing": crossing, "momentum_comparisons": momentum,
            "representation_comparison": conversion, "backend_comparison": backend_comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[5, 7, 9, 11])
    parser.add_argument("--cases", choices=("equal", "unequal"), nargs="+", default=["equal", "unequal"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh report path")
    if len(args.orders) < 2 or sorted(set(args.orders)) != args.orders or args.orders[0] < 1:
        parser.error("orders must be distinct, increasing positive integers")
    omega = 1/9 + .2j
    cases = {"equal": (omega, omega, omega, 3*omega),
             "unequal": (.11+.15j, .17+.16j, .23+.18j, .51+.49j)}
    report = {
        "status": "pointwise_CFT_audit_not_amplitude_precision_certificate",
        "definition": "C_XY = integral_0^infinity dP/pi sum_sigma C_sigma(12P) C_sigma(34P) F_X Fbar_Y; B_process=C_MY+w2*w3/(1-z)*C_PY",
        "excluded": ["free-time-boson factor", "spectator fermion correlator", "moduli integral", "sphere normalization", "candidate amplitude formula"],
        "physical_conventions": "c=13.5, descendant-aware seeds and component phases unchanged; anti-chiral complex energies NOT conjugated",
        "production_series": "q_s=z, q_t=1-z; even levels through N, odd through N+1/2",
        "legacy_scope": "diagnostic only; exact former algebraic conversion of the SAME coefficients, not h recursion or production",
        "error_metric": "abs(new-old)/abs(reference), also raw values, absolute and symmetric differences, cancellation scales; none is a rigorous error bound",
        "provenance": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(s.__file__), Path(s.__file__).with_name("spin23_ns_c_recursion.py"),
            Path(s.ref.__file__).with_name("liouville_momentum_quadrature.py"))},
        "cases": [],
    }
    for label in args.cases:
        report["cases"].append(audit_case(label, cases[label], args.orders))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
