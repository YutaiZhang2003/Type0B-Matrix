#!/usr/bin/env python3
"""Falsification-oriented audit of the Spin(23) sphere crossing check.

This script asks whether the small crossing residual could be an implementation
artifact.  It verifies that the direct and crossed channels do not share block
objects, measures the large mismatch at fixed internal momentum, applies
deliberately wrong parity/routing controls, and compares the accelerated backend
with independently instantiated high-precision recursion objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
FIT_DIR = CODE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
OUTPUT = (
    WORKSPACE_DIR
    / "Data Set"
    / "crossing_equation"
    / "spin23_crossing_faithfulness_c_recursion_elliptic_audit.json"
)
if str(FIT_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_DIR))

import heterotic_so23_1to3 as reference  # noqa: E402
import heterotic_so23_1to3_fast as fast  # noqa: E402


MOMENTA = (0.20, 0.35, 0.50, 0.65)
Z_AUDIT = np.asarray(
    [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99],
    dtype=complex,
)


@dataclass
class ReferenceKernel:
    p: float
    weight: float
    primary: reference.SewingNSBlock
    even_structure: complex
    odd_structure: complex


def evaluate(
    data: Sequence[Any],
    z: np.ndarray,
    *,
    even_scale: complex = 1.0,
    odd_scale: complex = 1.0,
) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    zbar = np.conjugate(z)
    result = np.zeros_like(z)
    for kernel in data:
        even = kernel.primary.value(z, "e")
        odd = kernel.primary.value(z, "o")
        evenbar = kernel.primary.value(zbar, "e")
        oddbar = kernel.primary.value(zbar, "o")
        result += kernel.weight * (
            even_scale * kernel.even_structure * even * evenbar
            + odd_scale * kernel.odd_structure * odd * oddbar
        )
    return result


def evaluate_one_kernel(kernel: Any, z: complex) -> complex:
    """Evaluate one fixed-P spectral density, excluding its quadrature weight."""
    value = evaluate([kernel], np.asarray([z], dtype=complex))[0]
    return value / kernel.weight


def relative(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-300)
    return np.abs(left - right) / scale


def build_production(q_order: int) -> tuple[list[Any], list[Any], dict[str, Any]]:
    caches: dict[str, dict[Any, Any]] = {
        "structure": {},
        "block": {},
        "elliptic": {},
    }
    diagnostics: dict[str, Any] = {}
    common = dict(
        series_parameter="elliptic_nome",
        q_order=q_order,
        p_nodes="segmented",
        p_max=6.0,
        diagnostics=diagnostics,
        structure_cache=caches["structure"],
        block_cache=caches["block"],
        elliptic_cache=caches["elliptic"],
    )
    s_channel = fast.build_s_channel_data(MOMENTA, **common)
    crossed = (MOMENTA[2], MOMENTA[1], MOMENTA[0], MOMENTA[3])
    t_channel = fast.build_s_channel_data(crossed, **common)
    diagnostics["structure_cache_entries"] = len(caches["structure"])
    diagnostics["block_cache_entries"] = len(caches["block"])
    diagnostics["plane_block_cache_entries"] = len(caches["elliptic"])
    return s_channel, t_channel, diagnostics


def build_reference_channel(
    momenta: Sequence[complex],
    q_order: int,
    p_nodes: int,
    p_max: float,
) -> list[ReferenceKernel]:
    """Build the same quadrature with reference mp recursion and Barnes G."""
    w1, w2, w3, w4 = map(complex, momenta)
    h1, h2, h3, h4 = map(reference.h_of_p, (w1, w2, w3, w4))
    ps, weights = fast._p_quadrature(p_nodes, p_max)
    kernels: list[ReferenceKernel] = []
    for p, weight in zip(ps, weights):
        p = float(p)
        hp = reference.h_of_p(p)
        block = reference.NSBlockComputer(
            h4, h3, h2, h1, False, False, max_level2=2 * q_order + 1
        )
        kernels.append(
            ReferenceKernel(
                p=p,
                weight=float(weight) / np.pi,
                primary=reference.ResummedNSBlock(block, hp, q_order),
                even_structure=complex(
                    reference.c_even(w1, w2, p) * reference.c_even(w3, w4, p)
                ),
                odd_structure=complex(
                    reference.c_odd(w1, w2, p) * reference.c_odd(w3, w4, p)
                ),
            )
        )
    return kernels


def build_fast_spot_channel(
    momenta: Sequence[complex], q_order: int, p_nodes: int, p_max: float
) -> list[Any]:
    return fast.build_s_channel_data(
        series_parameter="elliptic_nome",
        energies=momenta,
        q_order=q_order,
        p_nodes=p_nodes,
        p_max=p_max,
        structure_cache={},
        block_cache={},
        elliptic_cache={},
    )


def channel_weight_tuple(kernel: Any) -> list[dict[str, float]]:
    block = kernel.primary.block
    result = []
    for value in (block.h1, block.h2, block.h3, block.h4):
        value = complex(value)
        result.append({"real": value.real, "imag": value.imag})
    return result


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-order", type=int, default=9)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference-q-order", type=int, default=5)
    parser.add_argument("--reference-p-nodes", type=int, default=16)
    args = parser.parse_args()

    print("building production direct and crossed channels", flush=True)
    s_channel, t_channel, diagnostics = build_production(args.q_order)
    assert len(s_channel) == len(t_channel)
    assert np.allclose(
        [kernel.p for kernel in s_channel], [kernel.p for kernel in t_channel]
    )

    s_primary_ids = {id(kernel.primary) for kernel in s_channel}
    t_primary_ids = {id(kernel.primary) for kernel in t_channel}
    shared_primary_objects = s_primary_ids.intersection(t_primary_ids)
    even_structure_differences = np.asarray(
        [
            abs(s.even_structure - t.even_structure)
            / max(abs(s.even_structure), abs(t.even_structure), 1.0e-300)
            for s, t in zip(s_channel, t_channel)
        ]
    )
    odd_structure_differences = np.asarray(
        [
            abs(s.odd_structure - t.odd_structure)
            / max(abs(s.odd_structure), abs(t.odd_structure), 1.0e-300)
            for s, t in zip(s_channel, t_channel)
        ]
    )
    same_structure_nodes = int(
        np.sum(
            (even_structure_differences < 1.0e-13)
            & (odd_structure_differences < 1.0e-13)
        )
    )

    gs = evaluate(s_channel, Z_AUDIT)
    gt = evaluate(t_channel, 1.0 - Z_AUDIT)
    correct_residual = relative(gs, gt)

    # Negative controls: these must fail if the comparison is sensitive to the
    # crossed routing and to the even/odd superconformal sectors.
    wrong_same_channel = evaluate(s_channel, 1.0 - Z_AUDIT)
    even_s = evaluate(s_channel, Z_AUDIT, odd_scale=0.0)
    even_t = evaluate(t_channel, 1.0 - Z_AUDIT, odd_scale=0.0)
    odd_s = evaluate(s_channel, Z_AUDIT, even_scale=0.0)
    odd_t = evaluate(t_channel, 1.0 - Z_AUDIT, even_scale=0.0)
    flipped_t = evaluate(t_channel, 1.0 - Z_AUDIT, odd_scale=-1.0)
    shifted_t = evaluate(t_channel, 1.0 - Z_AUDIT, odd_scale=1.01)

    negative_controls = {
        "reuse_s_channel_as_t": float(np.max(relative(gs, wrong_same_channel))),
        "even_sector_only": float(np.max(relative(even_s, even_t))),
        "odd_sector_only": float(np.max(relative(odd_s, odd_t))),
        "flip_t_odd_sign": float(np.max(relative(gs, flipped_t))),
        "increase_t_odd_by_one_percent": float(np.max(relative(gs, shifted_t))),
    }

    fixed_p: list[dict[str, Any]] = []
    for target_p in (0.30, 0.90, 1.80):
        index = int(np.argmin(np.abs(np.asarray([k.p for k in s_channel]) - target_p)))
        for z in (0.20, 0.50, 0.80):
            s_value = evaluate_one_kernel(s_channel[index], z)
            t_value = evaluate_one_kernel(t_channel[index], 1.0 - z)
            fixed_p.append(
                {
                    "target_p": target_p,
                    "quadrature_p": s_channel[index].p,
                    "z": z,
                    "relative_mismatch": float(
                        relative(np.asarray([s_value]), np.asarray([t_value]))[0]
                    ),
                }
            )

    print("building independent reference spot check", flush=True)
    reference_z = np.asarray([0.10, 0.35, 0.50, 0.65, 0.90], dtype=complex)
    crossed_momenta = (MOMENTA[2], MOMENTA[1], MOMENTA[0], MOMENTA[3])
    reference_s = build_reference_channel(
        MOMENTA, args.reference_q_order, args.reference_p_nodes, 4.0
    )
    reference_t = build_reference_channel(
        crossed_momenta, args.reference_q_order, args.reference_p_nodes, 4.0
    )
    fast_s = build_fast_spot_channel(
        MOMENTA, args.reference_q_order, args.reference_p_nodes, 4.0
    )
    fast_t = build_fast_spot_channel(
        crossed_momenta, args.reference_q_order, args.reference_p_nodes, 4.0
    )
    reference_gs = evaluate(reference_s, reference_z)
    reference_gt = evaluate(reference_t, 1.0 - reference_z)
    fast_gs = evaluate(fast_s, reference_z)
    fast_gt = evaluate(fast_t, 1.0 - reference_z)

    report = {
        "status": "pass",
        "external_momenta": list(MOMENTA),
        **reference.elliptic_conversion.representation_metadata("elliptic_nome",args.q_order),
        "scope": "backend/routing audit; not a finite-order crossing accuracy certification",
        "production_controls": {
            "q_order": args.q_order,
            "p_max": 6.0,
            "p_nodes": len(s_channel),
            "z_points": [float(value.real) for value in Z_AUDIT],
        },
        "channel_independence": {
            "shared_primary_block_objects": len(shared_primary_objects),
            "nodes_with_relatively_identical_structure_products": same_structure_nodes,
            "minimum_even_structure_product_relative_difference": float(
                np.min(even_structure_differences)
            ),
            "median_even_structure_product_relative_difference": float(
                np.median(even_structure_differences)
            ),
            "minimum_odd_structure_product_relative_difference": float(
                np.min(odd_structure_differences)
            ),
            "median_odd_structure_product_relative_difference": float(
                np.median(odd_structure_differences)
            ),
            "s_first_block_weights_h1_h2_h3_h4": channel_weight_tuple(s_channel[0]),
            "t_first_block_weights_h1_h2_h3_h4": channel_weight_tuple(t_channel[0]),
            "backend_diagnostics": diagnostics,
        },
        "correct_crossing": {
            "maximum_relative_residual": float(np.max(correct_residual)),
            "worst_z": float(Z_AUDIT[int(np.argmax(correct_residual))].real),
            "residual_by_z": [float(value) for value in correct_residual],
        },
        "negative_controls_maximum_relative_residual": negative_controls,
        "fixed_p_is_not_crossing_invariant": {
            "minimum_relative_mismatch": float(
                min(row["relative_mismatch"] for row in fixed_p)
            ),
            "maximum_relative_mismatch": float(
                max(row["relative_mismatch"] for row in fixed_p)
            ),
            "samples": fixed_p,
        },
        "independent_reference_spot_check": {
            "q_order": args.reference_q_order,
            "p_nodes": len(reference_s),
            "p_max": 4.0,
            "z_points": [float(value.real) for value in reference_z],
            "maximum_fast_vs_reference_s_relative_difference": float(
                np.max(relative(fast_gs, reference_gs))
            ),
            "maximum_fast_vs_reference_t_relative_difference": float(
                np.max(relative(fast_gt, reference_gt))
            ),
            "maximum_reference_crossing_residual": float(
                np.max(relative(reference_gs, reference_gt))
            ),
            "maximum_fast_crossing_residual": float(
                np.max(relative(fast_gs, fast_gt))
            ),
        },
    }

    # Hard failure conditions are deliberately conservative.  The negative
    # controls should miss crossing by orders of magnitude more than the real
    # calculation, and the two channels must not share block instances.
    if shared_primary_objects or same_structure_nodes:
        report["status"] = "fail"
    if min(negative_controls.values()) < 1.0e-5:
        report["status"] = "fail"
    if report["independent_reference_spot_check"][
        "maximum_fast_vs_reference_s_relative_difference"
    ] > 1.0e-8:
        report["status"] = "fail"
    if report["independent_reference_spot_check"][
        "maximum_fast_vs_reference_t_relative_difference"
    ] > 1.0e-8:
        report["status"] = "fail"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, default=json_default)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
