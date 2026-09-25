#!/usr/bin/env python3
"""Test the four possible even/odd phases of the all-singlet NS block.

This is a diagnostic, not a fit.  For one outgoing-energy permutation it
builds the conformal-block atlas once, then reevaluates the complete sphere
integral after assigning each of the four possible sign pairs to the
``A=(G_{-1/2}V)^4`` component block.  Comparing all six permutations tests
whether the observed ``S -> S S S`` asymmetry is solely an ``A``-component
phase error.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import spin23_singlet_amplitudes as singlet
from spin23_highdim_scan import read_row, row_energies


PHASE_PAIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _rephase_block(
    block: singlet._GenericBlock,
    desired_even: int,
    desired_odd: int,
) -> singlet._GenericBlock:
    """Return an ``A`` block with the requested absolute component phases."""

    current_even = singlet.component_phase(singlet.A_WORDS, 0)
    current_odd = singlet.component_phase(singlet.A_WORDS, 1)
    even_multiplier = desired_even / current_even
    odd_multiplier = desired_odd / current_odd
    return replace(
        block,
        even_z=even_multiplier * block.even_z,
        odd_z=odd_multiplier * block.odd_z,
        even_h=even_multiplier * block.even_h,
        odd_h=odd_multiplier * block.odd_h,
    )


def _rephase_channel(
    channel: singlet._ChannelData,
    desired_even: int,
    desired_odd: int,
) -> singlet._ChannelData:
    kernels = []
    for kernel in channel.kernels:
        blocks = dict(kernel.blocks)
        blocks["A"] = _rephase_block(
            blocks["A"],
            desired_even,
            desired_odd,
        )
        kernels.append(replace(kernel, blocks=blocks))
    return replace(channel, kernels=tuple(kernels))


def _rephase_atlas(
    atlas: singlet._AmplitudeAtlas,
    desired_even: int,
    desired_odd: int,
) -> singlet._AmplitudeAtlas:
    return replace(
        atlas,
        original_s=_rephase_channel(
            atlas.original_s,
            desired_even,
            desired_odd,
        ),
        original_t=_rephase_channel(
            atlas.original_t,
            desired_even,
            desired_odd,
        ),
        swapped_s=_rephase_channel(
            atlas.swapped_s,
            desired_even,
            desired_odd,
        ),
        swapped_t=_rephase_channel(
            atlas.swapped_t,
            desired_even,
            desired_odd,
        ),
    )


def evaluate_permutation(
    manifest: Path,
    row_index: int,
    *,
    q_order: int,
    p_max: float,
) -> list[dict[str, object]]:
    row = read_row(manifest, row_index)
    outgoing = row_energies(row)
    energies = (*outgoing, sum(outgoing))
    atlas = singlet._build_atlas(
        energies,
        q_order=q_order,
        p_nodes="segmented",
        p_max=p_max,
        p_cut=0.03,
        gram_condition_limit=1.0e13,
    )

    results: list[dict[str, object]] = []
    for even_phase, odd_phase in PHASE_PAIRS:
        phased_atlas = _rephase_atlas(atlas, even_phase, odd_phase)
        values = singlet._evaluate_at_order(
            phased_atlas,
            energies,
            order=q_order,
            epsilon0=0.08,
            epsilon1=0.06,
            theta_orders=(12, 12, 48),
            radial_order=20,
            disk_total_order=q_order + 11,
            crossed_disk_total_order=q_order + 11,
        )
        result: dict[str, object] = {
            "point_id": row["point_id"],
            "parent_id": row["parent_id"],
            "permutation": row["permutation"],
            "row_index": row_index,
            "q_order": q_order,
            "p_max": p_max,
            "a_even_phase": even_phase,
            "a_odd_phase": odd_phase,
            "ssss_re": values.ssss_raw.real,
            "ssss_im": values.ssss_raw.imag,
            "ssss_abs": abs(values.ssss_raw),
            "maximum_gram_condition": atlas.maximum_gram_condition,
        }
        for region, value in values.pieces["ssss"].items():
            result[f"{region}_re"] = value.real
            result[f"{region}_im"] = value.imag
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--q-order", type=int, default=4)
    parser.add_argument("--p-max", type=float, default=2.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = evaluate_permutation(
        args.manifest,
        args.row_index,
        q_order=args.q_order,
        p_max=args.p_max,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
