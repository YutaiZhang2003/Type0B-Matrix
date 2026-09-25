#!/usr/bin/env python3
"""Write the odd-resonance component and reconstruction-design manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from spin23_fivepoint_odd_resonance_catalog import (
    ODD_SCREENING_NUMBERS,
    RECONSTRUCTION_SAMPLES,
    component_manifest_rows,
    reconstruction_samples_for_sector,
)
from spin23_fivepoint_reconstruction import SECTORS


def _fraction_string(value) -> str:
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components-output",
        type=Path,
        default=Path(
            "data_exports/spin23_fivepoint_odd_resonance_components.csv"
        ),
    )
    parser.add_argument(
        "--design-output",
        type=Path,
        default=Path(
            "data_exports/spin23_fivepoint_odd_resonance_design.json"
        ),
    )
    args = parser.parse_args()

    rows = list(component_manifest_rows())
    args.components_output.parent.mkdir(parents=True, exist_ok=True)
    with args.components_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    design = {
        "status": "exact_kinematic_design_no_residues_evaluated",
        "conventions": {
            "screening_planes": list(ODD_SCREENING_NUMBERS),
            "incoming_momentum": "P0=i*(s+3)/2",
            "outgoing_variables": "Pj=i*tj and xj=i*Pj=-tj",
            "factorization_divisor": "d_A=1+sum_{j in A} x_j, |A|=2 or 3",
            "relative_component_weight": (
                "s!/(m!*ell!*2^ell)*pi^(|C|+ell)*product_{j in C}(a_j)"
            ),
        },
        "component_projection_count": len(
            {str(row["projection_label"]) for row in rows}
        ),
        "component_row_counts_by_s": {
            str(screening): count
            for screening, count in sorted(
                Counter(int(row["screening_number"]) for row in rows).items()
            )
        },
        "shared_sample_count": len(RECONSTRUCTION_SAMPLES),
        "minimality": (
            "The all-singlet contact basis has dimension 12, so fewer than "
            "12 shared rows cannot have full rank."
        ),
        "rank_scope": (
            "Exact rational rank was checked after canonical outgoing-leg "
            "permutation for each of the 26 labelled scalar projections."
        ),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "screening_number": sample.screening_number,
                "incoming_k": sample.incoming_k,
                "outgoing_t": [
                    _fraction_string(value) for value in sample.outgoing_t
                ],
                "outgoing_x": [
                    _fraction_string(value) for value in sample.outgoing_x
                ],
                "minimum_factorization_margin": _fraction_string(
                    sample.minimum_factorization_margin
                ),
            }
            for sample in RECONSTRUCTION_SAMPLES
        ],
        "sector_designs": {
            name: {
                "contact_dimension": sector.contact_dimension,
                "contact_degree": sector.contact_degree,
                "certified_exact_rank": sector.contact_dimension,
                "sample_ids": [
                    sample.sample_id
                    for sample in reconstruction_samples_for_sector(name)
                ],
            }
            for name, sector in SECTORS.items()
        },
    }
    args.design_output.parent.mkdir(parents=True, exist_ok=True)
    with args.design_output.open("w") as handle:
        json.dump(design, handle, indent=2)
        handle.write("\n")

    print(f"wrote {len(rows)} component rows to {args.components_output}")
    print(f"wrote {len(RECONSTRUCTION_SAMPLES)} samples to {args.design_output}")


if __name__ == "__main__":
    main()
