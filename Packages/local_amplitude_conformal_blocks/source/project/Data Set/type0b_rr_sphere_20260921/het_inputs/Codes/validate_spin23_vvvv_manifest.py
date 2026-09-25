#!/usr/bin/env python3
"""Validate the conservative no-pole conditions of a Spin(23) scan manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--minimum-pole-margin",type=float,default=0.0)
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()

    with args.manifest.open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("manifest is empty")

    margins=[]
    denominators=[]
    failures=[]
    for row in rows:
        energies=[
            complex(float(row[f"omega{i}_re"]),float(row[f"omega{i}_im"]))
            for i in (1,2,3)
        ]
        imaginary_parts=[value.imag for value in energies]
        margin=1.0-(sum(imaginary_parts)+max(imaginary_parts))
        margins.append(margin)
        pairs=((1,2),(0,2),(0,1))
        denominators.extend(abs(1+1j*(energies[j]+energies[k])) for j,k in pairs)
        if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in energies):
            failures.append(f"{row['point_id']}: nonfinite energy")
        if margin <= args.minimum_pole_margin:
            failures.append(f"{row['point_id']}: pole margin {margin}")

    summary={
        "manifest":str(args.manifest),
        "rows":len(rows),
        "families":dict(Counter(row["family"] for row in rows)),
        "settings":dict(Counter(row["setting"] for row in rows)),
        "minimum_pole_margin":min(margins),
        "minimum_candidate_denominator_magnitude":min(denominators),
        "failed_rows":len(failures),
        "failures":failures,
        "interpretation":(
            "The pole-margin criterion is sufficient and conservative for the undeformed "
            "internal-momentum contour; it is not a proof against every possible singularity."
        ),
    }
    text=json.dumps(summary,indent=2)+"\n"
    print(text,end="")
    if args.output is not None:
        args.output.write_text(text)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
