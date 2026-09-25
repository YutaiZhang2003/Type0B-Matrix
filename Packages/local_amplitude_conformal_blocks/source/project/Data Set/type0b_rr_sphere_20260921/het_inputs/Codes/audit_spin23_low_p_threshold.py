#!/usr/bin/env python3
"""Audit binary64 NS-block coefficients in the low-momentum fallback region.

This diagnostic evaluates every distinct ordered external-weight configuration
used for one vector-amplitude point.  It compares the binary64 recursion to the
high-precision reference recursion at the actual segmented-quadrature momenta
through a requested block order.  It does not modify the production precision
policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np


FIT_DIR=Path(__file__).resolve().parent/"heterotic_so23_1to3_vvvv_fit_bundle"
sys.path.insert(0,str(FIT_DIR))

import heterotic_so23_1to3 as reference  # noqa: E402
import heterotic_so23_1to3_fast as stable  # noqa: E402


def _read_row(path: Path,index: int) -> dict[str,str]:
    with path.open(newline="") as handle:
        for row_index,row in enumerate(csv.DictReader(handle)):
            if row_index == index:
                return row
    raise IndexError(f"manifest index {index} is out of range")


def _energies(row: dict[str,str]) -> tuple[complex,...]:
    values=[]
    for name in ("omega1","omega2","omega3"):
        values.append(complex(float(row[f"{name}_re"]),float(row[f"{name}_im"])))
    values.append(sum(values))
    return tuple(values)


def _block_configurations(energies: tuple[complex,...]) -> list[tuple[complex,...]]:
    configurations=set()
    cyclic=(energies[1],energies[2],energies[0],energies[3])
    for ordering in (energies,cyclic):
        swapped=(ordering[0],ordering[2],ordering[1],ordering[3])
        for channel_energies in (ordering,swapped):
            h1,h2,h3,h4=(stable.h_of_p(value) for value in channel_energies)
            configurations.add((h4,h3,h2,h1))  # s channel
            configurations.add((h4,h1,h2,h3))  # t channel
    return sorted(
        configurations,
        key=lambda values: tuple((value.real,value.imag) for value in values),
    )


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--q-order",type=int,default=6)
    parser.add_argument("--mp-dps",type=int,default=70)
    parser.add_argument("--p-max",type=float,default=0.18)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()

    started=time.perf_counter()
    mp.mp.dps=args.mp_dps
    row=_read_row(args.manifest,args.index)
    energies=_energies(row)
    configurations=_block_configurations(energies)
    momenta,_=stable._p_quadrature("segmented",4.0,0.03)
    momenta=[float(value) for value in momenta if value <= args.p_max]
    max_level2=2*args.q_order+1

    records=[]
    for configuration_index,(h4,h3,h2,h1) in enumerate(configurations):
        for momentum in momenta:
            h_internal=stable.h_of_p(momentum)
            for star in (False,True):
                observed=stable.FastNSBlockComputer(
                    h4,h3,h2,h1,star,star,max_level2=max_level2
                )
                expected=reference.NSBlockComputer(
                    h4,h3,h2,h1,star,star,max_level2=max_level2
                )
                errors=[]
                for level2 in range(1,max_level2+1):
                    value=complex(observed.coefficient(level2,h_internal))
                    target=complex(expected.coefficient(level2,h_internal))
                    errors.append(abs(value-target)/max(abs(target),1.0e-300))
                records.append({
                    "configuration_index":configuration_index,
                    "momentum":momentum,
                    "star":star,
                    "max_relative_coefficient_error":max(errors),
                    "max_cancellation_ratio":observed.max_condition,
                })

    result={
        "point_id":row["point_id"],
        "manifest_index":args.index,
        "q_order":args.q_order,
        "mp_dps":args.mp_dps,
        "configuration_count":len(configurations),
        "momentum_count":len(momenta),
        "max_relative_coefficient_error":max(
            record["max_relative_coefficient_error"] for record in records
        ),
        "runtime_seconds":time.perf_counter()-started,
        "records":records,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key != "records"},indent=2))


if __name__ == "__main__":
    main()
