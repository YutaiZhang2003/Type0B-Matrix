"""JSON-configured command line: python -m genus0_ns_elliptic_h_recursion."""

import argparse
import json
from pathlib import Path

import mpmath as mp

from . import compute_h_recursion, reconstruct_from_real_moduli, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path,help="JSON input; use decimal strings for weights/coordinates")
    parser.add_argument("--save-table",type=Path)
    parser.add_argument("--version",action="version",version=__version__)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = {"external_weights","internal_weights","order"}
    allowed = required | {"b","central_charge","dps","pole_tolerance","z","mobile_positions","parity"}
    if not required <= config.keys() or config.keys()-allowed:
        parser.error("missing required fields or unknown configuration fields")
    parameters = {k:v for k,v in config.items() if k not in ("z","mobile_positions","parity")}
    table = compute_h_recursion(**parameters)
    if args.save_table:
        table.save(args.save_table)
    output = {"version":__version__,"point_count":table.point_count,
              "coefficient_count":len(table.coefficients),"order":table.order,
              "dps":table.dps,"central_charge":mp.nstr(table.central_charge,table.dps)}
    if "z" in config:
        answer = reconstruct_from_real_moduli(table,z=config["z"],
                    mobile_positions=config.get("mobile_positions",()),parity=config.get("parity"))
        def encode(value):
            return {"real":mp.nstr(mp.re(value),table.dps),"imag":mp.nstr(mp.im(value),table.dps)}
        output.update(parity=answer.parity,q=encode(answer.nomes.q),
                      segment_nomes=list(map(encode,answer.nomes.segment_nomes)),
                      H=encode(answer.reduced_value),F=encode(answer.value))
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
