"""Test, rather than assume, a theta-ratio conversion of a plumbing factor.

If Z_psi(q,delta)=H(q)*|theta[delta](Omega)| in the conventions being used,
H must agree for all four all-NS characteristics. This necessary test uses
only the existing all-NS free evaluator; it does not modify that evaluator
or assert that a failing test is a defect in the checked block package.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nsrr_nsnsns_theta_omega_scan as scan
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm


def audit(q_geometry, omega, *, max_mode=32, tolerance=1e-8):
    rows=[]
    for lifts in ((1,1,1), (1,-1,1), (1,1,-1), (1,-1,-1)):
        characteristic=scan._spin_characteristic_from_lifts("theta",q_geometry,lifts)
        fermion=theta_physical_fermion_fredholm(q_geometry,lifts,max_mode=max_mode).nonchiral_value
        theta=abs(scan.riemann_theta_constant_genus2(omega,characteristic,tol=1e-15))
        rows.append({"lifts_geometry":list(lifts), "characteristic":characteristic,
                     "majorana_plumbing_value":float(fermion),
                     "absolute_theta_constant":float(theta),
                     "candidate_spin_independent_factor":float(fermion/theta)})
    reference=rows[0]["candidate_spin_independent_factor"]
    discrepancy=max(abs(row["candidate_spin_independent_factor"]/reference-1) for row in rows)
    return {"schema":"nsrr-free-spin-conversion-audit-v1", "rows":rows,
            "maximum_relative_incompatibility":discrepancy,
            "tolerance":tolerance, "compatible":discrepancy<=tolerance,
            "scope":"Necessary compatibility test for the new theta-ratio adapter, not a test of double-Virasoro/PBW correctness."}


def require_compatible_theta_ratio(q_geometry, omega, **kwargs):
    result=audit(q_geometry,omega,**kwargs)
    if not result["compatible"]:
        raise ArithmeticError(
            "The new theta-ratio free-spin conversion is not certified: "
            f"the all-NS frame factors disagree by {result['maximum_relative_incompatibility']:.6g}. "
            "Do not label the converted value a same-frame physical Ramond denominator.")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for point in scan._load(args.config)["points"]:
        chart=point["charts"]["source_nsrr"]
        row=audit(tuple(complex(x) for x in chart["q_values"]),scan.complex_matrix(chart["omega"]))
        row["t"]=point["t"]
        rows.append(row)
        print(f"t={point['t']}: relative incompatibility {row['maximum_relative_incompatibility']:.6g}",flush=True)
    scan.write_json(args.output,{"points":rows,"all_compatible":all(r["compatible"] for r in rows)})


if __name__ == "__main__":
    main()
