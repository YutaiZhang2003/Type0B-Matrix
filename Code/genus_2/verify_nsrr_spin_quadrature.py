#!/usr/bin/env python3
"""Independent reduction and selected-node reconstruction of the new sweep."""
from decimal import Decimal, localcontext
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.special import roots_genlaguerre

import check_nsrr_spin_quadrature as check


def independent_rule(envelope, n, index):
    x, w = roots_genlaguerre(n, -.5)
    indices = (index//(n*n), (index//n) % n, index % n)
    momenta, measure = [], 1.
    for q, j in zip(envelope, indices):
        a = -math.log(q)
        momenta.append(float(math.sqrt(x[j]/a)))
        measure *= float(w[j]*math.exp(x[j])/(2*math.pi*math.sqrt(a)))
    return momenta, measure


def decimal_sum(values):
    with localcontext() as ctx:
        ctx.prec = 50
        return float(sum((Decimal(str(v)) for v in values), Decimal(0)))


def source_reconstruct(c, shard):
    b = c["source"]["b"]
    momenta = shard["momenta"]
    constants = check.trial.GenericSuperLiouvilleConstants(b, dps=30)
    E, O = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
    coeff = {1: E/2, -1: O/2}
    q = tuple(map(complex, c["source_point"]["q_geometry"]))
    weights = [(b+1/b)**2/8+p*p/2+(1/16 if j < 2 else 0) for j, p in enumerate(momenta)]
    primary_squared = math.exp(sum(2*h*math.log(abs(z)) for h, z in zip(weights, q)))
    errors = []
    for row in shard["rows"]:
        terms = []
        for (f, eta, etap), encoded in zip(itertools.product((0, 1), (1, -1), (1, -1)), row["blocks"]):
            block = complex(*encoded)
            terms.append((-1)**f*(1j**f*coeff[eta])*(1j**f*coeff[etap])*abs(block)**2)
        z = sum(terms)*primary_squared*shard["measure"]
        errors.append(abs(z/row["Z_weighted"]-1))
    return max(errors)


def audit(output=check.DEFAULT_OUTPUT):
    c = check.trial.load(output/"config.json")
    check.validate(c)
    result = check.trial.load(output/"quadrature_summary.json")
    if result["config_digest"] != check.trial.digest(c):
        raise ValueError("summary has different inputs")
    for channel in ("source", "target"):
        if c[channel] != check.trial.load(Path(c["references"][channel])/"config.json"):
            raise ValueError("embedded reference config changed")
    rows = []
    expected_paths = set()
    for channel in ("target", "source"):
        envelope = c[channel]["q_envelope" if channel == "source" else "quadrature_reference_abs_q"]
        for n in c[channel+"_orders"]:
            shards, coordinate_error, measure_error = [], 0., 0.
            for i in range(n**3):
                path = check.shard_path(output, channel, n, i)
                expected_paths.add(path)
                s = check.trial.load(path)
                check.validate_shard(c, channel, n, i, s)
                p, w = independent_rule(envelope, n, i)
                coordinate_error = max(coordinate_error, *(abs(a/b-1) for a, b in zip(p, s["momenta"])))
                measure_error = max(measure_error, abs(w/s["measure"]-1))
                shards.append(s)
            z = decimal_sum(s["rows"][0]["Z_weighted"] for s in shards)
            saved = next(r for r in result["rows"] if r["channel"] == channel and r["N"] == n)
            reduced_error = abs(z/saved["Z"]-1)
            row = {"channel": channel, "N": n, "nodes": n**3, "decimal_sum_Z": z,
                   "independent_sum_relative_error": reduced_error,
                   "independent_rule_momentum_relative_error": coordinate_error,
                   "independent_rule_measure_relative_error": measure_error}
            if channel == "target":
                row["global_nonconverged_calls"] = sum(s["checks"]["global_nonconverged_calls"] for s in shards)
                row["global_max_occupation_used"] = max(s["checks"]["global_max_occupation_used"] for s in shards)
                row["global_worst_last_shell_relative"] = max(s["checks"]["global_worst_last_shell_relative"] for s in shards)
                if row["global_nonconverged_calls"]:
                    raise ArithmeticError("global resummation failed")
            else:
                row["branching_ward_residual_maximum"] = max(s["checks"]["branching_ward_residual"] for s in shards)
                row["analytic_ground_half_level_error_maximum"] = max(s["checks"]["analytic_ground_half_level_max_error"] for s in shards)
                row["explicit_PBW_completion_calls"] = sum(s["checks"]["explicit_PBW_completion_calls"] for s in shards)
                indices = (0, n*n+n+1, n**3-1)
                row["independent_source_node_indices"] = indices
                row["independent_source_contraction_relative_error"] = max(source_reconstruct(c, shards[i]) for i in indices)
                row["literal_lift_relative_spread"] = max(abs(decimal_sum(s["rows"][j]["Z_weighted"] for s in shards)/z-1) for j in range(4))
                if row["independent_source_contraction_relative_error"] > 1e-11:
                    raise ArithmeticError("source contraction reconstruction failed")
            if max(reduced_error, coordinate_error, measure_error) > 1e-12:
                raise ArithmeticError("independent numerical reduction failed")
            rows.append(row)
    if set((output/"shards").glob("*.json")) != expected_paths:
        raise ValueError("extra or missing node files")
    b = c["source"]["b"]
    kappa = (1.5+3*(b+1/b)**2)/1.5
    q_errors = [abs(r["Z"]/c["Z_free_"+r["channel"]]**kappa/r["Q_diagnostic"]-1) for r in result["rows"]]
    if max(q_errors) > 1e-12:
        raise ArithmeticError("free normalization reduction failed")
    report = {"summary_digest": check.trial.digest(result), "config_digest": check.trial.digest(c),
              "checks": rows, "total_new_nodes": len(expected_paths),
              "maximum_Q_reduction_relative_error": max(q_errors),
              "protected": check.fresh.protected_hashes(), "physical_Q_NSrr": None,
              "audit_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    check.trial.save(output/"verification.json", report)
    return report


if __name__ == "__main__":
    report = audit()
    print("Independent verification passed for", report["total_new_nodes"], "new nodes")
    for row in report["checks"]:
        print(row)
