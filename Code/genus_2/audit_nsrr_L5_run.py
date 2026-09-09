#!/usr/bin/env python3
"""Independent completion/primary/reduction audit of a bounded L5 trial run."""
import argparse
import hashlib
import math
from pathlib import Path

import nsrr_trial_cluster as cluster


def audit(output):
    trial = cluster.trial
    result = trial.load(output/"summary.json")
    status = trial.load(output/"status.json")
    config = result["config"]
    cluster.validate_config(config)
    count = len(config["reference_nodes"])
    if status["status"] != "complete" or status["completed_nodes"] != count or status["summary_digest"] != trial.digest(result):
        raise ValueError("run is incomplete or summary provenance changed")
    shards = [trial.load(output/"shards"/f"node-{i:03d}.json") for i in range(count)]
    term_error = max(cluster.audit_shard(config, i, s) for i, s in enumerate(shards))
    primary_error = reduction_error = 0.
    sums = {}
    bg = config["b"]+1/config["b"]
    for shard in shards:
        h = [bg**2/8+p*p/2+(1/16 if i < 2 else 0) for i, p in enumerate(shard["momenta_geometry"])]
        c = [trial.decode(z)/2 for z in shard["C_BRY"]]
        for row in shard["rows"]:
            point = next(p for p in config["points"] if p["t"] == row["t"])
            primary2 = math.exp(sum(2*w*math.log(abs(complex(q))) for w, q in zip(h, point["q_geometry"])))
            primary_error = max(primary_error, abs(primary2/abs(trial.decode(row["primary"]))**2-1))
            value = primary2*sum(c[0 if eta == 1 else 1]*c[0 if ep == 1 else 1]*abs(trial.decode(z))**2
                for (_, eta, ep), z in zip(trial.CHANNELS, row["blocks"]))
            key = row["t"], row["level"], tuple(row["lifts_geometry"])
            sums.setdefault(key, []).append(shard["measure"]*value)
    for row in result["rows"]:
        terms = sums[row["t"], row["level"], tuple(row["lifts_geometry"])]
        expected = complex(math.fsum(z.real for z in terms), math.fsum(z.imag for z in terms))
        reduction_error = max(reduction_error, abs(expected-row["total"])/max(abs(expected), 1e-280))
    if max(term_error, primary_error, reduction_error) > 1e-11:
        raise ArithmeticError("independent completion audit failed")
    spread = max((max(v)-min(v))/max(map(abs, v)) for p in config["points"] for l in config["levels"]
                 for v in [[r["total"] for r in result["rows"] if r["t"] == p["t"] and r["level"] == l]])
    report = {"summary_digest": trial.digest(result), "complete_nodes": count,
              "elapsed_seconds": status["elapsed_seconds"], "maximum_term_error": term_error,
              "maximum_primary_error": primary_error, "maximum_reduction_error": reduction_error,
              "maximum_lift_relative_spread": spread,
              "protected_kernel_sha256": trial.protected_hashes(),
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "physical_Z": None, "physical_Q": None}
    trial.save(output/"verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(audit(parser.parse_args().run_dir))
