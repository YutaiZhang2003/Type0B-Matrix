#!/usr/bin/env python3
"""Manifest-driven RC evaluations; retain provisional physics metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Codes"))


def encode(value):
    import numpy as np
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [encode(item) for item in value]
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify_sources():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed since packaging: {relative}")
    return hashlib.sha256((ROOT / "source_manifest.json").read_bytes()).hexdigest()


def run_task(task):
    from benchmark_so7e8_four_ramond_convergence import MODULI_GRIDS, FAMILIES
    from so7e8_four_ramond_liouville_integral import evaluate_four_ramond_liouville_convergent
    from so7e8_liouville_chamber import require_conservative_real_p_chamber
    from so7e8_two_ramond_liouville_integral import collision_convergence_margins

    momenta = tuple(complex(*pair) for pair in task["momenta"])
    require_conservative_real_p_chamber(momenta[:3], momenta[3])
    if not collision_convergence_margins(momenta).convergent:
        raise ValueError("Energy point is outside the collision convergence chamber")
    if task["kind"] == "fourr_crossing":
        from benchmark_so7e8_nonchiral_crossing import run
        return run(task["order"], task["p_nodes"], momenta=momenta)
    if task["kind"] == "mixed_integral":
        from so7e8_two_ramond_liouville_integral import evaluate_two_ramond_liouville_convergent
        species = task["species"]
        result = evaluate_two_ramond_liouville_convergent(
            momenta, ns_at_z=species[0], ns_at_one=species[1],
            maximum_twice_level=task["order"], p_nodes=task["p_nodes"],
            momentum_scheme="infinite_gauss", p_max=4.0,
            epsilon0=.12, epsilon1=.10, digits=max(80,4*task["order"]+40),
            crossed_block_backend="hybrid", **MODULI_GRIDS[task["grid"]])
        import numpy as np
        if not np.all(np.isfinite(list(result.values.coefficients.values()))):
            raise ArithmeticError("Nonfinite mixed coefficients")
        return dict(coefficients=dict(result.values.coefficients),
            pieces=dict(result.values.pieces),
            backend="elliptic_recursion with two-star inverse-Gram table above order 7",
            two_star_inverse_gram_fallback=species=="SS" and task["order"]>7,
            assembly="RN orientation + ordered PCO signs + squared-theta seed",
            sphere_atlas="folded; all-species inversion tested against original exterior",
            production_certified=False)
    if task["kind"] == "mixed_ward":
        from benchmark_so7e8_mixed_ward import run
        return run(order=task["order"],momenta=momenta)
    if task["kind"] == "mixed_picture_integral":
        from benchmark_so7e8_mixed_direct import run
        return run(task["order"],task["p_nodes"],task["grid"],
                   task["species"],task["picture"],momenta=momenta)
    if task["kind"] != "fourr_integral":
        raise ValueError(f"Unsupported task kind {task['kind']}")
    result = evaluate_four_ramond_liouville_convergent(
        momenta, families=FAMILIES,
        maximum_twice_level=task["order"], p_nodes=task["p_nodes"],
        momentum_scheme="infinite_gauss", p_max=4.0,
        infinite_gauss_scale=task.get("p_scale", 1.0),
        epsilon0=task.get("epsilon0", .12), epsilon1=task.get("epsilon1", .10),
        sld_block_backend="elliptic_recursion", spin7_backend="kz_ode",
        spin7_maximum_order=12, lens_channel="reordered",
        digits=max(80, 4*task["order"]+40), adjacent_level=False,
        spin7_order_diagnostic=False, **MODULI_GRIDS[task["grid"]],
    )
    import numpy as np
    if not np.all(np.isfinite(result.coefficients)):
        raise ArithmeticError("Nonfinite integrated coefficients")
    return dict(coefficients=result.coefficients, pieces=result.pieces,
                families=FAMILIES, tensor_basis="oriented Spin(7) ranks 0,1,2,3",
                spectral_measure="dP/pi on [0,infinity)",
                sphere_measure="d2z on the complex plane",
                sld_block_backend=result.sld_block_backend,
                spin7_backend=result.spin7_backend, lens_channel=result.lens_channel,
                sewing_basis=result.sewing_basis,
                sewing_cocycle_status=result.sewing_cocycle_status,
                production_certified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    task = json.loads(args.tasks.read_text())[args.index]
    source_hash = verify_sources()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"task_{args.index:04d}.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("source_manifest_sha256") == source_hash and previous.get("task") == task:
            print(f"Already complete: {output}", flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite a different result: {output}")
    started = time.perf_counter()
    print(json.dumps(dict(event="started", task=task, index=args.index)), flush=True)
    result = run_task(task)
    report = dict(task=task, index=args.index, result=encode(result),
                  seconds=time.perf_counter()-started,
                  source_manifest_sha256=source_hash, hostname=socket.gethostname(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                  slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"))
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output)
    print(json.dumps(dict(event="completed", output=str(output), seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
