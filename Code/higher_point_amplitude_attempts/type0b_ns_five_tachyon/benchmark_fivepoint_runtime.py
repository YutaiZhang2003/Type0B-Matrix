#!/usr/bin/env python3
"""Isolated-process memory/time comparison, without changing the amplitude.

The table workload uses production (6,7) quadrature nodes, physical external
weights and the (8,8), total-16 cutoff. It is not a full moduli-integration
runtime prediction. The optional integrand workload uses a smaller (2,3)
momentum rule at one moduli point, with the same recursion cutoff.
"""

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc

from type0b_ns_five_tachyon import BRYNSFiveTachyonIntegrand, _smooth_momentum_nodes, _to_fixed_gauge
from fivepoint_runtime import BoundedLRU, CoefficientStore, CompactCBlock
from ns_multipoint_c_recursion import NSSphereLinearCRecursion


def memory():
    current, peak = tracemalloc.get_traced_memory()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(retained_python_mib=current / 1024**2, peak_python_mib=peak / 1024**2,
                peak_rss_mib=rss / (1024**2 if sys.platform == "darwin" else 1024))


def worker(args):
    compact = args.worker_mode == "compact"
    started = time.perf_counter()
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="fivepoint-benchmark-") as directory:
        if args.integrand:
            kernel = BRYNSFiveTachyonIntegrand(
                outgoing_energies=(0.25 + 0.02j,) * 4, block_backend="c",
                compact_c_cache=compact, block_cache_limit=args.cache_limit,
                c_coefficient_cache_path=Path(directory) / "coefficients.sqlite",
                central_charge_shift=0, global_max_twice_levels=(8, 8),
                global_max_total_twice_level=16, momentum_orders=(2, 3),
                momentum_maximum=1, structure_precision=22, block_working_precision=45,
            )
            positions = _to_fixed_gauge(0.18 + 0.03j, 0.31 - 0.04j, (0, 1, 2, 3, 4))
            values = []
            timings = []
            for _ in range(2):
                before = time.perf_counter()
                value = kernel.fixed_gauge_integrand_positions(positions)
                values.append([value.real, value.imag])
                timings.append(time.perf_counter() - before)
            result = dict(mode=args.worker_mode, values=values, evaluation_seconds=timings,
                          diagnostics=kernel.cache_diagnostics(), **memory())
            kernel.close_runtime()
            return result

        pairs = [(p, q) for p, _ in _smooth_momentum_nodes(6, 2)
                 for q, _ in _smooth_momentum_nodes(7, 2)][:args.blocks]
        if args.blocks > len(pairs):
            raise ValueError("at most 168 distinct production momentum pairs")
        cache = BoundedLRU(args.cache_limit) if compact else {}
        store = CoefficientStore(Path(directory) / "coefficients.sqlite") if compact else None
        snapshots = []
        checksum = 0j
        compilations = 0
        for visit in range(2):
            for index, (p, q) in enumerate(pairs):
                if index not in cache:
                    kwargs = dict(
                        central_charge=13.5,
                        external_weights=tuple(0.5 * (1 + w*w) for w in
                                               (1 + 0.08j, *((0.25 + 0.02j,) * 4))),
                        external_descendants=(1, 1, 1, 0, 0),
                        internal_weights=(0.5 * (1 + p*p), 0.5 * (1 + q*q)),
                        vertex_sectors=(1, 0, 0), working_precision=45,
                    )
                    cache[index] = (CompactCBlock(coefficient_store=store, **kwargs)
                                    if compact else NSSphereLinearCRecursion(**kwargs))
                block = cache[index]
                before = getattr(block, "compiled_coefficients", 0)
                value = block.series_value((0.12 + visit * 0.01j, 0.23 - 0.03j),
                                           (8, 8), max_total_twice_level=16)
                compilations += getattr(block, "compiled_coefficients", 0) - before
                checksum += complex(value)
                if (index + 1) % 16 == 0 or index + 1 == len(pairs):
                    gc.collect()
                    snapshots.append(dict(
                        visit=visit, evaluated=index + 1, blocks_retained=len(cache),
                        scratch_entries=sum(len(b._coefficient_cache) for b in cache.values()),
                        final_entries=sum(len(getattr(b, "final_coefficients", {})) for b in cache.values()),
                        elapsed_seconds=time.perf_counter() - started, **memory(),
                    ))
        result = dict(mode=args.worker_mode, blocks=len(pairs), snapshots=snapshots,
                      seconds=time.perf_counter() - started,
                      checksum=[checksum.real, checksum.imag],
                      compiled_coefficients=compilations,
                      disk_hits=store.hits if store else 0,
                      evictions=getattr(cache, "evictions", 0), **memory())
        if store:
            store.close()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode", choices=("legacy", "compact"))
    parser.add_argument("--blocks", type=int, default=96)
    parser.add_argument("--cache-limit", type=int, default=16)
    parser.add_argument("--integrand", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_mode:
        print(json.dumps(worker(args), indent=2))
        return
    results = {}
    for mode in ("legacy", "compact"):
        command = [sys.executable, "-B", __file__, "--worker-mode", mode,
                   "--blocks", str(args.blocks), "--cache-limit", str(args.cache_limit)]
        if args.integrand:
            command.append("--integrand")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        results[mode] = json.loads(result.stdout)
        print(f"{mode} benchmark complete", file=sys.stderr, flush=True)
    field = "values" if args.integrand else "checksum"
    old = results["legacy"][field]
    new = results["compact"][field]
    old_values, new_values = (old, new) if args.integrand else ([old], [new])
    relative_difference = max(abs(complex(*a) - complex(*b)) / max(1, abs(complex(*a)))
                              for a, b in zip(old_values, new_values))
    results.update(schema="fivepoint-bounded-memory-benchmark-v1",
                   workload="small-momentum-integrand" if args.integrand else "production-node-tables",
                   recursion_twice_levels=[8, 8], total_twice_level=16,
                   cache_limit=args.cache_limit, relative_difference=relative_difference,
                   tracing_enabled=True, full_production_runtime_estimate=False)
    payload = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    if relative_difference > 1e-12:
        raise ArithmeticError("benchmark changed the numerical value")


if __name__ == "__main__":
    main()
