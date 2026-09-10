"""Directed check of the unsaturated convolution's opposite-sign kernel.

Compute F by physical PBW sewing, Fhat by the ordinary double-Virasoro
branch sum, and F_F by auxiliary oscillator sewing. No inverse, sector
projection, zero-mode insertion, or scalar clipping enters this check.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=3, choices=range(4))
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    import complex_arithmetic as ca
    ca.THRESHOLD = 0.0
    from check_level10_complex import load_runner
    from degenerate_kernel import make_check
    m = load_runner()

    class OrdinarySewing(make_check(m)):
        @lru_cache(None)
        def edge(self, slot, twice_level, parity, inserted=False):
            # This path constructs only branch labels and ordinary inverse
            # Virasoro Gram matrices; it never builds an insertion kernel.
            return super().edge(slot, twice_level, parity, inserted=False)

    b = Fraction(7, 5)
    momenta = tuple(map(Fraction, ("11/23", "13/29", "17/31")))
    sewing = OrdinarySewing(b, momenta)
    cutoff = 2 * args.level
    levels = tuple(m.level_triples(cutoff))
    auxiliary_started = time.perf_counter()
    auxiliary = m.auxiliary(cutoff, 1, insertion="identity")
    auxiliary_seconds = time.perf_counter() - auxiliary_started
    tube_product = tuple(m.F(int(i == 6)) for i in range(8))

    def maximum(series):
        return max((abs(x) for vector in series.values() for x in vector), default=0.0)

    def difference(left, right):
        return max((abs(x-y) for n in levels for x, y in
                    zip(left[n], right[n])), default=0.0)

    def sector_error(series, sign):
        return max((abs(x-sign*y) for vector in series.values()
                    for x, y in zip(m.star(tube_product, vector), vector)), default=0.0)

    def encode(series):
        return [dict(twice_levels=n, coefficients=[str(x) for x in series[n]])
                for n in levels]

    cases = [(p, f, eta, -eta) for p, f, eta in product((0, 1), (0, 1), (1, -1))]
    cases.append((0, 0, 1, 1))
    records = []
    timings = dict(auxiliary_seconds=auxiliary_seconds,
                   physical_seconds=0.0, enlarged_seconds=0.0, convolution_seconds=0.0)
    for p, f, eta, eta_prime in cases:
        stage = time.perf_counter()
        physical = {n: sewing.physical_coefficient(n, p, f, (eta, eta_prime)) for n in levels}
        timings["physical_seconds"] += time.perf_counter() - stage
        stage = time.perf_counter()
        enlarged = {n: sewing.enlarged_coefficient(n, p, f, (eta, eta_prime), cut=-1)
                    for n in levels}
        timings["enlarged_seconds"] += time.perf_counter() - stage
        stage = time.perf_counter()
        convolved = m.convolution(physical, auxiliary, cutoff)
        timings["convolution_seconds"] += time.perf_counter() - stage
        summary = dict(primary_parity=p, form_parity=f, etas=[eta, eta_prime],
                       equal_sign_control=eta == eta_prime,
                       maximum_physical_coefficient=maximum(physical),
                       maximum_direct_enlarged_coefficient=maximum(enlarged),
                       maximum_convolved_coefficient=maximum(convolved),
                       maximum_convolution_disagreement=difference(enlarged, convolved),
                       physical_sector_residual=sector_error(physical, eta*eta_prime),
                       ground_physical_all_lifts_plus=str(sum(physical[0, 0, 0], m.F(0))),
                       ground_enlarged_all_lifts_plus=str(sum(enlarged[0, 0, 0], m.F(0))))
        records.append(dict(**summary, physical=encode(physical),
                            direct_enlarged=encode(enlarged), convolution=encode(convolved)))
        print(json.dumps(summary), flush=True)
        sewing.vertex.cache_clear()

    timings["total_seconds"] = time.perf_counter() - started
    opposite = [r for r in records if not r["equal_sign_control"]]
    tolerance = 1e-90
    passed = (
        sector_error(auxiliary, 1) < tolerance
        and all(r["maximum_direct_enlarged_coefficient"] < tolerance
                and r["maximum_convolved_coefficient"] < tolerance
                and r["physical_sector_residual"] < tolerance
                and r["maximum_physical_coefficient"] > 0.1 for r in opposite)
        and all(r["maximum_convolution_disagreement"] < tolerance for r in records)
        and records[-1]["maximum_direct_enlarged_coefficient"] > 0.1
    )
    report = dict(status="passed" if passed else "failed", total_level=args.level,
                  precision_bits=ca.ctx.prec, scalar_clipping_threshold=ca.THRESHOLD,
                  numerical_tolerance=tolerance,
                  parameters=dict(b=str(b), momenta=list(map(str, momenta))),
                  monomials_per_case=len(levels), parity_components_per_monomial=8,
                  opposite_sign_cases=len(opposite),
                  opposite_sign_coefficient_slots=len(opposite)*len(levels)*8,
                  arithmetic="384-bit FLINT complex midpoints; not rigorous interval bounds",
                  conventions="Human Note form signs; ordinary unsaturated auxiliary sewing; eta_i^2=1",
                  methods=dict(physical="direct physical PBW Gram/Ward sewing",
                               enlarged="ordinary double-Virasoro Gram/Ward branch sum",
                               auxiliary="direct auxiliary oscillator Ward sewing"),
                  maximum_auxiliary_sector_residual=sector_error(auxiliary, 1),
                  timing_seconds=timings, auxiliary=encode(auxiliary), cases=records)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "auxiliary")}), flush=True)
    if not passed:
        raise SystemExit("The directed unsaturated-kernel check failed; inspect the saved residuals.")


if __name__ == "__main__":
    main()
