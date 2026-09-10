"""Independent low-level double-Virasoro sewing with an auxiliary projector.

Constructs the numerator from branch three-point coefficients, ordinary
Virasoro Ward forms, and an explicitly transformed insertion matrix. The
physical PBW block is used only as a reference after sewing.
"""

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from zero_mode_recovery import auxiliary_with_zero_mode, recover_physical_series
from nsrr_genus2_block import (
    HumanAuxiliaryThreePoint, direct_pbw_nsrr_series,
    level_triples, ramond_sector_residual, star_convolve_series,
)
from direct_state_check import DirectBranchingCoefficient, branch_in_pbw
from compute_target import BranchWeights, FreeFieldModule, VirasoroThreePoint, partitions
from theta_star_algebra import theta_quadratic_sign


class InsertedSewing:
    def __init__(self, b, momenta, p=0):
        self.p = p
        self.evaluator = DirectBranchingCoefficient(float(b), tuple(map(float, momenta)), p)
        self.evaluator.auxiliary_form = HumanAuxiliaryThreePoint(self.evaluator.free_modules)
        self.weights = BranchWeights(float(b), tuple(map(float, momenta)))
        self.maximum_band_error = 0.0

    @lru_cache(None)
    def edge(self, slot, twice_level):
        module = self.evaluator.free_modules[slot]
        pbw = self.evaluator.pbw_modules[slot]
        labels = (tuple(Fraction(k, 2) for k in range(-twice_level - 1, twice_level + 2))
                  if slot == 0 else tuple(Fraction(k, 4) for k in range(-2 * twice_level - 3, 2 * twice_level + 4, 2)))
        metadata, columns = [], []
        for n in labels:
            base = 4 * n * n - (Fraction(1, 4) if slot else 0)
            remaining = (twice_level - base) / 2
            if remaining < 0 or remaining.denominator != 1:
                continue
            for alpha in ((0,) if slot == 0 else (0, 1)):
                primary = {}
                for (aux, physical), outer in self.evaluator.branch(slot, n, alpha).items():
                    for oscillator, inner in pbw.to_fock(physical).items():
                        state = module.join_state(aux, oscillator)
                        primary[state] = primary.get(state, 0j) + outer * inner
                for a in range(int(remaining) + 1):
                    for first, second in product(partitions(a), partitions(int(remaining) - a)):
                        column = branch_in_pbw(module, pbw, module.descendant(primary, first, second))
                        metadata.append((n, alpha, first, second))
                        columns.append(column)
        rows = tuple(sorted(set().union(*(column.keys() for column in columns)), key=repr))
        transform = np.array([[column.get(row, 0j) for column in columns] for row in rows])
        assert transform.shape[0] == transform.shape[1], (slot, twice_level, transform.shape)
        gram_product = np.array([
            [self.evaluator.auxiliary_form.inner(slot, a, b) * pbw.inner(x, y)
             for b, y in rows] for a, x in rows
        ])
        gram = transform.T @ gram_product @ transform
        inverse = np.linalg.inv(gram)
        if slot:
            ground_sign = np.array([(-1) ** aux[1] for aux, _ in rows])
            insertion = np.linalg.solve(transform, ground_sign[:, None] * transform)
            for i, left in enumerate(metadata):
                for j, right in enumerate(metadata):
                    if abs(left[0] - right[0]) != Fraction(1, 2):
                        self.maximum_band_error = max(self.maximum_band_error, abs(insertion[i, j]))
        else:
            insertion = np.eye(len(metadata))
        parities = np.array([(int(2 * n) + self.p) % 2 if slot == 0 else alpha
                             for n, alpha, _, _ in metadata])
        return metadata, inverse, insertion, parities

    @lru_cache(None)
    def vertex(self, left, middle, right, eta):
        states = (left, middle, right)
        labels = tuple(s[0] for s in states)
        f = (int(2 * labels[0]) + middle[1] + right[1]) % 2
        # The archived branching oracle's odd-form eta label is opposite
        # to the physical PBW block's label. This transport is checked below
        # against all physical f,p,eta,eta-prime cases, not inferred from f=0.
        primary = self.evaluator.raw(labels, middle[1], right[1], (-1) ** f * eta)
        for copy in (0, 1):
            form = VirasoroThreePoint(self.weights.triple(labels, copy), self.weights.central_charges[copy])
            primary *= form.value(*(s[2 + copy] for s in states))
        return primary

    def series(self, cutoff, f, etas, cut=1, insertion="fixed"):
        answer = {}
        for levels in level_triples(cutoff):
            edges = [self.edge(slot, level) for slot, level in enumerate(levels)]
            value = [0j] * 8
            for r1_parity in (0, 1):
                parities = ((levels[0] + self.p) % 2, r1_parity, (f + levels[0] + r1_parity) % 2)
                bases, kernels = [], []
                for slot, (metadata, inverse, d, parity) in enumerate(edges):
                    indices = np.flatnonzero(parity == parities[slot])
                    bases.append([metadata[i] for i in indices])
                    operator = np.eye(len(metadata))
                    if slot == cut:
                        operator = {"identity": operator, "saturated": d, "fixed": (operator + d) / 2}[insertion]
                    kernels.append((operator @ inverse)[np.ix_(indices, indices)])
                if any(not basis for basis in bases):
                    continue
                tensors = [np.array([self.vertex(*states, eta) for states in product(*bases)])
                           .reshape(tuple(map(len, bases))) for eta in etas]
                contracted = np.einsum("abc,ad,be,cf,def->", tensors[0], *kernels, tensors[1], optimize=True)
                index = parities[0] | parities[1] << 1 | parities[2] << 2
                value[index] = (-1) ** levels[0] * theta_quadratic_sign(index) * contracted
            answer[levels] = tuple(value)
        return answer


def error(left, right):
    return max(abs(a - b) for key in set(left) | set(right)
               for a, b in zip(left.get(key, (0j,) * 8), right.get(key, (0j,) * 8)))


def degenerate_field_check(b):
    # The negative momentum realization contains the irreducible physical
    # vacuum: G_-1/2|0>=L_-1|0>=0. The opposite Fock chart does not.
    module = FreeFieldModule("NS", b, -(b + 1 / b) / 2)
    psi = {((1,), (), ()): 1}
    records = []
    for copy, t in ((1, 2 * b**2 / (1 - b**2)), (2, 2 / (b**2 - 1))):
        h = -0.5 - 3 * t / 4
        eigenstate = {k: v / 2 for k, v in module.apply_embedded(
            copy, 1, module.apply_embedded(copy, -1, psi)
        ).items()}
        first = module.apply_embedded(copy, -1, module.apply_embedded(copy, -1, psi))
        second = module.apply_embedded(copy, -2, psi)
        residual = max(abs(first.get(k, 0) + t * second.get(k, 0)) for k in set(first) | set(second))
        weight_error = max(abs(eigenstate.get(k, 0) - h * psi.get(k, 0)) for k in set(eigenstate) | set(psi))
        positive_error = max((abs(x) for n in (1, 2) for x in module.apply_embedded(copy, n, psi).values()), default=0)
        records.append(dict(copy=copy, weight=h, null_error=residual, weight_error=weight_error, positive_mode_error=positive_error))
    assert max(max(r[k] for k in ("null_error", "weight_error", "positive_mode_error")) for r in records) < 1e-10
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twice-level", type=int, default=2)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    b = Fraction(7, 5)
    momenta = (Fraction(11, 23), Fraction(13, 29), Fraction(17, 31))
    saturated = auxiliary_with_zero_mode(maximum_total_twice_level=args.twice_level, insertion="saturated")
    records = []
    for p in (0, 1):
        sewing = InsertedSewing(b, momenta, p)
        for f, cut, eta, eta_prime in product((0, 1), (1, 2), (-1, 1), (-1, 1)):
            auxiliary = auxiliary_with_zero_mode(maximum_total_twice_level=args.twice_level, cut=cut)
            physical = direct_pbw_nsrr_series(b=b, momenta=momenta, form_parity=f, primary_parity=p,
                                            etas=(eta, eta_prime), maximum_total_twice_level=args.twice_level)
            enlarged = sewing.series(args.twice_level, f, (eta, eta_prime), cut=cut)
            expected = star_convolve_series(auxiliary, physical, maximum_total_twice_level=args.twice_level)
            recovered = recover_physical_series(enlarged, auxiliary, maximum_total_twice_level=args.twice_level)
            record = dict(p=p, f=f, cut=cut, etas=[eta, eta_prime], convolution_error=error(enlarged, expected),
                          recovery_error=error(recovered, physical), branch_band_error=float(sewing.maximum_band_error))
            records.append(record)
            print(json.dumps(record), flush=True)
    report = dict(total_twice_level=args.twice_level, elapsed_seconds=time.perf_counter() - started,
                  saturated_sector_error=ramond_sector_residual(saturated, -1),
                  degenerate_field=degenerate_field_check(float(b)), cases=records)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    assert max(item["recovery_error"] for item in records) < 1e-8, report
    assert max(item["branch_band_error"] for item in records) < 1e-8, report


if __name__ == "__main__":
    main()
