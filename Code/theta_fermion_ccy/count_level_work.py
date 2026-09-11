"""Count the production CCY/PBW loop workload without evaluating any block.

CCY counts are for the full nonzero-residue recursion graph. Exactly zero
branching prefactors/residues can only reduce them. The saved level-ten run
is used to check these structural counts, not to extrapolate them to 15.
PBW counts include all requested tensor entries and dense contractions, but
not the internal Ward/mode-action recursion or the operations in inversion.
"""

from collections import Counter
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse
import hashlib
import json
import time


def convolution(left, right, cutoff):
    result = [0] * (cutoff + 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right[:cutoff + 1 - i]):
            result[i + j] += a * b
    return result


def moment_product(left, right, cutoff):
    """Moments (count, incoming, outgoing, incoming*outgoing) under sums."""
    result = [[0] * 4 for _ in range(cutoff + 1)]
    for i, (a, ai, ao, ap) in enumerate(left):
        for j, (b, bi, bo, bp) in enumerate(right[:cutoff + 1 - i]):
            row = result[i + j]
            row[0] += a * b
            row[1] += ai * b + a * bi
            row[2] += ao * b + a * bo
            row[3] += ap * b + ai * bo + ao * bi + a * bp
    return result


def branch_histogram(level):
    """Actual n-dependent budgets; identify split-edge exchange once."""
    cutoff = 2 * level
    ramond_level = lambda n: int(2*n*n - F(1, 8))
    ns = [F(i, 2) for i in range(-isqrt(cutoff), isqrt(cutoff) + 1)]
    ramond = [F(i, 4) for i in range(-4*level-1, 4*level+2, 2)
              if ramond_level(F(i, 4)) <= level]
    histogram = Counter()
    outer = set()
    pairs = set()
    for i in range(-4*level-1, 4*level+2, 2):
        left, right = F(i, 4), F(i+2, 4)
        if ramond_level(left) + ramond_level(right) > cutoff:
            continue
        pairs.add((left, right))
        for n in ns:
            for last in ramond:
                budget = cutoff - int(4*n*n) - ramond_level(left) \
                    - ramond_level(right) - 2*ramond_level(last)
                if budget >= 0:
                    histogram[budget] += 2  # two Virasoro copies
                    outer.update(((n, left, last), (n, right, last)))
    return histogram, len(outer), 4*len(pairs)


def ccy_cutoff_counts(cutoff):
    labels = [tuple(r*s for r in range(2, n+1)
                    for s in range(1, n//r+1)) for n in range(cutoff+1)]
    outgoing = list(map(len, labels))
    incoming = [sum(n-d != 1 for d in labels[n]) for n in range(cutoff+1)]
    state = [[0]*4 for _ in range(cutoff+1)]
    root = [[0]*4 for _ in range(cutoff+1)]
    terminal = [[0]*4 for _ in range(cutoff+1)]
    state[0][0] = root[0][0] = terminal[0][0] = 1
    shift_counts = [1] + [0]*cutoff
    root_terminal = [1] + [0]*cutoff
    for weight in (2, 1, 1, 2):
        all_local = [[0]*4 for _ in range(cutoff+1)]
        root_local = [[0]*4 for _ in range(cutoff+1)]
        terminal_local = [[0]*4 for _ in range(cutoff+1)]
        shifts_local = [0]*(cutoff+1)
        root_terminal_local = [0]*(cutoff+1)
        for n in range(cutoff//weight+1):
            root_local[weight*n] = [1, 0, outgoing[n], 0]
            shifts_local[weight*n] = int(n != 1)
            root_terminal_local[weight*n] = int(n < 2)
            for shift in range(n+1):
                if shift == 1:
                    continue
                level = n-shift
                row = [1, incoming[shift], outgoing[level], incoming[shift]*outgoing[level]]
                for k in range(4):
                    all_local[weight*n][k] += row[k]
                    if level < 2:
                        terminal_local[weight*n][k] += row[k]
        state = moment_product(state, all_local, cutoff)
        root = moment_product(root, root_local, cutoff)
        terminal = moment_product(terminal, terminal_local, cutoff)
        shift_counts = convolution(shift_counts, shifts_local, cutoff)
        root_terminal = convolution(root_terminal, root_terminal_local, cutoff)
    root_counts = [row[0] for row in root]
    product_pairs = convolution(root_counts, root_counts, cutoff)
    result = []
    for k in range(cutoff+1):
        total_states = sum(row[0] for row in state[:k+1])
        globals_terminal = sum(row[0] for row in terminal[:k+1])
        roots_nonterminal = sum(root_counts[:k+1]) - sum(root_terminal[:k+1])
        result.append({
            "requested_coefficients": sum(root_counts[:k+1]),
            "rational_blocks": total_states - globals_terminal,
            "terminal_globals": globals_terminal,
            "global_seed_constructions": total_states,
            "numeric_evaluations": roots_nonterminal + sum(
                a[1]-b[1] for a, b in zip(state[:k+1], terminal[:k+1])),
            "stored_pole_terms": sum(row[2] for row in state[:k+1]),
            "pole_term_multiplications_and_additions_each": sum(
                a[3]+b[2] for a, b in zip(state[:k+1], root[:k+1])),
            "pole_templates": sum(shift_counts[j]*outgoing[(k-j)//weight]
                for j in range(k+1) for weight in (2, 1, 1, 2)),
            "pole_geometry_entries": sum(outgoing[k//weight-shift]
                for weight in (2, 1, 1, 2) for shift in range(k//weight+1) if shift != 1),
            "virasoro_product_candidate_pairs": sum(product_pairs[:k+1]),
        })
    return result


def ccy_counts(level, cutoff_counts):
    hist, outer, middle = branch_histogram(level)
    counts = {key: sum(number*cutoff_counts[k][key] for k, number in hist.items())
              for key in cutoff_counts[0]}
    # One scalar product uses two copies, not one product per copy.
    counts["virasoro_product_candidate_pairs"] //= 2
    return dict(virasoro_series=sum(hist.values()), cutoff_histogram=dict(sorted(hist.items())),
                counts=counts, largest_series=cutoff_counts[2*level],
                requested_outer_label_triples=outer, middle_coefficients=middle)


def character(cutoff, fermion_steps, boson_steps):
    coeff = [1]+[0]*cutoff
    for step in boson_steps:
        for n in range(step, cutoff+1):
            coeff[n] += coeff[n-step]
    for step in fermion_steps:
        for n in range(cutoff, step-1, -1):
            coeff[n] += coeff[n-step]
    return coeff


def pbw_counts(level):
    # NS argument counts half-levels. R coefficient is dimension per parity:
    # each oscillator word has precisely one ground state of either parity.
    ns = character(2*level, range(1, 2*level+1, 2), range(2, 2*level+1, 2))
    ramond = character(level, range(1, level+1), range(1, level+1))
    triples = tensor_entries = multiplications = additions = 0
    max_tensor = 0
    for j, a in enumerate(ns):
        for k, b in enumerate(ramond):
            for ell, c in enumerate(ramond):
                if j+2*k+2*ell > 2*level:
                    continue
                triples += 1
                volume = a*b*c
                tensor_entries += 4*volume  # two eta forms and two parity sectors
                multiplications += 2*volume*(a+b+c+1)
                additions += 2*(volume*(a+b+c-2)-1)
                max_tensor = max(max_tensor, volume)
    return dict(
        monomials=triples, parity_output_slots=8*triples,
        ns_dimensions_by_twice_level=ns, ramond_dimensions_per_parity_by_level=ramond,
        gram_matrices=(2*level+1)+4*(level+1),
        gram_entries=sum(d*d for d in ns)+4*sum(d*d for d in ramond),
        gram_inverse_sum_dimension_cubed=sum(d**3 for d in ns)+4*sum(d**3 for d in ramond),
        requested_three_point_tensor_entries=tensor_entries,
        dense_contraction_complex_multiplications=multiplications,
        dense_contraction_complex_additions=additions,
        largest_tensor_entries=max_tensor,
        largest_ns_gram_dimension=max(ns), largest_ramond_gram_dimension=max(ramond),
        all_retained_gram_and_inverse_array_bytes=32*(sum(d*d for d in ns)+4*sum(d*d for d in ramond)),
        exclusions=["Ward recursion cache misses and internal sums",
                    "mode-action recursion used to form Gram entries",
                    "matrix inversion operations (sum d^3 is a work proxy, not an exact FLOP count)",
                    "Python object/cache memory; word construction; result writes"],
    )


def verify_saved(report, path):
    saved = json.loads(path.read_text())
    records = saved["numerator_diagnostics"]["ccy"]
    observed = {
        "rational_blocks": sum(r["cache"]["rational_blocks"]["misses"] for r in records),
        "terminal_globals": sum(r["cache"]["global"]["misses"] for r in records),
        "numeric_evaluations": sum(r["cache"]["numeric_evaluations"] for r in records),
        "pole_templates": sum(r["cache"]["pole_templates"]["misses"] for r in records),
        "pole_geometry_entries": sum(r["cache"]["pole_geometry"]["misses"] for r in records),
        "pole_term_multiplications_and_additions_each": sum(
            r["cache"]["transition"]["hits"]+r["cache"]["transition"]["misses"] for r in records),
    }
    predicted = report["levels"][str(saved["total_q_level"])] ["ccy"]
    matches = {key: value == predicted["counts"][key] for key, value in observed.items()}
    matches["virasoro_series"] = len(records) == predicted["virasoro_series"]
    if not all(matches.values()):
        raise AssertionError((observed, predicted, matches))
    return dict(saved_result=str(path), observed=observed, all_available_counts_match=True)


def calibrate_pbw(report, path):
    """Fit per-coefficient costs, using counted work, never a fit versus N.

    These alternative models expose sensitivity to the cost assumptions.
    The d^3 feature also correlates with Gram construction complexity; its
    fitted coefficient MUST NOT be reported as a measured inversion speed.
    """
    import numpy as np
    from scipy.optimize import nnls
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]

    def features(level):
        dims = report['levels'][str(level)]['pbw']
        ns = dims['ns_dimensions_by_twice_level']
        ramond = dims['ramond_dimensions_per_parity_by_level']
        seen, result = set(), []
        for total in range(2*level+1):
            for j in range(total+1):
                for twice_k in range(0, total-j+1, 2):
                    twice_l = total-j-twice_k
                    if twice_l % 2:
                        continue
                    k, ell = twice_k//2, twice_l//2
                    a, b, c = ns[j], ramond[k], ramond[ell]
                    gram = cubes = 0
                    for key, size, number in (((0, j), a, 1), ((1, k), b, 2), ((2, ell), c, 2)):
                        if key not in seen:
                            gram += number*size**2
                            cubes += number*size**3
                            seen.add(key)
                    volume = a*b*c
                    result.append(((j, twice_k, twice_l), [gram, 4*volume, cubes,
                        2*volume*(a+b+c+1), len(result)+1, 1]))
        return result

    baseline_level = max(sum(row['pbw_levels']) for row in rows)//2
    baseline = features(baseline_level)
    assert len(rows) == len(baseline)
    assert all(tuple(row['pbw_levels']) == key for row, (key, _) in zip(rows, baseline))
    x = np.array([value for _, value in baseline], dtype=float)
    y = np.diff([0]+[row['elapsed_seconds'] for row in rows])
    names = ['gram_entries', 'requested_tensor_entries', 'sum_dimension_cubed',
             'dense_contraction_products', 'json_record_writes_proxy', 'monomials']
    models = []
    for columns in ([0, 1, 4, 5], [0, 1, 2, 3, 4, 5]):
        scale = x[:, columns].sum(axis=0)
        seconds, _ = nnls(x[:, columns]/scale, y)
        rates = seconds/scale
        models.append(dict(features=[names[c] for c in columns],
            fitted_seconds_per_feature_unit=list(map(float, rates)),
            fitted_baseline_seconds=float(sum(seconds)),
            per_coefficient_residual_rms_seconds=float(np.sqrt(np.mean((x[:, columns]@rates-y)**2))),
            predicted_seconds={str(level): float(np.array([value for _, value in features(level)])[:, columns].sum(axis=0)@rates)
                               for level in map(int, report['levels'])}))
    return dict(calibration_log=str(path), models=models,
        limitation='Counts are direct. Seconds assume unchanged cost per counted feature. Internal Ward/mode recursion and memory effects are not independently measured; the two models are not a confidence interval.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[10, 15])
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--verify-saved", type=Path)
    parser.add_argument("--calibrate-pbw-log", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    cutoff_counts = ccy_cutoff_counts(2*max(args.levels))
    report = dict(method="integer combinatorial count; no numerical blocks evaluated",
        ccy_scope="Full scheduled graph before exactly-zero residue or branching skips; same per-block caches, cutoffs and split-edge symmetry as production",
        parameters=dict(p=0, f=0, etas=[1, -1]),
        levels={str(n): dict(ccy=ccy_counts(n, cutoff_counts), pbw=pbw_counts(n)) for n in args.levels},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.verify_saved:
        report["saved_level10_count_check"] = verify_saved(report, args.verify_saved)
    report["counting_seconds"] = time.perf_counter()-started
    if args.calibrate_pbw_log:
        report['pbw_count_based_cost_models'] = calibrate_pbw(report, args.calibrate_pbw_log)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
