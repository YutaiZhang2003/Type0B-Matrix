#!/usr/bin/env python3
"""All-position SO(n)-singlet null-kernel checks, without a circle restriction."""
import itertools
import json
from pathlib import Path

import sympy as s

from mqm_kz_dynamical_checks import brauer_pairing_matrices


def main():
    n = s.symbols("n", integer=True, positive=True)
    x = s.symbols("x0:4", real=True)
    basis, _, _ = brauer_pairing_matrices(n)

    def B(pairing, labels):
        return s.Integer(all(labels[a] == labels[b] for a, b in pairing))

    def ell(fun, i, labels):
        # L^{01}=i ell.  The common i does not affect rank/kernel.
        if labels[i] not in (0, 1):
            return s.Integer(0)
        out = list(labels)
        out[i] = 1 - labels[i]
        return (1 if labels[i] == 0 else -1) * fun(tuple(out))

    def om(fun, i, j, labels):
        out = list(labels)
        out[i], out[j] = out[j], out[i]
        value = fun(tuple(out))
        if labels[i] != labels[j]:
            return value
        # Output patterns use only 0,1,2.  Every other contracted color
        # contributes equally, so this computes Q at symbolic n.
        for c, multiplicity in [(0, 1), (1, 1), (2, 1), (3, n - 3)]:
            out = list(labels)
            out[i] = out[j] = c
            value -= multiplicity * fun(tuple(out))
        return s.expand(value)

    patterns = sorted(set(itertools.permutations((0, 1, 2, 2)))
                      | set(itertools.permutations((0, 0, 0, 1)))
                      | set(itertools.permutations((0, 1, 1, 1))))
    rows, labels_for_rows = [], []
    for i in range(4):
        for labels in patterns:
            row = []
            for pairing in basis:
                fun = lambda a, pairing=pairing: B(pairing, a)
                value = 0
                for j in range(4):
                    if j == i:
                        continue
                    num = (
                        2 * (n - 1) * ell(fun, j, labels)
                        - 3 * ell(lambda a: om(fun, i, j, a), i, labels)
                        + (n - 1) * om(lambda a: ell(fun, i, a), i, j, labels)
                    )
                    value += s.factor(num) / (3 * (n - 1) * (x[i] - x[j]))
                row.append(s.factor(value))
            if row != [0, 0, 0]:
                rows.append(row)
                labels_for_rows.append({"site": i, "output_indices": labels})
    C = s.Matrix(rows)
    psi = s.Matrix([
        1 / ((x[0] - x[1]) * (x[2] - x[3])),
        -1 / ((x[0] - x[2]) * (x[1] - x[3])),
        1 / ((x[0] - x[3]) * (x[1] - x[2])),
    ])
    residual = (C * psi).applyfunc(s.factor)
    assert residual == s.zeros(C.rows, 1)
    sample = {n: 23, **dict(zip(x, [0, 1, 3, 6]))}
    _, row_pivots = C.subs(sample).T.rref()
    assert len(row_pivots) == 2
    selected_rows = list(row_pivots)
    _, col_pivots = C[selected_rows, :].subs(sample).rref()
    selected_cols = list(col_pivots)
    assert len(selected_cols) == 2
    selected = C.extract(selected_rows, selected_cols)
    determinant = s.factor(selected.det())
    print("selected rows", selected_rows)
    print("selected columns", selected_cols)
    print("determinant", determinant)
    result = {
        "all_checks_passed": True,
        "n_symbolic": True,
        "number_nonzero_component_rows": C.rows,
        "KZ_Pfaffian_kernel_residual": "identically zero at arbitrary distinct coordinates",
        "selected_row_labels": [labels_for_rows[i] for i in selected_rows],
        "selected_columns": selected_cols,
        "selected_two_by_three_rows": [[str(s.factor(v)) for v in row]
                                       for row in C[selected_rows, :].tolist()],
        "selected_minor": str(determinant),
        "scope": "SO(n) invariant tensor subspace at N4 for n23/24; common kernel rank statement follows only where the explicit minor is nonzero.",
    }
    path = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_null_symbolic_results.json')
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
