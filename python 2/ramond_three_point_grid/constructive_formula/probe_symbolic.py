#!/usr/bin/env python3
"""Print exact scaled Ramond master numerators at one symbolic-momentum sheet.

This is an exploratory helper for discovering a constructive matrix-valued
replacement for the scalar NS blow-up factor.  It never fits the answer to a
preselected product: it asks the state-level Ward evaluator for the raw master
and clears the known branch-leg denominators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
sys.path.insert(0, str(GRID_DIR))

import fit_signed_sectors as fit  # noqa: E402


def rational(text: str) -> sp.Rational:
    return sp.Rational(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n1", type=rational)
    parser.add_argument("n2", type=rational)
    parser.add_argument("n3", type=rational)
    parser.add_argument("--b", type=rational, default=sp.Rational(3, 2))
    args = parser.parse_args()

    p1, p2, p3 = sp.symbols("P1 P2 P3")
    sample = (args.b, p1, p2, p3)
    labels = (args.n1, args.n2, args.n3)
    print(f"labels={labels}, b={args.b}, Q={args.b + 1 / args.b}")
    for epsilon2 in (0, 1):
        for eta in (1, -1):
            value = sp.factor(
                fit.scaled_raw(labels, (epsilon2, 0, 0, eta), sample),
                extension=sp.I,
            )
            numerator, denominator = sp.fraction(sp.cancel(value))
            polynomial = sp.Poly(numerator, p1, p2, p3, domain="EX")
            print(
                f"epsilon2={epsilon2}, eta={eta}, "
                f"degree={polynomial.total_degree()}, denominator={denominator}"
            )
            print(value)


if __name__ == "__main__":
    main()
