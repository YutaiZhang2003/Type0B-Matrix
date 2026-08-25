#!/usr/bin/env python3
"""Small exact benchmark for the cubic fermionic stage."""

from __future__ import annotations

import argparse
import random
import time

import sympy as sp

from .core import pfaffian


def skew_matrix(size, seed=17):
    random.seed(seed)
    answer = sp.zeros(size)
    for row in range(size):
        for column in range(row + 1, size):
            value = sp.Rational(random.randint(-31, 31), random.randint(1, 37))
            answer[row, column] = value
            answer[column, row] = -value
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    args = parser.parse_args()
    if args.size % 2:
        raise SystemExit("size must be even")
    matrix = skew_matrix(args.size)
    began = time.perf_counter()
    value = pfaffian(matrix)
    elapsed = time.perf_counter() - began
    if args.size <= 20 and sp.factor(value**2 - matrix.det()) != 0:
        raise AssertionError("Pfaffian determinant check failed")
    print(f"exact Pfaffian size={args.size}: {elapsed:.6f}s")


if __name__ == "__main__":
    main()
