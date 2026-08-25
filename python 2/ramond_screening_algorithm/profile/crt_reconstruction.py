#!/usr/bin/env python3
"""CRT/rational-reconstruction check for the modular transition solver.

The production audit should reconstruct only the final three-point scalar.
This small test reconstructs an entire low-level transition column so that
the exactness of the finite-field procedure can be checked independently.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt

import numpy as np
import sympy as sp

import modular_transition as modular


def crt_pair(first, first_modulus, second, second_modulus):
    correction = (second - first) % second_modulus
    correction *= pow(first_modulus % second_modulus, -1, second_modulus)
    value = first + first_modulus * (correction % second_modulus)
    modulus = first_modulus * second_modulus
    return value % modulus, modulus


def rational_reconstruct(residue, modulus):
    """Return the unique small rational represented by a modular residue."""

    residue %= modulus
    bound = isqrt(modulus // 2)
    old_remainder, remainder = modulus, residue
    old_denominator, denominator = 0, 1
    while remainder > bound:
        quotient = old_remainder // remainder
        old_remainder, remainder = remainder, old_remainder - quotient * remainder
        old_denominator, denominator = (
            denominator,
            old_denominator - quotient * denominator,
        )
    numerator = remainder
    if denominator == 0 or abs(denominator) > bound:
        return None
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    if gcd(numerator, denominator) != 1:
        return None
    if (numerator - residue * denominator) % modulus:
        return None
    return Fraction(numerator, denominator)


def primes_one_mod_eight(start=1_000_000):
    candidate = int(sp.nextprime(start))
    while True:
        if candidate % 8 == 1:
            yield candidate
        candidate = int(sp.nextprime(candidate))


def reconstruct_ns_column(level2=9, modes2=(9,)):
    rational_q = sp.Rational(13, 6)
    rational_p = sp.Rational(2, 5)
    residues_real = None
    residues_imaginary = None
    modulus = 1
    basis = None

    for prime_count, prime in enumerate(primes_one_mod_eight(), start=1):
        root_i, _ = modular.roots(prime)
        q_value = modular.rational_mod(rational_q, prime)
        momentum = modular.rational_mod(rational_p, prime)
        basis, matrix_plus = modular.ns_transition(
            level2, q_value, momentum, root_i, prime
        )
        _, matrix_minus = modular.ns_transition(
            level2, q_value, momentum, -root_i % prime, prime
        )
        rhs = modular.ns_target_rhs(basis, modes2)
        plus = modular.solve_mod(matrix_plus, rhs, prime)[:, 0]
        minus = modular.solve_mod(matrix_minus, rhs, prime)[:, 0]
        inverse_two = pow(2, -1, prime)
        real = (plus + minus) * inverse_two % prime
        imaginary = (plus - minus) * inverse_two % prime
        imaginary = imaginary * pow(root_i, -1, prime) % prime

        if residues_real is None:
            residues_real = [int(value) for value in real]
            residues_imaginary = [int(value) for value in imaginary]
            modulus = prime
        else:
            old_modulus = modulus
            for index in range(len(basis)):
                residues_real[index], _ = crt_pair(
                    residues_real[index], old_modulus, int(real[index]), prime
                )
                residues_imaginary[index], _ = crt_pair(
                    residues_imaginary[index],
                    old_modulus,
                    int(imaginary[index]),
                    prime,
                )
            modulus = old_modulus * prime

        real_q = [rational_reconstruct(value, modulus) for value in residues_real]
        imaginary_q = [
            rational_reconstruct(value, modulus) for value in residues_imaginary
        ]
        if any(value is None for value in real_q + imaginary_q):
            continue
        candidate = sp.Matrix(
            [
                sp.Rational(real.numerator, real.denominator)
                + sp.I * sp.Rational(imaginary.numerator, imaginary.denominator)
                for real, imaginary in zip(real_q, imaginary_q)
            ]
        )

        old_q = modular.ns_reference.Q
        modular.ns_reference.Q = rational_q
        try:
            exact_basis, exact_matrix = modular.ns_reference.transition(
                level2, rational_p
            )
        finally:
            modular.ns_reference.Q = old_q
            modular.ns_reference.transition.cache_clear()
        if exact_basis != basis:
            raise AssertionError("basis order changed")
        exact_rhs = sp.zeros(len(basis), 1)
        exact_rhs[basis.index(((), tuple(sorted(modes2, reverse=True))))] = 1
        if exact_matrix * candidate == exact_rhs:
            return prime_count, modulus, candidate
    raise RuntimeError("reconstruction did not stabilize")


def main():
    prime_count, modulus, vector = reconstruct_ns_column()
    print(f"reconstructed an exact level-9 NS column with {prime_count} primes")
    print(f"CRT modulus has {modulus.bit_length()} bits")
    print(f"nonzero coefficients: {sum(value != 0 for value in vector)}")


if __name__ == "__main__":
    main()
