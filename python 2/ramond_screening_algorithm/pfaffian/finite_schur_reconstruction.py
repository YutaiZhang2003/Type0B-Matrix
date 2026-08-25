"""Finite-field reconstruction in a bounded-width Schur basis.

This module is the interpolation layer for the state-free Ramond
screening calculation.  It never expands a Pfaffian quotient in the
screening variables.  The caller supplies only a black-box evaluation

    evaluate((t_1, ..., t_N), p) -> F(t_1, ..., t_N) in GF(p)

of a symmetric polynomial ``F`` whose Schur support is contained in the
rectangle ``k^N`` (equivalently, every partition has first row at most
``k``).  Exactly

    binomial(N + k, k)

black-box evaluations recover every Schur coefficient.

The interpolation nodes are all N-subsets of N+k distinct field points.
Indeed,

    Delta(t) s_lambda(t) = det(t_i**e_j),

where the strictly increasing exponents ``e_j`` form an N-subset of
``{0, ..., N+k-1}``.  Thus the evaluation matrix is the N-th compound
matrix of a square Vandermonde matrix.  Jacobi's complementary-minor
identity evaluates its inverse with k-by-k determinants.  If
``M=binomial(N+k,k)``, the transform therefore costs ``O(M^2 k^3)``
field operations after the M black-box calls.  In particular it does not
construct ``v_n``, ``W_n``, a PBW basis, or a multivariate symbolic
Pfaffian expansion.

The production routine is :func:`reconstruct_schur_mod`.  Running this
file performs an exact low-rank symbolic audit and the requested N=11,
k=2 benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb, gcd, isqrt
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence


Partition = tuple[int, ...]
FieldEvaluator = Callable[[tuple[int, ...], int], int]


def _trim_partition(parts: Iterable[int]) -> Partition:
    result = tuple(int(part) for part in parts)
    while result and result[-1] == 0:
        result = result[:-1]
    if any(part <= 0 for part in result):
        raise ValueError("a partition may have zeros only after its last part")
    if any(left < right for left, right in zip(result, result[1:])):
        raise ValueError("partition parts must be weakly decreasing")
    return result


def transpose_partition(partition: Sequence[int]) -> Partition:
    """Return the conjugate (transposed Young diagram) of ``partition``."""

    partition = _trim_partition(partition)
    if not partition:
        return ()
    return tuple(
        sum(row >= column for row in partition)
        for column in range(1, partition[0] + 1)
    )


def partition_from_exponents(exponents: Sequence[int]) -> Partition:
    """Map an increasing exponent subset to its Schur partition.

    If ``e_0 < ... < e_(N-1)``, then

        lambda = (e_(N-1)-(N-1), ..., e_1-1, e_0).
    """

    exponents = tuple(map(int, exponents))
    if any(left >= right for left, right in zip(exponents, exponents[1:])):
        raise ValueError("exponents must be strictly increasing")
    return _trim_partition(reversed(tuple(e - index for index, e in enumerate(exponents))))


def exponents_from_partition(partition: Sequence[int], variable_count: int) -> tuple[int, ...]:
    """Inverse of :func:`partition_from_exponents` for N variables."""

    partition = _trim_partition(partition)
    variable_count = int(variable_count)
    if len(partition) > variable_count:
        raise ValueError("partition has more rows than variables")
    padded = partition + (0,) * (variable_count - len(partition))
    return tuple(padded[variable_count - 1 - index] + index for index in range(variable_count))


def partitions_in_rectangle(variable_count: int, width: int) -> tuple[Partition, ...]:
    """All partitions in ``width^variable_count``, in compound order."""

    variable_count = int(variable_count)
    width = int(width)
    if variable_count < 0 or width < 0:
        raise ValueError("variable_count and width must be nonnegative")
    return tuple(
        partition_from_exponents(exponents)
        for exponents in combinations(range(variable_count + width), variable_count)
    )


def _det_mod(matrix: Sequence[Sequence[int]], prime: int) -> int:
    """Determinant over GF(prime), with small-minor fast paths."""

    size = len(matrix)
    if size == 0:
        return 1
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 1:
        return matrix[0][0] % prime
    if size == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        ) % prime

    work = [[entry % prime for entry in row] for row in matrix]
    answer = 1
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column]),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            answer = -answer
        pivot_value = work[column][column]
        answer = answer * pivot_value % prime
        inverse = pow(pivot_value, -1, prime)
        for row in range(column + 1, size):
            if not work[row][column]:
                continue
            factor = work[row][column] * inverse % prime
            for entry in range(column + 1, size):
                work[row][entry] = (
                    work[row][entry] - factor * work[column][entry]
                ) % prime
    return answer % prime


def _vandermonde(values: Sequence[int], prime: int) -> int:
    answer = 1
    for left, x_left in enumerate(values):
        for x_right in values[left + 1 :]:
            answer = answer * (x_right - x_left) % prime
    return answer


def _complement(subset: Sequence[int], size: int) -> tuple[int, ...]:
    chosen = set(subset)
    return tuple(index for index in range(size) if index not in chosen)


def _schur_minor(
    powers: Sequence[Sequence[int]],
    rows: Sequence[int],
    columns: Sequence[int],
    prime: int,
) -> int:
    return _det_mod(
        [[powers[row][column] for column in columns] for row in rows],
        prime,
    )


@dataclass(frozen=True)
class ReconstructionInfo:
    """Operation-size information returned on request by the reconstructor."""

    variable_count: int
    width: int
    sample_count: int
    callback_count: int
    inverse_minor_size: int
    complementary_minor_count: int
    accumulation_term_count: int

    @property
    def width_two_core_multiplications(self) -> int | None:
        """Exact inverse-transform multiply count when ``width == 2``.

        This counts two products for each 2-by-2 determinant, one product
        by the sampled value, and one final product by ``det(V)^(-1)``
        per coefficient.  Callback and Vandermonde setup are excluded.
        """

        if self.width != 2:
            return None
        return 3 * self.complementary_minor_count + self.sample_count

    @property
    def width_two_core_additions(self) -> int | None:
        """Exact inverse-transform add/subtract count for width two."""

        if self.width != 2:
            return None
        return 2 * self.complementary_minor_count

    @property
    def width_two_total_multiplications(self) -> int | None:
        """Exact multiply count outside the black-box callback.

        Modular exponentiation for the single field inverse is treated as
        one inversion, not expanded into a modulus-dependent addition
        chain.
        """

        core = self.width_two_core_multiplications
        if core is None:
            return None
        n = self.variable_count
        m = n + self.width
        sample_setup = self.sample_count * (comb(n, 2) + 1)
        global_vandermonde = comb(m, 2)
        power_table = m * (m - 1)
        return core + sample_setup + global_vandermonde + power_table

    @property
    def width_two_total_additions(self) -> int | None:
        """Exact add/subtract count outside the black-box callback."""

        core = self.width_two_core_additions
        if core is None:
            return None
        n = self.variable_count
        m = n + self.width
        return core + self.sample_count * comb(n, 2) + comb(m, 2)


def reconstruct_schur_mod(
    evaluate: FieldEvaluator,
    variable_count: int,
    width: int = 2,
    prime: int = 2_147_483_647,
    nodes: Sequence[int] | None = None,
    *,
    return_info: bool = False,
) -> dict[Partition, int] | tuple[dict[Partition, int], ReconstructionInfo]:
    """Recover the bounded-width Schur coefficients of ``evaluate``.

    Parameters
    ----------
    evaluate:
        A callback ``evaluate(t, prime)`` returning the value modulo
        ``prime``.  It is called exactly ``binomial(N+k,k)`` times.
    variable_count:
        Number N of screening variables.
    width:
        The proven upper bound k on the first row of every Schur label.
    prime:
        A prime field modulus.  Primality is not tested; all required
        denominators must be invertible modulo it.
    nodes:
        N+k pairwise distinct field elements.  The default is
        ``1, ..., N+k``.  A caller whose quotient has a pole at one of
        those nodes should pass a shifted set.
    return_info:
        Also return a :class:`ReconstructionInfo` record.

    The callback may evaluate a quotient, provided that quotient is a
    regular symmetric polynomial in the rectangle ``k^N`` and none of
    the selected sample tuples hits a removable denominator singularity.
    """

    variable_count = int(variable_count)
    width = int(width)
    prime = int(prime)
    if variable_count < 0 or width < 0:
        raise ValueError("variable_count and width must be nonnegative")
    total_points = variable_count + width
    if nodes is None:
        nodes = tuple(range(1, total_points + 1))
    else:
        nodes = tuple(int(node) % prime for node in nodes)
    if len(nodes) != total_points:
        raise ValueError(f"expected {total_points} interpolation nodes")
    if len(set(node % prime for node in nodes)) != total_points:
        raise ValueError("interpolation nodes must be distinct modulo prime")

    nodes = tuple(node % prime for node in nodes)
    subsets = tuple(combinations(range(total_points), variable_count))
    expected_count = comb(total_points, width)
    assert len(subsets) == expected_count

    # Multiplication by the ascending Vandermonde converts F into an
    # alternating polynomial whose coefficients are precisely the Schur
    # coefficients in the exponent-subset basis.
    samples: list[int] = []
    for subset in subsets:
        point = tuple(nodes[index] for index in subset)
        value = int(evaluate(point, prime)) % prime
        samples.append(value * _vandermonde(point, prime) % prime)

    determinant = _vandermonde(nodes, prime)
    if determinant == 0:
        raise ZeroDivisionError("singular Vandermonde matrix modulo prime")
    inverse_determinant = pow(determinant, -1, prime)

    complements = tuple(_complement(subset, total_points) for subset in subsets)
    subset_parities = tuple(sum(subset) & 1 for subset in subsets)
    powers_list = []
    for node in nodes:
        row = [1]
        for _ in range(1, total_points):
            row.append(row[-1] * node % prime)
        powers_list.append(tuple(row))
    powers = tuple(powers_list)

    coefficients: dict[Partition, int] = {}
    for exponent_subset, exponent_complement, exponent_parity in zip(
        subsets, complements, subset_parities
    ):
        coefficient = 0
        for sample, point_complement, point_parity in zip(
            samples, complements, subset_parities
        ):
            # Jacobi complementary-minor identity:
            # det(V^-1[E,A]) = (-1)^(sum(E)+sum(A))
            #                      det(V[A^c,E^c]) / det(V).
            inverse_minor_numerator = _schur_minor(
                powers,
                point_complement,
                exponent_complement,
                prime,
            )
            term = sample * inverse_minor_numerator
            if exponent_parity != point_parity:
                coefficient = (coefficient - term) % prime
            else:
                coefficient = (coefficient + term) % prime
        partition = partition_from_exponents(exponent_subset)
        coefficients[partition] = coefficient * inverse_determinant % prime

    if return_info:
        return coefficients, ReconstructionInfo(
            variable_count=variable_count,
            width=width,
            sample_count=expected_count,
            callback_count=len(samples),
            inverse_minor_size=width,
            complementary_minor_count=expected_count**2,
            accumulation_term_count=expected_count**2,
        )
    return coefficients


def complete_homogeneous_mod(
    values: Sequence[int], maximum_degree: int, prime: int
) -> tuple[int, ...]:
    """Return ``h_0, ..., h_maximum_degree`` over GF(prime)."""

    maximum_degree = int(maximum_degree)
    if maximum_degree < 0:
        return ()
    complete = [1] + [0] * maximum_degree
    for value in values:
        value %= prime
        # Ascending order implements multiplication by
        # (1-value*z)^(-1), including every power of this variable.
        for degree in range(1, maximum_degree + 1):
            complete[degree] = (
                complete[degree] + value * complete[degree - 1]
            ) % prime
    return tuple(complete)


def schur_mod(partition: Sequence[int], values: Sequence[int], prime: int) -> int:
    """Evaluate a Schur polynomial by Jacobi--Trudi over GF(prime)."""

    partition = _trim_partition(partition)
    if len(partition) > len(values):
        return 0
    if not partition:
        return 1
    size = len(partition)
    maximum_degree = partition[0] + size - 1
    complete = complete_homogeneous_mod(values, maximum_degree, prime)

    def h(degree: int) -> int:
        if degree < 0:
            return 0
        return complete[degree]

    return _det_mod(
        [
            [h(partition[row] - row + column) for column in range(size)]
            for row in range(size)
        ],
        prime,
    )


def dual_cauchy_evaluator(parameters: Sequence[int]) -> FieldEvaluator:
    """A dense, cheaply evaluated test polynomial of width ``len(parameters)``.

    The returned callback evaluates

        product_(i,a) (1 + t_i parameters_a).

    By the dual Cauchy identity its coefficient of ``s_lambda(t)`` is
    exactly ``s_(lambda^T)(parameters)``.  This supplies an independent
    closed-form audit of every reconstructed coefficient.
    """

    parameters = tuple(map(int, parameters))

    def evaluate(point: tuple[int, ...], prime: int) -> int:
        answer = 1
        for variable in point:
            for parameter in parameters:
                answer = answer * (1 + variable * parameter) % prime
        return answer

    return evaluate


def crt_pair(left: int, left_modulus: int, right: int, right_modulus: int) -> tuple[int, int]:
    """Combine two coprime scalar congruences."""

    if gcd(left_modulus, right_modulus) != 1:
        raise ValueError("CRT moduli must be coprime")
    correction = (
        (right - left) * pow(left_modulus, -1, right_modulus)
    ) % right_modulus
    modulus = left_modulus * right_modulus
    return (left + left_modulus * correction) % modulus, modulus


def combine_schur_crt(
    modular_coefficients: Sequence[Mapping[Partition, int]],
    primes: Sequence[int],
    *,
    centered: bool = True,
) -> tuple[dict[Partition, int], int]:
    """Combine coefficient dictionaries obtained at several primes."""

    if len(modular_coefficients) != len(primes) or not primes:
        raise ValueError("provide one nonempty coefficient map per prime")
    keys = tuple(modular_coefficients[0])
    if any(set(coefficients) != set(keys) for coefficients in modular_coefficients):
        raise ValueError("all coefficient maps must have identical partitions")

    result = {key: int(modular_coefficients[0][key]) % int(primes[0]) for key in keys}
    modulus = int(primes[0])
    for coefficients, prime in zip(modular_coefficients[1:], primes[1:]):
        prime = int(prime)
        for key in keys:
            result[key], _ = crt_pair(result[key], modulus, coefficients[key], prime)
        modulus *= prime
    if centered:
        result = {
            key: value - modulus if value > modulus // 2 else value
            for key, value in result.items()
        }
    return result, modulus


def rational_reconstruct(residue: int, modulus: int) -> Fraction:
    """Recover the unique small rational represented modulo ``modulus``.

    Both numerator and denominator are bounded by ``sqrt(modulus/2)``.
    More primes can be accumulated with :func:`combine_schur_crt` if this
    bound is insufficient.
    """

    residue %= modulus
    bound = isqrt(modulus // 2)
    old_remainder, remainder = modulus, residue
    old_denominator, denominator = 0, 1
    while remainder and abs(remainder) > bound:
        quotient = old_remainder // remainder
        old_remainder, remainder = remainder, old_remainder - quotient * remainder
        old_denominator, denominator = (
            denominator,
            old_denominator - quotient * denominator,
        )
    if remainder == 0 or denominator == 0:
        raise ValueError("no rational reconstruction within the standard bound")
    numerator = remainder
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    common = gcd(abs(numerator), denominator)
    numerator //= common
    denominator //= common
    if (
        abs(numerator) > bound
        or denominator > bound
        or gcd(denominator, modulus) != 1
        or numerator * pow(denominator, -1, modulus) % modulus != residue
    ):
        raise ValueError("no rational reconstruction within the standard bound")
    return Fraction(numerator, denominator)


def _schur_symbolic(partition, values):
    """Small SymPy Jacobi--Trudi evaluator used only by the audit."""

    import sympy as sp

    partition = _trim_partition(partition)
    if not partition:
        return sp.Integer(1)
    maximum_degree = partition[0] + len(partition) - 1
    complete = [sp.Integer(1)] + [sp.Integer(0)] * maximum_degree
    for value in values:
        for degree in range(1, maximum_degree + 1):
            complete[degree] += value * complete[degree - 1]

    def h(degree):
        return sp.Integer(0) if degree < 0 else complete[degree]

    return sp.det(
        sp.Matrix(
            [
                [h(partition[row] - row + column) for column in range(len(partition))]
                for row in range(len(partition))
            ]
        )
    )


def exact_symbolic_audit() -> None:
    """Check every coefficient and rebuild a low polynomial exactly."""

    import sympy as sp

    variable_count, width = 3, 2
    parameters = (2, -3)
    prime = 1_000_003
    coefficients, info = reconstruct_schur_mod(
        dual_cauchy_evaluator(parameters),
        variable_count,
        width,
        prime,
        return_info=True,
    )
    expected = {
        partition: schur_mod(transpose_partition(partition), parameters, prime)
        for partition in partitions_in_rectangle(variable_count, width)
    }
    assert coefficients == expected
    assert info.callback_count == comb(variable_count + width, width)

    xs = sp.symbols(f"x0:{variable_count}")
    target = sp.prod(
        1 + x * parameter for x in xs for parameter in parameters
    )
    centered = {
        partition: value - prime if value > prime // 2 else value
        for partition, value in coefficients.items()
    }
    rebuilt = sum(
        coefficient * _schur_symbolic(partition, xs)
        for partition, coefficient in centered.items()
    )
    assert sp.expand(target - rebuilt) == 0

    # Audit the optional exact arithmetic utilities independently.
    fraction = Fraction(-37, 23)
    primes = (1_000_003, 1_000_033)
    residues = [
        fraction.numerator * pow(fraction.denominator, -1, prime) % prime
        for prime in primes
    ]
    combined, modulus = crt_pair(residues[0], primes[0], residues[1], primes[1])
    assert rational_reconstruct(combined, modulus) == fraction


def benchmark(variable_count: int = 11, width: int = 2) -> dict[str, int | float]:
    """Run the dense dual-Cauchy benchmark and verify every coefficient."""

    prime = 2_147_483_647
    parameters = tuple(17 + 6 * index for index in range(width))
    started = perf_counter()
    coefficients, info = reconstruct_schur_mod(
        dual_cauchy_evaluator(parameters),
        variable_count,
        width,
        prime,
        return_info=True,
    )
    elapsed = perf_counter() - started
    expected = {
        partition: schur_mod(transpose_partition(partition), parameters, prime)
        for partition in partitions_in_rectangle(variable_count, width)
    }
    assert coefficients == expected
    return {
        "N": variable_count,
        "width": width,
        "coefficients": len(coefficients),
        "callback_evaluations": info.callback_count,
        "inverse_minor_size": info.inverse_minor_size,
        "complementary_minors": info.complementary_minor_count,
        "core_multiplications": info.width_two_core_multiplications or -1,
        "core_additions": info.width_two_core_additions or -1,
        "total_multiplications": info.width_two_total_multiplications or -1,
        "total_additions": info.width_two_total_additions or -1,
        "seconds": elapsed,
    }


def main() -> None:
    exact_symbolic_audit()
    print("exact symbolic audit: PASS")
    result = benchmark()
    print(
        "N={N}, width={width}: {coefficients} Schur coefficients from "
        "{callback_evaluations} callbacks; inverse minors {inverse_minor_size}x"
        "{inverse_minor_size}, count={complementary_minors}; "
        "core mul/add={core_multiplications}/{core_additions}; "
        "total outside callback={total_multiplications}/{total_additions}; "
        "{seconds:.6f} s".format(**result)
    )


if __name__ == "__main__":
    main()
