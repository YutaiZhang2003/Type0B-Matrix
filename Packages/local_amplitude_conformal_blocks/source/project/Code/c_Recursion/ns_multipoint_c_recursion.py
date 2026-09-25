#!/usr/bin/env python3
"""Multipoint NS central-charge recursion in plumbing coordinates.

This module implements the coefficient form of the Belavin--Geiko
``c``-recursion for two standard trivalent graphs:

* the sphere ``N``-point linear channel;
* the torus ``N``-point necklace channel (``N >= 2``).

Sphere insertions may be bottom components or their level-one-half
``G_-1/2`` components.  A block is labelled by one relative three-form
sector, zero or one, at each trivalent vertex.  The xor of the sphere vertex
sectors equals the xor of the external component markings.  At a pole on an
internal edge an odd null vector toggles the sectors at the two endpoints of
that edge.  This graph-level transport is the part of the multipoint
recursion which is invisible in a four-point scalar implementation.  The
torus necklace interface retains bottom external components for now.

Levels are represented by non-negative integers ``twice_levels``.  Thus the
monomial associated with ``(t_1, ..., t_M)`` is
``prod_i q_i**(t_i/2)``.  The returned coefficient does not include the
leading powers of the plumbing parameters.

The central charge is the ordinary super-Virasoro charge.  Belavin--Geiko's
``hat c`` is related by ``c_ordinary = 3*hat_c/2``; the shared pole kernels in
``ns_recursion_recipe`` already make this conversion.

The implementation assumes generic, non-confluent fixed-weight ``c`` poles.
At a collision it raises ``ZeroDivisionError`` instead of silently choosing
a finite-part prescription.  The collision-aware genus-two code in this
directory provides the model for a future multipoint wrapper.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import math
from typing import Iterable, Sequence

import mpmath

from ns_global_osp_block import osp_norm, osp_sector_vertex
from ns_recursion_recipe import ns_ordinary_edge_scalar_kernel_mp


def _validate_twice_levels(
    values: Sequence[int], *, expected_length: int
) -> tuple[int, ...]:
    if len(values) != expected_length:
        raise ValueError(
            f"twice_levels must contain exactly {expected_length} entries"
        )
    levels = tuple(values)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in levels
    ):
        raise ValueError("twice_levels must contain non-negative integers")
    return levels


def _validate_sectors(
    values: Sequence[int],
    *,
    expected_length: int,
    graph_name: str,
    required_parity: int = 0,
) -> tuple[int, ...]:
    if len(values) != expected_length:
        raise ValueError(
            f"vertex_sectors must contain exactly {expected_length} entries"
        )
    if any(value not in (0, 1) for value in values):
        raise ValueError("vertex_sectors must contain only zeroes and ones")
    if required_parity not in (0, 1):
        raise ValueError("required sector parity must be zero or one")
    sectors = tuple(int(value) for value in values)
    if sum(sectors) % 2 != int(required_parity):
        raise ValueError(
            f"{graph_name} requires total vertex-sector parity "
            f"{int(required_parity)}"
        )
    return sectors


def _state_from_twice_level(twice_level: int) -> tuple[int, int]:
    epsilon = int(twice_level) % 2
    return (int(twice_level) - epsilon) // 2, epsilon


def _iter_ns_c_poles(maximum_product: int) -> Iterable[tuple[int, int]]:
    """Yield fixed-weight NS c-poles with ``r*s <= maximum_product``."""

    for r in range(2, int(maximum_product) + 1):
        for s in range(1, int(maximum_product) // r + 1):
            if (r + s) % 2 == 0:
                yield r, s


@lru_cache(maxsize=None)
def ns_non_global_vacuum_coefficients(
    max_twice_level: int, spin_lift: int = 1
) -> tuple[int, ...]:
    r"""Return coefficients of the non-global NS vacuum factor.

    With ``x = Q**(1/2)`` the factor is

    ``prod_(n>=2) (1 + spin_lift*x**(2*n-1))/(1-x**(2*n))``.

    In a necklace ``Q`` is the product of all plumbing parameters, so a term
    of twice-level ``k`` shifts every edge level by the same integer ``k``.
    ``spin_lift=1`` is the ordinary NS trace and ``spin_lift=-1`` is the
    temporally twisted lift.
    """

    if not isinstance(max_twice_level, int) or max_twice_level < 0:
        raise ValueError("max_twice_level must be a non-negative integer")
    if spin_lift not in (-1, 1):
        raise ValueError("spin_lift must be +1 or -1")

    coefficients = [0] * (max_twice_level + 1)
    coefficients[0] = 1
    for n in range(2, max_twice_level // 2 + 2):
        fermion_power = 2 * n - 1
        if fermion_power <= max_twice_level:
            previous = coefficients.copy()
            for degree in range(fermion_power, max_twice_level + 1):
                coefficients[degree] += (
                    spin_lift * previous[degree - fermion_power]
                )

        boson_power = 2 * n
        if boson_power <= max_twice_level:
            # Ascending in-place multiplication by 1/(1-x**boson_power).
            for degree in range(boson_power, max_twice_level + 1):
                coefficients[degree] += coefficients[degree - boson_power]
    return tuple(coefficients)


class _NSMultipointCRecursionBase:
    """Shared coefficient recursion for an ordinary-edge plumbing graph."""

    edge_count: int
    vertex_count: int

    def __init__(
        self,
        *,
        central_charge,
        internal_weights: Sequence[object],
        vertex_sectors: Sequence[int],
        working_precision: int,
        pole_tolerance: float,
        graph_name: str,
        required_sector_parity: int = 0,
    ) -> None:
        if not isinstance(working_precision, int) or working_precision < 20:
            raise ValueError("working_precision must be an integer at least 20")
        if not math.isfinite(pole_tolerance) or pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")
        self.working_precision = int(working_precision)
        self.pole_tolerance = float(pole_tolerance)
        with mpmath.workdps(self.working_precision):
            self.central_charge = mpmath.mpc(central_charge)
            self.internal_weights = tuple(
                mpmath.mpc(weight) for weight in internal_weights
            )
        if len(self.internal_weights) != self.edge_count:
            raise ValueError(
                f"internal_weights must contain exactly {self.edge_count} entries"
            )
        self.vertex_sectors = _validate_sectors(
            vertex_sectors,
            expected_length=self.vertex_count,
            graph_name=graph_name,
            required_parity=required_sector_parity,
        )
        self._coefficient_cache: dict[tuple[object, ...], object] = {}

    def _global_coefficient(
        self,
        twice_levels: tuple[int, ...],
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        raise NotImplementedError

    def _regular_coefficient(
        self,
        twice_levels: tuple[int, ...],
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        raise NotImplementedError

    def _edge_kernel_arguments(
        self,
        edge: int,
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ) -> tuple[
        tuple[object, object],
        tuple[object, object],
        int,
        int,
        int,
        int,
    ]:
        """Return weight pairs, sectors, and endpoint vertex indices."""

        raise NotImplementedError

    def _compatible_level_parities(
        self, vertex_sectors: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """Return uniquely fixed edge parities, when the graph has them."""

        return None

    def _edge_residue(
        self,
        *,
        edge: int,
        r: int,
        s: int,
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        (
            first_weights,
            second_weights,
            first_sector,
            second_sector,
            first_vertex,
            second_vertex,
        ) = self._edge_kernel_arguments(edge, internal_weights, vertex_sectors)
        pole, scalar, child_endpoint_sectors = (
            ns_ordinary_edge_scalar_kernel_mp(
                r=r,
                s=s,
                internal_weight=internal_weights[edge],
                left_weights=first_weights,
                right_weights=second_weights,
                left_sector=first_sector,
                right_sector=second_sector,
            )
        )

        # This is the same component-to-fixed-parity transport phase used by
        # HighPrecisionNSSphereFourPointBlock.  It belongs to an ordinary
        # edge with two distinct incidences, hence applies to both supported
        # multipoint graphs.
        scalar *= -1 if (r * s) % 2 else 1

        children = list(vertex_sectors)
        children[first_vertex], children[second_vertex] = child_endpoint_sectors
        return pole, scalar, tuple(children)

    def _coefficient(
        self,
        twice_levels: tuple[int, ...],
        central_charge,
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        key = (
            twice_levels,
            central_charge,
            internal_weights,
            vertex_sectors,
        )
        cached = self._coefficient_cache.get(key)
        if cached is not None:
            return cached

        result = mpmath.mpc(
            self._regular_coefficient(
                twice_levels, internal_weights, vertex_sectors
            )
        )
        for edge, available_level in enumerate(twice_levels):
            for r, s in _iter_ns_c_poles(available_level):
                product_rs = r * s
                pole, residue, child_sectors = self._edge_residue(
                    edge=edge,
                    r=r,
                    s=s,
                    internal_weights=internal_weights,
                    vertex_sectors=vertex_sectors,
                )
                denominator = central_charge - pole.c
                scale = max(
                    mpmath.mpf(1),
                    abs(central_charge),
                    abs(pole.c),
                )
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise ZeroDivisionError(
                        "multipoint c-recursion encountered a confluent or "
                        f"on-contour ({r},{s}) pole on edge {edge}: "
                        f"c={central_charge!r}, c_rs={pole.c!r}"
                    )

                child_levels = list(twice_levels)
                child_levels[edge] -= product_rs
                child_weights = list(internal_weights)
                child_weights[edge] += mpmath.mpf(product_rs) / 2
                result += residue / denominator * self._coefficient(
                    tuple(child_levels),
                    pole.c,
                    tuple(child_weights),
                    child_sectors,
                )

        self._coefficient_cache[key] = result
        return result

    def global_coefficient(self, twice_levels: Sequence[int]):
        """Return the global ``osp(1|2)`` coefficient at one multi-level."""

        levels = _validate_twice_levels(
            twice_levels, expected_length=self.edge_count
        )
        with mpmath.workdps(self.working_precision):
            return self._global_coefficient(
                levels, self.internal_weights, self.vertex_sectors
            )

    def compatible_level_parities(self) -> tuple[int, ...] | None:
        """Return the edge twice-level parities selected by this block.

        Tree graphs have a unique answer fixed by the external component and
        trinion-sector markings.  Necklace graphs may return ``None`` when a
        separate parity sum remains.
        """

        return self._compatible_level_parities(self.vertex_sectors)

    def regular_coefficient(self, twice_levels: Sequence[int]):
        """Return the large-``c`` regular coefficient at one multi-level."""

        levels = _validate_twice_levels(
            twice_levels, expected_length=self.edge_count
        )
        with mpmath.workdps(self.working_precision):
            return self._regular_coefficient(
                levels, self.internal_weights, self.vertex_sectors
            )

    def coefficient(self, twice_levels: Sequence[int]):
        """Return the full fixed-weight ``c``-recursive coefficient."""

        levels = _validate_twice_levels(
            twice_levels, expected_length=self.edge_count
        )
        with mpmath.workdps(self.working_precision):
            return self._coefficient(
                levels,
                self.central_charge,
                self.internal_weights,
                self.vertex_sectors,
            )

    def coefficient_table(
        self, max_twice_levels: Sequence[int]
    ) -> dict[tuple[int, ...], object]:
        """Return all coefficients in a rectangular multi-level cutoff."""

        maxima = _validate_twice_levels(
            max_twice_levels, expected_length=self.edge_count
        )
        return {
            levels: self.coefficient(levels)
            for levels in product(*(range(value + 1) for value in maxima))
        }

    def series_value(
        self,
        q_values: Sequence[object],
        max_twice_levels: Sequence[int],
        *,
        max_total_twice_level: int | None = None,
        q_log_values: Sequence[object] | None = None,
        minimum_twice_levels: Sequence[int] | None = None,
    ):
        r"""Evaluate the reduced plumbing series at a finite cutoff.

        The returned value is

        ``sum_levels coefficient(levels) prod_i q_i**(levels_i/2)``.

        It does not include the channel-dependent leading powers of the
        plumbing parameters.  ``max_twice_levels`` supplies an independent
        rectangular cutoff on every edge.  When
        ``max_total_twice_level`` is provided, terms above that total
        twice-level are omitted as well.  ``q_log_values`` may supply a
        coherent lift of every plumbing logarithm.  This is required for an
        antiholomorphic block on a branch cut: independently applying the
        principal power to ``conj(q)`` gives the wrong sign for half-integer
        NS levels when ``q`` is negative real.
        """

        maxima = _validate_twice_levels(
            max_twice_levels, expected_length=self.edge_count
        )
        minima = (
            (0,) * self.edge_count
            if minimum_twice_levels is None
            else _validate_twice_levels(
                minimum_twice_levels, expected_length=self.edge_count
            )
        )
        if len(q_values) != self.edge_count:
            raise ValueError(
                f"q_values must contain exactly {self.edge_count} entries"
            )
        if max_total_twice_level is not None and (
            not isinstance(max_total_twice_level, int)
            or isinstance(max_total_twice_level, bool)
            or max_total_twice_level < 0
        ):
            raise ValueError(
                "max_total_twice_level must be a non-negative integer or None"
            )

        with mpmath.workdps(self.working_precision):
            q_tuple = tuple(mpmath.mpc(value) for value in q_values)
            if any(
                not (mpmath.isfinite(value.real) and mpmath.isfinite(value.imag))
                for value in q_tuple
            ):
                raise ValueError("q_values must be finite")
            log_tuple = (
                tuple(mpmath.log(value) for value in q_tuple)
                if q_log_values is None
                else tuple(mpmath.mpc(value) for value in q_log_values)
            )
            if len(log_tuple) != self.edge_count or any(
                not (mpmath.isfinite(value.real) and mpmath.isfinite(value.imag))
                for value in log_tuple
            ):
                raise ValueError(
                    "q_log_values must contain one finite logarithm per edge"
                )

            level_parities = self._compatible_level_parities(
                self.vertex_sectors
            )
            level_ranges = (
                tuple(range(parity, maximum + 1, 2))
                for parity, maximum in zip(level_parities, maxima)
            ) if level_parities is not None else (
                tuple(range(maximum + 1)) for maximum in maxima
            )
            result = mpmath.mpc(0)
            for levels in product(*level_ranges):
                if any(level < minimum for level, minimum in zip(levels, minima)):
                    continue
                if (
                    max_total_twice_level is not None
                    and sum(levels) > max_total_twice_level
                ):
                    continue
                monomial = mpmath.fprod(
                    mpmath.exp(log_q * (mpmath.mpf(level) / 2))
                    for log_q, level in zip(log_tuple, levels)
                )
                result += self._coefficient(
                    levels,
                    self.central_charge,
                    self.internal_weights,
                    self.vertex_sectors,
                ) * monomial
            return result

    def recursive_series_value(
        self,
        q_values: Sequence[object],
        recursion_max_twice_level: int,
        global_max_twice_levels: Sequence[int],
        *,
        global_max_total_twice_level: int | None = None,
        q_log_values: Sequence[object] | None = None,
        minimum_twice_levels: Sequence[int] | None = None,
        maximum_accumulated_twice_levels: Sequence[int | None] | None = None,
    ):
        r"""Evaluate the functional c-recursion with finite global leaves.

        ``recursion_max_twice_level`` bounds the accumulated null-vector
        twice-level along a recursion path.  Every leaf is the regular
        large-c graph block, evaluated with ``global_max_twice_levels``.
        For a sphere that regular block is precisely the global
        ``osp(1|2)`` network.  This is the multipoint analogue of evaluating
        the four-point c-recursion with exact hypergeometric leaves and is
        much faster than constructing every full finite-c coefficient when
        only a numerical block value is needed.  ``q_log_values`` has the
        same coherent-branch meaning as in :meth:`series_value`.
        """

        if (
            not isinstance(recursion_max_twice_level, int)
            or isinstance(recursion_max_twice_level, bool)
            or recursion_max_twice_level < 0
        ):
            raise ValueError(
                "recursion_max_twice_level must be a non-negative integer"
            )
        maxima = _validate_twice_levels(
            global_max_twice_levels, expected_length=self.edge_count
        )
        minima = (
            (0,) * self.edge_count
            if minimum_twice_levels is None
            else _validate_twice_levels(
                minimum_twice_levels, expected_length=self.edge_count
            )
        )
        if maximum_accumulated_twice_levels is None:
            accumulated_maxima: tuple[int | None, ...] = (None,) * self.edge_count
        else:
            if len(maximum_accumulated_twice_levels) != self.edge_count:
                raise ValueError(
                    "maximum_accumulated_twice_levels must contain one entry per edge"
                )
            accumulated_maxima = tuple(maximum_accumulated_twice_levels)
            if any(
                value is not None
                and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                )
                for value in accumulated_maxima
            ):
                raise ValueError(
                    "accumulated twice-level maxima must be non-negative integers or None"
                )
        if len(q_values) != self.edge_count:
            raise ValueError(
                f"q_values must contain exactly {self.edge_count} entries"
            )
        if global_max_total_twice_level is not None and (
            not isinstance(global_max_total_twice_level, int)
            or isinstance(global_max_total_twice_level, bool)
            or global_max_total_twice_level < 0
        ):
            raise ValueError(
                "global_max_total_twice_level must be a non-negative integer "
                "or None"
            )

        with mpmath.workdps(self.working_precision):
            q_tuple = tuple(mpmath.mpc(value) for value in q_values)
            if any(
                not (mpmath.isfinite(value.real) and mpmath.isfinite(value.imag))
                for value in q_tuple
            ):
                raise ValueError("q_values must be finite")
            log_tuple = (
                tuple(mpmath.log(value) for value in q_tuple)
                if q_log_values is None
                else tuple(mpmath.mpc(value) for value in q_log_values)
            )
            if len(log_tuple) != self.edge_count or any(
                not (mpmath.isfinite(value.real) and mpmath.isfinite(value.imag))
                for value in log_tuple
            ):
                raise ValueError(
                    "q_log_values must contain one finite logarithm per edge"
                )

            regular_cache: dict[tuple[object, ...], object] = {}
            recursion_cache: dict[tuple[object, ...], object] = {}

            def regular_value(
                internal_weights,
                vertex_sectors,
                required_minima,
                remaining_maxima,
            ):
                key = (
                    internal_weights,
                    vertex_sectors,
                    required_minima,
                    remaining_maxima,
                )
                if key in regular_cache:
                    return regular_cache[key]
                level_parities = self._compatible_level_parities(
                    vertex_sectors
                )
                level_ranges = (
                    tuple(range(parity, maximum + 1, 2))
                    for parity, maximum in zip(level_parities, maxima)
                ) if level_parities is not None else (
                    tuple(range(maximum + 1)) for maximum in maxima
                )
                result = mpmath.mpc(0)
                for levels in product(*level_ranges):
                    if any(
                        level < minimum
                        for level, minimum in zip(levels, required_minima)
                    ):
                        continue
                    if any(
                        maximum is not None and level > maximum
                        for level, maximum in zip(levels, remaining_maxima)
                    ):
                        continue
                    if (
                        global_max_total_twice_level is not None
                        and sum(levels) > global_max_total_twice_level
                    ):
                        continue
                    monomial = mpmath.fprod(
                        mpmath.exp(log_q * (mpmath.mpf(level) / 2))
                        for log_q, level in zip(log_tuple, levels)
                    )
                    result += self._regular_coefficient(
                        levels, internal_weights, vertex_sectors
                    ) * monomial
                regular_cache[key] = result
                return result

            def recurse(
                budget,
                central_charge,
                internal_weights,
                vertex_sectors,
                required_minima,
                remaining_maxima,
            ):
                key = (
                    budget,
                    central_charge,
                    internal_weights,
                    vertex_sectors,
                    required_minima,
                    remaining_maxima,
                )
                if key in recursion_cache:
                    return recursion_cache[key]

                result = mpmath.mpc(
                    regular_value(
                        internal_weights,
                        vertex_sectors,
                        required_minima,
                        remaining_maxima,
                    )
                )
                for edge in range(self.edge_count):
                    for r, s in _iter_ns_c_poles(budget):
                        shift = r * s
                        if (
                            remaining_maxima[edge] is not None
                            and shift > remaining_maxima[edge]
                        ):
                            continue
                        pole, residue, child_sectors = self._edge_residue(
                            edge=edge,
                            r=r,
                            s=s,
                            internal_weights=internal_weights,
                            vertex_sectors=vertex_sectors,
                        )
                        denominator = central_charge - pole.c
                        scale = max(
                            mpmath.mpf(1),
                            abs(central_charge),
                            abs(pole.c),
                        )
                        if abs(denominator) <= self.pole_tolerance * scale:
                            raise ZeroDivisionError(
                                "multipoint functional c-recursion encountered "
                                f"a confluent or on-contour ({r},{s}) pole on "
                                f"edge {edge}: c={central_charge!r}, "
                                f"c_rs={pole.c!r}"
                            )
                        child_weights = list(internal_weights)
                        child_weights[edge] += mpmath.mpf(shift) / 2
                        child_minima = list(required_minima)
                        child_minima[edge] = max(
                            0, child_minima[edge] - shift
                        )
                        child_maxima = list(remaining_maxima)
                        if child_maxima[edge] is not None:
                            child_maxima[edge] -= shift
                        result += (
                            mpmath.exp(
                                log_tuple[edge] * (mpmath.mpf(shift) / 2)
                            )
                            * residue
                            / denominator
                            * recurse(
                                budget - shift,
                                pole.c,
                                tuple(child_weights),
                                child_sectors,
                                tuple(child_minima),
                                tuple(child_maxima),
                            )
                        )
                recursion_cache[key] = result
                return result

            return recurse(
                recursion_max_twice_level,
                self.central_charge,
                self.internal_weights,
                self.vertex_sectors,
                minima,
                accumulated_maxima,
            )

    def clear_cache(self) -> None:
        """Discard memoized recursive coefficients."""

        self._coefficient_cache.clear()


class NSSphereLinearCRecursion(_NSMultipointCRecursionBase):
    r"""NS sphere ``N``-point block in the standard linear channel.

    External weights are ordered from zero to infinity.  For ``N=5`` the
    three vertices are

    ``rho(h1,d2,d1)``, ``rho(h2,d3,h1)``, ``rho(d5,d4,h2)``.

    There are ``N-3`` internal edges and ``N-2`` vertex sectors.
    ``external_descendants[i]=1`` marks ``G_-1/2 V_i``.  The xor of all
    vertex sectors must equal the xor of these external markings.
    """

    def __init__(
        self,
        *,
        central_charge,
        external_weights: Sequence[object],
        internal_weights: Sequence[object],
        external_descendants: Sequence[int] | None = None,
        vertex_sectors: Sequence[int] | None = None,
        working_precision: int = 60,
        pole_tolerance: float = 1.0e-30,
    ) -> None:
        if len(external_weights) < 4:
            raise ValueError("a sphere linear block requires at least four points")
        self.external_weights = tuple(external_weights)
        if external_descendants is None:
            external_descendants = (0,) * len(self.external_weights)
        if len(external_descendants) != len(self.external_weights) or any(
            value not in (0, 1) for value in external_descendants
        ):
            raise ValueError(
                "external_descendants must contain one zero/one marking per "
                "external weight"
            )
        self.external_descendants = tuple(
            int(value) for value in external_descendants
        )
        self.edge_count = len(self.external_weights) - 3
        self.vertex_count = self.edge_count + 1
        if vertex_sectors is None:
            vertex_sectors = (
                sum(self.external_descendants) % 2,
            ) + (0,) * (self.vertex_count - 1)
        with mpmath.workdps(working_precision):
            self.external_weights = tuple(
                mpmath.mpc(weight) for weight in self.external_weights
            )
        super().__init__(
            central_charge=central_charge,
            internal_weights=internal_weights,
            vertex_sectors=vertex_sectors,
            working_precision=working_precision,
            pole_tolerance=pole_tolerance,
            graph_name="sphere-linear",
            required_sector_parity=sum(self.external_descendants) % 2,
        )

    def _global_coefficient(
        self,
        twice_levels: tuple[int, ...],
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        states = tuple(_state_from_twice_level(level) for level in twice_levels)
        result = mpmath.mpc(1)

        n0, epsilon0 = states[0]
        result *= osp_sector_vertex(
            sector=vertex_sectors[0],
            n1=n0,
            n2=0,
            n3=0,
            epsilon1=epsilon0,
            epsilon2=self.external_descendants[1],
            epsilon3=self.external_descendants[0],
            d1=internal_weights[0],
            d2=self.external_weights[1],
            d3=self.external_weights[0],
        )

        for vertex in range(1, self.edge_count):
            upper_n, upper_epsilon = states[vertex]
            lower_n, lower_epsilon = states[vertex - 1]
            result *= osp_sector_vertex(
                sector=vertex_sectors[vertex],
                n1=upper_n,
                n2=0,
                n3=lower_n,
                epsilon1=upper_epsilon,
                epsilon2=self.external_descendants[vertex + 1],
                epsilon3=lower_epsilon,
                d1=internal_weights[vertex],
                d2=self.external_weights[vertex + 1],
                d3=internal_weights[vertex - 1],
            )

        last_n, last_epsilon = states[-1]
        result *= osp_sector_vertex(
            sector=vertex_sectors[-1],
            n1=0,
            n2=0,
            n3=last_n,
            epsilon1=self.external_descendants[-1],
            epsilon2=self.external_descendants[-2],
            epsilon3=last_epsilon,
            d1=self.external_weights[-1],
            d2=self.external_weights[-2],
            d3=internal_weights[-1],
        )

        denominator = mpmath.mpc(1)
        for weight, (occupation, epsilon) in zip(internal_weights, states):
            denominator *= osp_norm(weight, occupation, epsilon)
        return result / denominator

    def _compatible_level_parities(
        self, vertex_sectors: tuple[int, ...]
    ) -> tuple[int, ...]:
        # At the first endpoint, the edge parity is the local three-form
        # sector xor the two prescribed external component parities.  Every
        # middle trinion then fixes the next edge parity.
        parities = [
            vertex_sectors[0]
            ^ self.external_descendants[0]
            ^ self.external_descendants[1]
        ]
        for vertex in range(1, self.edge_count):
            parities.append(
                parities[-1]
                ^ vertex_sectors[vertex]
                ^ self.external_descendants[vertex + 1]
            )
        expected_last = (
            vertex_sectors[-1]
            ^ self.external_descendants[-2]
            ^ self.external_descendants[-1]
        )
        if parities[-1] != expected_last:
            raise ValueError("inconsistent sphere vertex-sector assignment")
        return tuple(parities)

    def _regular_coefficient(
        self,
        twice_levels: tuple[int, ...],
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        # A sphere has no non-global vacuum handle factor.
        return self._global_coefficient(
            twice_levels, internal_weights, vertex_sectors
        )

    def _edge_kernel_arguments(
        self,
        edge: int,
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        first_other = (
            self.external_weights[0]
            if edge == 0
            else internal_weights[edge - 1]
        )
        first_middle = self.external_weights[edge + 1]
        second_other = (
            self.external_weights[-1]
            if edge == self.edge_count - 1
            else internal_weights[edge + 1]
        )
        second_middle = self.external_weights[edge + 2]
        return (
            (first_other, first_middle),
            (second_other, second_middle),
            vertex_sectors[edge],
            vertex_sectors[edge + 1],
            edge,
            edge + 1,
        )


class NSTorusNecklaceCRecursion(_NSMultipointCRecursionBase):
    r"""NS torus ``N``-point block in the annulus necklace channel.

    Vertex ``i`` is ordered as ``rho(h_i,d_i,h_(i+1))``, with indices taken
    cyclically.  The implementation starts at ``N=2`` because a one-point
    torus block is a self-loop rather than an ordinary edge with two distinct
    vertices; use the existing torus one-point c-recursion for that case.

    The regular seed is the global necklace block convolved with the
    non-global NS vacuum factor in ``Q = product_i q_i``.
    """

    def __init__(
        self,
        *,
        central_charge,
        external_weights: Sequence[object],
        internal_weights: Sequence[object],
        vertex_sectors: Sequence[int] | None = None,
        spin_lift: int = 1,
        working_precision: int = 60,
        pole_tolerance: float = 1.0e-30,
    ) -> None:
        if len(external_weights) < 2:
            raise ValueError(
                "the ordinary-edge necklace driver requires at least two points"
            )
        if len(internal_weights) != len(external_weights):
            raise ValueError(
                "a torus necklace requires one internal edge per external point"
            )
        if spin_lift not in (-1, 1):
            raise ValueError("spin_lift must be +1 or -1")
        self.edge_count = len(external_weights)
        self.vertex_count = self.edge_count
        self.spin_lift = int(spin_lift)
        if vertex_sectors is None:
            vertex_sectors = (0,) * self.vertex_count
        with mpmath.workdps(working_precision):
            self.external_weights = tuple(
                mpmath.mpc(weight) for weight in external_weights
            )
        super().__init__(
            central_charge=central_charge,
            internal_weights=internal_weights,
            vertex_sectors=vertex_sectors,
            working_precision=working_precision,
            pole_tolerance=pole_tolerance,
            graph_name="torus-necklace",
        )

    def _global_coefficient(
        self,
        twice_levels: tuple[int, ...],
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        states = tuple(_state_from_twice_level(level) for level in twice_levels)
        result = mpmath.mpc(1)
        for vertex in range(self.vertex_count):
            ket_edge = (vertex + 1) % self.edge_count
            bra_n, bra_epsilon = states[vertex]
            ket_n, ket_epsilon = states[ket_edge]
            result *= osp_sector_vertex(
                sector=vertex_sectors[vertex],
                n1=bra_n,
                n2=0,
                n3=ket_n,
                epsilon1=bra_epsilon,
                epsilon2=0,
                epsilon3=ket_epsilon,
                d1=internal_weights[vertex],
                d2=self.external_weights[vertex],
                d3=internal_weights[ket_edge],
            )

        denominator = mpmath.mpc(1)
        for weight, (occupation, epsilon) in zip(internal_weights, states):
            denominator *= osp_norm(weight, occupation, epsilon)
        return result / denominator

    def _regular_coefficient(
        self,
        twice_levels: tuple[int, ...],
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        cutoff = min(twice_levels)
        vacuum = ns_non_global_vacuum_coefficients(cutoff, self.spin_lift)
        result = mpmath.mpc(0)
        for diagonal_shift, vacuum_coefficient in enumerate(vacuum):
            if vacuum_coefficient == 0:
                continue
            child_levels = tuple(
                level - diagonal_shift for level in twice_levels
            )
            result += vacuum_coefficient * self._global_coefficient(
                child_levels, internal_weights, vertex_sectors
            )
        return result

    def _edge_kernel_arguments(
        self,
        edge: int,
        internal_weights: tuple[object, ...],
        vertex_sectors: tuple[int, ...],
    ):
        first_vertex = edge
        second_vertex = (edge - 1) % self.vertex_count
        first_other_edge = (edge + 1) % self.edge_count
        second_other_edge = (edge - 1) % self.edge_count
        return (
            (
                internal_weights[first_other_edge],
                self.external_weights[first_vertex],
            ),
            (
                internal_weights[second_other_edge],
                self.external_weights[second_vertex],
            ),
            vertex_sectors[first_vertex],
            vertex_sectors[second_vertex],
            first_vertex,
            second_vertex,
        )


__all__ = [
    "NSSphereLinearCRecursion",
    "NSTorusNecklaceCRecursion",
    "ns_non_global_vacuum_coefficients",
]
