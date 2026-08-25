#!/usr/bin/env python3
"""Collision-aware CCY recursion for closed trivalent plumbing graphs.

This module implements the central-charge recursion of Cho--Collier--Yin for
the graph data in :mod:`virasoro_plumbing_graph`.  It returns the chiral
Virasoro block with descendant plumbing powers only:

    F(c, h_e, q_e)
      = U(c, h_e, q_e)
        + sum_e sum_{r>=2,s>=1}
          q_e^(rs) R_{e,rs}(h) /
          (c-c_{rs}(h_e))
          F(c_{rs}(h_e), h_e+rs, h_{f != e}, q_f).

The regular term is the graph-global ``SL(2)`` block times the large-``c``
Schottky vacuum seed.  This is the graph form of the genus-two CCY
construction and is used for all five marked genus-three channels.
:func:`pants_large_c_regular_term` independently extracts the same limit from
direct descendant contractions and is retained as a diagnostic.  The CCY
simple-pole recursion is evaluated at generic internal weights.  Exact
coincident poles are resolved by a symmetric generic-weight limit of the
complete block while the requested physical central charge is held fixed.

The graph combinatorics determine the global block and the null-vector fusion
polynomials.  They do *not* determine a Schottky marking; the vacuum seed is
therefore a separate callback.  Known callbacks are supplied for the
genus-two theta and glasses charts and all five marked genus-three channels.

Primary propagation factors ``prod_e q_e**h_e``, three-point structure
constants, and internal-momentum integrations are not included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Mapping, Sequence

try:
    from ccy_genus2_block import (
        ConfluentPoleError,
        PartialFractionInC,
        _as_complex,
        _is_finite_complex,
        _validate_order,
        b_from_c_rs_h,
        c_rs_from_h,
        collision_regulated_partial_fraction_value,
        fusion_polynomial_for_weights,
        genus2_vacuum_seed_schottky,
        minus_dc_dh_times_a_rs,
        rho_lminus1_triple,
        sl2_descendant_norm,
    )
    from ccy_genus2_glasses_block import glasses_vacuum_seed_schottky
    from genus2_vacuum_blocks import schottky_vacuum_block
    from genus3_plumbing_channels import (
        Genus3PlumbingChannel,
        generators_for_genus3_channel,
        genus3_channel_by_name,
        genus3_channel_for_graph,
    )
    from genus3_plumbing_tetrahedron import generators_for_tetrahedron
    from genus3_global_resummation import (
        clear_genus3_global_resummation_caches,
        genus3_channel_global_sl2_block_resummed,
    )
    from virasoro_plumbing_graph import (
        INFINITY_SLOT,
        ONE_SLOT,
        ZERO_SLOT,
        TrivalentPlumbingGraph,
        direct_plumbing_graph_block,
        genus2_glasses_graph,
        genus2_theta_graph,
        genus3_tetrahedral_graph,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import (
        ConfluentPoleError,
        PartialFractionInC,
        _as_complex,
        _is_finite_complex,
        _validate_order,
        b_from_c_rs_h,
        c_rs_from_h,
        collision_regulated_partial_fraction_value,
        fusion_polynomial_for_weights,
        genus2_vacuum_seed_schottky,
        minus_dc_dh_times_a_rs,
        rho_lminus1_triple,
        sl2_descendant_norm,
    )
    from plumbing.ccy_genus2_glasses_block import glasses_vacuum_seed_schottky
    from plumbing.genus2_vacuum_blocks import schottky_vacuum_block
    from plumbing.genus3_plumbing_channels import (
        Genus3PlumbingChannel,
        generators_for_genus3_channel,
        genus3_channel_by_name,
        genus3_channel_for_graph,
    )
    from plumbing.genus3_plumbing_tetrahedron import generators_for_tetrahedron
    from plumbing.genus3_global_resummation import (
        clear_genus3_global_resummation_caches,
        genus3_channel_global_sl2_block_resummed,
    )
    from plumbing.virasoro_plumbing_graph import (
        INFINITY_SLOT,
        ONE_SLOT,
        ZERO_SLOT,
        TrivalentPlumbingGraph,
        direct_plumbing_graph_block,
        genus2_glasses_graph,
        genus2_theta_graph,
        genus3_tetrahedral_graph,
    )


VacuumSeedEvaluator = Callable[[tuple[complex, ...], int, int], complex]
RegularTermEvaluator = Callable[
    [TrivalentPlumbingGraph, tuple[complex, ...], tuple[complex, ...], int],
    complex,
]


@dataclass(frozen=True)
class CCYPlumbingGraphBlockResult:
    """One total-degree truncated graph-level CCY block."""

    value: complex
    graph_name: str
    genus: int
    central_charge: complex
    edge_weights: tuple[complex, ...]
    q_values: tuple[complex, ...]
    order: int
    include_vacuum_seed: bool
    regular_term_scheme: str
    vacuum_word_len: int
    vacuum_oscillator_level_max: int
    partial_fraction_pole_count: int
    partial_fraction_coefficient_count: int
    partial_fraction_max_pole_order: int
    collision_regulated: bool = False
    collision_regulator_error: float = 0.0
    collision_regulator_scale: float = 0.0


def _graph_edge_values(
    graph: TrivalentPlumbingGraph,
    values: Sequence[complex] | Mapping[str, complex],
    *,
    label: str,
) -> tuple[complex, ...]:
    if isinstance(values, Mapping):
        edge_names = {edge.name for edge in graph.edges}
        missing = [edge.name for edge in graph.edges if edge.name not in values]
        extra = sorted(set(values) - edge_names)
        if missing or extra:
            raise ValueError(f"{label} keys do not match graph edges: missing={missing}, extra={extra}")
        return tuple(complex(values[edge.name]) for edge in graph.edges)
    if len(values) != len(graph.edges):
        raise ValueError(f"{label} must contain {len(graph.edges)} entries")
    return tuple(complex(value) for value in values)


def _graph_edge_real_values(
    graph: TrivalentPlumbingGraph,
    values: Sequence[float] | Mapping[str, float],
    *,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, Mapping):
        edge_names = {edge.name for edge in graph.edges}
        missing = [edge.name for edge in graph.edges if edge.name not in values]
        extra = sorted(set(values) - edge_names)
        if missing or extra:
            raise ValueError(
                f"{label} keys do not match graph edges: "
                f"missing={missing}, extra={extra}"
            )
        ordered = tuple(float(values[edge.name]) for edge in graph.edges)
    else:
        if len(values) != len(graph.edges):
            raise ValueError(
                f"{label} must contain {len(graph.edges)} entries"
            )
        ordered = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError(f"{label} must contain only finite values")
    return ordered


@lru_cache(maxsize=None)
def _fixed_total_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if length == 0:
        return ((),) if total == 0 else ()
    return tuple(
        (first,) + tail
        for first in range(total + 1)
        for tail in _fixed_total_compositions(total - first, length - 1)
    )


def _vertex_edge_indices(graph: TrivalentPlumbingGraph) -> tuple[tuple[int, int, int], ...]:
    indices: list[list[int | None]] = [[None, None, None] for _ in range(graph.vertex_count)]
    for edge_index, edge in enumerate(graph.edges):
        for endpoint in edge.endpoints:
            indices[endpoint.vertex][endpoint.slot] = edge_index
    if any(item is None for vertex in indices for item in vertex):
        raise ValueError("graph is missing a vertex slot")
    return tuple(tuple(int(item) for item in vertex) for vertex in indices)


def global_sl2_plumbing_graph_block(
    graph: TrivalentPlumbingGraph,
    *,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    order: int,
) -> complex:
    r"""Return the total-degree truncated graph-global ``SL(2)`` block.

    At edge level ``n_e`` only ``L_-1**n_e`` propagates.  The inverse global
    Gram element is

    ``1 / (n_e! (2 h_e)_{n_e})``,

    and every trivalent vertex contributes the closed three-point matrix
    element ``rho_lminus1_triple`` in its ordered infinity/one/zero slots.
    """

    order = _validate_order(order)
    weights = _graph_edge_values(graph, edge_weights, label="edge_weights")
    q_tuple = _graph_edge_values(graph, q_values, label="q_values")
    vertex_edges = _vertex_edge_indices(graph)
    total = 0.0 + 0.0j
    for total_level in range(order + 1):
        for levels in _fixed_total_compositions(total_level, len(graph.edges)):
            coefficient = 1.0 + 0.0j
            for edge_index, level in enumerate(levels):
                coefficient *= (
                    q_tuple[edge_index] ** level
                    / sl2_descendant_norm(weights[edge_index], level)
                )
            for edge_at_vertex in vertex_edges:
                infinity_edge, one_edge, zero_edge = edge_at_vertex
                coefficient *= rho_lminus1_triple(
                    levels[infinity_edge],
                    levels[one_edge],
                    levels[zero_edge],
                    weights[infinity_edge],
                    weights[one_edge],
                    weights[zero_edge],
                )
            total += coefficient
    return complex(total)


@lru_cache(maxsize=512)
def _pants_large_c_regular_term_cached(
    graph: TrivalentPlumbingGraph,
    edge_weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    order: int,
) -> complex:
    """Cached implementation of the graph-local large-``c`` limit."""

    if order <= 1:
        # No level-two Virasoro state occurs, so the direct contraction is
        # already the graph-global SL(2) block and has no c dependence.
        return global_sl2_plumbing_graph_block(
            graph,
            edge_weights=edge_weights,
            q_values=q_values,
            order=order,
        )

    # The direct finite-level contraction is analytic in x=1/c near x=0.
    # Quadratic Richardson extrapolation at c=(C,2C,4C) removes the first two
    # inverse-c corrections.  C=2e4 keeps the Verma Gram inversions well
    # conditioned while making the residual O(C^-3) term sub-double-precision
    # at the plumbing orders for which the direct contraction is practical.
    central_charge_scale = 2.0e4
    values = tuple(
        direct_plumbing_graph_block(
            graph,
            central_charge=central_charge_scale * multiplier,
            edge_weights=edge_weights,
            q_values=q_values,
            max_total_level=order,
        ).value
        for multiplier in (1.0, 2.0, 4.0)
    )
    return complex(values[0] / 3.0 - 2.0 * values[1] + 8.0 * values[2] / 3.0)


def pants_large_c_regular_term(
    graph: TrivalentPlumbingGraph,
    edge_weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    order: int,
) -> complex:
    r"""Return the CCY regular term in the graph's pants-coordinate frame.

    The regular part of central-charge recursion is

    ``U_G(h,q) = lim_{c -> infinity} F_G(c,h,q)``.

    It is evaluated from the same direct descendant contractions used to
    validate the recursion, followed by a three-point Richardson
    extrapolation in ``1/c``.  It provides an independent coefficient-level
    check of the global-block-times-vacuum formula, including for graphs with
    tadpoles.
    """

    validated_order = _validate_order(order)
    weights = _graph_edge_values(graph, edge_weights, label="edge_weights")
    q_tuple = _graph_edge_values(graph, q_values, label="q_values")
    return _pants_large_c_regular_term_cached(
        graph,
        weights,
        q_tuple,
        validated_order,
    )


@lru_cache(maxsize=128)
def genus3_tetrahedral_vacuum_seed_schottky(
    q_values: tuple[complex, ...],
    max_word_len: int = 3,
    oscillator_level_max: int = 12,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> complex:
    """Return the rank-three Schottky vacuum seed in the K4 chart."""

    if len(q_values) != 6:
        raise ValueError("the tetrahedral chart requires six plumbing parameters")
    result = schottky_vacuum_block(
        generators_for_tetrahedron(q_values),
        max_word_length=_validate_order(max_word_len),
        max_mode=_validate_order(oscillator_level_max),
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
        channel="genus3_tetrahedral",
        q_values=tuple(complex(value) for value in q_values),
    )
    if word_tail_tolerance is not None and (
        result.primitive_word_tail_estimate is None
        or result.primitive_word_tail_estimate > float(word_tail_tolerance)
    ):
        raise RuntimeError(
            "tetrahedral CCY vacuum seed exhausted its word-length safety cap "
            "before reaching the requested primitive-word tail"
        )
    return result.value


@lru_cache(maxsize=256)
def genus3_channel_vacuum_seed_schottky(
    channel_name: str,
    q_values: tuple[complex, ...],
    max_word_len: int = 3,
    oscillator_level_max: int = 12,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> complex:
    """Return the rank-three Schottky vacuum seed in any fixed channel."""

    channel = genus3_channel_by_name(channel_name)
    if len(q_values) != len(channel.graph.edges):
        raise ValueError(
            f"channel {channel.name!r} requires {len(channel.graph.edges)} plumbing parameters"
        )
    result = schottky_vacuum_block(
        generators_for_genus3_channel(channel, q_values),
        max_word_length=_validate_order(max_word_len),
        max_mode=_validate_order(oscillator_level_max),
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
        channel=channel.graph.name,
        q_values=tuple(complex(value) for value in q_values),
    )
    if word_tail_tolerance is not None and (
        result.primitive_word_tail_estimate is None
        or result.primitive_word_tail_estimate > float(word_tail_tolerance)
    ):
        raise RuntimeError(
            f"{channel.name} CCY vacuum seed exhausted its word-length "
            "safety cap before reaching the requested primitive-word tail"
        )
    return result.value


def known_schottky_vacuum_seed(
    graph: TrivalentPlumbingGraph,
    q_values: tuple[complex, ...],
    max_word_len: int,
    oscillator_level_max: int,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> complex:
    """Evaluate the registered Schottky seed for a known plumbing chart."""

    if graph == genus2_theta_graph():
        # The graph edges follow rho's (infinity, one, zero) slot order,
        # whereas the Schottky theta chart is (zero, one, infinity).
        return genus2_vacuum_seed_schottky(
            q_values[2],
            q_values[1],
            q_values[0],
            max_word_len=max_word_len,
            oscillator_level_max=oscillator_level_max,
            word_tail_tolerance=word_tail_tolerance,
            minimum_word_length=minimum_word_length,
        )
    if graph == genus2_glasses_graph():
        return glasses_vacuum_seed_schottky(
            *q_values,
            max_word_len=max_word_len,
            oscillator_level_max=oscillator_level_max,
            word_tail_tolerance=word_tail_tolerance,
            minimum_word_length=minimum_word_length,
        )
    if graph == genus3_tetrahedral_graph():
        return genus3_tetrahedral_vacuum_seed_schottky(
            q_values,
            max_word_len=max_word_len,
            oscillator_level_max=oscillator_level_max,
            word_tail_tolerance=word_tail_tolerance,
            minimum_word_length=minimum_word_length,
        )
    genus3_channel = genus3_channel_for_graph(graph)
    if genus3_channel is not None:
        return genus3_channel_vacuum_seed_schottky(
            genus3_channel.name,
            q_values,
            max_word_len=max_word_len,
            oscillator_level_max=oscillator_level_max,
            word_tail_tolerance=word_tail_tolerance,
            minimum_word_length=minimum_word_length,
        )
    raise ValueError(
        f"no Schottky vacuum seed is registered for graph {graph.name!r}; "
        "pass vacuum_seed_evaluator explicitly"
    )


def _fusion_pair_slots(null_slot: int) -> tuple[int, int]:
    """Return the ordered two spectator slots in the CCY rho convention."""

    if null_slot == INFINITY_SLOT:
        return ZERO_SLOT, ONE_SLOT
    if null_slot == ONE_SLOT:
        return ZERO_SLOT, INFINITY_SLOT
    if null_slot == ZERO_SLOT:
        return INFINITY_SLOT, ONE_SLOT
    raise ValueError(f"invalid trivalent slot {null_slot}")


def _edge_fusion_weight_pairs(
    graph: TrivalentPlumbingGraph,
    edge_index: int,
    weights: tuple[complex, ...],
    level: int,
) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """Return the two endpoint fusion pairs for one null edge.

    For a self-loop, sequential null-vector factorization shifts the second
    occurrence of the same edge by ``rs`` in one of the two fusion
    polynomials.  This reproduces the CCY torus one-point residue

    ``P[h_spectator, h+rs] P[h_spectator, h]``.
    """

    edge = graph.edges[edge_index]
    vertex_edges = _vertex_edge_indices(graph)
    pairs: list[tuple[complex, complex]] = []
    for endpoint_number, endpoint in enumerate(edge.endpoints):
        pair_weights: list[complex] = []
        for spectator_slot in _fusion_pair_slots(endpoint.slot):
            spectator_edge = vertex_edges[endpoint.vertex][spectator_slot]
            spectator_weight = weights[spectator_edge]
            if spectator_edge == edge_index and endpoint_number == 0:
                spectator_weight += level
            pair_weights.append(spectator_weight)
        pairs.append((pair_weights[0], pair_weights[1]))
    return pairs[0], pairs[1]


def graph_edge_residue_prefactor(
    graph: TrivalentPlumbingGraph,
    *,
    edge_index: int,
    r: int,
    s: int,
    edge_weights: Sequence[complex] | Mapping[str, complex],
) -> complex:
    r"""Return ``-dc/dh A_rs P_endpoint1 P_endpoint2`` for one graph edge."""

    weights = _graph_edge_values(graph, edge_weights, label="edge_weights")
    if not 0 <= int(edge_index) < len(graph.edges):
        raise IndexError("edge_index is outside the plumbing graph")
    edge_index = int(edge_index)
    level = int(r) * int(s)
    h_edge = weights[edge_index]

    def direct(current_h: complex) -> complex:
        current_weights = list(weights)
        current_weights[edge_index] = complex(current_h)
        first_pair, second_pair = _edge_fusion_weight_pairs(
            graph,
            edge_index,
            tuple(current_weights),
            level,
        )
        b_pole = b_from_c_rs_h(r, s, current_h)
        return (
            minus_dc_dh_times_a_rs(r, s, current_h)
            * fusion_polynomial_for_weights(r, s, b_pole, *first_pair)
            * fusion_polynomial_for_weights(r, s, b_pole, *second_pair)
        )

    try:
        value = direct(h_edge)
        if _is_finite_complex(value):
            return complex(value)
    except ZeroDivisionError:
        pass

    scale = max(1.0, abs(h_edge))
    samples: list[tuple[float, complex]] = []
    for relative_step in (1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8):
        step = relative_step * scale
        try:
            value = direct(h_edge + step)
        except ZeroDivisionError:
            continue
        if _is_finite_complex(value):
            samples.append((step, complex(value)))
    if not samples:
        raise ZeroDivisionError(
            f"could not resolve graph CCY residue for edge={edge_index}, r={r}, s={s}, h={h_edge!r}"
        )
    if len(samples) == 1:
        return samples[-1][1]
    step_a, value_a = samples[-2]
    step_b, value_b = samples[-1]
    return (step_a * value_b - step_b * value_a) / (step_a - step_b)


def ccy_plumbing_graph_block_partial_fraction(
    graph: TrivalentPlumbingGraph,
    *,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_seed_evaluator: VacuumSeedEvaluator | None = None,
    regular_term_evaluator: RegularTermEvaluator | None = None,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    pole_tolerance: float = 1.0e-12,
) -> PartialFractionInC:
    """Return the graph-level CCY block as a partial fraction in ``c``."""

    order = _validate_order(order)
    weights = _graph_edge_values(graph, edge_weights, label="edge_weights")
    q_tuple = _graph_edge_values(graph, q_values, label="q_values")
    if regular_term_evaluator is not None and not include_vacuum_seed:
        raise ValueError(
            "regular_term_evaluator supplies the full large-c term and requires "
            "include_vacuum_seed=True"
        )
    if regular_term_evaluator is None and include_vacuum_seed:
        evaluator = vacuum_seed_evaluator or (
            lambda q, words, modes: known_schottky_vacuum_seed(
                graph,
                q,
                words,
                modes,
                word_tail_tolerance=vacuum_word_tail_tolerance,
                minimum_word_length=vacuum_minimum_word_length,
            )
        )
        vacuum_seed = complex(
            evaluator(q_tuple, int(vacuum_word_len), int(vacuum_oscillator_level_max))
        )
    else:
        vacuum_seed = 1.0 + 0.0j

    @lru_cache(maxsize=None)
    def recurse(current_weights: tuple[complex, ...], remaining: int) -> PartialFractionInC:
        if regular_term_evaluator is None:
            seed = vacuum_seed * global_sl2_plumbing_graph_block(
                graph,
                edge_weights=current_weights,
                q_values=q_tuple,
                order=remaining,
            )
        else:
            seed = complex(
                regular_term_evaluator(
                    graph,
                    current_weights,
                    q_tuple,
                    remaining,
                )
            )
        total = PartialFractionInC(constant=seed)
        for edge_index, (h_edge, q_edge) in enumerate(zip(current_weights, q_tuple)):
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    pole_c = c_rs_from_h(r, s, h_edge)
                    residue = (q_edge**level) * graph_edge_residue_prefactor(
                        graph,
                        edge_index=edge_index,
                        r=r,
                        s=s,
                        edge_weights=current_weights,
                    )
                    shifted = list(current_weights)
                    shifted[edge_index] = h_edge + level
                    subblock = recurse(tuple(shifted), remaining - level)
                    total.add_residue_times_laurent_at(
                        pole=pole_c,
                        residue=residue,
                        subblock=subblock,
                        pole_tolerance=pole_tolerance,
                    )
        return total

    return recurse(weights, order)


def ccy_plumbing_graph_block(
    graph: TrivalentPlumbingGraph,
    *,
    central_charge: complex,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_seed_evaluator: VacuumSeedEvaluator | None = None,
    regular_term_evaluator: RegularTermEvaluator | None = None,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    pole_tolerance: float = 1.0e-12,
    collision_regulator_scale: float = 1.0e-3,
    collision_regulator_direction: (
        Sequence[float] | Mapping[str, float] | None
    ) = None,
) -> CCYPlumbingGraphBlockResult:
    """Evaluate the CCY block, regulating exact coincident-weight poles."""

    c_value = _as_complex(central_charge)
    weights = _graph_edge_values(graph, edge_weights, label="edge_weights")
    q_tuple = _graph_edge_values(graph, q_values, label="q_values")
    regulator_direction = (
        None
        if collision_regulator_direction is None
        else _graph_edge_real_values(
            graph,
            collision_regulator_direction,
            label="collision_regulator_direction",
        )
    )
    collision_regulator_scale = float(collision_regulator_scale)
    if (
        not math.isfinite(collision_regulator_scale)
        or collision_regulator_scale <= 0.0
    ):
        raise ValueError("collision regulator scale must be finite and positive")

    def build_partial(current_weights: tuple[complex, ...]) -> PartialFractionInC:
        return ccy_plumbing_graph_block_partial_fraction(
            graph,
            edge_weights=current_weights,
            q_values=q_tuple,
            order=order,
            include_vacuum_seed=include_vacuum_seed,
            vacuum_seed_evaluator=vacuum_seed_evaluator,
            regular_term_evaluator=regular_term_evaluator,
            vacuum_word_len=vacuum_word_len,
            vacuum_oscillator_level_max=vacuum_oscillator_level_max,
            vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
            vacuum_minimum_word_length=vacuum_minimum_word_length,
            pole_tolerance=pole_tolerance,
        )

    regulated = False
    regulator_error = 0.0
    regulator_scale = 0.0
    try:
        partial = build_partial(weights)
        value = partial.value(c_value, pole_tolerance=pole_tolerance)
    except ConfluentPoleError:
        regulated_value = collision_regulated_partial_fraction_value(
            build_partial_fraction=build_partial,
            weights=weights,
            central_charge=c_value,
            pole_tolerance=pole_tolerance,
            relative_scale=collision_regulator_scale,
            direction=regulator_direction,
        )
        partial = regulated_value.representative_partial_fraction
        value = regulated_value.value
        regulated = True
        regulator_error = regulated_value.error_estimate
        regulator_scale = regulated_value.relative_scale

    return CCYPlumbingGraphBlockResult(
        value=value,
        graph_name=graph.name,
        genus=graph.genus,
        central_charge=c_value,
        edge_weights=weights,
        q_values=q_tuple,
        order=_validate_order(order),
        include_vacuum_seed=bool(include_vacuum_seed),
        regular_term_scheme=(
            "custom-large-c"
            if regular_term_evaluator is not None
            else ("schottky-vacuum-times-global" if include_vacuum_seed else "global")
        ),
        vacuum_word_len=int(vacuum_word_len),
        vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
        partial_fraction_pole_count=partial.pole_count,
        partial_fraction_coefficient_count=partial.coefficient_count,
        partial_fraction_max_pole_order=partial.max_pole_order,
        collision_regulated=regulated,
        collision_regulator_error=regulator_error,
        collision_regulator_scale=regulator_scale,
    )


def ccy_genus3_tetrahedral_block(
    *,
    central_charge: complex,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    order: int,
    include_vacuum_seed: bool = True,
    regular_term_evaluator: RegularTermEvaluator | None = None,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    pole_tolerance: float = 1.0e-12,
    collision_regulator_scale: float = 1.0e-3,
    collision_regulator_direction: (
        Sequence[float] | Mapping[str, float] | None
    ) = None,
) -> CCYPlumbingGraphBlockResult:
    """Evaluate the genus-three K4 block in tetrahedral edge order.

    Sequence inputs use edge order ``(01, 02, 03, 12, 13, 23)``.  Mapping
    inputs may instead use the explicit keys ``q01``, ..., ``q23``.
    """

    return ccy_plumbing_graph_block(
        genus3_tetrahedral_graph(),
        central_charge=central_charge,
        edge_weights=edge_weights,
        q_values=q_values,
        order=order,
        include_vacuum_seed=include_vacuum_seed,
        regular_term_evaluator=regular_term_evaluator,
        vacuum_word_len=vacuum_word_len,
        vacuum_oscillator_level_max=vacuum_oscillator_level_max,
        vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
        vacuum_minimum_word_length=vacuum_minimum_word_length,
        pole_tolerance=pole_tolerance,
        collision_regulator_scale=collision_regulator_scale,
        collision_regulator_direction=collision_regulator_direction,
    )


def ccy_genus3_channel_block(
    *,
    channel: str | Genus3PlumbingChannel,
    central_charge: complex,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    order: int,
    regular_term_scheme: str = "schottky",
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    global_block_tolerance: float = 1.0e-9,
    global_block_minimum_cap: int = 8,
    global_block_maximum_cap: int = 24,
    global_block_cap_step: int = 2,
    manage_global_block_cache: bool = True,
    pole_tolerance: float = 1.0e-12,
    collision_regulator_scale: float = 1.0e-3,
    collision_regulator_direction: (
        Sequence[float] | Mapping[str, float] | None
    ) = None,
) -> CCYPlumbingGraphBlockResult:
    """Evaluate the CCY block in any of the five marked genus-three channels.

    ``regular_term_scheme="schottky"`` uses the total-degree truncated global
    block.  ``"schottky-resummed"`` instead selects the best native-frame
    all-level global-block contraction for any of the five marked channels.
    ``"pants"`` retains the independent direct large-c descendant contraction
    for diagnostic comparisons.
    """

    resolved = genus3_channel_by_name(channel) if isinstance(channel, str) else channel
    scheme = str(regular_term_scheme).strip().lower()
    if scheme not in {"pants", "schottky", "schottky-resummed"}:
        raise ValueError(
            "regular_term_scheme must be 'pants', 'schottky', or "
            "'schottky-resummed'"
        )
    if scheme == "schottky-resummed" and not include_vacuum_seed:
        raise ValueError(
            "schottky-resummed supplies the full large-c regular term and "
            "requires include_vacuum_seed=True"
        )

    regular_evaluator: RegularTermEvaluator | None
    result_label: str | None = None
    if scheme == "pants" and include_vacuum_seed:
        regular_evaluator = pants_large_c_regular_term
    elif scheme == "schottky-resummed":
        q_tuple = _graph_edge_values(resolved.graph, q_values, label="q_values")
        vacuum_seed = known_schottky_vacuum_seed(
            resolved.graph,
            q_tuple,
            int(vacuum_word_len),
            int(vacuum_oscillator_level_max),
            word_tail_tolerance=vacuum_word_tail_tolerance,
            minimum_word_length=vacuum_minimum_word_length,
        )

        def regular_evaluator(
            current_graph: TrivalentPlumbingGraph,
            current_weights: tuple[complex, ...],
            current_q: tuple[complex, ...],
            remaining: int,
        ) -> complex:
            del remaining
            if current_graph != resolved.graph:
                raise AssertionError("resummed CCY callback changed plumbing graph")
            global_result = genus3_channel_global_sl2_block_resummed(
                resolved,
                edge_weights=current_weights,
                q_values=current_q,
                tolerance=global_block_tolerance,
                minimum_cap=global_block_minimum_cap,
                maximum_cap=global_block_maximum_cap,
                cap_step=global_block_cap_step,
            )
            return vacuum_seed * global_result.value

        result_label = "schottky-vacuum-times-channel-resummed-global"
    else:
        regular_evaluator = None

    if scheme == "schottky-resummed" and manage_global_block_cache:
        clear_genus3_global_resummation_caches()
    try:
        result = ccy_plumbing_graph_block(
            resolved.graph,
            central_charge=central_charge,
            edge_weights=edge_weights,
            q_values=q_values,
            order=order,
            include_vacuum_seed=include_vacuum_seed,
            regular_term_evaluator=regular_evaluator,
            vacuum_word_len=vacuum_word_len,
            vacuum_oscillator_level_max=vacuum_oscillator_level_max,
            vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
            vacuum_minimum_word_length=vacuum_minimum_word_length,
            pole_tolerance=pole_tolerance,
            collision_regulator_scale=collision_regulator_scale,
            collision_regulator_direction=collision_regulator_direction,
        )
    finally:
        if scheme == "schottky-resummed" and manage_global_block_cache:
            clear_genus3_global_resummation_caches()
    return (
        replace(result, regular_term_scheme=result_label)
        if result_label is not None
        else result
    )


__all__ = [
    "CCYPlumbingGraphBlockResult",
    "RegularTermEvaluator",
    "VacuumSeedEvaluator",
    "ccy_genus3_channel_block",
    "ccy_genus3_tetrahedral_block",
    "ccy_plumbing_graph_block",
    "ccy_plumbing_graph_block_partial_fraction",
    "genus3_channel_vacuum_seed_schottky",
    "genus3_tetrahedral_vacuum_seed_schottky",
    "genus3_channel_global_sl2_block_resummed",
    "global_sl2_plumbing_graph_block",
    "graph_edge_residue_prefactor",
    "known_schottky_vacuum_seed",
    "pants_large_c_regular_term",
]
