#!/usr/bin/env python3
"""Checks for the CCY plumbing-frame sewing convention."""

from __future__ import annotations

import cmath
import math

import mpmath as mp

try:
    from ccy_plumbing_conventions import (
        GENUS2_CFT_DESCENDANT_EDGE_ORDERS,
        GENUS2_FREE_BOSON_EDGE_ORDERS,
        GENUS2_PLUMBING_EDGE_ORDERS,
        ccy_primary_propagator,
        ccy_raw_sewing_propagator,
        genus2_plumbing_coordinate_metadata,
        genus2_channel_q_values,
        liouville_threshold_modulus_factor,
        liouville_threshold_weight,
        theta_geometry_to_ccy_order,
        validate_genus2_plumbing_coordinates,
    )
    from ccy_genus2_block import ccy_genus2_block
    from liouville_genus2_ccy import (
        liouville_genus2_ccy_density,
        liouville_genus2_ccy_partition,
        liouville_weight_from_momentum,
    )
    from liouville_genus2_glasses import (
        liouville_genus2_glasses_density,
        liouville_genus2_glasses_partition,
    )
    from liouville_torus import (
        UpsilonB,
        log_yin_structure_constant_momentum,
        yin_structure_constant_momentum,
    )
    from plumbing_algorithms import (
        glasses_eval_row,
        glasses_seam_matrix,
        local_coordinate_map,
        plumbing_transition,
    )
    from virasoro_plumbing_graph import SLOT_NAMES, genus2_glasses_graph, genus2_theta_graph
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_plumbing_conventions import (
        GENUS2_CFT_DESCENDANT_EDGE_ORDERS,
        GENUS2_FREE_BOSON_EDGE_ORDERS,
        GENUS2_PLUMBING_EDGE_ORDERS,
        ccy_primary_propagator,
        ccy_raw_sewing_propagator,
        genus2_plumbing_coordinate_metadata,
        genus2_channel_q_values,
        liouville_threshold_modulus_factor,
        liouville_threshold_weight,
        theta_geometry_to_ccy_order,
        validate_genus2_plumbing_coordinates,
    )
    from plumbing.ccy_genus2_block import ccy_genus2_block
    from plumbing.liouville_genus2_ccy import (
        liouville_genus2_ccy_density,
        liouville_genus2_ccy_partition,
        liouville_weight_from_momentum,
    )
    from plumbing.liouville_genus2_glasses import (
        liouville_genus2_glasses_density,
        liouville_genus2_glasses_partition,
    )
    from plumbing.liouville_torus import (
        UpsilonB,
        log_yin_structure_constant_momentum,
        yin_structure_constant_momentum,
    )
    from plumbing.plumbing_algorithms import (
        glasses_eval_row,
        glasses_seam_matrix,
        local_coordinate_map,
        plumbing_transition,
    )
    from plumbing.virasoro_plumbing_graph import (
        SLOT_NAMES,
        genus2_glasses_graph,
        genus2_theta_graph,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_named_genus2_edge_orders() -> None:
    theta_named = {
        "q_infinity": 0.013 + 0.003j,
        "q_one": 0.021 - 0.002j,
        "q_zero": 0.034 + 0.001j,
    }
    glasses_named = {
        "q_bridge": 0.017 - 0.004j,
        "q_right": 0.026 + 0.002j,
        "q_left": 0.039 - 0.001j,
    }
    theta = genus2_channel_q_values("theta", theta_named)
    glasses = genus2_channel_q_values("glasses", glasses_named)
    require(
        theta == tuple(
            theta_named[name] for name in GENUS2_PLUMBING_EDGE_ORDERS["theta"]
        ),
        "theta named q values were not canonicalized geometrically",
    )
    require(
        glasses == tuple(
            glasses_named[name] for name in GENUS2_PLUMBING_EDGE_ORDERS["glasses"]
        ),
        "glasses named q values were not canonicalized geometrically",
    )
    require(
        theta_geometry_to_ccy_order(theta)
        == (theta_named["q_infinity"], theta_named["q_one"], theta_named["q_zero"]),
        "theta geometric-to-CCY conversion changed its named slot order",
    )
    try:
        genus2_channel_q_values(
            "theta",
            {"q_zero": 0.01, "q_one": 0.02, "q_wrong": 0.03},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("theta ordering accepted missing/extra named edges")

    theta_logs_named = {
        name: cmath.log(value) for name, value in theta_named.items()
    }
    coordinates = validate_genus2_plumbing_coordinates(
        "theta",
        theta_named,
        log_q_values=theta_logs_named,
    )
    metadata = genus2_plumbing_coordinate_metadata(coordinates)
    require(
        metadata["plumbing_edge_order"] == "q_zero,q_one,q_infinity"
        and metadata["q1_edge_name"] == "q_zero"
        and metadata["q3_edge_name"] == "q_infinity",
        "theta serialization did not retain explicit geometric edge names",
    )
    swapped_logs = dict(theta_logs_named)
    swapped_logs["q_zero"], swapped_logs["q_infinity"] = (
        swapped_logs["q_infinity"],
        swapped_logs["q_zero"],
    )
    try:
        validate_genus2_plumbing_coordinates(
            "theta",
            theta_named,
            log_q_values=swapped_logs,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("theta coordinate contract accepted swapped log(q) edges")

    deep_log = -3606.0 + 0.37j
    deep_surrogate = cmath.exp(-690.0 + 0.37j)
    deep = validate_genus2_plumbing_coordinates(
        "glasses",
        (0.02 + 0.0j, deep_surrogate, 0.01 + 0.0j),
        log_q_values=(cmath.log(0.02), deep_log, cmath.log(0.01)),
    )
    require(
        deep.named_log_q_values["q_right"] == deep_log,
        "deep-cusp logarithm was not retained on the named glasses edge",
    )


def check_theta_endpoint_pairing_matches_period_map() -> None:
    """Match the two-sphere period seams to the CCY theta-graph endpoints."""

    q_by_puncture = {
        "zero": 0.031 + 0.007j,
        "one": 0.019 - 0.004j,
        "infinity": 0.011 + 0.002j,
    }
    global_points = {
        "zero": 0.27 + 0.19j,
        "one": 1.27 + 0.19j,
        "infinity": 2.3 - 0.4j,
    }
    period_puncture_names = {
        "zero": "zero",
        "one": "one",
        "infinity": "infty",
    }
    for name, period_name in period_puncture_names.items():
        q_value = q_by_puncture[name]
        source_global = global_points[name]
        target_global = plumbing_transition(
            period_name,
            period_name,
            q_value,
        )(source_global)
        source_local = local_coordinate_map(period_name)(source_global)
        target_local = local_coordinate_map(period_name)(target_global)
        require(
            abs(source_local * target_local / q_value - 1.0) < 2.0e-14,
            f"period map does not sew {name} on sphere 0 to {name} on sphere 1",
        )

    graph = genus2_theta_graph()
    expected_ccy_slot = {
        "q1": "infinity",
        "q2": "one",
        "q3": "zero",
    }
    for edge in graph.edges:
        endpoint_vertices = tuple(endpoint.vertex for endpoint in edge.endpoints)
        endpoint_slots = tuple(SLOT_NAMES[endpoint.slot] for endpoint in edge.endpoints)
        expected_slot = expected_ccy_slot[edge.name]
        require(
            endpoint_vertices == (0, 1)
            and endpoint_slots == (expected_slot, expected_slot),
            f"CCY theta edge {edge.name} is not {expected_slot}-{expected_slot}",
        )

    q_geometry = tuple(
        q_by_puncture[name] for name in ("zero", "one", "infinity")
    )
    require(
        theta_geometry_to_ccy_order(q_geometry)
        == (
            q_by_puncture["infinity"],
            q_by_puncture["one"],
            q_by_puncture["zero"],
        ),
        "geometric equal-puncture seams were not assigned to their CCY slots",
    )
    print("\ntheta two-sphere endpoint pairing")
    print("  period geometry: 0-0, 1-1, infinity-infinity")
    print("  CCY slots:       q1=infinity, q2=one, q3=zero")


def check_glasses_endpoint_pairing_matches_period_map() -> None:
    """Match the glasses period seams to its two handles and bridge graph."""

    q_named = {
        "q_left": 0.031 + 0.007j,
        "q_right": 0.019 - 0.004j,
        "q_bridge": 0.011 + 0.002j,
    }
    seam_data = (
        ("left handle", "zero", "infty", q_named["q_left"], 0.27 + 0.19j),
        ("right handle", "zero", "infty", q_named["q_right"], 0.23 - 0.17j),
        ("bridge", "one", "one", q_named["q_bridge"], 1.24 + 0.16j),
    )
    for name, source, target, q_value, source_global in seam_data:
        target_global = plumbing_transition(source, target, q_value)(source_global)
        source_local = local_coordinate_map(source)(source_global)
        target_local = local_coordinate_map(target)(target_global)
        require(
            abs(source_local * target_local / q_value - 1.0) < 2.0e-14,
            f"period map does not realize the glasses {name} as u v=q",
        )

    # Audit the actual collocation matrix, rather than only the generic
    # transition helper: each row must pull a one-form across the same map.
    basis_order = 3
    samples_per_seam = 8
    radii = (0.21, 0.19, 0.17)
    matrix, index = glasses_seam_matrix(
        q_named["q_left"],
        q_named["q_right"],
        q_named["q_bridge"],
        basis_order,
        samples_per_seam,
        *radii,
    )
    expected_rows = []
    for sphere, q_value, radius in (
        (0, q_named["q_left"], radii[0]),
        (1, q_named["q_right"], radii[1]),
    ):
        transition = plumbing_transition("zero", "infty", q_value)
        for k in range(samples_per_seam):
            source_global = radius * cmath.exp(2.0j * cmath.pi * k / samples_per_seam)
            target_global = transition(source_global)
            expected_rows.append(
                glasses_eval_row(index, sphere, source_global)
                - transition.deriv(source_global)
                * glasses_eval_row(index, sphere, target_global)
            )
    transition = plumbing_transition("one", "one", q_named["q_bridge"])
    for k in range(samples_per_seam):
        source_global = 1.0 + radii[2] * cmath.exp(
            2.0j * cmath.pi * k / samples_per_seam
        )
        target_global = transition(source_global)
        expected_rows.append(
            glasses_eval_row(index, 0, source_global)
            - transition.deriv(source_global)
            * glasses_eval_row(index, 1, target_global)
        )
    seam_matrix_error = max(
        abs(actual_value - expected_value) / max(1.0, abs(expected_value))
        for actual, expected in zip(matrix, expected_rows)
        for actual_value, expected_value in zip(actual, expected)
    )
    require(
        seam_matrix_error < 2.0e-13,
        f"glasses collocation seams disagree with the endpoint maps by {seam_matrix_error}",
    )

    graph = genus2_glasses_graph()
    expected_endpoints = {
        "q_left": ((0, "infinity"), (0, "zero")),
        "q_right": ((1, "infinity"), (1, "zero")),
        "q_bridge": ((0, "one"), (1, "one")),
    }
    for edge in graph.edges:
        endpoints = tuple(
            (endpoint.vertex, SLOT_NAMES[endpoint.slot])
            for endpoint in edge.endpoints
        )
        require(
            endpoints == expected_endpoints[edge.name],
            f"glasses graph edge {edge.name} has endpoints {endpoints}",
        )

    expected_order = ("q_left", "q_right", "q_bridge")
    require(
        GENUS2_PLUMBING_EDGE_ORDERS["glasses"] == expected_order
        and GENUS2_CFT_DESCENDANT_EDGE_ORDERS["glasses"] == expected_order
        and GENUS2_FREE_BOSON_EDGE_ORDERS["glasses"] == expected_order,
        "glasses geometry, CFT, and free-boson consumers do not share one order",
    )
    require(
        genus2_channel_q_values("glasses", q_named)
        == tuple(q_named[name] for name in expected_order),
        "named glasses edges were permuted at the consumer boundary",
    )
    print("\nglasses two-sphere endpoint pairing")
    print("  left handle:  sphere 0 infinity-zero")
    print("  right handle: sphere 1 infinity-zero")
    print("  bridge:       sphere 0 one - sphere 1 one")
    print("  all consumers: q_left, q_right, q_bridge (no permutation)")
    print(f"  collocation endpoint residual: {seam_matrix_error:.3e}")


def check_primary_propagator_matches_liouville_sample() -> None:
    b = 1.0
    q_values = (0.03 + 0.01j, 0.025 - 0.004j, 0.012 + 0.002j)
    result = liouville_genus2_ccy_partition(
        b=b,
        q1=q_values[0],
        q2=q_values[1],
        q3=q_values[2],
        block_order=0,
        p_max=0.7,
        quadrature_order=1,
        dps=22,
        include_vacuum_seed=False,
        store_samples=True,
    )
    sample = result.samples[0]
    weights = (
        liouville_weight_from_momentum(b, sample.p1),
        liouville_weight_from_momentum(b, sample.p2),
        liouville_weight_from_momentum(b, sample.p3),
    )
    expected = ccy_primary_propagator(q_values, weights)
    relative_error = abs(sample.propagator - expected) / max(abs(expected), 1.0e-300)

    print("CCY separated primary propagator")
    print(f"  sample propagator={sample.propagator!r}")
    print(f"  expected product={expected!r}")
    print(f"  relative error={relative_error:.6e}")
    require(relative_error < 1.0e-14, "Liouville wrapper is not using the CCY q^h primary propagator")


def check_theta_geometry_is_reversed_at_descendant_tensor() -> None:
    b = 1.0
    q_geometry = (0.031 + 0.007j, 0.019 - 0.004j, 0.011 + 0.002j)
    result = liouville_genus2_ccy_partition(
        b=b,
        q1=q_geometry[0],
        q2=q_geometry[1],
        q3=q_geometry[2],
        block_order=2,
        p_max=0.7,
        quadrature_order=1,
        dps=22,
        include_vacuum_seed=False,
        store_samples=True,
    )
    sample = result.samples[0]
    h_geometry = tuple(
        liouville_weight_from_momentum(b, momentum)
        for momentum in (sample.p1, sample.p2, sample.p3)
    )
    q_ccy = theta_geometry_to_ccy_order(q_geometry)
    h_ccy = theta_geometry_to_ccy_order(h_geometry)
    expected = ccy_genus2_block(
        c=25.0,
        h1=h_ccy[0],
        h2=h_ccy[1],
        h3=h_ccy[2],
        q1=q_ccy[0],
        q2=q_ccy[1],
        q3=q_ccy[2],
        order=2,
        include_vacuum_seed=False,
        resum_global_block=True,
    ).value
    wrong_order = ccy_genus2_block(
        c=25.0,
        h1=h_geometry[0],
        h2=h_geometry[1],
        h3=h_geometry[2],
        q1=q_geometry[0],
        q2=q_geometry[1],
        q3=q_geometry[2],
        order=2,
        include_vacuum_seed=False,
        resum_global_block=True,
    ).value
    print("\ntheta geometry/CCY slot order")
    print(f"  wrapper block={sample.block!r}")
    print(f"  correctly ordered block={expected!r}")
    print(f"  old-order displacement={abs(sample.block - wrong_order):.6e}")
    require(
        abs(sample.block - expected) < 1.0e-14,
        "theta geometry was not reversed into CCY infinity/one/zero slot order",
    )
    require(
        abs(sample.block - wrong_order) > 1.0e-6,
        "ordering regression data do not distinguish the old wiring",
    )


def check_literal_plumbing_coordinate() -> None:
    """Check that geometry and CCY propagation use the same ``u v=q``."""

    q_value = 0.013 - 0.004j
    source_global = 0.27 + 0.19j
    for source, target in (
        ("zero", "one"),
        ("one", "infty"),
        ("infty", "zero"),
    ):
        source_local = local_coordinate_map(source)(source_global)
        target_global = plumbing_transition(source, target, q_value)(source_global)
        target_local = local_coordinate_map(target)(target_global)
        require(
            abs(source_local * target_local / q_value - 1.0) < 2.0e-14,
            f"{source}->{target} transition does not satisfy u v=q",
        )

    q_values = (0.011, 0.009, 0.007)
    weights_a = (1.0, 1.0, 1.0)
    weights_b = (1.17, 1.31, 1.43)
    ratio_four_threshold = ccy_primary_propagator(
        tuple(4.0 * q for q in q_values),
        weights_a,
    ) / ccy_primary_propagator(q_values, weights_a)
    ratio_four_generic = ccy_primary_propagator(
        tuple(4.0 * q for q in q_values),
        weights_b,
    ) / ccy_primary_propagator(q_values, weights_b)
    ratio_sixteen_threshold = ccy_primary_propagator(
        tuple(16.0 * q for q in q_values),
        weights_a,
    ) / ccy_primary_propagator(q_values, weights_a)

    print("\nliteral CCY plumbing coordinate")
    print(
        "  nonchiral threshold ratio under q->4q  = "
        f"{abs(ratio_four_threshold) ** 2:.12g}"
    )
    print(
        "  nonchiral generic-weight ratio q->4q   = "
        f"{abs(ratio_four_generic) ** 2:.12g}"
    )
    print(
        "  nonchiral threshold ratio under q->16q = "
        f"{abs(ratio_sixteen_threshold) ** 2:.12g}"
    )
    require(
        abs(abs(ratio_four_threshold) ** 2 / (2.0**12) - 1.0) < 1.0e-14,
        "a hypothetical q->4q on three threshold edges should expose 2^12",
    )
    require(
        abs(abs(ratio_sixteen_threshold) ** 2 / (2.0**24) - 1.0) < 1.0e-14,
        "q->16q would give 2^24, not the observed nonchiral factor",
    )
    require(
        abs(abs(ratio_four_generic / ratio_four_threshold) ** 2 - 1.0) > 1.0,
        "q->4q should be visibly momentum dependent away from threshold",
    )


def check_diagnostic_shift_is_extra_factor() -> None:
    q_values = (0.15 + 0.0j, 0.04 + 0.01j, 0.03 - 0.005j)
    weights = (1.2, 1.7, 2.1)
    shift = 0.9
    raw = ccy_raw_sewing_propagator(q_values, weights)
    shifted = ccy_raw_sewing_propagator(q_values, weights, diagnostic_shift=shift)
    expected_ratio = 1.0 + 0.0j
    for q_value in q_values:
        expected_ratio *= q_value ** (-shift)
    relative_error = abs((shifted / raw) - expected_ratio) / max(abs(expected_ratio), 1.0e-300)

    print("\nCCY diagnostic shift bookkeeping")
    print(f"  relative error={relative_error:.6e}")
    require(relative_error < 1.0e-14, "diagnostic shift is not a pure extra factor on top of CCY sewing")


def check_liouville_threshold_factor() -> None:
    q_values = (0.15 + 0.0j, 0.15 + 0.0j, 0.15 + 0.0j)
    threshold = liouville_threshold_weight(1.0)
    factor = liouville_threshold_modulus_factor(q_values, b=1.0)
    expected = abs(q_values[0] * q_values[1] * q_values[2]) ** (2.0 * threshold)

    print("\nLiouville threshold factor")
    print(f"  Q^2/4={threshold:.12g}")
    print(f"  factor={factor:.12e}")
    require(threshold == 1.0, "b=1 Liouville threshold should be one")
    require(abs(factor - expected) < 1.0e-18, "threshold modulus factor does not match |prod q^(Q^2/4)|^2")


def check_logarithmic_underflow_propagator() -> None:
    logs = (-3.0 + 0.2j, -3606.0 + 0.4j, -4.0 - 0.3j)
    q_values = tuple(cmath.exp(complex(max(value.real, -690.0), value.imag)) for value in logs)
    weights = (1.02, 1.001, 1.03)
    got = ccy_raw_sewing_propagator(
        q_values,
        weights,
        diagnostic_shift=1.0,
        log_q_values=logs,
    )
    expected = cmath.exp(sum((weight - 1.0) * log_q for weight, log_q in zip(weights, logs)))
    require(abs(got / expected - 1.0) < 1.0e-14, "logarithmic propagator used the surrogate q")


def check_complex_dozz_phase_is_preserved() -> None:
    common = dict(
        b=0.9,
        block_order=0,
        p_max=0.6,
        quadrature_order=1,
        dps=22,
        mu=0.8 + 0.3j,
        include_cosmological_prefactor=True,
        include_vacuum_seed=False,
        store_samples=True,
    )
    theta = liouville_genus2_ccy_partition(
        **common,
        q1=0.025 + 0.003j,
        q2=0.021 - 0.002j,
        q3=0.018 + 0.001j,
    )
    theta_sample = theta.samples[0]
    theta_expected = (
        theta_sample.measure_weight
        * (theta_sample.structure_constant**2)
        * abs(theta_sample.propagator * theta_sample.block) ** 2
    )
    theta_abs_weight = (
        theta_sample.measure_weight
        * abs(theta_sample.structure_constant) ** 2
        * abs(theta_sample.propagator * theta_sample.block) ** 2
    )

    glasses = liouville_genus2_glasses_partition(
        **common,
        q_left=0.025 + 0.003j,
        q_right=0.021 - 0.002j,
        q_bridge=0.018 + 0.001j,
    )
    glasses_sample = glasses.samples[0]
    glasses_expected = (
        glasses_sample.measure_weight
        * glasses_sample.structure_left
        * glasses_sample.structure_right
        * abs(glasses_sample.propagator * glasses_sample.block) ** 2
    )
    glasses_abs_weight = (
        glasses_sample.measure_weight
        * abs(glasses_sample.structure_left * glasses_sample.structure_right)
        * abs(glasses_sample.propagator * glasses_sample.block) ** 2
    )

    print("\ncomplex DOZZ phase preservation")
    print(f"  theta contribution={theta_sample.contribution!r}")
    print(f"  glasses contribution={glasses_sample.contribution!r}")
    require(abs(theta_sample.contribution - theta_expected) < 1.0e-28, "theta wrapper dropped the DOZZ phase")
    require(abs(glasses_sample.contribution - glasses_expected) < 1.0e-28, "glasses wrapper dropped the DOZZ phase")
    require(
        abs(theta_expected - theta_abs_weight) > 1.0e-20,
        "theta complex-mu sample did not produce a visible phase test",
    )
    require(
        abs(glasses_expected - glasses_abs_weight) > 1.0e-20,
        "glasses complex-mu sample did not produce a visible phase test",
    )


def check_genus2_combined_log_large_momentum() -> None:
    r"""Make a finite genus-two density from ``infinity * zero`` factors."""

    momentum = 100.0
    special = UpsilonB(1.0, dps=70)
    log_structure = log_yin_structure_constant_momentum(
        special,
        momentum,
        momentum,
        momentum,
    )
    weight = 1.0 + momentum**2
    log_q = -float(mp.re(log_structure)) / (3.0 * weight)
    q_value = math.exp(log_q)

    common = dict(
        special=special,
        b=1.0,
        block_order=0,
        include_vacuum_seed=False,
        resum_global_block=False,
    )
    theta = liouville_genus2_ccy_density(
        **common,
        q1=q_value,
        q2=q_value,
        q3=q_value,
        p1=momentum,
        p2=momentum,
        p3=momentum,
    )
    glasses = liouville_genus2_glasses_density(
        **common,
        q_left=q_value,
        q_right=q_value,
        q_bridge=q_value,
        p_left=momentum,
        p_right=momentum,
        p_bridge=momentum,
    )
    expected = 1.0 / math.pi**3

    # These are precisely the two intermediates that the old code formed.
    old_structure = yin_structure_constant_momentum(
        special,
        momentum,
        momentum,
        momentum,
    )
    old_propagator = q_value ** (3.0 * weight)

    print("\ngenus-two combined-log large-momentum stress test")
    print(f"  old DOZZ intermediate={old_structure!r}")
    print(f"  old propagator intermediate={old_propagator!r}")
    print(f"  theta density={theta!r}")
    print(f"  glasses density={glasses!r}")
    require(
        math.isinf(old_structure.real) and old_propagator == 0.0,
        "large-momentum stress point did not exercise infinity-times-zero",
    )
    require(
        abs(theta / expected - 1) < 5.0e-10,
        "theta combined-log density lost the DOZZ/propagator cancellation",
    )
    require(
        abs(glasses / expected - 1) < 5.0e-10,
        "glasses combined-log density lost the DOZZ/propagator cancellation",
    )


def run() -> None:
    check_named_genus2_edge_orders()
    check_theta_endpoint_pairing_matches_period_map()
    check_glasses_endpoint_pairing_matches_period_map()
    check_primary_propagator_matches_liouville_sample()
    check_theta_geometry_is_reversed_at_descendant_tensor()
    check_literal_plumbing_coordinate()
    check_diagnostic_shift_is_extra_factor()
    check_liouville_threshold_factor()
    check_logarithmic_underflow_propagator()
    check_complex_dozz_phase_is_preserved()
    check_genus2_combined_log_large_momentum()
    print("\nall CCY plumbing convention checks passed")


if __name__ == "__main__":
    run()
