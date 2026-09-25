"""Descendant c-recursion vs independent Ward/Gram and vector-block oracles."""

import itertools
from unittest.mock import patch

import mpmath as mp
import numpy as np
import pytest

import spin23_singlet_amplitudes as singlet
from spin23_ns_c_recursion import block_coefficients, global_seed, RecursionPoleCollision
from spin23_ns_sphere_blocks import (
    G_MINUS_HALF, direct_sphere_block_series, global_sphere_block_beta_series,
    recursive_sphere_block_series,
)


PATTERNS = tuple(a for a in itertools.product((0, 1), repeat=4) if sum(a) % 2 == 0)
WEIGHTS = (0.41 + 0.02j, 0.63 - 0.03j, 0.82 + 0.01j, 1.11 + 0.04j)
ENERGIES = (0.11 + 0.15j, 0.17 + 0.16j, 0.23 + 0.18j, 0.51 + 0.49j)


@pytest.mark.parametrize("alphas", PATTERNS)
def test_large_c_seed_is_the_global_descendant_tensor(alphas):
    words = tuple(G_MINUS_HALF if a else () for a in alphas)
    for component in ("even", "odd"):
        oracle = global_sphere_block_beta_series(
            h_internal=0.83, external_weights=WEIGHTS, external_words=words,
            maximum_twice_level=9, beta=int(component == "odd"),
        )
        for n, expected in oracle.coefficients.items():
            np.testing.assert_allclose(global_seed(n, 0.83, WEIGHTS, alphas), expected,
                                       rtol=2e-13, atol=2e-13)


def test_two_star_seed_is_not_a_weight_shifted_primary_seed():
    h = 0.83
    h1, h2, h3, h4 = WEIGHTS
    actual = global_seed(1, h, WEIGHTS, (0, 1, 1, 0))
    np.testing.assert_allclose(actual, -(h + h3 - h4) * (h + h2 - h1) / (2 * h))
    assert abs(actual - 1 / (2 * h)) > 0.1
    # Four upper components have a nontrivial leading term too: never return 1 blindly.
    np.testing.assert_allclose(global_seed(0, h, WEIGHTS, (1, 1, 1, 1)),
                               -(h3 + h4 - h) * (h1 + h2 - h))


@pytest.mark.parametrize("alphas", PATTERNS)
@pytest.mark.parametrize("c,h", [(13.5, 0.76645), (9.7 + 0.2j, 1.17 + 0.07j)])
def test_all_even_patterns_against_full_gram(alphas, c, h):
    words = tuple(G_MINUS_HALF if a else () for a in alphas)
    result = block_coefficients(
        c=c, h_internal=h, external_weights=WEIGHTS, external_patterns=(alphas,),
        maximum_twice_level=9,
    )
    for component in ("even", "odd"):
        oracle = direct_sphere_block_series(
            c=c, h_internal=h, external_weights=WEIGHTS, external_words=words,
            maximum_twice_level=9, component=component,
        )
        for n, expected in oracle.coefficients.items():
            np.testing.assert_allclose(result.coefficients[alphas][n], expected,
                                       rtol=3e-11, atol=3e-12)


def test_low_momentum_through_production_q7_against_mp_gram():
    momenta = np.asarray([0.006])
    weights = tuple(singlet.fast.h_of_p(w) for w in ENERGIES)
    solvers, _ = singlet._gram_solvers(momenta, 15, condition_limit=1e13)
    oracle = singlet._coefficient_table(
        weights, momenta, solvers, 7, tuple(singlet.WORD_PATTERNS),
        high_precision_condition=1, high_precision_digits=70,
    )
    result = singlet._recursive_coefficient_table(weights, momenta, 7, tuple(singlet.WORD_PATTERNS))
    assert result.high_precision_recursion_node_count == 1
    assert result.high_precision_solve_count == 0
    for name in singlet.WORD_PATTERNS:
        for beta in (0, 1):
            np.testing.assert_allclose(result.coefficients[name][beta], oracle.coefficients[name][beta],
                                       rtol=2e-11, atol=2e-12)


def test_crossed_ward_identities_are_independent_checks_of_recursion():
    momenta = np.asarray([0.006, 0.73, 2.1])
    table = singlet._recursive_coefficient_table(WEIGHTS, momenta, 5, tuple(singlet.WORD_PATTERNS))
    left, right = singlet._crossed_pair_coefficients_from_ward_identity(table)
    for name, expected in (("L", left), ("R", right)):
        for beta in (0, 1):
            np.testing.assert_allclose(table.coefficients[name][beta], expected[beta],
                                       rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("p", [0.006, 0.3, 0.73, 2.1])
def test_q9_reference_vector_blocks_and_precision_stability(p):
    h = singlet.fast.h_of_p(p)
    patterns = ((0, 0, 0, 0), (0, 1, 1, 0))
    result = block_coefficients(
        c=13.5, h_internal=h, external_weights=WEIGHTS, external_patterns=patterns,
        maximum_twice_level=19, force_high_precision=p <= 0.18,
    )
    with mp.workdps(80):
        for a in patterns:
            block = singlet.ref.NSBlockComputer(*WEIGHTS[::-1], star2=bool(a[1]), star3=bool(a[2]))
            expected = [complex(block.coefficient(n, mp.mpc(h))) for n in range(20)]
            np.testing.assert_allclose(result.coefficients[a], expected, rtol=2e-9, atol=2e-10)


def test_precision_fallback_and_context_isolation():
    before = mp.mp.dps
    result = block_coefficients(
        c=13.5, h_internal=0.77, external_weights=WEIGHTS, external_patterns=PATTERNS,
        maximum_twice_level=9, cancellation_limit=1e-6,
    )
    assert result.high_precision
    assert mp.mp.dps == before
    assert result.maximum_cancellation is not None


def test_higher_descendants_and_odd_external_parity_are_not_silently_substituted():
    from ns_algebra.ns_sca import L
    with pytest.raises(NotImplementedError):
        recursive_sphere_block_series(c=13.5, h_internal=0.83, external_weights=WEIGHTS,
                                     external_words=((L(-1),), (), (), ()),
                                     maximum_twice_level=4, component="even")
    with pytest.raises(NotImplementedError):
        block_coefficients(c=13.5, h_internal=0.83, external_weights=WEIGHTS,
                           external_patterns=((1, 0, 0, 0),), maximum_twice_level=4)


def test_actual_external_recursion_pole_raises_without_a_regulator():
    from spin23_ns_c_recursion import _Recursor
    engine = _Recursor(WEIGHTS)
    pole, _, _ = engine.pole(0.83, 3, 1)
    with pytest.raises(RecursionPoleCollision):
        block_coefficients(c=pole, h_internal=0.83, external_weights=WEIGHTS,
                           external_patterns=((0, 1, 1, 0),), maximum_twice_level=3,
                           pole_tolerance=1e-11)


def test_all_amplitudes_and_six_moduli_pieces_agree_without_gram_in_production():
    settings = dict(q_order=3, lower_q_order=2, p_nodes=4, p_max=3.0,
                    theta_orders=(8, 8, 16), radial_order=10, disk_total_order=12,
                    crossed_disk_total_order=12, include_v_to_vss=True)
    oracle = singlet.evaluate_singlet_amplitudes(ENERGIES, block_backend="inverse_gram", **settings)
    with patch.object(singlet, "_level_template", side_effect=AssertionError("Gram template in production")), \
         patch.object(singlet, "_gram_solvers", side_effect=AssertionError("Gram solve in production")), \
         patch.object(singlet, "_crossed_pair_coefficients_from_ward_identity", side_effect=AssertionError("Ward construction in production")):
        result = singlet.evaluate_singlet_amplitudes(ENERGIES, **settings)
    assert result.block_backend == "c_recursion"
    assert result.maximum_gram_condition is None
    assert result.high_precision_gram_solve_count == 0
    assert result.template_seconds == 0
    for observed, expected in ((result.values, oracle.values),
                               (result.lower_order_values, oracle.lower_order_values)):
        for process in ("ssvv", "ssss", "v_to_vss"):
            for piece, target in expected.pieces[process].items():
                np.testing.assert_allclose(observed.pieces[process][piece], target,
                                           rtol=3e-10, atol=2e-12)


def test_sewing_spectral_channel_discrepancies_are_measured_under_refinement():
    from validate_spin23_c_recursion import crossing_check
    result = crossing_check(q_order=5)
    assert len(result["rows"]) == 6
    assert result["series_parameter"] == "sewing"
    assert result["improves_with_order"]
    # The old elliptic-q5 bound is deliberately not assigned to sewing q5.


def test_equal_energy_channels_reuse_one_union_of_descendant_tables():
    w = 0.11 + 0.15j
    with patch.object(singlet, "_recursive_coefficient_table", wraps=singlet._recursive_coefficient_table) as build:
        atlas = singlet._build_atlas((w, w, w, 3 * w), q_order=3, p_nodes=4, p_max=3.0,
                                    p_cut=0.03, gram_condition_limit=1e13)
    assert build.call_count == 1
    assert set(build.call_args.args[3]) == set(singlet.WORD_PATTERNS)
    assert atlas.original_s.kernels is atlas.swapped_s.kernels
    assert atlas.original_t.kernels is atlas.swapped_t.kernels
