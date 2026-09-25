"""Ordinary sewing q, independent of elliptic-nome/pillow utilities."""

from unittest.mock import patch

import numpy as np
import pytest

import spin23_singlet_amplitudes as singlet

ref = singlet.ref
fast = singlet.fast
ENERGIES = (0.11 + 0.15j, 0.17 + 0.16j, 0.23 + 0.18j, 0.51 + 0.49j)


@pytest.mark.parametrize("star2,star3", [(False, False), (True, True), (False, True), (True, False)])
def test_sewing_block_is_the_original_c_recursion_series(star2, star3):
    weights = tuple(fast.h_of_p(w) for w in ENERGIES)
    block = fast.FastNSBlockComputer(*weights[::-1], star2=star2, star3=star3)
    h = fast.h_of_p(0.73)
    q = np.asarray([0.23 + 0.1j, 0.6 - 0.2j, -0.8 + 0.1j])
    with patch.object(ref, "_modular_series", side_effect=AssertionError("nome conversion")):
        sewing = ref.SewingNSBlock(block, h, 5)
    for parity in ("e", "o"):
        np.testing.assert_allclose(sewing.value(q, parity), ref._direct_block(block, h, q, parity, 5),
                                   rtol=3e-14, atol=3e-14)


def test_singlet_uses_the_same_sewing_coefficients_in_all_six_components():
    weights = tuple(fast.h_of_p(w) for w in ENERGIES)
    h = fast.h_of_p(0.73)
    table = singlet._recursive_coefficient_table(weights, np.asarray([0.73]), 4, tuple(singlet.WORD_PATTERNS))
    q = np.asarray([0.25 + 0.12j])
    with patch.object(ref, "_modular_series", side_effect=AssertionError("nome conversion")):
        for name, words in singlet.WORD_PATTERNS.items():
            even, odd = table.coefficients[name]
            block = singlet._generic_block(h_internal=h, external_weights=weights, words=words,
                                           even_coefficients=even[0], odd_coefficients=odd[0], q_order=4)
            h1, h2, _, _ = singlet._effective_weights(weights, words)
            for beta in (0, 1):
                coefficients = table.coefficients[name][beta][0]
                expected = (singlet.component_phase(words, beta) * q**(h - h1 - h2 + beta/2)
                            * sum(c*q**n for n, c in enumerate(coefficients)))
                np.testing.assert_allclose(block.direct_value(beta, q, 4), expected, rtol=1e-14, atol=1e-14)


def test_sewing_grid_retains_the_plane_measure_without_nome_evaluation():
    with patch.object(ref, "_q_grid", side_effect=AssertionError("nome evaluation")):
        q, weights = fast.sewing_annulus_grid(0.2, 0.0, (4, 4, 12), 8)
    assert np.all(np.abs(q) < 1)
    np.testing.assert_allclose(sum(weights), np.pi*(1 - 0.2**2), rtol=2e-14)


def test_full_singlet_and_vector_paths_do_not_call_elliptic_utilities():
    with patch.object(ref, "_q_grid", side_effect=AssertionError("nome evaluation")), \
         patch.object(ref, "_modular_series", side_effect=AssertionError("nome conversion")), \
         patch.object(ref, "_evaluate_elliptic_block", side_effect=AssertionError("pillow block")):
        result = singlet.evaluate_singlet_amplitudes(
            series_parameter="sewing",
            energies=ENERGIES, q_order=2, lower_q_order=1, p_nodes=3, p_max=3.0,
            theta_orders=(4, 4, 8), radial_order=6, disk_total_order=8, crossed_disk_total_order=8,
        )
        vector = fast.one_ordering_regularized(
            ENERGIES, series_parameter="sewing", q_order=2, p_nodes=3, p_max=3.0,
            theta_orders=(4, 4, 8), radial_order=6, disk_total_order=8,
            lens_radial_order=6, lens_angular_order=12,
        )
        reference = ref.vector_amplitude_coefficients(
            ENERGIES, series_parameter="sewing", q_order=2, p_nodes=3, p_max=3.0,
            theta_orders=(4, 4, 8), radial_order=6, lens_radial_order=6, lens_angular_order=12,
        )
    assert result.series_parameter == "sewing"
    assert result.momentum_quadrature["scheme"] == "threshold_weighted"
    assert np.isfinite(result.values.ssvv_raw)
    assert np.all(np.isfinite(vector))
    assert np.all(np.isfinite(reference))


def test_vector_bulk_is_identical_to_its_ordinary_ope_polynomial():
    data = fast.build_s_channel_data(ENERGIES, 3, 3, 3.0, series_parameter="sewing")
    q = np.asarray([0.25 + 0.1j, -0.45 + 0.3j])
    weights = np.asarray([0.3, 0.2])
    for kernel in data:
        np.testing.assert_allclose(fast._integrate_one_p_kernel(kernel, ENERGIES, (q, weights)),
                                   fast._s_direct_kernel_integral(kernel, ENERGIES, q, weights, 3),
                                   rtol=3e-14, atol=3e-14)


def test_an_unknown_series_request_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="series_parameter"):
        singlet.evaluate_singlet_amplitudes(ENERGIES, series_parameter="pillow")


def test_crossing_diagnostics_use_production_sewing_blocks():
    import audit_spin23_crossing_faithfulness as audit
    import check_spin23_sphere_crossing as crossing

    data = fast.build_s_channel_data(ENERGIES, 2, 3, 3.0, series_parameter="sewing")
    z = np.asarray([0.25 + 0.1j, 0.55 + 0.1j])
    with patch.object(ref, "_q_grid", side_effect=AssertionError("nome evaluation")), \
         patch.object(ref, "_evaluate_elliptic_block", side_effect=AssertionError("nome block")):
        np.testing.assert_allclose(audit.evaluate(data, z), crossing.evaluate_channel(data, z), rtol=1e-14)
        rows, *_ = crossing.crossing_rows(label="test", q_order=2, z=z,
                                          s_channel=data, t_channel=data)
    assert all(row["series_parameter"] == "sewing" for row in rows)
    np.testing.assert_allclose([row["q_s_abs"] for row in rows], abs(z))
    np.testing.assert_allclose([row["q_t_abs"] for row in rows], abs(1-z))


@pytest.mark.parametrize("starred", [False, True])
def test_historical_nome_coefficients_follow_triangular_series_conversion(starred):
    """Verify the explanation of the OLD utility, not a production nome path."""
    weights = tuple(fast.h_of_p(w) for w in ENERGIES)
    block = fast.FastNSBlockComputer(*weights[::-1], star2=starred, star3=starred)
    h = fast.h_of_p(0.73)
    old = ref.EllipticNSBlock(block, h, 2)
    d1, d2, d3, d4 = weights
    d2 += 0.5 * starred
    d3 += 0.5 * starred
    prefactor_linear = 8 * (d1-d2-d3+d4-h)
    even1 = 16 * complex(block.coefficient(2, h)) + prefactor_linear
    odd0 = (-1 if starred else 1) * complex(block.coefficient(1, h))
    odd1 = (-1 if starred else 1) * complex(block.coefficient(3, h))
    np.testing.assert_allclose(old.even_h[2], even1, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(old.odd_h[1], 4*odd0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(old.odd_h[3], 4*(16*odd1+(prefactor_linear-4)*odd0),
                               rtol=1e-13, atol=1e-13)
