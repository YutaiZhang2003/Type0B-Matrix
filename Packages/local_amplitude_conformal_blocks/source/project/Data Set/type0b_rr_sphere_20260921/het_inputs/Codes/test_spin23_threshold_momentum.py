"""Analytic moments, infinite-tail Jacobian and amplitude-route checks."""

import mpmath as mp
import numpy as np
import pytest

import spin23_singlet_amplitudes as singlet
from liouville_momentum_quadrature import ThresholdConfig, threshold_weighted_rule


@pytest.mark.parametrize("beta", [-0.5, 0.0, 0.7, 2.0, 4.0])
def test_shifted_gaussian_moments_and_region_integrals(beta):
    config = ThresholdConfig(beta=beta, a=1.4, s=1.2)
    rule = threshold_weighted_rule((20, 96, 40), config)
    p = rule.momenta
    for power in (0, 1, 4):
        values = np.exp(config.log_weight(p)) * p**power
        with mp.workdps(45):
            f = lambda x: x**(beta+power)*mp.exp(-mp.mpf('1.4')*x*x+mp.mpf('1.2')*x)
            limits = (0, config.endpoint, config.tail, mp.inf)
            for region, (lo, hi) in enumerate(zip(limits[:-1], limits[1:])):
                observed = np.dot(rule.weights[rule.regions == region], values[rule.regions == region])
                expected = float(mp.quad(f, [lo, hi]))
                np.testing.assert_allclose(observed, expected, rtol=4e-11, atol=1e-14)
        np.testing.assert_allclose(rule.integrate(values), rule.integrate(p**power, factored=True),
                                   rtol=3e-14, atol=1e-14)


def test_tail_map_and_jacobian_include_entire_half_line():
    config = ThresholdConfig(a=0.9, s=1.7, tail=2.0)
    rule = threshold_weighted_rule((4, 8, 16), config)
    mask = rule.regions == 2
    p, w = rule.momenta[mask], rule.weights[mask]
    x = config.a*(p**2-config.tail**2)-config.s*(p-config.tail)
    # f(P)=(dt/dP)e^-t has integral exactly one on [tail,infinity).
    np.testing.assert_allclose(np.dot(w, (2*config.a*p-config.s)*np.exp(-x)), 1.0, rtol=1e-14)
    assert np.all(p > config.tail)
    assert rule.metadata()["tail_domain"].startswith("[tail, infinity)")


def test_weight_and_threshold_changes_do_not_change_complex_integral():
    expected = complex(mp.quad(lambda p: p**2*mp.exp(-(1.3+0.2j)*p*p+(0.7+0.1j)*p), [0, 1, mp.inf]))
    for e, t, a, s in ((0.09, 2.1, 0.8, -0.3), (0.18, 3.2, 1, 0), (0.27, 4.5, 1.6, 0.9)):
        rule = threshold_weighted_rule((16, 96, 48), dict(endpoint=e, tail=t, a=a, s=s))
        p = rule.momenta
        values = p**2*np.exp(-(1.3+0.2j)*p*p+(0.7+0.1j)*p)
        np.testing.assert_allclose(rule.integrate(values), expected, rtol=3e-12, atol=1e-14)


@pytest.mark.parametrize("nodes", [3, 8, 24, 96, (8, 40, 16), "segmented"])
def test_node_budgets_and_positive_weights(nodes):
    rule = threshold_weighted_rule(nodes)
    assert len(rule.momenta) == sum(rule.counts)
    if isinstance(nodes, int):
        assert len(rule.momenta) == nodes
    assert np.all(rule.momenta > 0)
    assert np.all(rule.weights > 0)
    assert np.all(np.diff(rule.momenta) > 0)
    assert set(rule.regions) == {0, 1, 2}


@pytest.mark.parametrize("options", [{"beta": -1}, {"a": 0}, {"a": -1}, {"s": 7},
                                      {"endpoint": 0}, {"tail": 0.1}, {"s": 1j},
                                      {"a": float('nan')}, {"bulk_breakpoints": [1, 0.5]}, {"typo": 1}])
def test_invalid_envelopes_and_thresholds_are_rejected(options):
    with pytest.raises(ValueError):
        threshold_weighted_rule(24, options)


@pytest.mark.parametrize("nodes", [0, 2, True, 24.5, [4, 0, 8], [4, 5.5, 8], [2]*7])
def test_invalid_node_counts_are_not_silently_reinterpreted(nodes):
    with pytest.raises(ValueError):
        threshold_weighted_rule(nodes)


def test_all_three_engines_use_identical_dP_weights_and_one_pi_factor():
    options = dict(endpoint=0.12, tail=2.8, beta=2, a=0.8, s=0.4)
    expected = threshold_weighted_rule((3, 6, 3), options)
    p, w = singlet._momentum_quadrature((3, 6, 3), 0, 0,
        scheme="threshold_weighted", momentum_threshold_options=options)
    fp, fw = singlet.fast._p_quadrature((3, 6, 3), 0, 0,
        scheme="threshold_weighted", threshold_options=options)
    rp, rw = singlet.ref._reference_momentum_rule((3, 6, 3), 0, "threshold_weighted", options)
    for nodes, weights in ((p, w), (fp, fw), (rp, rw)):
        np.testing.assert_array_equal(nodes, expected.momenta)
        np.testing.assert_array_equal(weights, expected.weights)
    energies = (0.1+0.15j,)*3+(0.3+0.45j,)
    common = dict(momentum_scheme="threshold_weighted", momentum_threshold_options=options)
    s = singlet.fast.build_s_channel_data(energies, 1, (3, 6, 3), 0, **common)
    t = singlet.fast.build_t_channel_data(energies, 1, (3, 6, 3), 0, **common)
    for kernels in (s, t):
        np.testing.assert_allclose([k.weight for k in kernels], w/np.pi, rtol=1e-15)
        np.testing.assert_array_equal([k.p for k in kernels], p)
