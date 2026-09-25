"""Validate the diagnostic itself against the production component contraction."""

from types import SimpleNamespace

import numpy as np
import pytest

import check_spin23_cft_repairs as audit


@pytest.fixture(scope="module")
def atlas():
    return audit.build((.11+.15j, .17+.16j, .23+.18j, .51+.49j), 3, (4, 16, 8))


@pytest.mark.parametrize("chart", ("original_s", "original_t"))
def test_each_component_is_the_production_contraction_with_one_measure(atlas, chart):
    data = getattr(atlas, chart)
    z = np.asarray([.42+.13j, .53+.10j])
    values, scales = audit.cft_components(data, z, 3)
    for left, right in audit.COMPONENTS:
        terms = [kernel.quadrature_weight * audit.s._contract_grid(
            kernel, left, right, z, z.conj(), q=None, qbar=None,
            theta3=None, theta3bar=None, order=3,
        ) for kernel in data.kernels]
        expected = np.sum(terms, axis=0)
        np.testing.assert_allclose(values[left+right], expected, rtol=2e-14, atol=1e-15)
        assert np.all(scales[left+right] >= abs(expected)*(1-1e-14))


@pytest.mark.parametrize("chart", ("original_s", "original_t"))
def test_adding_back_pco_and_free_fields_recovers_grid_integral(atlas, chart):
    data = getattr(atlas, chart)
    z = np.asarray([.42+.13j, .53+.10j])
    coordinate = 1-z if chart.endswith("_t") else z
    values, _ = audit.cft_components(data, coordinate, 3)
    w1, w2, w3, _ = data.energies
    time_factor = np.exp(-2*w1*w2*np.log(abs(z))-2*w2*w3*np.log(abs(1-z)))
    for process, right in audit.PROCESSES.items():
        if process == "vvvv":
            continue
        pco = values["M"+right] + w2*w3/(1-z)*values["P"+right]
        spectator = 1/(1-z.conj()) if process == "ssvv" else 1
        expected = audit.s._grid_integral(data, coordinate, np.ones(z.size),
            original_z=z, q=None, theta3=None, order=3, process=process)
        np.testing.assert_allclose(np.sum(time_factor*spectator*pco), expected, rtol=2e-14, atol=1e-15)


@pytest.mark.parametrize("name", ("P", "M"))
def test_legacy_adapter_agrees_with_retained_vector_elliptic_utility(atlas, name):
    block = atlas.original_s.kernels[12].blocks[name]
    weights = audit.s._ordered_weights(atlas.original_s.energies)
    def raw_coefficient(level, _):
        parity = level % 2
        array = block.even_z if parity == 0 else block.odd_z
        return array[level//2] / audit.s.component_phase(block.words, parity)
    computer = SimpleNamespace(h1=weights[0], h2=weights[1], h3=weights[2], h4=weights[3],
                               star2=name == "M", star3=name == "M", coefficient=raw_coefficient)
    old = audit.s.ref.EllipticNSBlock(computer, block.h_internal, q_order=3)
    z = np.asarray([.06+.02j, .42+.13j, -.8+.1j, .49+.84j, .94+.02j])
    qhat, theta3 = audit.s.ref._q_grid(z)
    for parity, label in enumerate(("e", "o")):
        expected = audit.s.ref._evaluate_elliptic_block(old, z, qhat, theta3, label)
        np.testing.assert_allclose(audit.legacy_value(block, parity, z, 3), expected, rtol=3e-13, atol=1e-14)


def test_legacy_matches_all_component_series_near_ope(atlas):
    z = np.asarray([1e-4+2e-5j])
    for channel in (atlas.original_s, atlas.original_t):
        for block in channel.kernels[12].blocks.values():
            for parity in (0, 1):
                np.testing.assert_allclose(audit.legacy_value(block, parity, z, 3),
                                           block.direct_value(parity, z, 3), rtol=2e-11, atol=1e-13)


def test_selected_chart_and_relative_comparison(atlas):
    snapshot = audit.snapshot(atlas, 3)
    for row in snapshot["points"]:
        z = complex(*row["z"])
        assert row["chart"] == ("t" if abs(1-z) < abs(z) else "s")
        assert row["abs_sewing_coordinate"] < 1
    comparison = audit.compare(snapshot, snapshot)
    assert comparison["max_CFT_relative"] == comparison["max_PCO_relative"] == 0
    with pytest.raises(ValueError):
        audit.cft_components(atlas.original_s, np.asarray([.5+.1j]), 4)


def test_antichiral_weights_are_not_conjugated(atlas):
    # Complex external momenta make a conjugated-block shortcut incorrect.
    block = atlas.original_s.kernels[12].blocks["P"]
    z = np.asarray([.42+.13j])
    assert abs(block.direct_value(0, z.conj(), 3)[0]
               - block.direct_value(0, z, 3)[0].conjugate()) > .01


@pytest.mark.parametrize("momentum", (.006, .73, 2.1))
def test_order11_coefficients_have_independent_precision_and_ward_controls(momentum):
    from spin23_ns_c_recursion import block_coefficients
    omega = 1/9+.2j
    weights = audit.s._ordered_weights((omega, omega, omega, 3*omega))
    names = tuple(audit.s.WORD_PATTERNS)
    table = audit.s._recursive_coefficient_table(weights, np.asarray([momentum]), 11, names)
    patterns = tuple(tuple(int(bool(word)) for word in audit.s.WORD_PATTERNS[name]) for name in names)
    precise = block_coefficients(c=13.5, h_internal=audit.s.fast.h_of_p(momentum),
        external_weights=weights, external_patterns=patterns, maximum_twice_level=23,
        force_high_precision=True, digits=100)
    for name, pattern in zip(names, patterns):
        for parity in (0, 1):
            np.testing.assert_allclose(table.coefficients[name][parity][0],
                precise.coefficients[pattern][parity::2], rtol=3e-12, atol=3e-12)
    left, right = audit.s._crossed_pair_coefficients_from_ward_identity(table)
    for name, expected in (("L", left), ("R", right)):
        for parity in (0, 1):
            np.testing.assert_allclose(table.coefficients[name][parity], expected[parity], rtol=3e-12, atol=3e-12)
