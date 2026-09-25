"""Convention checks for the high-dimensional amplitude scan adapter."""

from __future__ import annotations

import math

import numpy as np

from spin23_vvvv_scan_evaluator import _old_channel_order


def test_fit_and_scan_tensor_orderings_are_related_by_cba():
    assert np.array_equal(
        _old_channel_order(np.array([1+2j,3+4j,5+6j])),
        np.array([5+6j,3+4j,1+2j]),
    )


def test_old_channel_candidate_has_expected_spectator_pair():
    w1,w2,w3=0.11+0.13j,0.17+0.16j,0.23+0.18j
    w0=w1+w2+w3
    common=-math.pi*w0*w1*w2*w3
    expected=np.array([
        common/(1+1j*(w2+w3)),
        common/(1+1j*(w1+w3)),
        common/(1+1j*(w1+w2)),
    ])
    from heterotic_so23_1to3_fast import pair_channel_ansatz

    assert np.allclose(_old_channel_order(pair_channel_ansatz([w1,w2,w3,w0])),expected)


def test_singlet_scan_options_select_and_forward_c_recursion():
    from spin23_singlet_scan_evaluator import _numerical_options
    assert _numerical_options({})["block_backend"] == "c_recursion"
    assert _numerical_options({})["momentum_scheme"] == "threshold_weighted"
    options = _numerical_options({"recursion_digits": 80, "recursion_reference_p_max": 0.2,
                                  "recursion_cancellation_limit": 1e4, "momentum_scheme": "infinite_gauss"})
    assert options["recursion_digits"] == 80
    assert options["recursion_reference_p_max"] == 0.2
    assert options["recursion_cancellation_limit"] == 1e4
    assert options["momentum_scheme"] == "infinite_gauss"
    assert _numerical_options({"block_backend": "inverse_gram"})["block_backend"] == "inverse_gram"


def test_scan_forwards_threshold_controls_and_rejects_stale_resume(tmp_path):
    import pytest
    from spin23_singlet_scan_evaluator import _numerical_options, _settings_signature, _completed_successfully
    from spin23_vvvv_scan_evaluator import _numerical_options as vector_options
    settings = {"p_nodes": [8, 32, 16], "momentum_threshold_options": {"beta": 2, "a": 1.5, "s": 0.4}}
    for options in (_numerical_options(settings), vector_options(settings, {})):
        assert options["momentum_scheme"] == "threshold_weighted"
        assert options["momentum_threshold_options"] == settings["momentum_threshold_options"]
    signature = _settings_signature(settings)
    path = tmp_path / "point.csv"
    original = f"status,numerical_settings_sha256\nok,{signature}\n"
    path.write_text(original)
    assert _completed_successfully(path, signature)
    with pytest.raises(ValueError, match="fresh result directory"):
        _completed_successfully(path, _settings_signature({**settings, "momentum_threshold_options": {"a": 0.5}}))
    assert path.read_text() == original
    path.write_text("status\nok\n")
    with pytest.raises(ValueError, match="unrecorded"):
        _completed_successfully(path, signature)


def test_threshold_manifest_profiles_preserve_meaningful_momentum_controls():
    import json
    from pathlib import Path
    from liouville_momentum_quadrature import threshold_weighted_rule
    from spin23_singlet_scan_evaluator import _numerical_options
    from spin23_highdim_scan import settings_table
    generated = settings_table()
    assert all(p["momentum_scheme"] == "threshold_weighted" and "p_max" not in p for p in generated.values())
    assert generated["p_extended"]["momentum_threshold_options"]["tail"] == 4.5
    code_dir = Path(__file__).resolve().parent
    for filename in ("spin23_singlet_threshold_settings.json", "spin23_vvvv_threshold_settings.json"):
        settings = json.loads((code_dir / filename).read_text())
        for profile in settings.values():
            options = _numerical_options(profile)
            assert options["momentum_scheme"] == "threshold_weighted"
            rule = threshold_weighted_rule(options["p_nodes"], options["momentum_threshold_options"])
            assert rule.metadata()["node_count"] == sum(profile["p_nodes"])
        assert settings["p_extended"]["momentum_threshold_options"]["tail"] > settings["production"]["momentum_threshold_options"]["tail"]
    for filename in ("spin23_singlet_settings.json", "spin23_vvvv_stable_settings.json",
                     "spin23_formula_validation_settings.json", "spin23_singlet_convergence_settings.json",
                     "spin23_singlet_benchmark_settings.json"):
        assert all(profile["momentum_scheme"] == "cutoff"
                   for profile in json.loads((code_dir / filename).read_text()).values())


def test_blind_profiles_record_new_backend_and_preserve_historical_outputs():
    import run_spin23_ssvv_equal_blind as equal
    import run_spin23_ssvv_real_blind as real
    for driver in (equal, real):
        assert driver.DEFAULT_OUTPUT.name.endswith("_c_recursion_elliptic_threshold.json")
        for settings in driver.PROFILES.values():
            assert settings["block_backend"] == "c_recursion"
            assert settings["series_parameter"] in ("sewing", "elliptic_nome")
            assert settings["recursion_digits"] >= 70
            assert not any(k.startswith("gram_") for k in settings)
