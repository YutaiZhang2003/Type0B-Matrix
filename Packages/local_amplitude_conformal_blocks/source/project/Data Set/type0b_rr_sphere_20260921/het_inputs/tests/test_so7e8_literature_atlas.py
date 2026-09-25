"""Bounded checks of the atlas transports, branches and convergence policy."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Codes"))

from so7e8_literature_atlas import (
    BlockGeometry, CHARTS, atlas_point_sets, block_grid_values,
    channel_coordinates, geometric_assignment, four_ns_moduli_layout,
    mixed_u_to_base, mixed_channel_integrand, pointwise_adjacent_change,
    select_resolved_channel, overlap_diagnostic, four_ramond_channel_to_base,
)
from so7e8_literature_campaign import FrozenBlocks, unpairs, block_values
from so7e8_literature_amplitude import mixed_candidate_integrand, require_physical_amplitude_ready

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/"data_exports/so7e8_literature_cannon_N32_20260920"


def bank(index):
    path = PARENT/"banks"/f"{index:04d}.json"
    if not path.exists():
        pytest.skip("completed parent bank not present")
    return json.loads(path.read_text())


@pytest.mark.parametrize("index", (7, 39, 71))
def test_vectorized_native_blocks_keep_every_variant_and_both_cut_lips(index):
    record = bank(index)
    task = record["task"]
    p = tuple(unpairs(task["momenta"]))
    points = np.array([.35+.1j, .35-.1j, -.2+.1j, -.2-.1j, 1.2+.1j, 1.2-.1j])
    geometry = BlockGeometry.build(points)
    for pp, table in zip((p, tuple(x.conjugate() for x in p)), unpairs(record["payload"]["coefficients"])):
        blocks = FrozenBlocks(task["family"], pp, task["P"], table)
        expected = block_values(blocks, points)
        assert block_grid_values(blocks, geometry) == pytest.approx(expected, rel=2e-12, abs=1e-13)


def test_three_puncture_coverage_and_preserved_analytic_disks():
    settings = json.loads((PARENT/"manifest.json").read_text())["config"]["moduli"]["settings"]
    points, report = atlas_point_sets(settings)
    for observable in CHARTS:
        assert report[observable]["maximum_selected_nome"] < .066
        assert all(report[observable]["channel_point_counts"].values())
    assert set(points) == {o+"/"+c for o in CHARTS for c in ("s", "t", "u")}
    grids, disks = four_ns_moduli_layout(settings)
    for patch in ("bulk", "lens"):
        x, w = grids["original_"+patch]
        y, v = grids["exterior_"+patch]
        assert y == pytest.approx(1/x)
        assert v == pytest.approx(w/abs(x)**4)
    assert len(disks) == 2 and all(d["total_order"] == 17 for d in disks)
    z = np.array([.001+.0001j, .999+.0001j, 1000+.1j])
    assert list(geometric_assignment(z, "mixed")[0]) == [0, 1, 2]
    assert list(geometric_assignment(z, "rrrr")[0]) == [0, 1, 2]
    with pytest.raises(ValueError, match="cut lip"):
        channel_coordinates(1.2, "mixed")


def test_unused_unresolved_channel_does_not_veto_a_resolved_local_channel():
    diagnostics = dict(s=dict(block_converged=True, order=5),
                       t=dict(block_converged=False, order=10))
    chosen = select_resolved_channel(.2+.1j, "mixed", diagnostics)
    assert chosen["channel"] == "s" and chosen["block_converged"]
    overlap = overlap_diagnostic(dict(block_converged=True, value=1),
                                 dict(block_converged=False, value=100))
    assert not overlap["compared"]
    mismatch = overlap_diagnostic(dict(block_converged=True, value=1),
                                  dict(block_converged=True, value=100))
    assert mismatch["status"] == "needs_spectral_or_convention_diagnosis"
    capped = select_resolved_channel(.2+.1j, "mixed", dict(s=dict(block_converged=False, order=10)))
    assert capped["status"] == "unresolved" and not capped["block_converged"]
    # A small geometric nome also cannot substitute for the actual order check.
    assert not select_resolved_channel(.001+.0001j, "mixed", {})["block_converged"]


def test_other_moduli_cannot_hide_local_truncation_errors():
    a = np.array([[1e20, 1e-20], [1e20, 2e-20]], complex)
    b = np.array([[1e20, 2e-20], [1e20, 2e-20]], complex)
    assert pointwise_adjacent_change(a, b) == pytest.approx([0, .5])


def test_regional_release_does_not_require_every_channel_at_every_point():
    report = dict(mixed_physical_dictionary_verified=True,
        four_ramond_physical_dictionary_verified=True, absolute_normalization_verified=True,
        moduli_atlas_complete=True, moduli_grid_adjacent_orders_pass=True,
        resolved_overlaps_consistent=True, analytic_disks_checked=True,
        assembled_crossing_pass=False)
    require_physical_amplitude_ready(report)
    report["moduli_atlas_complete"] = False
    with pytest.raises(ValueError, match="moduli_atlas_complete"):
        require_physical_amplitude_ready(report)


def test_mixed_ns_exchange_mobius_factor_pictures_and_oriented_tensor():
    # Independently recompute the swapped external order at fixed P. The
    # same NS propagates here, so this is a legitimate fixed-P covariance
    # test, not a claim of crossing between different internal channels.
    from literature_self_dual_correlator import AnalyticSewing, SelfDualConstants
    from literature_self_dual_blocks import analytic_antiholomorphic
    from literature_component_blocks import sewn_integrand
    record = bank(7)
    task = record["task"]
    p, P = tuple(unpairs(task["momenta"])), task["P"]
    table = unpairs(record["payload"]["coefficients"])[..., :5]
    blocks = FrozenBlocks("mixed_ns", p, P, table[0])
    dual = FrozenBlocks("mixed_ns", tuple(x.conjugate() for x in p), P, table[1])
    constants = SelfDualConstants(50)
    swapped = AnalyticSewing("mixed_ns", (p[0], p[1], p[3], p[2]), P, 4, constants=constants)
    times = (p[0], -p[1], p[2], p[3])
    swapped_times = (times[0], times[1], times[3], times[2])
    for z in (.35+.1j, .35-.1j):
        v = z/(z-1)
        for species in ("SS", "SV", "VS", "VV"):
            for picture in ("one", "infinity"):
                a = mixed_candidate_integrand(z, species, times, lambda e:
                    sewn_integrand(blocks, constants, P, z, e,
                                   antiholomorphic_value=analytic_antiholomorphic(dual)), picture=picture)
                b = mixed_candidate_integrand(v, species[::-1], swapped_times,
                    lambda e: swapped.value(v, e), picture="infinity" if picture == "one" else "one")
                assert mixed_u_to_base(b, z, species) == pytest.approx(a, rel=3e-7, abs=1e-13)


def test_four_ramond_exchanges_square_to_identity_with_the_area_weight():
    value = np.array([1+.2j, 2-.3j, -.6+1j, .7-.4j])
    z = 2.3+.7j
    for channel, changed in (("t", 1-z), ("u", 1/z)):
        other = four_ramond_channel_to_base(value, z, channel)
        restored = four_ramond_channel_to_base(other, changed, channel)
        assert restored == pytest.approx(value, rel=2e-14, abs=2e-14)


def test_u_callback_retains_small_coordinate_and_swapped_component_sectors():
    z = 12+.3j
    times = (.02+.22j, -.09-.69j, .04+.24j, .03+.23j)
    seen = []
    def native(w, e):
        seen.append((w, e))
        return 1+.3j
    mixed_channel_integrand(z, "SV", times, "u", native)
    assert seen and all(w == 1/(1-z) for w, e in seen)
    assert all(isinstance(e[0], tuple) and isinstance(e[3], tuple)
               and isinstance(e[1], int) and isinstance(e[2], int) for w, e in seen)
    assert all(w.imag > 0 for w, e in seen)


def test_real_tail_instability_is_resolved_by_a_whole_alternate_channel():
    from resolve_so7e8_literature_atlas import channel_order_errors
    manifest = json.loads((PARENT/"manifest.json").read_text())
    z = np.array([.46214127820568157-.05960060208205986j,
                  .46214127820568157+.05960060208205986j])
    errors = {}
    for channel in ("s", "t"):
        records = [(task, bank(j)["payload"]) for j, task in enumerate(manifest["bank_tasks"])
                   if (task["energy"], task["observable"], task["channel"]) == ("new3", "rrrr", channel)]
        errors[channel], orders = channel_order_errors(records, z if channel == "s" else 1-z)
        assert len(orders) == 32 and set(orders) == {5}
        with pytest.raises(ValueError, match="32 distinct"):
            channel_order_errors(records[:-1], z)
    assert min(errors["s"]) > .02
    assert max(errors["t"]) < .0005


def test_regional_resolver_switches_coordinates_without_dropping_a_point(monkeypatch):
    import resolve_so7e8_literature_atlas as resolver
    points = {"rrrr/s": np.array([.46+.06j]), "rrrr/t": np.array([.2+.1j]),
              "rrrr/u": np.array([.1+.1j])}
    def checked(records, coordinates, **kwargs):
        errors = np.full(len(coordinates), .03 if records == "s" else .001)
        return errors, [5]*32
    monkeypatch.setattr(resolver, "channel_order_errors", checked)
    report = resolver.resolve_energy("rrrr", points, {c: c for c in ("s", "t", "u")})
    assert report["complete_order_resolved_coverage"]
    assert len(report["overrides"]) == 1 and not report["unresolved"]
    change = report["overrides"][0]
    assert change["original_channel"] == "s" and change["selected_channel"] == "t"
    assert complex(*change["selected_native_coordinate"]) == pytest.approx(.54-.06j)
    assert change["selected_change"] == .001
