from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Codes"))

from prepare_so7e8_literature_cannon import configuration, tasks
from so7e8_literature_campaign import (
    FrozenBlocks, adjacent_change, block_values, coefficient_table, digest, read_manifest, atomic_json,
)
from literature_self_dual_blocks import ExtrapolatedLiteratureBlocks


def test_scan_budget_sector_thresholds_and_requested_moduli_design():
    config = configuration()
    energies, banks, amplitudes, rules = tasks(config)
    assert len(energies) == 8 and len(banks) == 1024 and len(amplitudes) == 40
    assert config["blocks"]["first_comparison"] == [4,5]
    assert config["blocks"]["maximum_order"] == 10
    assert config["moduli"]["settings"]["z_radial_nodes"] == 24
    assert config["moduli"]["settings"]["z_angular_nodes"] == 60
    for sector,beta in (("NS",2),("R",0)):
        assert len(rules[sector]["momenta"]) == 32
        assert rules[sector]["metadata"]["parameters"]["beta"] == beta
        assert rules[sector]["metadata"]["region_counts"] == dict(endpoint=5,bulk=19,tail=8)
    assert next(t for t in banks if t["observable"] == "mixed")["momenta"][1] == energies["pilot"][3]


def test_bank_roundtrip_preserves_sign_components_and_full_complex_values():
    block = ExtrapolatedLiteratureBlocks("mixed_ns",(.21,.39,.31,.43),.71,4)
    frozen = FrozenBlocks("mixed_ns", block.p, .71, coefficient_table(block))
    points = (.35+.1j,.5-.1j)
    assert block_values(frozen,points) == pytest.approx(block_values(block,points), abs=1e-13)


def test_adjacent_order_check_cannot_hide_a_phase_or_weak_component_change():
    assert adjacent_change([1,-1j],[1,1j])["maximum_relative_change"] == 2
    assert adjacent_change([1,1e-5],[1,2e-5])["maximum_relative_change"] == .5
    assert adjacent_change([0,0],[0,0])["maximum_relative_change"] == 0
    with pytest.raises(ValueError):
        adjacent_change([1],[np.inf])


def test_release_integrity_rejects_changed_numerical_settings(tmp_path):
    manifest = dict(config=configuration(),source_sha256={})
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(tmp_path/"manifest.json", manifest)
    assert read_manifest(tmp_path,check_sources=False)["config"]["momentum"]["nodes_per_channel"] == 32
    manifest["config"]["momentum"]["nodes_per_channel"] = 33
    atomic_json(tmp_path/"manifest.json", manifest)
    with pytest.raises(ValueError,match="integrity"):
        read_manifest(tmp_path,check_sources=False)
