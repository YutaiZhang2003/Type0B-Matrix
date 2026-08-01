#!/usr/bin/env python3
"""Focused checks for the combined fundamental-domain period index."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

try:
    from bolza_torus_plumbing_reach import transform_omega
    from build_genus2_fundamental_period_index import build_combined_index
    from genus2_period_table import Genus2PeriodMapTable, PeriodTableEntry
except ImportError:  # pragma: no cover
    from plumbing.bolza_torus_plumbing_reach import transform_omega
    from plumbing.build_genus2_fundamental_period_index import build_combined_index
    from plumbing.genus2_period_table import Genus2PeriodMapTable, PeriodTableEntry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    omega_fund = np.asarray(
        [[0.10 + 1.20j, 0.05 + 0.15j], [0.05 + 0.15j, -0.20 + 1.45j]],
        dtype=np.complex128,
    )
    raw_to_fund = np.eye(4, dtype=np.int64)
    raw_to_fund[0, 2] = -1
    fund_to_raw = np.eye(4, dtype=np.int64)
    fund_to_raw[0, 2] = 1
    omega_raw = transform_omega(fund_to_raw, omega_fund)
    q = (0.04 + 0.002j, 0.05 - 0.003j, 0.03 + 0.001j)
    entry = PeriodTableEntry(
        row_id="synthetic-theta",
        topology="theta",
        q=q,
        omega=omega_raw,
        actual_backend="holomorphic-form-collocation",
        precision_tier="binary64-adaptive",
        error_estimate=1.0e-12,
        geometry_margin=0.1,
        certified=True,
    )
    with tempfile.TemporaryDirectory(prefix="g2-fundamental-index-") as temporary:
        root = Path(temporary)
        canonical = root / "table_fundamental.csv.gz"
        canonical.write_bytes(b"synthetic canonical table\n")
        base_index = root / "index_features.npz"
        table = Genus2PeriodMapTable([entry], q_abs_min=1.0e-14, q_abs_max=0.8)
        table.write_portable_index(base_index, table_sha256="a" * 64)
        fundamental_index = root / "fundamental_index.npz"
        np.savez_compressed(
            fundamental_index,
            schema_version=np.asarray([1], dtype=np.int64),
            base_table_sha256=np.asarray(["a" * 64]),
            row_ids=np.asarray([entry.row_id]),
            omega_fund=np.asarray(
                [(omega_fund[0, 0], omega_fund[0, 1], omega_fund[1, 1])],
                dtype=np.complex128,
            ),
            sp4_raw_to_fund=np.asarray([raw_to_fund], dtype=np.int64),
            b_period_branches=np.zeros((1, 3), dtype=np.int64),
            domain_margins=np.asarray([0.02]),
            transform_residuals=np.asarray([2.0e-14]),
            correction_indices=np.asarray([0], dtype=np.int64),
            correction_depth_limits=np.asarray([3], dtype=np.int64),
            attempted_depths=np.asarray(["3"]),
        )
        combined = root / "period_query_index.npz"
        summary = build_combined_index(
            base_index, fundamental_index, canonical, combined
        )
        assert summary["row_count"] == 1
        assert summary["canonical_table_sha256"] == _sha256(canonical)
        loaded = Genus2PeriodMapTable.from_portable_index(
            combined, verify_table_path=canonical
        )
        assert loaded.has_fundamental_index
        seeds = loaded.nearest_fundamental_seeds("theta", omega_fund, count=1)
        assert len(seeds) == 1
        seed = seeds[0]
        assert seed.row_id == entry.row_id
        assert seed.table_distance < 1.0e-13
        assert np.max(np.abs(seed.omega_marked - omega_raw)) < 1.0e-13
        assert np.array_equal(np.asarray(seed.matrix_fund_to_raw), fund_to_raw)
        assert max(abs(a - b) for a, b in zip(seed.seed.q, q)) < 1.0e-13

        legacy = Genus2PeriodMapTable.from_portable_index(base_index)
        assert not legacy.has_fundamental_index
        assert legacy.nearest_fundamental_seeds("theta", omega_fund) == ()

        wrong = root / "wrong.csv.gz"
        wrong.write_bytes(b"wrong table\n")
        try:
            Genus2PeriodMapTable.from_portable_index(
                combined, verify_table_path=wrong
            )
        except ValueError:
            pass
        else:
            raise AssertionError("combined index accepted the wrong canonical table")

    print("combined fundamental period-index checks passed")


if __name__ == "__main__":
    run()
