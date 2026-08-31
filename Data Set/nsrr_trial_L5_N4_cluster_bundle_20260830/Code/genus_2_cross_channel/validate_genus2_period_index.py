#!/usr/bin/env python3
"""Fail-fast validation for the production genus-two period-table index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from genus2_period_table import Genus2PeriodMapTable
except ImportError:  # pragma: no cover
    from plumbing.genus2_period_table import Genus2PeriodMapTable


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--table-csv", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    table = Genus2PeriodMapTable.from_portable_index(
        args.index,
        verify_table_path=args.table_csv,
    )
    if not table.has_fundamental_index:
        raise ValueError(
            "production requires a schema-v3 period index carrying fundamental-domain "
            "Omega coordinates and exact Sp(4,Z) markings"
        )
    topologies = {entry.topology for entry in table.entries}
    if topologies != {"theta", "glasses"}:
        raise ValueError(f"production period index lacks a topology: {sorted(topologies)}")
    print(
        f"validated schema-v3 fundamental period index: {len(table.entries)} rows, "
        "theta+glasses"
    )


if __name__ == "__main__":
    run()
