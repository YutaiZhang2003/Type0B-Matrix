# Legacy direct-chi anchor (inactive)

This directory is intentionally outside every active Python path.

The preserved `DirectHalfNSAnchor` agrees with Yuchen's production anchors at
the integral NS boundary but not at the half-integral boundaries. Across the
low boundary matrix, the former comparison found 32 mismatches out of 96 for
each value of `p1`; the disagreement was entirely at `n1=+/-1/2`.

The files are historical snapshots. The active boundary source is
`Code/ramond_branching_recursion/direct_state_check.py` together with
`Code/full_ramond_block_runtime/compute_full_block.py::BranchingGrid`.
