#!/usr/bin/env python3
"""Entry point to the existing C++-backed, bilinear frozen-grid assembler.

Arguments are those of Code/genus_2/run_nsrr_resummed_frozen_grid.py. Input
bundles contain the momentum rule, geometry, free factors and saved NS target.
Physical tube signs and the normalization policy remain explicit inputs.
"""
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[2]
for relative in ("Code/genus_2", "Code/full_ramond_block_runtime", "Code/c_Recursion", "Code/genus_2_cross_channel"):
    sys.path.insert(0, str(REPO / relative))

if __name__ == "__main__":
    runpy.run_path(str(REPO / "Code/genus_2/run_nsrr_resummed_frozen_grid.py"), run_name="__main__")
