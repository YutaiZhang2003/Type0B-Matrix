#!/usr/bin/env python3
"""Launch the C++ partition pipeline in the paper's F, C_a, C_{f,eta} notation.

Arguments are passed unchanged to bin/partition. Convert historical frozen
inputs first with prepare_partition_inputs.py. No legacy sewing is selected.
"""
from pathlib import Path
import subprocess
import sys

BINARY=Path(__file__).resolve().parents[1]/'bin'/'partition'
if __name__=='__main__':
    if not BINARY.is_file():
        sys.exit('Build first: make -C C++ partition')
    sys.exit(subprocess.call([str(BINARY),*sys.argv[1:]]))
