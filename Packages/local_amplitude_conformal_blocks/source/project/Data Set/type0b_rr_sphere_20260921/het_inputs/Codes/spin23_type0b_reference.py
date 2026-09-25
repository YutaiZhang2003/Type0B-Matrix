"""Locate the pinned public recursion dependency without copying its code."""
import os
from pathlib import Path
import subprocess
import sys

REFERENCE_COMMIT='e79337838d47bd38589a7e2561252b81b24de04b'


def reference_root():
    persistent=Path(__file__).resolve().parents[1]/'data_exports/references/Type0B-Matrix'
    root=Path(os.environ.get('SPIN23_TYPE0B_REFERENCE',
        str(persistent) if persistent.exists() else
        '/private/tmp/heterotic_genus1_type0b_reference')).resolve()
    if not (root/'Code/c_Recursion/ns_multipoint_c_recursion.py').exists():
        raise FileNotFoundError('Set SPIN23_TYPE0B_REFERENCE to the Type0B-Matrix checkout')
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=REFERENCE_COMMIT:raise RuntimeError('The recursion reference revision is not the validated pin')
    return root


def enable_reference_imports():
    root=reference_root()
    for directory in ('Code/h_recursion','Code/c_Recursion'):
        path=str(root/directory)
        if path not in sys.path:sys.path.append(path)
    return root
