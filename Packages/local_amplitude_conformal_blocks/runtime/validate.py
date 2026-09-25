"""Scientific regression checks, run in isolated processes to keep frames separate."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def validate():
    reports={}
    env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    for case in ('sphere_banks','ns_crossing','torus_ns','torus_r'):
        output=subprocess.check_output([sys.executable,'-I','-B',str(ROOT/'tests/check_science.py'),case],env=env,cwd=ROOT,text=True)
        reports[case]=json.loads(output)
    return dict(kind='validation',passed=True,checks=reports)
