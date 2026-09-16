"""Fresh complete total-level-six validation; no previously saved coefficients loaded."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
def main():
    start=time.perf_counter();stages=[]
    commands=[('build',['make','all']),('theta_nsrr',[sys.executable,'theta_tests.py'])]
    commands += [(name,['./graph_validate',name,'6']) for name in ['theta_ns','glasses_rr','glasses_nr','glasses_nn','mercedes']]
    commands += [('ns_physical_port',[sys.executable,'ns_physical_port_check.py']),
                 ('coverage_and_sector',[sys.executable,'audit_saved_results.py'])]
    for name,command in commands:
        t=time.perf_counter()
        with (ROOT/(name+'_fresh.log')).open('w') as log:
            run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        stages.append(dict(name=name,command=command,wall_seconds=time.perf_counter()-t,exit_code=run.returncode))
        manifest=dict(total_level=6,dps=40,platform=platform.platform(),stages=stages,
                      wall_seconds=time.perf_counter()-start,passed=all(x['exit_code']==0 for x in stages),
                      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in ROOT.iterdir() if p.suffix in ['.hpp','.cpp','.inc','.py'] or p.name in ['Makefile','theta_makefile']})
        (ROOT/'fresh_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,stages[-1],flush=True)
        if run.returncode:raise SystemExit(run.returncode)
if __name__=='__main__':main()
