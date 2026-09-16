"""Fresh, bounded genus-three Mercedes-channel validation."""
from pathlib import Path
import hashlib, json, platform, subprocess, sys, time

D=Path(__file__).resolve().parent
ROOT=D.parents[2]
commands=[[sys.executable,'export_ns.py'],['make'],['./mercedes']]
times=[]
for command in commands:
    start=time.perf_counter()
    if command[0]=='./mercedes':
        with (D/'results.json').open('w') as out, (D/'run.log').open('w') as err:
            run=subprocess.run(command,cwd=D,stdout=out,stderr=err)
    else:
        run=subprocess.run(command,cwd=D)
    times.append(time.perf_counter()-start)
    if run.returncode:
        raise SystemExit(f'{command} failed; inspect saved output')
d=json.loads((D/'results.json').read_text())
assert d['failed_cases']==0 and d['ordinary_cases']==d['inserted_cases']==32
summaries={}
for group in (False,True):
    cases=[c for c in d['case_results'] if c['inserted']==group]
    summaries['inserted' if group else 'ordinary']={
        key:dict(components=sum(c[key]['components'] for c in cases),
                 nonzero_components=sum(c[key]['nonzero_components'] for c in cases),
                 max_absolute=max(c[key]['max_absolute'] for c in cases),
                 max_scaled=max(c[key]['max_scaled'] for c in cases),
                 failed_components=sum(c[key]['failed_components'] for c in cases))
        for key in ('forward','recovery','uninserted_vanishing','physical_sector','unequal_split','spin_evaluations')}
sources=[D/'export_ns.py',D/'run.py',D/'mercedes.cpp',D/'central_data.hpp',D/'Makefile']
sources+=list((ROOT/'C++/include/ramond').glob('*.hpp'))
manifest=dict(passed=True,platform=platform.platform(),commands=commands,
              stage_wall_seconds=dict(zip(('central_export','compile','all_checks'),times)),
              total_wall_seconds=sum(times),arithmetic=d['precision'],
              b='7/5',momenta=['11/23','13/29','17/31','19/37','23/41','29/43'],
              slot_edges=[[0,1,2],[0,3,5],[1,4,3],[2,5,4]],
              halfedge_pairs=[[0,3],[1,6],[2,9],[4,8],[7,11],[10,5]],
              ramond_loop=[3,4,5],split_edge=3,
              intrinsic_primary_parities=[0]*6,
              zero_based_labels=True,diagonal_multidegrees_per_case=216,
              tolerance=1e-8,
              summaries=summaries,
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              result_sha256={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('results.json','coefficients.jsonl','central_manifest.json')},
              scope='All six original edges have independent level <=1. Full inserted test also retains unequal powers on the split pair, each <=1. Virasoro descendants are <=1, so the CCY answer is its exact global term; no higher Kac residues or nonconstant Schottky vacuum factor enters.')
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','result_sha256')},indent=2))
