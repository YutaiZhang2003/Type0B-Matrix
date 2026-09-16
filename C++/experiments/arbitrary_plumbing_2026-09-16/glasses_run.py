#!/usr/bin/env python3
"""Build and rerun the directed low-level glasses tests; no expensive run."""
from pathlib import Path
import subprocess,sys,json,hashlib,time,platform
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
command=['clang++','-O1','-std=c++17','-I',str(ROOT/'C++/include'),'-I/opt/homebrew/include']
link=['-L/opt/homebrew/lib','-lmpc','-lmpfr','-lgmpxx','-lgmp']
link+=['-framework','Accelerate'] if sys.platform=='darwin' else ['-llapack','-lblas']
start=time.perf_counter();commands=[]
for name,out in [('glasses_diagonal','glasses_results.json'),('glasses_full_split','glasses_full_split_results.json'),('glasses_export','glasses_ramond_local_data.json'),('glasses_fermion_export','glasses_fermion_loop_data.json')]:
 cmd=command+[str(D/(name+'.cpp'))]+link+['-o',str(D/name)]
 subprocess.run(cmd,check=True);commands.append(cmd)
 with (D/out).open('w') as f:subprocess.run([str(D/name)],stdout=f,check=True)
cmd=[sys.executable,str(D/'glasses_ns_assignments.py')];subprocess.run(cmd,check=True);commands.append(cmd)
results=[json.loads((D/n).read_text()) for n in ['glasses_results.json','glasses_full_split_results.json','glasses_ns_assignments_results.json']]
assert all(r['failed_cases']==0 for r in results)
sources=sorted(D.glob('glasses*.cpp'))+sorted(D.glob('glasses*.py'))+sorted((ROOT/'C++/include/ramond').glob('*.hpp'))
sources += [ROOT/p for p in ['Code/c_Recursion/ns_genus12_finite_c_check.py','Code/c_Recursion/mixed_ns_ramond_descendant_blocks.py','Code/double_virasoro/all_ns/two_virasoro_fusion.py','Code/genus_2_cross_channel/free_majorana_pair_of_pants.py']]
manifest={'platform':platform.platform(),'build_and_tests_seconds':time.perf_counter()-start,'commands':commands,'sources_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'failed_cases':0,'R_R_diagonal_components':sum(c['components'] for c in results[0]['cases']),'R_R_full_split_components':sum(c['components'] for c in results[1]['cases']),'NS_NS_and_NS_R_components':sum(c['components'] for c in results[2]['cases'])}
(D/'glasses_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['commands','sources_sha256']},indent=2))
