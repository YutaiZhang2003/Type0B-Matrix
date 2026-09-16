from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/'Code/c_Recursion'),str(root/'Code/genus_2_cross_channel')]
from ns_genus12_finite_c_check import DirectThetaOracle
b=7/5;q=b+1/b;p=(11/23,13/29,17/31);h=tuple(q*q/8-x*x/2 for x in p)
engine=DirectThetaOracle(c=1.5+3*q*q,weights=h)
start=time.perf_counter();errors=[]
folder=root/'C++/experiments/total_level6_2026-09-16'
for line in (folder/'theta_ns_L6_coefficients.jsonl').read_text().splitlines():
 row=json.loads(line); f=row['case']; k=tuple(row['level2'])
 actual=sum((complex(float(x[1]),float(x[2])) for x in row['physical']),0j)
 expected=engine.coefficient(twice_levels=k,sectors=(f,f))
 errors.append((abs(actual-expected)/max(1,abs(actual),abs(expected)),k,f))
result={'comparison':'C++ NS physical PBW versus independent existing Python DirectThetaOracle','total_level':6,'components':len(errors),'reference_arithmetic':'machine complex; C++ coefficients computed at 40 digits','tolerance':1e-9,'max_scaled':max(errors)[0],'worst':max(errors)[1:],'runtime_seconds':time.perf_counter()-start,'passed':max(errors)[0]<1e-9}
(folder/'ns_physical_port_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
