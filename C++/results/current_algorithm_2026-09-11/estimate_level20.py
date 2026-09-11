"""Integer work estimates for the current C++ loops; no numerical block evaluation."""
from collections import Counter
from functools import lru_cache
from math import isqrt
from pathlib import Path
import json

MAX = 20
nulls = [[r*s for r in range(2,n+1) for s in range(1,n//r+1)] for n in range(MAX+1)]
out = list(map(len,nulls))
inc = [sum(n-d != 1 for d in nulls[n]) for n in range(MAX+1)]

@lru_cache(None)
def counts(dim,L,R):
    seeds=weighted=transitions=coefficients=shifts=0
    for a in range(L+1):
        for b in range(L-a+1):
            for c in range((L-a-b if dim==3 else R-a)+1):
                for d in range(1 if dim==3 else min(L-a-b,R-a-c)+1):
                    ns=(a,b,c) if dim==3 else (a,b,c,d)
                    coefficients+=1
                    # Number and first moment of reachable shifts below each coefficient.
                    w=[1 if n==0 else n for n in ns]
                    v=1
                    for x in w: v*=x
                    seeds+=v
                    remaining_sum=0
                    for i,n in enumerate(ns):
                        rem=n*(n+1)//2-(n-1 if n>=1 else 0)
                        prod=1
                        for j,x in enumerate(w):
                            if j!=i: prod*=x
                        remaining_sum+=rem*prod
                    # Positive affine proxy for global arithmetic per seed, not an exact FLOP count.
                    weighted+=v+remaining_sum
                    if 1 in ns: continue
                    shifts+=1
                    incoming=sum(inc[n] for n in ns)+(1 if sum(ns)==0 else 0)
                    if dim==3: outgoing=3*out[L-sum(ns)]
                    else:
                        left=L-a-b-d; right=R-a-c-d
                        outgoing=2*out[min(left,right)]+out[left]+out[right]
                    transitions+=incoming*outgoing
    return dict(coefficients=coefficients,global_seed_terms=seeds,
                weighted_global_work=weighted,forward_transition_terms=transitions,
                shift_states=shifts,shift_membership_tests=coefficients*shifts)

def histogram(N,inserted):
    hist=Counter()
    rl=lambda k:(k*k-1)//8
    r=[k for k in range(-4*N-1,4*N+2,2) if rl(k)<=N]
    for j in range(-isqrt(2*N),isqrt(2*N)+1):
        if inserted:
            # One orientation per unordered middle pair; two Virasoro copies.
            pairs=[(k,k+2) for k in range(-4*N-1,4*N+2,2)]
        else: pairs=[(k,k) for k in r]
        for k,kp in pairs:
            for z in r:
                budget=(2*N-j*j-2*rl(z))//2
                if inserted:
                    L,R=budget-rl(k),budget-rl(kp)
                    if min(L,R)>=0: hist[(L,R)]+=2
                else:
                    L=budget-rl(k)
                    if L>=0: hist[(L,L)]+=2
    return hist

def character(n,steps_b,steps_f,copies=1):
    c=[1]+[0]*n
    for k in steps_b:
        for j in range(k,n+1): c[j]+=c[j-k]
    for _ in range(copies):
        for k in steps_f:
            for j in range(n,k-1,-1): c[j]+=c[j-k]
    return c

part=character(20,range(1,21),[])
pp=[sum(part[j]*part[n-j] for j in range(n+1)) for n in range(21)]
nsdim=character(44,range(2,45,2),range(1,45,2),2)
rdim=character(30,range(1,31),range(1,31),2)

def actions(N,inserted):
    nsmax=2*isqrt(2*N)
    if inserted:
        pairs=[(k,k+2) for k in range(-4*N-1,4*N+2,2)
               if (k*k-1)//8+((k+2)**2-1)//8<=2*N]
        rmax=max(abs(x) for pair in pairs for x in pair)
    else:
        rmax=max(k for k in range(1,4*N+2,2) if (k*k-1)//8<=N)
    records=[]
    for k in range(4,nsmax+1,2):
        columns=pp[k-3]; level2=k*k//4-2
        records.append(dict(sector="NS",n4=k,action="plus",multiplicity=2,
                            target_twice_level=level2,columns=columns,row_bound=nsdim[level2]))
    for k in range(1,rmax+1,2):
        level=(k*k-1)//8
        records.append(dict(sector="R",n4=k,action="minus",multiplicity=4,
                            target_twice_level=2*(level+1),columns=pp[0 if k==1 else k-1]+2,
                            row_bound=2*rdim[level+1]))
        if inserted and k>1:
            records.append(dict(sector="R",n4=k,action="plus",multiplicity=2,
                                target_twice_level=2*(level-1),columns=pp[k-3],
                                row_bound=2*rdim[level-1]))
    return dict(ns_max_n4=nsmax,r_max_n4=rmax,systems=records,
                # Full Fock dimensions bound sparse actual supports.
                dense_qr_proxy=sum(x["multiplicity"]*x["row_bound"]*x["columns"]**2 for x in records),
                column_volume_proxy=sum(x["multiplicity"]*x["row_bound"]*x["columns"] for x in records))

report={"scope":"Structural nonzero-residue counts and full-Fock action bounds, no numerical evaluations",
        "levels":{}}
for N in (10,15,20):
    report["levels"][str(N)]={}
    for mode in ("ordinary","inserted"):
        hist=histogram(N,mode=="inserted")
        total=Counter()
        for (L,R),copies in hist.items():
            for key,value in counts(4 if mode=="inserted" else 3,L,R).items():
                total[key]+=copies*value
        report["levels"][str(N)][mode]=dict(virasoro_engines_before_zero_branch_skips=sum(hist.values()),
                    work=dict(total),actions=actions(N,mode=="inserted"),
                    budgets=[dict(left=k[0],right=k[1],engines=v) for k,v in sorted(hist.items())])

saved=json.loads(Path(__file__).with_name("timings.json").read_text())
report["ccy_calibrations"]={}
report["runtime_estimates"]={}
for mode in ("ordinary","inserted"):
    estimates={}
    for key in ("global_seed_terms","weighted_global_work","forward_transition_terms","shift_membership_tests"):
        values={n:report["levels"][str(n)][mode]["work"][key] for n in (10,15,20)}
        timings={r["level"]:r["timing_seconds"]["ccy"] for r in saved["runs"] if r["mode"]==mode}
        estimates[key]=dict(work=values,seconds20_from15=timings[15]*values[20]/values[15],
                           seconds15_from10=timings[10]*values[15]/values[10])
    report["ccy_calibrations"][mode]=estimates
    times={r["level"]:r["timing_seconds"] for r in saved["runs"] if r["mode"]==mode}
    levels={n:report["levels"][str(n)][mode] for n in (10,15,20)}
    fits={}
    for key in ("global_seed_terms","weighted_global_work","forward_transition_terms","shift_membership_tests"):
        x1=levels[10]["virasoro_engines_before_zero_branch_skips"]
        x2=levels[15]["virasoro_engines_before_zero_branch_skips"]
        y1=levels[10]["work"][key]; y2=levels[15]["work"][key]
        det=x1*y2-x2*y1
        a=(times[10]["ccy"]*y2-times[15]["ccy"]*y1)/det
        b=(x1*times[15]["ccy"]-x2*times[10]["ccy"])/det
        fits[key]=dict(per_engine=a,per_work_unit=b,
                       estimated_ccy20_seconds=a*levels[20]["virasoro_engines_before_zero_branch_skips"]
                                                 +b*levels[20]["work"][key])
    action_proxies={key:times[15]["actions"]*levels[20]["actions"][key]/levels[15]["actions"][key]
                    for key in ("column_volume_proxy","dense_qr_proxy")}
    ccy_values=[x["seconds20_from15"] for x in estimates.values()]+[x["estimated_ccy20_seconds"] for x in fits.values()]
    report["runtime_estimates"][mode]=dict(
        ccy_engine_plus_work_fits=fits,
        action20_seconds_by_proxy=action_proxies,
        stage_model_sum_seconds=[min(ccy_values)+min(action_proxies.values()),
                                 max(ccy_values)+max(action_proxies.values())],
        caveats=["Working precision40digits, same parameters and C++ implementation.",
                 "Counts include structurally reachable paths before parameter-dependent zero residues.",
                 "Fock row dimensions bound actual sparse action supports; action proxies are not timings.",
                 "Two-level fits have no independent high-level calibration; these are planning scenarios, not confidence bounds.",
                 "The largest new action is unmeasured; memory pressure and rank/refinement failures can invalidate its time prediction.",
                 "Stage sum excludes a small additional allowance for outer Ward, products, fermion and restoration.",
                 "No level20 numerical block, action, or PBW computation was performed."])
report["runtime_estimates"]["ordinary"]["rounded_planning_minutes"]=[12,15]
report["runtime_estimates"]["inserted"]["rounded_planning_hours"]=[2.5,4.5]
Path(__file__).with_name("level20_work_estimate.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({n:{mode:{k:v for k,v in data.items() if k not in ("budgets","actions")}
                   for mode,data in row.items()} for n,row in report["levels"].items()},indent=2))
print(json.dumps(report["ccy_calibrations"],indent=2))
print(json.dumps({n:{mode:{k:v for k,v in data["actions"].items() if k!="systems"}
                        for mode,data in row.items()} for n,row in report["levels"].items()},indent=2))
