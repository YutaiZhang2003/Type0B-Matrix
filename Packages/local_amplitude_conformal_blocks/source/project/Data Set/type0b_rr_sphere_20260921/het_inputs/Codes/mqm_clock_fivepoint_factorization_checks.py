"""Compare clock five-point predictions with existing wall-on pole data.

The repository does not yet certify the integrated five-point regular
part. This script compares the actual clock with lower-point sewing and
the explicitly empirical contact-degree ansatz; it does not fit a clock
interaction to either input. User-owned source/data files are read only.
"""
import ast
from itertools import combinations
from pathlib import Path
import json
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
source=ast.parse((ROOT/'Codes/mqm_clock_quintic_generator_checks.py').read_text())
names={"add","multiply","scale","clock_F4_over_w"}
definitions=ast.Module(body=[
    node for node in source.body
    if isinstance(node,(ast.Import,ast.ImportFrom))
    or isinstance(node,ast.FunctionDef) and node.name in names
],type_ignores=[])
namespace={"I":s.I,"D":lambda k:1+s.I*k}
exec(compile(ast.fix_missing_locations(definitions),"quintic_source_definitions","exec"),namespace)
clock=namespace["clock_F4_over_w"]
x=s.symbols("x1 x2 x3 x4",real=True)
X=sum(x)
checks=[]


def equal(name,a,b=0):
    difference=s.cancel(a-b)
    if difference!=0:
        raise AssertionError((name,s.factor(difference)))
    checks.append(name)


def truth(name,value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def clock_from_sources(incoming,outgoing,values):
    """Independent nilpotent-source coefficient at exact momenta."""
    energies=tuple(-s.I*t for t in values)
    E=sum(energies)
    result=-s.I*clock(incoming,E,tuple(zip(outgoing,energies)))/s.sqrt(2)
    result*=s.prod(1+t for a,t in zip(outgoing,values) if a==0)
    return s.cancel(result)


def elementary(degree,values=x):
    return sum(s.prod(values[j] for j in selected)
               for selected in combinations(range(len(values)),degree))


def clock_reduced(incoming,outgoing):
    """Compact direct-source formulas, avoiding huge redundant expansion."""
    if incoming==0 and outgoing==(0,0,0,0):
        pairs=tuple(combinations(range(4),2))
        return (X+1)**2*(
            2*X**2-X-1-elementary(2)
            -X*sum(x[i]*x[j]/(1+x[i]+x[j]) for i,j in pairs)
            +s.prod(x)*sum(
                1/((1+x[0]+x[j])*(1+sum(x[k] for k in range(1,4) if k!=j)))
                for j in range(1,4)))
    if incoming==0 and outgoing==(0,0,1,1):
        a,b=x[0]+x[1],x[2]+x[3]
        return (X+1)**2*(-1-a-2*b+x[0]*x[1]/(1+a))/(1+b)
    if incoming==0 and outgoing==(1,1,2,2):
        return (X+1)**2/((1+x[0]+x[1])*(1+x[2]+x[3]))
    if incoming==1 and outgoing==(1,0,0,0):
        return (X+1)*(-1-2*X+sum(x[i]*x[j]/(1+x[i]+x[j])
                                for i,j in combinations(range(1,4),2)))
    if incoming==1 and outgoing==(1,2,2,0):
        return (X+1)/(1+x[1]+x[2])
    raise ValueError((incoming,outgoing))


def predicted_contact(incoming,outgoing):
    if incoming==0 and outgoing==(0,0,0,0):
        return ((X+1)*(2*X+1)*(X*X-1-elementary(2))
                -4*X*elementary(3)-2*elementary(4))
    if incoming==0 and outgoing==(0,0,1,1):
        a,b=x[0]+x[1],x[2]+x[3]
        return -4*a*a-5*a*b-3*a-2*b*b-3*b+2*x[0]*x[1]-1
    if incoming==0 and outgoing==(1,1,2,2):
        return s.Integer(2)
    if incoming==1 and outgoing==(1,0,0,0):
        return -(X+1)*(2*X+1)+elementary(2,x[1:])
    if incoming==1 and outgoing==(1,2,2,0):
        return s.Integer(1)
    raise ValueError((incoming,outgoing))


def pole_part(incoming,outgoing):
    """Pole part after division by 2*i*lambda*pi^2*X*prod x."""
    result=0
    pairs=[]
    for selected in combinations(range(4),2):
        i,j=selected
        if outgoing[i]!=outgoing[j]:
            continue
        rest=tuple(k for k in range(4) if k not in selected)
        k,l=rest
        if incoming!=0 and sorted((outgoing[k],outgoing[l]))!=[0,incoming]:
            continue
        pairs.append(selected)
        da=1+x[i]+x[j]
        eb=x[i]*x[j] if outgoing[i]==0 else 1
        y=x[k]+x[l]
        if incoming==0:
            truth("incoming-singlet allowed complement "+str((outgoing,selected)),
                  outgoing[k]==outgoing[l])
            if outgoing[k]==0:
                ward=x[k]**2+x[k]*x[l]+x[l]**2-1
                result-=eb*y*y*ward/((1+y)*da)
            else:
                result+=eb*y*y/((1+y)*da)
        else:
            result+=eb*y/da
    for first,second in combinations(pairs,2):
        if set(first)&set(second):
            continue
        ea=s.prod(x[j] for j in first) if outgoing[first[0]]==0 else 1
        eb=s.prod(x[j] for j in second) if outgoing[second[0]]==0 else 1
        result-=ea*eb/((1+sum(x[j] for j in first))*(1+sum(x[j] for j in second)))
    return result,pairs


cases=(
    ("S_to_SSSS",0,(0,0,0,0),4),
    ("S_to_SSVV",0,(0,0,1,1),2),
    ("S_to_VVVV_12_34",0,(1,1,2,2),0),
    ("V_to_VSSS",1,(1,0,0,0),2),
    ("V_to_VVVS_23",1,(1,2,2,0),0),
)
records={}
for name,incoming,outgoing,degree_cap in cases:
    actual=clock_reduced(incoming,outgoing)
    pole,pairs=pole_part(incoming,outgoing)
    contact=predicted_contact(incoming,outgoing)
    numerator,denominator=s.fraction(contact)
    truth("pole subtraction leaves a polynomial "+name,denominator==1)
    degree=s.Poly(contact,*x).total_degree()
    truth("empirical regular-degree bound "+name,degree<=degree_cap)
    if name=="S_to_SSSS":
        # Cancel each disjoint pair partition separately, then check the
        # remaining polynomial. This is exactly the full identity and avoids
        # expanding a needlessly enormous common rational denominator.
        polynomial=(X+1)**2*(2*X**2-X-1-elementary(2))
        for second in range(1,4):
            ia=(0,second)
            ib=tuple(j for j in range(1,4) if j!=second)
            aa=sum(x[j] for j in ia)
            bb=sum(x[j] for j in ib)
            ua=s.prod(x[j] for j in ia)
            ub=s.prod(x[j] for j in ib)
            ca=(X+1)**2*(-X*(ua/(1+aa)+ub/(1+bb))+ua*ub/((1+aa)*(1+bb)))
            pa=(-ua*bb**2*(bb**2-ub-1)-ub*aa**2*(aa**2-ua-1)-ua*ub)/((1+aa)*(1+bb))
            qa=ua*(-3*X**2+X*(3*aa-1)-aa*aa)
            qb=ub*(-3*X**2+X*(3*bb-1)-bb*bb)
            equal("all-singlet pair-partition rational identity "+str(second),
                  ca-pa,qa+qb+2*ua*ub)
            polynomial+=qa+qb+2*ua*ub
        equal("all-singlet polynomial remainder identity",polynomial,contact)
    else:
        equal("all pair and double residues with one common normalization "+name,
              actual,pole+contact)
    # The formula only has pair divisors; no independent triple denominator.
    reconstructed=1
    for selected in pairs:
        reconstructed*=1+sum(x[j] for j in selected)
    # Inspect the explicitly generated rational terms: all denominator
    # factors have degree one and are one of the allowed pair divisors.
    allowed={s.expand(1+sum(x[j] for j in pair)) for pair in pairs}
    denominator_factors=[s.expand(node.base) for node in s.preorder_traversal(actual)
                         if isinstance(node,s.Pow) and node.exp.is_negative]
    truth("no triple or other denominator "+name,
          all(factor in allowed for factor in denominator_factors))
    for values in ((s.Rational(1,5),s.Rational(2,7),s.Rational(3,8),s.Rational(4,9)),
                   (-s.Rational(1,8),-s.Rational(3,5),-s.Rational(5,8),-s.Rational(13,20)),
                   (s.Rational(2,3),-s.Rational(4,5),s.Rational(7,6),s.Rational(3,4))):
        equal("independent direct clock source "+str((name,values)),
              clock_from_sources(incoming,outgoing,values),actual.subs(dict(zip(x,values))))
    records[name]={"clock_reduced":str(actual),
                   "pole_part":str(pole),"contact_polynomial":str(s.factor(contact)),
                   "contact_degree":degree,"allowed_degree":degree_cap,
                   "nonzero_pair_labels":[[j+1 for j in pair] for pair in pairs]}
    print("Verified",name,"with regular degree",degree,flush=True)

a=x[0]+x[1]
b=x[2]+x[3]
equal("S four-vector closed clock formula",
      clock_reduced(0,(1,1,2,2)),(1+X)**2/((1+a)*(1+b)))
equal("S four-vector predicted contact",
      s.sympify(records["S_to_VVVV_12_34"]["contact_polynomial"]),2)
equal("V four-vector predicted contact",
      s.sympify(records["V_to_VVVS_23"]["contact_polynomial"]),1)

# First wall screening plane X=-2: this is a predicted value, not a
# comparison with the incomplete separated-screen or fixed-moduli pilots.
first_resonance=s.factor(((1+X)**2/((1+a)*(1+b))).subs(x[3],-2-x[0]-x[1]-x[2]))
equal("S four-vector first-resonance prediction",first_resonance,-1/(1+a)**2)
sample=tuple(map(s.Rational,("-0.125","-0.600","-0.625","-0.650")))
sample_subs=dict(zip(x,sample))
sample_U=sum(sample)*s.prod(sample)
sample_clock=s.factor(2*s.I*s.pi**2*sample_U*
                       clock_reduced(0,(1,1,2,2)).subs(sample_subs))
sample_pole=s.factor(2*s.I*s.pi**2*sample_U*
                      pole_part(0,(1,1,2,2))[0].subs(sample_subs))
equal("sample full clock differs from pole by predicted contact",
      sample_clock-sample_pole,4*s.I*s.pi**2*sample_U)

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "normalization":"x=i*omega, X=sum x, U=X*prod x. Clock A5/(sqrt(2)*U)=P. One common comparison factor M5=i*sqrt(2)*lambda*pi^2*A5 gives M5/(2*i*lambda*pi^2*U)=P.",
        "master_sectors":records,
        "lower_data_scope":"Incoming-singlet pole constraints use stipulated exact lower amplitudes. Incoming-vector constraints additionally assume the repository's numerically supported fixed-incoming V-to-VSS candidate.",
        "first_resonance_S_VVVV_reduced":str(first_resonance),
        "sample_at_P_out_i_times_0125_0600_0625_0650":{
            "factorization_scale":"lambda=1",
            "clock_full_prediction":str(sample_clock),
            "clock_full_prediction_numeric":str(s.N(sample_clock,17)),
            "repository_pole_part":str(sample_pole),
            "repository_pole_part_numeric":str(s.N(sample_pole,17))},
        "scope":"All fixed-incoming clock master sectors satisfy existing lower-point pole data and the empirical contact-degree ansatz. The regular polynomials are clock predictions; no certified integrated worldsheet five-point datum is used."}
(ROOT/'data_exports/mqm/mqm_clock_fivepoint_factorization_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("named_checks","master_sectors")},indent=2))
