"""Principled Weyl and moment-product orderings of the geometric charge I.

Derive finite-cutoff Weyl-to-normal corrections, subtract their analytically
identified conserved divergences, and test the resulting exact low-energy
operators. No coefficient is chosen from a block commutant.
"""
import ast
from collections import Counter
from fractions import Fraction as Q
from itertools import product, permutations
from pathlib import Path
import json
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
namespace={}
for filename,names in (
    ('Codes/mqm_clock_generator_checks.py',{"n","h","h4","basis","generator","ket_index"}),
    ('Codes/mqm_clock_nonlocal_charge_checks.py',
     {"clean","moment","multiply","invariant","wick_matrix","quartic_poly"})):
    tree=ast.parse((ROOT/filename).read_text())
    chosen=ast.Module(body=[node for node in tree.body
        if isinstance(node,(ast.Import,ast.ImportFrom))
        or isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
    exec(compile(ast.fix_missing_locations(chosen),filename+":selected_definitions","exec"),namespace)
basis=namespace["basis"]
generator=namespace["generator"]
ket_index=namespace["ket_index"]
wick_matrix=namespace["wick_matrix"]
quartic_poly=namespace["quartic_poly"]
checks=[]


def truth(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)


def equal(name,a,b=0):
    if s.simplify(a-b)!=0:raise AssertionError((name,s.simplify(a-b)))
    checks.append(name)


def matrix_equal(name,a,b):
    if a.shape!=b.shape or any(s.simplify(v)!=0 for v in a-b):
        raise AssertionError(name)
    checks.append(name)


def H(N):
    return sum((Q(1,m) for m in range(1,N+1)),Q(0))


def rp(r,N):
    """Mr=i**r Rr, where Rr has real rational Fourier coefficients."""
    frequencies=(*range(-N,0),*range(1,N+1))
    out={}
    for xs in product(frequencies,repeat=r):
        y=-sum(xs)
        if y==0 or abs(y)>N:continue
        key=tuple(sorted([(1,k) for k in xs]+[(2,y)]))
        coefficient=Q(1)
        for k in xs:coefficient/=k
        out[key]=out.get(key,Q(0))+coefficient
    return {k:v for k,v in out.items() if v}


def multiply(a,b):
    out={}
    for ka,ca in a.items():
        for kb,cb in b.items():
            key=tuple(sorted(ka+kb))
            out[key]=out.get(key,Q(0))+ca*cb
    return {k:v for k,v in out.items() if v}


def add(a,b,scale=Q(1)):
    out=dict(a)
    for key,value in b.items():out[key]=out.get(key,Q(0))+scale*value
    return {k:v for k,v in out.items() if v}


def invariant(N):
    return add(multiply(rp(1,N),rp(3,N)),
               multiply(rp(2,N),rp(2,N)),-Q(3,4))


def delta(poly):
    """Delta=(1/2) sum_species sum_m m d_(z_m) d_(z_-m)."""
    out={}
    for key,coefficient in poly.items():
        counts=Counter(key)
        for (a,k),count in counts.items():
            if k<0 or not counts.get((a,-k)):continue
            remaining=list(key)
            remaining.remove((a,k))
            remaining.remove((a,-k))
            remaining=tuple(remaining)
            value=coefficient*Q(k,2)*count*counts[(a,-k)]
            out[remaining]=out.get(remaining,Q(0))+value
    return {k:v for k,v in out.items() if v}


def corrections(poly):
    terms=[poly]
    for order in range(1,4):
        terms.append({k:v/order for k,v in delta(terms[-1]).items()})
    return terms


def sympy_poly(poly):
    return {k:s.Rational(v.numerator,v.denominator) for k,v in poly.items()}


def matrix(poly,E):
    return wick_matrix(sympy_poly(poly),3,E,max(1,E))


def area_square_poly(N):
    # M1=i R1, and M1 is the ordinary x-y rotation.
    return {k:-v for k,v in multiply(rp(1,N),rp(1,N)).items()}


def q2_coefficient(a,k,N=None):
    base=3-2*H(k) if a==1 else Q(1,k)-2*H(k)
    if N is not None:base+=H(N)-H(N-k)
    return Q(3,2*k)*base


def renormalized_poly(N,E,finite_tail=False):
    p=invariant(N)
    quartic=add(delta(p),area_square_poly(N),-3*H(N))
    result=add(p,quartic)
    for a in (1,2):
        for k in range(1,E+1):
            coefficient=q2_coefficient(a,k,N if finite_tail else None)
            if coefficient:result[((a,-k),(a,k))]=coefficient
    return result


def symmetrized_word_matrix(poly,E):
    """Independent Weyl definition for the three N=1 degree-six monomials."""
    modes,states=basis(3,E,max(1,E))
    mi={mode:j for j,mode in enumerate(modes)}
    index={state:j for j,state in enumerate(states)}
    out=s.zeros(len(states))
    for key,coefficient in poly.items():
        words=set(permutations(key))
        for word in words:
            for col,state in enumerate(states):
                target=list(state)
                value=s.Rational(coefficient.numerator,coefficient.denominator)/len(words)
                for species,k in reversed(word):
                    j=mi[(abs(k),species)]
                    if k>0:
                        if not target[j]:
                            value=0
                            break
                        value*=s.sqrt(k*target[j])
                        target[j]-=1
                    else:
                        target[j]+=1
                        value*=s.sqrt((-k)*target[j])
                if value:
                    row=index[tuple(target)]
                    out[row,col]+=value
    return out.applyfunc(s.simplify)


# Delta is fixed by Weyl symmetrization, not selected to cancel a commutator.
equal("elementary Weyl pair contributes half a quantum",
      delta({((1,-2),(1,2)):Q(1)})[()],1)
small=invariant(1)
small_terms=corrections(small)
small_weyl={}
for term in small_terms:small_weyl=add(small_weyl,term)
for E in (1,2,3):
    matrix_equal(f"literal all-word Weyl symmetrization E{E}",
                 symmetrized_word_matrix(small,E),matrix(small_weyl,E))

coefficient_records=[]
for N in range(1,11):
    p=invariant(N)
    if N<=4:
        truth(f"rational charge polynomial matches prior construction N{N}",
              sympy_poly(p)==namespace["invariant"](N))
    terms=corrections(p)
    c0=Q(9*N,4)-Q(3,4)*H(N)
    equal(f"exact Weyl vacuum subtraction N{N}",terms[3].get((),Q(0)),c0)
    for k in range(1,N+1):
        # Independent convolution sums from C_N and D_N.
        cc=sum((Q(1,4*abs(p)*abs(k-p))
            for p in range(-N,N+1) if p and k-p and abs(k-p)<=N),Q(0))
        cd=sum((Q(abs(k-p),4*abs(p))
            for p in range(-N,N+1) if p and k-p and abs(k-p)<=N),Q(0))
        cc_closed=Q(1,2*k)*(2*H(k)-Q(1,k)+H(N-k)-H(N))
        cd_closed=Q(1,4)*(2*N-3*k+2*k*H(k)+k*(H(N-k)-H(N)))
        equal(f"C squared convolution N{N} k{k}",cc,cc_closed)
        equal(f"C D convolution N{N} k{k}",cd,cd_closed)
        for a in (1,2):
            coefficient=terms[2][((a,-k),(a,k))]
            predicted=3*H(N)/k+q2_coefficient(a,k,N)
            equal(f"quadratic Weyl coefficient N{N} k{k} species{a}",
                  coefficient,predicted)
    coefficient_records.append({"N":N,"vacuum_constant":str(c0),
                                "quartic_log_coefficient":str(3*H(N))})

block_records=[]
for E in range(1,5):
    G3=generator(3,E,E)
    G4=wick_matrix(quartic_poly(3,E),3,E,E)
    p=invariant(E)
    terms=corrections(p)
    ren=matrix(renormalized_poly(E,E),E)
    matrix_equal(f"renormalized charge Hermitian E{E}",ren,ren.H)
    for N in (E,E+1,E+3):
        termsN=corrections(invariant(N))
        total={}
        for term in termsN:total=add(total,term)
        raw=matrix(total,E)
        rotation=wick_matrix({k:s.I*s.Rational(v.numerator,v.denominator)
                              for k,v in rp(1,N).items()},3,E,E)
        c0=Q(9*N,4)-Q(3,4)*H(N)
        subtracted=raw-s.Rational(c0.numerator,c0.denominator)*s.eye(raw.rows)-3*s.Rational(H(N))*rotation**2
        predicted=matrix(renormalized_poly(N,E,True),E)
        matrix_equal(f"exact finite-cutoff operator subtraction E{E} N{N}",
                     subtracted,predicted)
        matrix_equal(f"renormalized limit independent quartic cutoff E{E} N{N}",
                     ren,matrix(renormalized_poly(N,E),E))
    comm3=G3*ren-ren*G3
    comm4=G4*ren-ren*G4
    record={"E":E,"dimension":ren.rows,
            "renormalized_Weyl_commutes_G3":comm3==s.zeros(ren.rows),
            "renormalized_Weyl_commutes_G4":comm4==s.zeros(ren.rows)}
    if E>=2:
        truth(f"renormalized Weyl charge fails G3 E{E}",comm3!=s.zeros(ren.rows))
        truth(f"renormalized Weyl charge fails G4 E{E}",comm4!=s.zeros(ren.rows))
    # A second specified rule: symmetrized products of ordered moments.
    Ms=[wick_matrix({k:s.I**r*s.Rational(v.numerator,v.denominator)
                     for k,v in rp(r,E).items()},3,E,E) for r in (1,2,3)]
    M1,M2,M3=Ms
    moment_ordered=(M1*M3+M3*M1)/2-s.Rational(3,4)*M2**2
    matrix_equal(f"moment-product charge Hermitian E{E}",
                 moment_ordered,moment_ordered.H)
    moment_comm=G3*moment_ordered-moment_ordered*G3
    record["moment_product_commutes_G3"]=moment_comm==s.zeros(ren.rows)
    if E>=2:truth(f"moment-product charge fails G3 E{E}",moment_comm!=s.zeros(ren.rows))
    if E==2:
        x2=ket_index(3,E,E,[(2,1)])
        y2=ket_index(3,E,E,[(2,2)])
        sx=ket_index(3,E,E,[(1,0),(1,1)])
        sy=ket_index(3,E,E,[(1,0),(1,2)])
        equal("renormalized Weyl E2 x2 to S1x1 defect",
              comm3[sx,x2],-3*s.sqrt(2)/2)
        equal("renormalized Weyl E2 y2 to S1y1 defect",
              comm3[sy,y2],-9*s.sqrt(2)/4)
        equal("moment-product E2 x2 to S1x1 defect",
              moment_comm[sx,x2],-3*s.sqrt(2)/2)
        record["G3_counterexample"]=str(comm3[sx,x2])
    if E==3:
        xxx=ket_index(3,E,E,[(1,1)]*3)
        xxy=ket_index(3,E,E,[(1,1),(1,1),(1,2)])
        s2x=ket_index(3,E,E,[(2,0),(1,1)])
        s2y=ket_index(3,E,E,[(2,0),(1,2)])
        ssy=ket_index(3,E,E,[(1,0),(1,0),(1,2)])
        equal("renormalized Weyl E3 xxx to S2x defect",
              comm3[s2x,xxx],3*s.sqrt(30)/5)
        equal("renormalized Weyl E3 xxy to S2y defect",
              comm3[s2y,xxy],3*s.sqrt(10))
        equal("renormalized Weyl E3 G4 xxy to SSy defect",
              comm4[ssy,xxy],-s.Rational(9,2))
        record["G4_counterexample"]=str(s.simplify(comm4[ssy,xxy]))
    block_records.append(record)

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "finite_cutoff_coefficients":coefficient_records,"blocks":block_records,
        "normalization":"Mr=integral dt/(2pi) x^r y'; elementary current zero modes omitted; f_k=i z_k/k.",
        "analytic_subtractions":"c_N=(9N-3H_N)/4 times identity, plus 3H_N L_xy^2.",
        "limit_quadratic_coefficients":"A_k=3(3-2H_k)/(2k), B_k=3(1/k-2H_k)/(2k); coefficient of z_-k z_k.",
        "scope":"A well-defined renormalized Weyl observable and two failed quantum charge-ordering prescriptions; no no-go for other corrections, no string-charge identification."}
(ROOT/'data_exports/mqm/mqm_clock_weyl_charge_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="named_checks"},indent=2))
