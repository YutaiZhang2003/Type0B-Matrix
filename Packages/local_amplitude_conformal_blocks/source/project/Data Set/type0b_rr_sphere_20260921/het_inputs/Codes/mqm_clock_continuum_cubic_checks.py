"""Checks supporting an intrinsic noncompact-Fock bound for clock G3.

The proof is the continuum weighted Schur estimate in the companion memo;
these tests check kernel inequalities, all-state lattice combinatorics and
the physical circle normalization. They are not a finite-mode extrapolation.
"""
import ast
from fractions import Fraction
from pathlib import Path
import json
import math
import numpy as np
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'Codes/mqm_clock_generator_checks.py').read_text())
ns={}
chosen=ast.Module(body=[node for node in tree.body
    if isinstance(node,(ast.Import,ast.ImportFrom))
    or isinstance(node,ast.FunctionDef) and node.name in {"basis","n","h"}],
    type_ignores=[])
exec(compile(ast.fix_missing_locations(chosen),"cubic_basis_definitions","exec"),ns)
basis=ns["basis"]
h=ns["h"]
checks=[]


def equal(name,a,b=0):
    if s.simplify(a-b)!=0:raise AssertionError((name,s.simplify(a-b)))
    checks.append(name)


def truth(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)


x,y=s.symbols("x y",positive=True)
w=x+y
P=1+x*x+y*y+x*y
equal("global squared SSS kernel inequality identity",
      (1+x*x)*(1+y*y)*(1+w*w)-P*P,x*x*y*y*w*w)
equal("mixed kernel squared deficit",1-1/(1+x*x),x*x/(1+x*x))
# Use a fixed parent energy and an independent integration variable.
t,W=s.symbols("t W",positive=True)
equal("fixed-parent weighted join integral",
      s.integrate(W/s.Integer(2),(t,0,W)),W*W/2)
for n in range(1,7):
    ks=s.symbols("k0:"+str(n),positive=True)
    equal(f"row rate equals half total energy squared n{n}",
          sum(k*k for k in ks)/2+
          sum(ks[i]*ks[j] for i in range(n) for j in range(i+1,n)),
          sum(ks)**2/2)


def block(d,K,delta=s.Integer(1),dominating=True):
    modes,states=basis(d,K,max(K,1))
    mi={mode:j for j,mode in enumerate(modes)}
    si={state:j for j,state in enumerate(states)}
    terms={}
    for parent in range(2,K+1):
        for p in range(1,parent):
            q=parent-p
            flavors=[(0,0,0)]
            for a in range(1,d):
                flavors.extend(((0,a,a),(a,0,a),(a,a,0)))
            for I,J,L in flavors:
                tensor=s.sqrt(2) if dominating else h(((I,-delta*parent),(J,delta*p),(L,delta*q)))
                coefficient=delta**2*tensor*s.sqrt(parent*p*q)/2
                key=(mi[(parent,I)],tuple(sorted((mi[(p,J)],mi[(q,L)]))))
                terms[key]=s.simplify(terms.get(key,0)+coefficient)
    entries={}
    for col,state in enumerate(states):
        for (ann,cre),coefficient in terms.items():
            if state[ann]==0:continue
            target=list(state)
            factor=s.sqrt(target[ann])
            target[ann]-=1
            for j in cre:
                target[j]+=1
                factor*=s.sqrt(target[j])
            row=si[tuple(target)]
            value=s.simplify(coefficient*factor)
            entries[row,col]=entries.get((row,col),0)+value
            entries[col,row]=entries.get((col,row),0)+value
    return s.SparseMatrix(len(states),len(states),entries)


block_records=[]
for d,maximum in ((1,7),(2,6),(3,5),(24,2)):
    for K in range(1,maximum+1):
        G=block(d,K)
        modes,states=basis(d,K,K)
        weights=s.Matrix([1/s.sqrt(s.Integer(d)**sum(state)*
             s.prod(s.Integer(m)**occ*s.factorial(occ)
                    for (m,a),occ in zip(modes,state))) for state in states])
        action=G*weights
        rate_cap=s.Rational(K*(K-1),2)
        margins=[]
        for row,state in enumerate(states):
            particles=[(m,a) for (m,a),occ in zip(modes,state) for _ in range(occ)]
            expected=sum(s.Rational(m*(m-1),2) if a==0
                         else s.Rational(m*(m-1),d) for m,a in particles)
            expected+=sum(m*n for i,(m,a) in enumerate(particles)
                          for n,b in particles[i+1:] if a==0 or b==0 or a==b)
            actual=s.simplify(action[row]/weights[row]/s.sqrt(2*d))
            equal(f"all-state Schur row d{d} K{K} row{row}",actual,expected)
            margin=rate_cap-expected
            truth(f"all-state nonnegative Schur margin d{d} K{K} row{row}",margin>=0)
            margins.append(margin)
        if d<=2:
            truth(f"scalar/two-species dominating rows saturate d{d} K{K}",
                  all(margin==0 for margin in margins))
        block_records.append({"species":d,"integer_energy":K,"dimension":G.rows,
                              "smallest_Schur_margin":str(min(margins))})

# Actual clock kernels, with the physical decompactification coefficient.
# Unit-delta oscillators give G_delta=delta^2 times the integer-current sum
# when the continuum generator uses the measure convention in the memo.
spectral_records=[]
for d,K,delta in ((1,5,s.Rational(1)),(2,2,s.Rational(1)),
                  (2,4,s.Rational(1,2)),(2,8,s.Rational(1,4)),
                  (3,4,s.Rational(1)),(3,6,s.Rational(1,2))):
    G=block(d,K,delta,False)
    values=np.linalg.eigvalsh(np.array(G.evalf(),dtype=float))
    norm=float(np.max(np.abs(values))) if len(values) else 0.0
    E=delta*K
    cap=s.sqrt(2*d)*delta**2*K*(K-1)/2
    truth(f"actual spectral bound d{d} K{K} delta{delta}",norm<=float(cap)+1e-10)
    spectral_records.append({"species":d,"integer_energy":K,
                             "frequency_spacing":str(delta),"physical_energy":str(E),
                             "operator_norm":norm,"exact_bound":str(cap)})

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "complete_blocks":block_records,"actual_kernel_spectral_checks":spectral_records,
        "continuum_bound":"||G3||_(H0<=E) <= sqrt(d/2) E^2 in the declared unit-delta integral convention.",
        "measure_conversion":"For current modes with bracket omega*2pi*delta and Fourier measure dω/(2pi), the unit-delta generator and bound both acquire 1/sqrt(2pi).",
        "scope":"Intrinsic cubic continuum generator only; no bound for all higher logarithmic generators, complete quantum clock, or heterotic loop prescription."}
(ROOT/'data_exports/mqm/mqm_clock_continuum_cubic_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="named_checks"},indent=2))
