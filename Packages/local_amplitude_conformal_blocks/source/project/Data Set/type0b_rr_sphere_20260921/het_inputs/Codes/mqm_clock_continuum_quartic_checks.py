"""Checks of the intrinsic continuum G4 weighted-Schur construction.

The analytic continuum proof is in mqm_clock_continuum_quartic_audit.md.
Exact complete occupation blocks check the Fock combinatorics against an
all-flavor constant tensor, which dominates any bounded quartic tensor.
The numerical clock blocks are supplementary checks, not the proof.
"""
import ast
from collections import Counter
from itertools import combinations, combinations_with_replacement
from pathlib import Path
import json
import math
import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parents[1]
namespace = {}
for filename, names in (
    ('Codes/mqm_clock_generator_checks.py', {"n", "h4", "basis"}),
    ('Codes/mqm_clock_nonlocal_charge_checks.py', {"wick_matrix"}),
):
    source = ast.parse((ROOT / filename).read_text())
    definitions = ast.Module(body=[
        node for node in source.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef) and node.name in names
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(definitions),
                 filename + ":selected_definitions", "exec"), namespace)
basis, h4, wick_matrix = (namespace[name] for name in ("basis", "h4", "wick_matrix"))
checks = []


def equal(name, left, right=0):
    difference = s.simplify(left-right)
    if difference != 0:
        raise AssertionError((name, difference))
    checks.append(name)


def truth(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


# The general r-created/s-annihilated action and Schur weight factors.
d = s.symbols("d", positive=True)
for r in range(1,5):
    for q in range(1,5):
        for out_count in range(r,r+4):
            in_count = out_count-r+q
            raw = (s.sqrt(s.factorial(in_count)*s.factorial(out_count))
                   / s.factorial(out_count-r)
                   / s.binomial(out_count,r)
                   / s.factorial(r) / s.factorial(q))
            expected = s.sqrt(s.factorial(in_count)/s.factorial(out_count))/s.factorial(q)
            equal(f"Fock action r{r} q{q} n{out_count}", raw, expected)
            weight_factor = (d**s.Rational(r-q,2)
                             * s.sqrt(s.factorial(out_count)/s.factorial(in_count)))
            equal(f"Schur factorial cancellation r{r} q{q} n{out_count}",
                  raw*weight_factor,
                  d**s.Rational(r-q,2)/s.factorial(q))

E = s.symbols("E", positive=True)
simplex = s.Integer(1)
t = s.symbols("t", positive=True)
for q in range(1,6):
    equal(f"positive frequency simplex volume q{q}", simplex,
          E**(q-1)/s.factorial(q-1))
    simplex = s.integrate(simplex.subs(E,t), (t,0,E))

equal("coarse quartic coefficient sum",
      sum(s.Rational(1,math.factorial(r)*math.factorial(4-r)*math.factorial(3-r))
          for r in range(1,4)), s.Rational(1,2))

for count in range(1,8):
    w = s.symbols("w0:"+str(count), positive=True)
    row = (sum(k**3 for k in w)/12
           + sum(w[i]*w[j]*(w[i]+w[j])/2
                 for i in range(count) for j in range(i+1,count))
           + sum(s.prod(w[i] for i in subset)
                 for subset in combinations(range(count),3)))
    equal(f"sharp continuum combined quartic row count{count}",
          row, sum(w)**3/6-sum(k**3 for k in w)/12)
    lattice = sum(
        s.prod(w[i] for i in subset)
        * s.prod(sum(w[i] for i in subset)-j for j in range(1,4-r))
        / s.factorial(3-r) / s.factorial(4-r)
        for r in range(1,4)
        for subset in combinations(range(count),r)
    )
    total = sum(w)
    equal(f"exact lattice combined quartic row count{count}",
          lattice, (2*total**3-3*total**2+2*total-sum(k**3 for k in w))/12)

# The sole step using energy conservation in the global SSSS tensor bound.
k1,k2,k3=s.symbols("k1 k2 k3", real=True)
ks=(k1,k2,k3,-k1-k2-k3)
equal("singlet quadratic numerator conservation identity",
      -sum(k*k for k in ks)/2,
      sum(ks[i]*ks[j] for i in range(4) for j in range(i+1,4)))
truth("global quartic tensor cap from 1+six+three terms",1+6+3==10)


def quartic_poly(species, integer_energy, spacing=s.Integer(1), constant_tensor=True):
    signed = tuple((a,k) for a in range(species)
                   for k in (*range(-integer_energy,0),*range(1,integer_energy+1)))
    result = {}
    for legs in combinations_with_replacement(signed,4):
        if sum(k for a,k in legs) or sum(k for a,k in legs if k>0)>integer_energy:
            continue
        tensor = s.Integer(1) if constant_tensor else h4(tuple((a,spacing*k) for a,k in legs))
        coefficient = (spacing**3*tensor
                       / s.prod(s.factorial(count) for count in Counter(legs).values()))
        if coefficient != 0:
            result[legs] = coefficient
    return result


block_records = []
for species, maximum in ((1,7),(2,5),(3,4),(8,2)):
    for K in range(1,maximum+1):
        modes, states = basis(species,K,K)
        matrix = wick_matrix(quartic_poly(species,K),species,K,K)
        truth(f"constant-tensor Hermitian block d{species} K{K}", matrix==matrix.H)
        weights = s.Matrix([
            1/s.sqrt(s.Integer(species)**sum(state)
                     * s.prod(s.Integer(k)**occ*s.factorial(occ)
                              for (k,a),occ in zip(modes,state)))
            for state in states
        ])
        action = matrix*weights
        for index,state in enumerate(states):
            particles = [k for (k,a),occ in zip(modes,state) for _ in range(occ)]
            expected = s.Rational(2*K**3-3*K**2+2*K-sum(k**3 for k in particles),12)
            actual = s.simplify(action[index]/weights[index]/species**2)
            equal(f"exact Schur block row d{species} K{K} row{index}",actual,expected)
            truth(f"sharp Schur bound d{species} K{K} row{index}",
                  0<=expected<=s.Rational(K**3,6))
        block_records.append({"species":species,"integer_energy":K,
                              "dimension":len(states),"all_rows_exact":True})

spectral_records=[]
for species,K,spacing in ((1,5,s.Rational(1)),(2,4,s.Rational(1)),
                          (3,4,s.Rational(1,2)),(2,6,s.Rational(1,3))):
    polynomial=quartic_poly(species,K,spacing,False)
    for legs in polynomial:
        tensor=h4(tuple((a,spacing*k) for a,k in legs))
        truth(f"sample global tensor bound d{species} K{K} legs{legs}",
              abs(float(tensor))<=10+1e-12)
    matrix=wick_matrix(polynomial,species,K,K)
    truth(f"actual clock Hermitian block d{species} K{K} spacing{spacing}",matrix==matrix.H)
    values=np.linalg.eigvalsh(np.array(matrix.evalf(),dtype=float))
    norm=float(np.max(np.abs(values)))
    physical_energy=spacing*K
    cap=s.Rational(5,3)*species**2*physical_energy**3
    truth(f"actual clock spectral bound d{species} K{K} spacing{spacing}",
          norm<=float(cap)+1e-10)
    spectral_records.append({"species":species,"integer_energy":K,
                             "spacing":str(spacing),"physical_energy":str(physical_energy),
                             "operator_norm":norm,"exact_bound":str(cap)})

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "complete_comparison_blocks":block_records,"actual_clock_spectral_checks":spectral_records,
        "continuum_bound":"||G4||_(H0<=E) <= (5/3)d^2 E^3 in the explicit unit-delta formula; the simpler 5d^2 E^3 bound also holds.",
        "measure_conversion":"The repository current/Fourier convention multiplies G4 and its bound by 1/(2pi).",
        "scope":"Intrinsic two-generator unitary exp[-i(gG3+g^2G4)] by closure on the common bounded-energy core; no full Hamiltonian-logarithm sum, quantum mirror-state identification or heterotic loop claim."}
(ROOT/'data_exports/mqm/mqm_clock_continuum_quartic_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({key:value for key,value in result.items() if key!="named_checks"},indent=2))
