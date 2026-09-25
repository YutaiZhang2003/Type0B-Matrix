"""Exact ordering test of I=M1*M3-3/4*M2**2 for normalized circle moments.

Normal order the full homogeneous degree-six polynomial, not separately
ordered moment operators. Elementary scalar/current zero modes are absent.
"""
import ast
from collections import Counter
from itertools import combinations_with_replacement, product
from pathlib import Path
import json
import sympy as s

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT/'Codes/mqm_clock_generator_checks.py').read_text())
selected = {"n", "h", "h4", "basis", "generator", "ket_index"}
dependency = ast.Module(body=[
    node for node in source.body
    if isinstance(node, (ast.Import, ast.ImportFrom))
    or isinstance(node, ast.FunctionDef) and node.name in selected
], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(dependency),
             "mqm_clock_generator_checks.py:selected_definitions", "exec"), namespace)
basis = namespace["basis"]
generator = namespace["generator"]
ket_index = namespace["ket_index"]
h4 = namespace["h4"]

checks = []


def truth(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def equal(name, actual, expected=0):
    if s.simplify(actual-expected) != 0:
        raise AssertionError((name,s.simplify(actual-expected)))
    checks.append(name)


def clean(poly):
    return {key:s.simplify(value) for key,value in poly.items() if s.simplify(value) != 0}


def moment(r, cutoff):
    """Mr=integral x**r y' dt/(2pi); f_k=i*z_k/k."""
    frequencies = tuple(range(-cutoff,0))+tuple(range(1,cutoff+1))
    result = {}
    for momenta in product(frequencies,repeat=r+1):
        if sum(momenta):
            continue
        key = tuple(sorted([(1,k) for k in momenta[:-1]]+[(2,momenta[-1])]))
        coefficient = s.I**r/s.prod(momenta[:-1])
        result[key] = result.get(key,0)+coefficient
    return clean(result)


def multiply(a,b):
    result = {}
    for left,c1 in a.items():
        for right,c2 in b.items():
            key = tuple(sorted(left+right))
            result[key] = result.get(key,0)+c1*c2
    return clean(result)


def invariant(cutoff):
    first = multiply(moment(1,cutoff),moment(3,cutoff))
    second = multiply(moment(2,cutoff),moment(2,cutoff))
    for key,value in second.items():
        first[key] = first.get(key,0)-s.Rational(3,4)*value
    return clean(first)


def wick_matrix(poly, d, E, cutoff):
    """Apply every full normal-ordered current monomial to the exact block."""
    modes, states = basis(d,E,cutoff)
    index = {state:j for j,state in enumerate(states)}
    mode_index = {mode:j for j,mode in enumerate(modes)}
    entries = {}
    for monomial,coefficient in poly.items():
        truth_energy = sum(k for species,k in monomial)
        if truth_energy:
            raise AssertionError(("energy-nonconserving monomial", monomial))
        if sum(k for species,k in monomial if k>0) > E:
            continue
        annihilate = [mode_index[(k,species)] for species,k in monomial if k>0]
        create = [mode_index[(-k,species)] for species,k in monomial if k<0]
        coefficient *= s.sqrt(s.prod(abs(k) for species,k in monomial))
        for col,state in enumerate(states):
            target = list(state)
            factor = s.Integer(1)
            for j in annihilate:
                if target[j] == 0:
                    factor = 0
                    break
                factor *= s.sqrt(target[j])
                target[j] -= 1
            if factor == 0:
                continue
            for j in create:
                target[j] += 1
                factor *= s.sqrt(target[j])
            row = index[tuple(target)]
            value = s.simplify(coefficient*factor)
            entries[row,col] = s.simplify(entries.get((row,col),0)+value)
    return s.SparseMatrix(len(states),len(states),entries)


def quartic_poly(d,E):
    """G4=(1/24) sum symmetric h4 :z z z z:."""
    signed_modes = tuple((a,k) for a in range(d)
                         for k in (*range(-E,0),*range(1,E+1)))
    result = {}
    for legs in combinations_with_replacement(signed_modes,4):
        if sum(k for a,k in legs):
            continue
        if sum(k for a,k in legs if k>0)>E:
            continue
        coefficient = h4(legs)/s.prod(s.factorial(count) for count in Counter(legs).values())
        if coefficient != 0:
            result[legs] = coefficient
    return result


d,E,cutoff = 3,3,3
modes,states = basis(d,E,cutoff)
G3 = generator(d,E,cutoff)
Ipoly = invariant(3)
Iw = wick_matrix(Ipoly,d,E,cutoff)
G4 = wick_matrix(quartic_poly(d,E),d,E,cutoff)
truth("complete three-species E3 block has dimension 22",len(states)==22)
truth("full invariant has homogeneous degree six",all(len(key)==6 for key in Ipoly))
truth("full invariant conserves total frequency",all(sum(k for a,k in key)==0 for key in Ipoly))
truth("normal-ordered entire invariant is Hermitian",Iw==Iw.H)
truth("quartic generator block is Hermitian",G4==G4.H)

for source_cutoff in (1,2,4):
    other = wick_matrix(invariant(source_cutoff),d,E,cutoff)
    truth(f"charge cutoff {source_cutoff} gives same E3 block",other==Iw)

truth("frequency-one M2 vanishes before quantization",moment(2,1)=={})
surviving = {key:value for key,value in Ipoly.items()
             if sum(k for a,k in key if k>0)<=E}
truth("all surviving charge legs have frequency one",
      all(abs(k)==1 for key in surviving for a,k in key))
truth("three surviving full-Wick monomials",len(surviving)==3)

xxx = ket_index(d,E,cutoff,[(1,1)]*3)
xxy = ket_index(d,E,cutoff,[(1,1)]*2+[(1,2)])
xyy = ket_index(d,E,cutoff,[(1,1)]+[(1,2)]*2)
s2x = ket_index(d,E,cutoff,[(2,0),(1,1)])
s2y = ket_index(d,E,cutoff,[(2,0),(1,2)])
ssy = ket_index(d,E,cutoff,[(1,0)]*2+[(1,2)])
equal("I on xxy has eigenvalue 12",Iw[xxy,xxy],12)
equal("I connects xxx to xyy",Iw[xyy,xxx],-6*s.sqrt(3))
truth("I kills S2 x1",Iw[:,s2x]==s.zeros(len(states),1))
truth("I kills S2 y1",Iw[:,s2y]==s.zeros(len(states),1))
equal("G3 joins xx to S2 with y spectator",G3[s2y,xxy],s.sqrt(s.Rational(2,5)))
equal("G3 joins yy to S2 with x spectator",G3[s2x,xyy],s.sqrt(s.Rational(2,5)))

comm3 = G3*Iw-Iw*G3
truth("G3 and fully Wick ordered I do not commute",comm3!=s.zeros(len(states)))
truth("G3 commutator is anti-Hermitian",comm3.H==-comm3)
equal("G3-I commutator on xxx to S2 x1",comm3[s2x,xxx],-6*s.sqrt(s.Rational(6,5)))
equal("G3-I commutator on xxy to S2 y1",comm3[s2y,xxy],12*s.sqrt(s.Rational(2,5)))

comm4 = G4*Iw-Iw*G4
truth("G4 and fully Wick ordered I do not commute",comm4!=s.zeros(len(states)))
truth("G4 commutator is anti-Hermitian",comm4.H==-comm4)
equal("G4 xx to SS transition with y spectator",G4[ssy,xxy],-s.Rational(3,10))
equal("G4-I commutator on xxy to SS y1",comm4[ssy,xxy],-s.Rational(18,5))

for low_E in range(3):
    low = wick_matrix(Ipoly,d,low_E,cutoff)
    truth(f"degree-six I vanishes below E3: E{low_E}",low==s.zeros(low.rows))

result={"passed":True,"checks":len(checks),"named_checks":checks,
        "block_dimension":len(states),
        "normalization":"Mr=integral_0^(2pi) x^r y' dt/(2pi); f_k=i*z_k/k; orthonormal Fock states.",
        "surviving_I_monomials":[{"legs":key,"coefficient":str(value)} for key,value in surviving.items()],
        "G3_commutators":{"S2x_from_xxx":str(comm3[s2x,xxx]),
                          "S2y_from_xxy":str(comm3[s2y,xxy])},
        "G4_commutator_SSy_from_xxy":str(comm4[ssy,xxy]),
        "scope":"Counterexample to naive full Wick ordering of one conserved classical geometric invariant; no claim of absence of corrected quantum charges or of a string discrete-charge identification."}
(ROOT/'data_exports/mqm/mqm_clock_nonlocal_charge_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
