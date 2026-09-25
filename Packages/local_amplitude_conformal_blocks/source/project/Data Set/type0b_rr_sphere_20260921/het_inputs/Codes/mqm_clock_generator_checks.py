"""Exact finite-energy checks of Hamiltonian generators of CLOCK.

This tests a unitary quantization of the first nonlinear term, not the full
quantum clock or an MQM Hamiltonian. Cubic blocks and quartic transitions
use exact radicals.
"""
from functools import lru_cache
from itertools import product
from collections import Counter
from pathlib import Path
import json
import sympy as s

checks = []
blocks = []


def equal(name, actual, expected=0):
    if s.simplify(actual - expected) != 0:
        raise AssertionError((name, s.simplify(actual - expected)))
    checks.append(name)


def truth(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def n(k):
    return s.sqrt(1 + k*k)


def h(legs):
    """Symmetric cubic tensor, with species 0 the singlet."""
    species = [i for i, k in legs]
    if species == [0, 0, 0]:
        return s.sqrt(2)*(1 + sum(k*k for i, k in legs)/s.Integer(2))/s.prod(n(k) for i, k in legs)
    if species.count(0) == 1:
        vectors = [i for i in species if i]
        if vectors[0] == vectors[1]:
            k_s = next(k for i, k in legs if i == 0)
            return s.sqrt(2)/n(k_s)
    return s.Integer(0)


@lru_cache(None)
def basis(d, E, cutoff):
    modes = tuple((m, i) for m in range(1, cutoff + 1) for i in range(d))
    states = []

    def visit(index, remaining, prefix):
        if index == len(modes):
            if remaining == 0:
                states.append(tuple(prefix))
            return
        m, _ = modes[index]
        for multiplicity in range(remaining//m + 1):
            visit(index + 1, remaining - m*multiplicity, prefix + [multiplicity])

    visit(0, E, [])
    return modes, tuple(states)


@lru_cache(None)
def generator(d, E, cutoff):
    modes, states = basis(d, E, cutoff)
    index = {state: j for j, state in enumerate(states)}
    mode_index = {mode: j for j, mode in enumerate(modes)}
    terms = {}
    for w in range(2, min(cutoff, E) + 1):
        for p in range(1, w):
            q = w-p
            flavors = [(0, 0, 0)]
            for a in range(1, d):
                flavors.extend(((0, a, a), (a, 0, a), (a, a, 0)))
            for I, J, K in flavors:
                coefficient = h(((I, -w), (J, p), (K, q)))*s.sqrt(w*p*q)/2
                annihilate = mode_index[(w, I)]
                create = tuple(sorted((mode_index[(p, J)], mode_index[(q, K)])))
                key = (annihilate, create)
                terms[key] = s.simplify(terms.get(key, 0) + coefficient)
    entries = {}
    for col, state in enumerate(states):
        for (annihilate, create), coefficient in terms.items():
            if state[annihilate] == 0:
                continue
            target = list(state)
            factor = s.sqrt(target[annihilate])
            target[annihilate] -= 1
            for j in create:
                target[j] += 1
                factor *= s.sqrt(target[j])
            row = index[tuple(target)]
            value = s.simplify(coefficient*factor)
            entries[(row, col)] = s.simplify(entries.get((row, col), 0) + value)
            entries[(col, row)] = s.simplify(entries.get((col, row), 0) + value)
    return s.SparseMatrix(len(states), len(states), entries)


def ket_index(d, E, cutoff, excitations):
    modes, states = basis(d, E, cutoff)
    counts = {mode: excitations.count(mode) for mode in excitations}
    return states.index(tuple(counts.get(mode, 0) for mode in modes))


# Obtain quadratic clock kernels before introducing the generator. The z
# currents absorb only the singlet reflection phase and descendant norm.
x, y = s.symbols('x y', real=True)
w = x+y
D = lambda k: 1+s.I*k
P = 1+x*x+y*y+x*y
clock_sss = -s.I*s.sqrt(2)*w*P/(D(x)*D(y))*n(x)/D(-x)*n(y)/D(-y)/n(w)
clock_svv = -s.I*s.sqrt(2)*w/n(w)
clock_vsv = -s.I*s.sqrt(2)*w/D(x)*n(x)/D(-x)
equal('SSS first nonlinear current from Hamiltonian commutator',
      clock_sss, -s.I*w*h(((0, -w), (0, x), (0, y))))
equal('SVV first nonlinear current from Hamiltonian commutator',
      clock_svv, -s.I*w*h(((0, -w), (1, x), (1, y))))
equal('VSV first nonlinear current from Hamiltonian commutator',
      clock_vsv, -s.I*w*h(((1, -w), (0, x), (1, y))))

# Complete blocks, including every flavor state at the indicated energies.
for d, maximum in ((1, 6), (2, 5), (3, 5), (24, 2)):
    for E in range(maximum+1):
        cutoff = max(1, E)
        G = generator(d, E, cutoff)
        modes, states = basis(d, E, cutoff)
        truth(f'd{d} E{E} exact Hermiticity', G == G.H)
        truth(f'd{d} E{E} every state has H0=E',
              all(sum(m*occ for (m, i), occ in zip(modes, state)) == E for state in states))
        Gextra = generator(d, E, cutoff+1)
        extra_modes, extra_states = basis(d, E, cutoff+1)
        embedding = [extra_states.index(state+(0,)*d) for state in states]
        truth(f'd{d} E{E} cutoff stabilization', G == Gextra.extract(embedding, embedding))
        truth(f'd{d} E{E} second-order unitary identity',
              -G*G/2-G*G/2+G.H*G == s.zeros(G.rows))
        blocks.append({'species':d, 'energy':E, 'dimension':G.rows,
                       'nonzero_entries':len(G.todok())})

d, E, cutoff = 24, 2, 2
G = generator(d, E, cutoff)
v2 = ket_index(d, E, cutoff, [(2, 1)])
sv = ket_index(d, E, cutoff, [(1, 0), (1, 1)])
truth('full 24-species E2 vector invariant subspace',
      set(i for i in range(G.rows) if G[i, v2] != 0) == {sv})
truth('full 24-species E2 SV invariant subspace',
      set(i for i in range(G.rows) if G[i, sv] != 0) == {v2})
equal('full E2 vector G3=sqrt(2) sigma_x', G[sv, v2], s.sqrt(2))

s2 = ket_index(d, E, cutoff, [(2, 0)])
ss = ket_index(d, E, cutoff, [(1, 0), (1, 0)])
vv = [ket_index(d, E, cutoff, [(1, a), (1, a)]) for a in range(1, d)]
equal('full E2 singlet S2 to SS', G[ss, s2], s.sqrt(s.Rational(8,5)))
equal('full E2 singlet S2 to normalized vector-pair singlet',
      sum(G[j, s2] for j in vv)/s.sqrt(23), s.sqrt(s.Rational(46,5)))
equal('full E2 singlet splitting norm', (G[:, s2].T*G[:, s2])[0], s.Rational(54,5))
equal('full E2 singlet elastic real coefficient', -(G[:, s2].T*G[:, s2])[0]/2, -s.Rational(27,5))
truth('full E2 singlet only stated channels',
      set(i for i in range(G.rows) if G[i,s2] != 0) == {ss, *vv})

angle = s.symbols('g', real=True)
Uv = s.Matrix([[s.cos(s.sqrt(2)*angle), -s.I*s.sin(s.sqrt(2)*angle)],
               [-s.I*s.sin(s.sqrt(2)*angle), s.cos(s.sqrt(2)*angle)]])
truth('exact E2 vector exponential unitary', s.simplify(Uv.H*Uv) == s.eye(2))
equal('E2 vector elastic coefficient -g2', s.diff(Uv[0,0], angle, 2).subs(angle,0)/2, -1)
equal('E2 vector cubic scattering sign', s.diff(Uv[1,0], angle).subs(angle,0), -s.I*s.sqrt(2))
equal('E2 exact vector current vacuum commutator remains two',
      2*(s.cos(s.sqrt(2)*angle)**2+s.sin(s.sqrt(2)*angle)**2), 2)
Gsinglet = s.Matrix([[0, G[ss,s2], sum(G[j,s2] for j in vv)/s.sqrt(23)],
                    [G[ss,s2],0,0],
                    [sum(G[j,s2] for j in vv)/s.sqrt(23),0,0]])
truth('E2 singlet cubic minimal polynomial', Gsinglet**3 == s.Rational(54,5)*Gsinglet)
equal('E2 exact singlet current vacuum commutator remains two',
      2*(s.cos(s.sqrt(s.Rational(54,5))*angle)**2+s.sin(s.sqrt(s.Rational(54,5))*angle)**2), 2)


def flavor_rotation(d, E, cutoff, a, b):
    modes, states = basis(d,E,cutoff)
    state_index = {state:j for j,state in enumerate(states)}
    mode_index = {mode:j for j,mode in enumerate(modes)}
    entries = {}
    for col,state in enumerate(states):
        for m in range(1,cutoff+1):
            for create,annihilate,coefficient in ((a,b,-s.I),(b,a,s.I)):
                ia, ic = mode_index[(m,annihilate)], mode_index[(m,create)]
                if not state[ia]:
                    continue
                target = list(state)
                factor = s.sqrt(target[ia]*(target[ic]+1))
                target[ia] -= 1
                target[ic] += 1
                row = state_index[tuple(target)]
                entries[row,col] = entries.get((row,col),0)+coefficient*factor
    return s.SparseMatrix(len(states),len(states),entries)


for d,E in ((3,4),(24,2)):
    Grot = generator(d,E,E)
    L = flavor_rotation(d,E,E,1,2)
    truth(f'd{d} E{E} SO vector rotation conserved', Grot*L == L*Grot)

# At any integer w the first splitting is entirely in S_n V_(w-n).
# Use the actual Fock operator factors and symmetric h rather than QV.
vector_norms=[]
for w in range(2, 21):
    coefficients = [s.simplify(h(((1,-w),(0,p),(1,w-p)))*s.sqrt(w*p*(w-p))) for p in range(1,w)]
    norm2 = s.simplify(sum(c*c for c in coefficients))
    Q = 2*w*w*sum(s.Rational(p*(w-p),1+p*p) for p in range(1,w))
    equal(f'w{w} canonical split norm QV/w', norm2, Q/w)
    equal(f'w{w} unitary elastic repair cancels current excess', 2*w*(-norm2/2)+Q)
    if w <= 5:
        Gsmall = generator(2,w,w)
        source = ket_index(2,w,w,[(w,1)])
        equal(f'w{w} complete two-species block splitting norm',
              (Gsmall[:,source].T*Gsmall[:,source])[0], norm2)
    vector_norms.append({'energy':w, 'Q_V':str(Q), 'Re_delta_R_V':str(-norm2/2)})

# A discriminating quartic: two distinct vector flavors and distinct B
# frequencies avoid Bose-normalization degeneracies. U3 gives the cut real
# part but misses the dispersive imaginary term supplied by the clock.
d, E, cutoff = 3, 4, 4
G = generator(d,E,cutoff)
source = ket_index(d,E,cutoff,[(4,1)])
target = ket_index(d,E,cutoff,[(1,1),(1,2),(2,2)])
u3_quartic = s.simplify(-(G[target,:]*G[:,source])[0]/2)
clock_quartic = -s.I*s.sqrt(4*1*1*2)/D(3)
equal('VVVV E4 U3 produces real cut part', u3_quartic, s.re(clock_quartic))
equal('VVVV E4 imaginary quartic remains', s.im(clock_quartic), -s.sqrt(2)/5)
truth('U3 alone differs from full quartic clock', s.simplify(u3_quartic-clock_quartic) != 0)


def h4(legs):
    """Root-derived Lie-logarithm kernel, checked against independent blocks."""
    species = [a for a,k in legs]
    momenta = [k for a,k in legs]
    pairings = ((0,1,2,3),(0,2,1,3),(0,3,1,2))
    if species.count(0) == 4:
        rational = -(1+sum(k*k for k in momenta)/s.Integer(2))
        rational += s.prod(momenta)*sum(1/(1+s.Integer(momenta[i]+momenta[j])**2) for i,j,k,l in pairings)
        return s.simplify(rational/s.prod(n(k) for k in momenta))
    if species.count(0) == 2:
        singlets = [k for a,k in legs if a == 0]
        vectors = [a for a,k in legs if a != 0]
        if vectors[0] != vectors[1]:
            return s.Integer(0)
        p,q = singlets
        return (-1-s.Rational(p*q,1+(p+q)**2))/(n(p)*n(q))
    if species.count(0) == 0:
        return sum(s.Rational(1,1+(momenta[i]+momenta[j])**2)
                   for i,j,k,l in pairings if species[i] == species[j] and species[k] == species[l])
    return s.Integer(0)


quartic_checks=[]
for energies in ((1,1,2),(1,2,3),(2,2,2)):
    a,b,c = energies
    E = a+b+c
    W = E*a*b*c
    B = 1+s.Rational(E*E+a*a+b*b+c*c,2)
    for name,incoming,outgoing in (
        ('SSSS',0,(0,0,0)),('SSVV',0,(0,1,1)),
        ('VSSV',1,(1,0,0)),('VVVV_distinct',1,(1,2,2)),
        ('VVVV_same',1,(1,1,1))):
        G = generator(3,E,E)
        source = ket_index(3,E,E,[(E,incoming)])
        excitations = list(zip(energies,outgoing))
        target = ket_index(3,E,E,excitations)
        bose = s.sqrt(s.prod(s.factorial(multiplicity) for multiplicity in Counter(excitations).values()))
        half_square = s.simplify(-(G[target,:]*G[:,source])[0]/2)
        legs = [(incoming,-E)]+list(zip(outgoing,energies))
        quartic_matrix = s.simplify(h4(legs)*s.sqrt(W)/bose)
        if name == 'SSSS':
            clock = s.I*s.sqrt(W)*(W*(1/D(a+b)+1/D(a+c)+1/D(b+c))+(1+2*s.I*E)*B)/(n(E)*n(a)*n(b)*n(c))
        elif name == 'SSVV':
            clock = s.I*s.sqrt(W)*D(E)*(1+s.I*(2*E-a))/D(b+c)/(n(E)*n(a))
        elif name == 'VSSV':
            clock = s.I*s.sqrt(W)*(1+2*s.I*E+s.Rational(b*c)/D(b+c))/(n(b)*n(c))
        elif name == 'VVVV_distinct':
            clock = -s.I*s.sqrt(W)/D(b+c)
        else:
            clock = -s.I*s.sqrt(W)*(1/D(a+b)+1/D(a+c)+1/D(b+c))
        clock /= bose
        equal(f'{name} {energies} G3 cut equals clock real part', half_square, s.re(clock))
        equal(f'{name} {energies} G4 equals negative clock imaginary part', quartic_matrix, -s.im(clock))
        equal(f'{name} {energies} full quartic exponential', half_square-s.I*quartic_matrix, clock)
        quartic_checks.append({'channel':name,'energies':energies,
                               'G4_matrix_element':str(quartic_matrix),
                               'minus_half_G3_square':str(half_square)})

result={'passed':True,'checks':len(checks),'named_checks':checks,
        'complete_blocks':blocks,'vector_norms':vector_norms,
        'G4_quartic_checks':quartic_checks,
        'quartic_example':{'U3_g2_coefficient':str(u3_quartic),
                           'clock_g2_coefficient':str(s.simplify(clock_quartic))},
        'scope':'Exact cubic blocks, quartic generator transition checks, and finite-energy quantization; no noncompact quantum limit, microscopic MQM dynamics, or odd charge completion.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_generator_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
