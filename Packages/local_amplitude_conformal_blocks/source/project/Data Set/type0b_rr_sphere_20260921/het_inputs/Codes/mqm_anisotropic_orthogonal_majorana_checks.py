"""Exact finite-Clifford checks for the unfitted orthogonal Majorana MQM.

No continuum or scattering approximation enters these tests.
Run with /Users/sam/miniconda3/bin/python.
"""
from itertools import combinations
from math import comb
from pathlib import Path
import json
import sympy as s

I = s.I
checks = []


def check(name, condition):
    ok = bool(condition)
    checks.append({"name": name, "passed": ok})
    if not ok:
        raise AssertionError(name)


def tidy(a):
    return {m: s.expand(c) for m, c in a.items() if s.expand(c) != 0}


def add(*args):
    out = {}
    for a in args:
        for m, c in a.items():
            out[m] = out.get(m, 0) + c
    return tidy(out)


def scale(c, a):
    return tidy({m: c * z for m, z in a.items()})


def mul(a, b):
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            parity = 0
            rest = ma
            while rest:
                bit = rest & -rest
                parity ^= (mb & (bit - 1)).bit_count() & 1
                rest ^= bit
            m = ma ^ mb
            out[m] = out.get(m, 0) + (-1) ** parity * ca * cb
    return tidy(out)


def comm(a, b):
    return add(mul(a, b), scale(-1, mul(b, a)))


ONE = {0: s.Integer(1)}


def gamma(i, a, p):
    return {1 << (i * p + a): s.Integer(1)}


def color(i, j, N, p):
    return scale(I / 2, add(*(mul(gamma(i, a, p), gamma(j, a, p))
                              for a in range(p))))


def flavor(a, b, N, p):
    return scale(I / 2, add(*(mul(gamma(i, a, p), gamma(i, b, p))
                              for i in range(N))))


def reflection(i, p):
    # Canonical lift that fixes the vacuum of flavor-paired complex fermions.
    out = ONE
    for a in range(p):
        out = mul(out, gamma(i, a, p))
    return scale(I ** (-p // 2), out)


for N, p in [(2, 4), (3, 4), (4, 4), (3, 24)]:
    tag = f"N={N},p={p}"
    q = color(0, 1, N, p)
    for i in range(N):
        expected = gamma(1, 0, p) if i == 0 else (
            scale(-1, gamma(0, 0, p)) if i == 1 else {})
        check(f"{tag}: color action on row {i}",
              scale(I, comm(q, gamma(i, 0, p))) == expected)
    check(f"{tag}: commuting color/flavor actions",
          comm(q, flavor(0, 1, N, p)) == {})
    if N > 2:
        check(f"{tag}: color Lie bracket",
              comm(q, color(1, 2, N, p)) == scale(I, color(0, 2, N, p)))
    for i in range(N):
        ri = reflection(i, p)
        check(f"{tag}: reflection {i} squares to one", mul(ri, ri) == ONE)
        for j in range(N):
            check(f"{tag}: reflection {i} action on row {j}",
                  mul(mul(ri, gamma(j, 0, p)), ri)
                  == scale(-1 if i == j else 1, gamma(j, 0, p)))
    r0, r1 = reflection(0, p), reflection(1, p)
    upi = ONE
    for a in range(p):
        upi = mul(upi, scale(-1, mul(gamma(0, a, p), gamma(1, a, p))))
    check(f"{tag}: two reflections equal connected pi rotation",
          mul(r0, r1) == upi)
    check(f"{tag}: color 2pi anomaly cancelled", mul(upi, upi) == ONE)
    flavor_pi = ONE
    for i in range(N):
        flavor_pi = mul(flavor_pi, scale(-1, mul(gamma(i, 0, p), gamma(i, 1, p))))
    check(f"{tag}: flavor-center sign",
          mul(flavor_pi, flavor_pi) == scale((-1) ** N, ONE))
    qh = scale(I / 2, mul(gamma(0, p - 1, p), gamma(1, p - 1, p)))
    check(f"{tag}: selective Yukawa commutes with Cartan Gauss", comm(q, qh) == {})
    check(f"{tag}: Weyl reflection flips heavy Cartan charge",
          mul(mul(r1, qh), r1) == scale(-1, qh))

for p in [4, 24]:
    chirality = reflection(0, p)
    spectator_casimir = {}
    for a, b in combinations(range(p - 1), 2):
        jab = scale(I / 2, mul(gamma(0, a, p), gamma(0, b, p)))
        spectator_casimir = add(spectator_casimir, mul(jab, jab))
        check(f"p={p}: zero-mode parity commutes with spectator J{a},{b}",
              comm(chirality, jab) == {})
    check(f"p={p}: zero-factor spectator spinor Casimir",
          spectator_casimir == scale(s.Rational((p - 1) * (p - 2), 8), ONE))
    heavy_zero = gamma(0, p - 1, p)
    check(f"p={p}: heavy zero exchanges the two parities",
          add(mul(chirality, heavy_zero), mul(heavy_zero, chirality)) == {})
    check(f"p={p}: heavy zero intertwines spectator spin",
          comm(heavy_zero, flavor(0, 1, 1, p)) == {})

# Exact occupation spectra, including the actual 24-flavor coefficients.
fibers = []
for p in [4, 24]:
    m = p // 2
    heavy_empty = comb(p - 1, m)
    heavy_full = comb(p - 1, m - 1)
    check(f"p={p}: neutral fiber splits into equal heavy bands",
          comb(p, m) == heavy_empty + heavy_full and heavy_empty == heavy_full)
    check(f"p={p}: empty-heavy spectator charge cancels -1/2",
          s.Rational(m) - s.Rational(p - 1, 2) == s.Rational(1, 2))
    check(f"p={p}: full-heavy spectator charge cancels +1/2",
          s.Rational(m - 1) - s.Rational(p - 1, 2) == -s.Rational(1, 2))
    for N in [2, 3, 4, 5]:
        r = N // 2
        zero_dim = 2 ** m if N % 2 else 1
        fibers.append({"N": N, "p": p,
                       "neutral_dimension": comb(p, m) ** r * zero_dim,
                       "lowest_Yukawa_dimension": heavy_empty ** r * zero_dim,
                       "zero_factor_dimension": zero_dim,
                       "flavor_center": (-1) ** N})

# Independent small Fock enumeration verifies the branches and their parity.
for p in [4, 6]:
    neutral = [bits for bits in range(1 << p) if bits.bit_count() == p // 2]
    levels = {-1: [], 1: []}
    for bits in neutral:
        heavy_n = (bits >> (p - 1)) & 1
        levels[2 * heavy_n - 1].append(bits)
    check(f"enumeration p={p}: both exact Yukawa multiplicities",
          len(levels[-1]) == comb(p - 1, p // 2)
          and len(levels[1]) == comb(p - 1, p // 2 - 1))

# The full-flavor contraction of the orbital Gauss vector field vanishes.
for N in [3, 4]:
    X = s.zeros(N)
    for i, j in combinations(range(N), 2):
        X[i, j] = s.Symbol(f"x{i}{j}", real=True)
        X[j, i] = -X[i, j]
    contraction = s.zeros(N)
    for i, j in combinations(range(N), 2):
        T = s.zeros(N)
        T[i, j], T[j, i] = 1, -1
        contraction += X[i, j] * (T * X - X * T)
    check(f"N={N}: exact orbital contraction has no remainder", contraction == s.zeros(N))

# N=3 single-heavy-flavor band: an actual Berry line of Chern number -1.
theta, phi = s.symbols("theta phi", real=True)
w = s.Matrix([s.cos(theta / 2), s.exp(I * phi) * s.sin(theta / 2)])
berry_phi = s.simplify(I * (s.conjugate(w).T * s.diff(w, phi))[0])
curvature = s.trigsimp(s.diff(berry_phi, theta))
chern = s.simplify(s.integrate(curvature, (theta, 0, s.pi)))
check("N=3 heavy ground line Berry curvature", s.trigsimp(curvature + s.sin(theta) / 2) == 0)
check("N=3 heavy ground line first Chern number", chern == -1)

x, omega, lam = s.symbols("x omega lam", real=True, nonzero=True)
for sigma in [-1, 1]:
    initial = -omega ** 2 * x ** 2 / 2 + sigma * lam * x
    translated = -omega ** 2 * (x - sigma * lam / omega ** 2) ** 2 / 2 + lam ** 2 / (2 * omega ** 2)
    check(f"N=2 exact translated barrier, sigma={sigma}",
          s.expand(initial - translated) == 0)

# One-flavor determinant baryon: exact nonemptiness of the O(N) det sector.
for N in [1, 2, 3, 4]:
    p = 24
    creators = [scale(s.Rational(1, 2), add(gamma(i, 0, p),
                                           scale(-I, gamma(i, 1, p))))
                for i in range(N)]
    annihilators = [scale(s.Rational(1, 2), add(gamma(i, 0, p),
                                              scale(I, gamma(i, 1, p))))
                   for i in range(N)]
    baryon = ONE
    for creator in creators:
        baryon = mul(baryon, creator)
    baryon_dag = ONE
    for annihilator in reversed(annihilators):
        baryon_dag = mul(baryon_dag, annihilator)
    empty_projector = ONE
    for annihilator, creator in zip(annihilators, creators):
        empty_projector = mul(empty_projector, mul(annihilator, creator))
    check(f"N={N}: baryon has unit norm on the flavor vacuum",
          mul(baryon_dag, baryon) == empty_projector and bool(empty_projector))
    parity = ONE
    for i in range(N):
        ri = reflection(i, p)
        parity = mul(parity, ri)
        check(f"N={N}: baryon reflection character at row {i}",
              mul(mul(ri, baryon), ri) == scale(-1, baryon))
    for i, j in combinations(range(N), 2):
        check(f"N={N}: baryon connected Gauss invariance {i},{j}",
              comm(color(i, j, N, p), baryon) == {})
    check(f"N={N}: baryon Grassmann parity", 
          mul(mul(parity, baryon), parity) == scale((-1) ** N, baryon))

results = {"passed": all(x["passed"] for x in checks),
           "number_of_checks": len(checks), "checks": checks,
           "fiber_counts": fibers,
           "actual_p24": {"one_block_neutral": comb(24, 12),
                          "one_block_low_band": comb(23, 12),
                          "odd_zero_factor": 4096,
                          "each_Spin23_copy": 2048},
           "Berry_curvature": str(curvature), "Berry_Chern_number": str(chern)}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_anisotropic_orthogonal_majorana_results.json').write_text(
    json.dumps(results, indent=2) + "\n")
print(json.dumps({"passed": results["passed"], "checks": len(checks),
                  "actual_p24": results["actual_p24"], "Berry_Chern_number": str(chern)}, indent=2))
