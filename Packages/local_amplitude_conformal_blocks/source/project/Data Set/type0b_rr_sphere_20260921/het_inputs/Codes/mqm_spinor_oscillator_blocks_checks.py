"""Exact algebra tests for unfitted Spin(23)-graded oscillator blocks.

Run with /Users/sam/miniconda3/bin/python.
No oscillator Fock truncation is used: differential identities act on
polynomials times exp(-|x|^2/2), a common Schwartz invariant domain.
"""
import json
from math import comb
from pathlib import Path
import sympy as s

checks = []


def check(name, value):
    if isinstance(value, s.MatrixBase):
        value = all(s.simplify(v) == 0 for v in value)
    elif not isinstance(value, bool):
        value = s.simplify(value) == 0
    assert value, name
    checks.append(name)


# Pauli words represent all 23 gamma matrices without storing 2048^2 entries.
table = {
    ("X", "Y"): (1, "Z"), ("Y", "X"): (3, "Z"),
    ("Y", "Z"): (1, "X"), ("Z", "Y"): (3, "X"),
    ("Z", "X"): (1, "Y"), ("X", "Z"): (3, "Y"),
}


def mul(a, b):
    phase = a[0] + b[0]
    word = []
    for x, y in zip(a[1], b[1]):
        if x == "I":
            word.append(y)
        elif y == "I":
            word.append(x)
        elif x == y:
            word.append("I")
        else:
            extra, z = table[x, y]
            phase += extra
            word.append(z)
    return phase % 4, "".join(word)


def neg(a):
    return (a[0] + 2) % 4, a[1]


def transpose(a):
    return (a[0] + 2 * a[1].count("Y")) % 4, a[1]


identity = (0, "I" * 11)
gammas = []
for j in range(11):
    gammas += [(0, "Z" * j + q + "I" * (10 - j)) for q in ("X", "Y")]
gammas.append((0, "Z" * 11))
for a in range(23):
    check(f"gamma square {a}", mul(gammas[a], gammas[a]) == identity)
    for b in range(a):
        check(f"gamma anticommutator {a},{b}",
              mul(gammas[a], gammas[b]) == neg(mul(gammas[b], gammas[a])))
C = identity
for j in range(1, 23, 2):
    C = mul(C, gammas[j])
check("C symmetric", transpose(C) == C)
for j, gamma in enumerate(gammas):
    check(f"charge conjugation {j}",
          mul(transpose(gamma), C) == neg(mul(C, gamma)))
symmetric = []
antisymmetric = []
product = identity
for p in range(12):
    B = mul(C, product)
    sign = (-1) ** (p * (p + 1) // 2)
    check(f"rank {p} transpose", transpose(B) == (B if sign == 1 else neg(B)))
    (symmetric if sign == 1 else antisymmetric).append(p)
    if p < 11:
        product = mul(product, gammas[p])
d = 2**11
check("symmetric spinor square dimension",
      sum(comb(23, p) for p in symmetric) == d * (d + 1) // 2)
check("antisymmetric spinor square dimension",
      sum(comb(23, p) for p in antisymmetric) == d * (d - 1) // 2)
check("all spinor endomorphisms",
      sum(comb(23, p) for p in range(12)) == d * d)


# Rank-one odd maps on C^(1|m); identities are independent of m.
m = 4


def E(a, b):
    out = s.zeros(m + 1)
    out[a, b] = 1
    return out


for a in range(1, m + 1):
    R, T = E(a, 0), E(0, a)
    for b in range(1, m + 1):
        Rb, Tb = E(b, 0), E(0, b)
        delta = int(a == b)
        check(f"rank one same creation {a},{b}", R * Rb + Rb * R)
        check(f"rank one same annihilation {a},{b}", T * Tb + Tb * T)
        check(f"rank one mixed {a},{b}",
              R * Tb + Tb * R - E(a, b) - delta * E(0, 0))
        q, qb = R + T, Rb + Tb
        check(f"Hermitian odd symmetric {a},{b}",
              q * qb + qb * q - E(a, b) - E(b, a) - 2 * delta * E(0, 0))
        check(f"Hermitian odd antisymmetric {a},{b}",
              q * qb - qb * q - E(a, b) + E(b, a))


# Exact differential checks on two oscillator coordinates, frequency 1/2.
x = s.symbols("x0:2", real=True)
omega = s.Rational(1, 2)


def P(i, f):
    return s.expand(-s.I * (s.diff(f, x[i]) - x[i] * f))


def X(i, f):
    return x[i] * f


def H(f):
    return s.expand(sum(P(i, P(i, f)) - omega**2 * x[i]**2 * f
                        for i in range(2)) / 2)


def Y(sign, i, f):
    return s.expand((P(i, f) + sign * omega * x[i] * f)
                    / s.sqrt(2 * omega))


tests = [s.Integer(1), x[0], x[1], x[0]**2, x[0] * x[1], x[1]**2]
for index, f in enumerate(tests):
    for sign in (1, -1):
        for i in range(2):
            check(f"inverted energy shift {index},{sign},{i}",
                  H(Y(sign, i, f)) - Y(sign, i, H(f))
                  + s.I * sign * omega * Y(sign, i, f))
    for i in range(2):
        for j in range(2):
            check(f"inverted CCR {index},{i},{j}",
                  Y(1, i, Y(-1, j, f)) - Y(-1, j, Y(1, i, f))
                  - s.I * int(i == j) * f)


def y(a, f):
    return X(a, f) if a < 2 else P(a - 2, f)


def J(a, b):
    return int(a < 2 and b == a + 2) - int(b < 2 and a == b + 2)


def B(a, b, f):
    return s.expand(y(a, y(b, f)) + y(b, y(a, f)))


for a in range(4):
    for b in range(a, 4):
        for c in range(4):
            for index, f in enumerate(tests[:3]):
                check(f"osp triple {a},{b},{c},{index}",
                      B(a, b, y(c, f)) - y(c, B(a, b, f))
                      - 2 * s.I * (J(b, c) * y(a, f) + J(a, c) * y(b, f)))

# First-floor spectator primary p-form weighs p/2, so p>4 cannot
# inhabit a floor of required spectator/oscillator grade two.
check("same-sign forms above first charge floor",
      [p for p in symmetric if p > 4] == [7, 8, 11])
check("rank four present", 4 in symmetric)

results = {
    "checks_passed": len(checks),
    "spinor_dimension": d,
    "symmetric_square_ranks": symmetric,
    "antisymmetric_square_ranks": antisymmetric,
    "rank_dimensions": {str(p): comb(23, p) for p in range(12)},
    "osp_even_dimension": d * (2 * d + 1),
    "osp_odd_dimension": 2 * d,
    "scope": "Exact finite algebra and differential identities; no string scattering calculation.",
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_spinor_oscillator_blocks_results.json').write_text(
    json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
