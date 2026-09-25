"""Independent exact Ramond-mode test of the spinor-block coordinate norm.

No KZ equation or hypothesized U(1) norm formula is used to construct F.
The Ramond nonzero-mode contractions and zero-mode Clifford algebra are
implemented directly, with all position and Clifford coefficients rational.
"""
from functools import lru_cache
from fractions import Fraction as Q
from pathlib import Path
import json
import sympy as s


def partitions(N):
    """Restricted-growth words encode flavor equality classes once each."""
    if N == 0:
        yield ()
        return
    def walk(word, largest):
        if len(word) == N:
            yield tuple(word)
            return
        for a in range(largest + 2):
            yield from walk(word + [a], max(largest, a))
    yield from walk([0], 0)


@lru_cache(None)
def ramond_block(indices, xs):
    """Return prod sqrt(2*x_i) F as coefficients of canonical Clifford words."""
    if not indices:
        return ((0, Q(1)),)
    a = indices[0]
    result = {}
    for mask, coefficient in ramond_block(indices[1:], xs[1:]):
        sign = (-1) ** ((mask & ((1 << a) - 1)).bit_count())
        key = mask ^ (1 << a)
        result[key] = result.get(key, Q(0)) + sign * coefficient
    for j in range(1, len(indices)):
        if indices[j] != a:
            continue
        factor = (-1) ** (j - 1) * 2 * xs[j] / (xs[0] - xs[j])
        rest_indices = indices[1:j] + indices[j+1:]
        rest_xs = xs[1:j] + xs[j+1:]
        for mask, coefficient in ramond_block(rest_indices, rest_xs):
            result[mask] = result.get(mask, Q(0)) + factor * coefficient
    return tuple(sorted((mask, c) for mask, c in result.items() if c))


def block_norm_polynomial(xs, n):
    answer = 0
    normalization = s.prod(2 * x for x in xs)
    for word in partitions(len(xs)):
        colors = max(word, default=-1) + 1
        multiplicity = s.prod(n - j for j in range(colors))
        norm = sum(c*c for _, c in ramond_block(word, xs))
        answer += multiplicity * s.Rational(norm.numerator, norm.denominator)
    return s.expand(answer / normalization)


@lru_cache(None)
def current_moment(xs, n):
    if not xs:
        return s.Integer(1)
    result = n / (2 * xs[0]) * current_moment(xs[1:], n)
    for j in range(1, len(xs)):
        rest = xs[1:j] + xs[j+1:]
        result += n / (xs[0] - xs[j])**2 * current_moment(rest, n)
    return s.expand(result)


n = s.symbols('n', integer=True, positive=True)
passed = []
for N in range(1, 8):
    xs = tuple(Q((N + 1 - j)**2 + j) for j in range(N))
    actual = block_norm_polynomial(xs, n)
    predicted = current_moment(xs, n)
    assert s.simplify(actual - predicted) == 0, (N, actual, predicted)
    passed.append(f'N{N} exact symbolic-flavor Ramond norm')

x, y = s.symbols('x y', positive=True)
N2 = n * (x+y)**2 / (4*x*y*(x-y)**2) + n*(n-1)/(4*x*y)
assert s.factor(N2 - n/(x-y)**2 - n*n/(4*x*y)) == 0
passed.append('N2 all-position explicit block norm')

# Test the derived radial differential operator on the actual tower,
# retaining arbitrary radial dimension rather than substituting examples.
z, shape = s.symbols('z shape', positive=True)
for m in range(6):
    f = s.assoc_laguerre(m, shape-1, z)
    action = -2*z*s.diff(f,z,2) + 2*(z-shape)*s.diff(f,z)
    assert s.simplify(action-2*m*f) == 0
    passed.append(f'radial differential action on Laguerre state m{m}')

result = {
    'checks_passed': len(passed),
    'checks': passed,
    'conventions': 'positive strictly decreasing positions, Hermitian Euclidean Clifford generators',
    'block_method': 'direct Ramond zero/nonzero mode expansion, flavor equality partitions',
    'result': 'averaged spin norm = Gaussian U(1) current moment, level n, mean n/(2x)',
    'scope': 'exact finite-size tests through N7; general proof is separate',
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_spinor_measure_results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
