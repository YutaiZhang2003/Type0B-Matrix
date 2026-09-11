"""Direct finite-c Virasoro sewing of the punctured theta graph.

Benchmark-only implementation. It reuses the repository's ordinary Virasoro
Ward identities and contracts inverse PBW Gram matrices on all four edges.
No CCY coefficient or saved block enters this calculation.
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys
import time

import mpmath as mp

BRANCHING = Path(__file__).resolve().parents[2]/"ramond_branching_recursion"
sys.path.insert(0, str(BRANCHING))
import compute_target as br


def number(value):
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator)/value.denominator
    return mp.mpc(value)


class DirectVirasoroPBW:
    def __init__(self, central_charge, weights, external_weight, *, dps=100):
        br.set_multiprecision(dps)
        self.c = number(central_charge)
        self.weights = tuple(map(number, weights))
        h1, hL, hR, h3 = self.weights
        self.first = br.VirasoroThreePoint((h1, hL, h3), self.c)
        self.second = br.VirasoroThreePoint((h1, hR, h3), self.c)
        self.middle = br.VirasoroThreePoint((hR, number(external_weight), hL), self.c)
        self.gram_forms = tuple(br.VirasoroThreePoint((h, h, h), self.c) for h in self.weights)
        self.gram_seconds = 0.0
        self.vertex_seconds = 0.0
        self.contraction_seconds = 0.0

    @lru_cache(None)
    def inverse_gram(self, edge, level):
        started = time.perf_counter()
        words = br.partitions(level)
        form = self.gram_forms[edge]
        gram = mp.matrix(len(words))
        for i, left in enumerate(words):
            for j in range(i, len(words)):
                expression = {words[j]: mp.mpf(1)}
                for mode in left:
                    acted = {}
                    for word, outer in expression.items():
                        for target, inner in form.act(0, mode, word).items():
                            acted[target] = acted.get(target, 0)+outer*inner
                    expression = acted
                gram[i, j] = gram[j, i] = expression.get((), 0)
        inverse = mp.inverse(gram)
        self.gram_seconds += time.perf_counter()-started
        return inverse

    @lru_cache(None)
    def first_vertex(self, a, b, d):
        inverse_a, inverse_d = self.inverse_gram(0, a), self.inverse_gram(3, d)
        started = time.perf_counter()
        A, B, D = map(br.partitions, (a, b, d))
        result = mp.matrix(len(A)*len(D), len(B))
        for j, middle in enumerate(B):
            raw = mp.matrix([[self.first.value(left, middle, right) for right in D] for left in A])
            normalized = inverse_a*raw*inverse_d
            for i in range(len(A)):
                for l in range(len(D)):
                    result[i*len(D)+l, j] = normalized[i, l]
        self.vertex_seconds += time.perf_counter()-started
        return result

    @lru_cache(None)
    def second_vertex(self, a, c, d):
        started = time.perf_counter()
        A, C, D = map(br.partitions, (a, c, d))
        result = mp.matrix([[self.second.value(left, middle, right) for middle in C]
                            for left in A for right in D])
        self.vertex_seconds += time.perf_counter()-started
        return result

    @lru_cache(None)
    def middle_vertex(self, c, b):
        inverse_c, inverse_b = self.inverse_gram(2, c), self.inverse_gram(1, b)
        started = time.perf_counter()
        C, B = map(br.partitions, (c, b))
        raw = mp.matrix([[self.middle.value(left, (), right) for right in B] for left in C])
        result = inverse_c*raw*inverse_b
        self.vertex_seconds += time.perf_counter()-started
        return result

    def coefficient(self, levels):
        a, b, c, d = levels
        first = self.first_vertex(a, b, d)
        second = self.second_vertex(a, c, d)
        middle = self.middle_vertex(c, b)
        started = time.perf_counter()
        sewn = first*middle.T
        result = mp.fsum(sewn[i, j]*second[i, j]
                         for i in range(sewn.rows) for j in range(sewn.cols))
        self.contraction_seconds += time.perf_counter()-started
        return result

    def full_series(self, *, indices):
        return {tuple(levels): self.coefficient(levels) for levels in indices}

    def diagnostics(self):
        return {"gram_and_inverse_seconds": self.gram_seconds,
                "ward_and_normalized_vertices_seconds": self.vertex_seconds,
                "final_contractions_seconds": self.contraction_seconds,
                "inverse_gram_count": self.inverse_gram.cache_info().currsize,
                "first_vertex_count": self.first_vertex.cache_info().currsize,
                "second_vertex_count": self.second_vertex.cache_info().currsize,
                "middle_vertex_count": self.middle_vertex.cache_info().currsize}
