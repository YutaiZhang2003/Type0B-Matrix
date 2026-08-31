"""Independent NS PBW sewing: algebra and primary Ward identities only.

No Kac weights, fusion polynomials, or c/h-recursion are used here.
Modes have twice-integer indices; PBW words are L-descending, then
G-descending, as in the human note.  The three-form normalization is
rho_0(phi,phi,phi)=rho_1(phi,G_-1/2 phi,phi)=1.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp


@lru_cache(maxsize=None)
def partitions(total, maximum=None, *, odd=False, strict=False):
    if total == 0:
        return ((),)
    if total < 0:
        return ()
    maximum = min(total, total if maximum is None else maximum)
    result = []
    for first in range(maximum, 0, -1):
        if odd and first % 2 == 0:
            continue
        result.extend(
            (first,) + tail
            for tail in partitions(total-first, first-int(strict), odd=odd, strict=strict)
        )
    return tuple(result)


@lru_cache(maxsize=None)
def basis(twice_level):
    result = []
    for fermion_level in range(twice_level + 1):
        if (twice_level-fermion_level) % 2:
            continue
        for gs in partitions(fermion_level, odd=True, strict=True):
            for ls in partitions((twice_level-fermion_level)//2):
                result.append(tuple(("L", -2*n) for n in ls) + tuple(("G", -r) for r in gs))
    return tuple(result)


def level(state):
    return -sum(index for _, index in state)


class NSModule:
    def __init__(self, c, h, symbolic=False):
        self.c, self.h, self.symbolic = c, h, symbolic
        self.one = sp.S.One if symbolic else mp.mpf(1)

    def clean(self, value):
        return sp.expand(value) if self.symbolic else value

    def bracket(self, a, b):
        ak, ai = a
        bk, bi = b
        index = ai+bi
        if ak == bk == "L":
            out = [(self.one*(ai-bi)/2, ("L", index))]
            if index == 0:
                out.append((self.c*ai*(ai*ai-4)/96, None))
        elif ak == "L":
            out = [(self.one*(ai-2*bi)/4, ("G", index))]
        elif bk == "L":
            out = [(self.one*(2*ai-bi)/4, ("G", index))]
        else:
            out = [(2*self.one, ("L", index))]
            if index == 0:
                out.append((self.c*(ai*ai-1)/12, None))
        return tuple((v, x) for v, x in out if v != 0)

    @lru_cache(maxsize=None)
    def action(self, mode, state):
        kind, index = mode
        if index < 0:
            return (((mode,)+state, self.one),)
        if index == 0:
            assert kind == "L"
            return ((state, self.h+self.one*level(state)/2),)
        if not state or index > level(state):
            return ()
        first, tail = state[0], state[1:]
        sign = -1 if kind == first[0] == "G" else 1
        out = {}
        for word, coefficient in self.action(mode, tail):
            word = (first,)+word
            out[word] = out.get(word, 0)+sign*coefficient
        for scale, replacement in self.bracket(mode, first):
            terms = ((tail, self.one),) if replacement is None else self.action(replacement, tail)
            for word, coefficient in terms:
                out[word] = out.get(word, 0)+scale*coefficient
        return tuple((word, self.clean(v)) for word, v in out.items() if v != 0)

    @lru_cache(maxsize=None)
    def inner(self, bra, ket):
        if level(bra) != level(ket):
            return 0*self.one
        if not bra:
            return self.one if not ket else 0*self.one
        kind, index = bra[0]
        return self.clean(sum(
            coefficient*self.inner(bra[1:], word)
            for word, coefficient in self.action((kind, -index), ket)
        ))

    @lru_cache(maxsize=None)
    def gram(self, twice_level):
        states = basis(twice_level)
        matrix = sp.zeros(len(states)) if self.symbolic else mp.matrix(len(states))
        for i, bra in enumerate(states):
            for j in range(i+1):
                matrix[i,j] = matrix[j,i] = self.inner(bra, states[j])
        return matrix

    @lru_cache(maxsize=None)
    def inverse(self, twice_level):
        assert self.symbolic
        return self.gram(twice_level).inv(method="DM")

    @lru_cache(maxsize=None)
    def cholesky(self, twice_level):
        assert not self.symbolic
        gram = self.gram(twice_level)
        scale = tuple(mp.sqrt(gram[i,i]) for i in range(gram.rows))
        normalized = mp.matrix([
            [gram[i,j]/(scale[i]*scale[j]) for j in range(gram.cols)]
            for i in range(gram.rows)
        ])
        return mp.cholesky(normalized), scale

    def solve(self, twice_level, vector):
        if self.symbolic:
            return self.inverse(twice_level)*vector
        lower, scale = self.cholesky(twice_level)
        size = len(scale)
        forward = [mp.mpf(0)]*size
        for i in range(size):
            forward[i] = (vector[i]/scale[i]-sum(lower[i,j]*forward[j] for j in range(i)))/lower[i,i]
        backward = [mp.mpf(0)]*size
        for i in range(size-1, -1, -1):
            backward[i] = (forward[i]-sum(lower[j,i]*backward[j] for j in range(i+1,size)))/lower[i,i]
        return mp.matrix([backward[i]/scale[i] for i in range(size)])


class NSPrimaryWard:
    """rho(bra, (G_-1/2)^upper phi_d, ket) from local Ward identities."""

    def __init__(self, c, weights, symbolic=False, ket_module=None):
        self.left, self.d, self.right = weights
        self.module = ket_module or NSModule(c, self.right, symbolic)
        self.one = self.module.one

    def commutator(self, mode, bra, upper, ket):
        kind, twice_index = mode
        difference = self.left-self.right+self.one*(level(bra)-level(ket))/2
        if kind == "L":
            factor = difference+self.one*twice_index/2*(self.d+self.one*upper/2)
            return factor, upper
        if upper == 0:
            return self.one, 1
        return difference+twice_index*self.d, 0

    @lru_cache(maxsize=None)
    def value(self, bra, upper, ket):
        if bra:
            kind, index = bra[0]
            tail = bra[1:]
            mode = kind, -index
            factor, child_upper = self.commutator(mode, tail, upper, ket)
            out = factor*self.value(tail, child_upper, ket)
            sign = -1 if kind == "G" and upper else 1
            out += sign*sum(
                coefficient*self.value(tail, upper, word)
                for word, coefficient in self.module.action(mode, ket)
            )
            return self.module.clean(out)
        if ket:
            mode, tail = ket[0], ket[1:]
            factor, child_upper = self.commutator(mode, (), upper, tail)
            sign = 1 if mode[0] == "G" and upper else -1
            return self.module.clean(sign*factor*self.value((), child_upper, tail))
        return self.one


class DirectNSSpherePBW:
    """Plane comb coefficients with bottom or G_-1/2 external components.

    ``external_descendants`` labels actual states in their NS modules, not
    new NS primaries of shifted weight.  Vertex labels are fixed by the
    external markings and the parities of the requested internal levels.
    """

    def __init__(self, c, external, internal, symbolic=False, *, external_descendants=None):
        self.c, self.external, self.internal = c, tuple(external), tuple(internal)
        self.symbolic = symbolic
        self.m = len(internal)
        assert len(external) == self.m+3
        markings = (0,)*len(external) if external_descendants is None else tuple(external_descendants)
        if len(markings) != len(external) or any(type(x) is not int or x not in (0,1) for x in markings):
            raise ValueError("external_descendants must contain one zero/one integer per external field")
        self.external_descendants = markings
        self.left_external_state = (("G",-1),) if markings[0] else ()
        self.right_external_state = (("G",-1),) if markings[-1] else ()
        self.modules = tuple(NSModule(c, h, symbolic) for h in internal)
        self.left = NSPrimaryWard(c, (internal[0],external[1],external[0]), symbolic)
        self.right = NSPrimaryWard(c, (external[-1],external[-2],internal[-1]), symbolic, self.modules[-1])
        self.middle = tuple(
            NSPrimaryWard(c, (internal[i+1], external[i+2], internal[i]), symbolic, self.modules[i])
            for i in range(self.m-1)
        )

    def matrix(self, values):
        return sp.Matrix(values) if self.symbolic else mp.matrix(values)

    @lru_cache(maxsize=None)
    def propagated(self, levels):
        edge = len(levels)-1
        states = basis(levels[-1])
        if edge == 0:
            vector = self.matrix([self.left.value(state,self.external_descendants[1],self.left_external_state)
                                  for state in states])
        else:
            previous = self.propagated(levels[:-1])
            previous_states = basis(levels[-2])
            vector = self.matrix([
                sum(self.middle[edge-1].value(bra,self.external_descendants[edge+1],ket)*previous[j]
                    for j,ket in enumerate(previous_states))
                for bra in states
            ])
        result = self.modules[edge].solve(levels[-1], vector)
        return result.applyfunc(sp.cancel) if self.symbolic else result

    @lru_cache(maxsize=None)
    def coefficient(self, levels):
        assert len(levels) == self.m
        propagated = self.propagated(tuple(levels))
        result = sum(self.right.value(self.right_external_state,self.external_descendants[-2],state)*propagated[i]
                     for i,state in enumerate(basis(levels[-1])))
        return sp.cancel(result) if self.symbolic else result
