"""Heterotic all-NS five-point density and jets of the SAME elliptic table.

Labels 0..3 are outgoing, label 4 incoming; PCO labels stay (1,2,3).
The initial physical target is S -> VVVV.  No Type0B physical PCO sum,
Ramond normalization, inverse Gram production, or independent nome maps.

The returned density omits dPa dPb/pi^2, the sphere zero mode, couplings,
the string-measure phase, and (1/2)^3 from the string-note PCO prescription.
This makes the local CFT quantity and the outer normalization distinguishable.
"""
from dataclasses import dataclass, replace
from copy import copy
from itertools import combinations, product

import mpmath as mp

from spin23_fivepoint_c_elliptic import NSFivePointCBlock, frame_at
from spin23_fivepoint_type0b_reference import layers
from spin23_sphere_fivepoint import ns_fivepoint_structure_weights


PCO = (1, 2, 3)
BOTTOM = (0, 4)
TIME_SUBSETS = ((), (1, 2), (1, 3), (2, 3))
SECTORS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
TENSOR_NAMES = ("delta12_delta34", "delta13_delta24", "delta14_delta23")


def _ordering(ordering):
    ordering = tuple(ordering)
    if sorted(ordering) != list(range(5)):
        raise ValueError("one permutation of the five original labels required")
    return ordering


def graded_separation_sign(ordering, selected):
    """Literal odd-factor word, including the two odd delta(gamma)'s.

    Local words follow HETSO23_BASIC_SETUP.tex: delta(gamma)*lambda
    at bottom vectors, lambda*(Lambda+i k psi V) at zero-picture vectors,
    delta(gamma)*anti-Lambda at the incoming singlet. Stable-sort to
    ghost / time fermion / hol-SL / anti-SL / spectator sectors.
    bc pairs are even. The three-point specialization gives +Ctilde.
    """
    word = []
    for label in reversed(_ordering(ordering)):
        if label in BOTTOM:
            word.append(("ghost", label))
        word.append(("anti_SL" if label == 4 else "spectator", label))
        if label in PCO:
            word.append(("time" if label in selected else "hol_SL", label))
    rank = {name: j for j, name in enumerate(("ghost", "time", "hol_SL", "anti_SL", "spectator"))}
    exponent = sum(rank[a[0]] > rank[b[0]] for i, a in enumerate(word) for b in word[i+1:])
    return (-1)**exponent


def _difference_powers(i, j):
    """Outer-minus-inner, normalized when the outer puncture is infinity."""
    if i <= j:
        raise ValueError("slots must be strictly outer-to-inner")
    return {
        (1, 0): (1, 1, 0, 0, 0), (2, 0): (0, 1, 0, 0, 0),
        (3, 0): (0, 0, 0, 0, 0), (2, 1): (0, 1, 1, 0, 0),
        (3, 1): (0, 0, 0, 0, 1), (3, 2): (0, 0, 0, 1, 0),
        (4, 0): (0, 0, 0, 0, 0), (4, 1): (0, 0, 0, 0, 0),
        (4, 2): (0, 0, 0, 0, 0), (4, 3): (0, 0, 0, 0, 0),
    }[(i, j)]


def _add(powers, increment, scalar=1):
    return tuple(a+scalar*b for a, b in zip(powers, increment))


def spectator_factor(ordering, tensor):
    """Pfaffian coefficient in the canonical outer-to-inner spectator word."""
    if tensor not in range(3):
        raise ValueError("tensor index must be 0, 1, or 2")
    slots = {label: i for i, label in enumerate(_ordering(ordering))}
    word = tuple(label for label in reversed(ordering) if label != 4)
    pairs = {frozenset(p) for p in PAIRINGS[tensor]}
    sign, powers = 1, (0,)*5
    while word:
        partner = next(j for j in range(1, len(word)) if frozenset((word[0], word[j])) in pairs)
        sign *= (-1)**(partner-1)
        powers = _add(powers, _difference_powers(slots[word[0]], slots[word[partner]]), -1)
        word = word[1:partner]+word[partner+1:]
    return sign, powers


def free_terms(energies, ordering, tensor):
    """Four exact monomial prefactors, in this chart's d2x d2y measure."""
    ordering = _ordering(ordering)
    k = tuple(map(mp.mpc, energies))
    slots = {label: i for i, label in enumerate(ordering)}
    # Both halves contribute y: this is |y|^2, not an atlas weight.
    common = (0, 1, 0, 0, 0)
    for i, j in combinations(range(4), 2):
        common = _add(common, _difference_powers(j, i), -k[ordering[i]]*k[ordering[j]])
    ghost = _difference_powers(*sorted((slots[i] for i in BOTTOM), reverse=True))
    spectator_sign, spectator = spectator_factor(ordering, tensor)
    out = []
    for selected in TIME_SUBSETS:
        hol = _add(common, ghost, -1)
        anti = _add(common, spectator)
        scalar = mp.mpc(spectator_sign*graded_separation_sign(ordering, selected))
        if selected:
            hol = _add(hol, _difference_powers(*sorted((slots[i] for i in selected), reverse=True)), -1)
            # X0 X0=+log, psi0 psi0=+1/z implies G_X=psi0*dX0,
            # hence G_X e^(ikX0)=i*k*psi0 e^(ikX0)/z. Equivalently
            # use chi=i*psi0, chi chi=-1/z and a real +k coefficient.
            # The two-time-fermion term has i^2=-1. This is not a fit.
            scalar *= -mp.fprod(k[i] for i in selected)
        d = tuple(int(i in PCO and i not in selected) for i in ordering)
        dbar = tuple(int(i == 4) for i in ordering)
        out.append((selected, d, dbar, scalar, hol, anti))
    return tuple(out)


@dataclass(frozen=True)
class DensityPair:
    holomorphic: NSFivePointCBlock
    antiholomorphic: NSFivePointCBlock
    scalar: complex
    holomorphic_powers: tuple
    antiholomorphic_powers: tuple
    selected: tuple = ()
    sectors: tuple = ()


class EllipticDensity:
    """Complete component pairs with exact angular projection of boundary jets."""
    def __init__(self, pairs, *, precision=40):
        self.pairs, self.precision = tuple(pairs), precision
        if not self.pairs:
            raise ValueError("empty density")
        with mp.workdps(precision):
            exponents = [tuple(block.leading[i]+powers[i]+mp.mpf(block.parities[i])/2 for i in (0, 1))
                for pair in self.pairs for block, powers in (
                    (pair.holomorphic, pair.holomorphic_powers),
                    (pair.antiholomorphic, pair.antiholomorphic_powers))]
            self.betas = tuple(2*min((e[i] for e in exponents), key=lambda x: x.real) for i in (0, 1))
            for e in exponents:
                for i in (0, 1):
                    shift = 2*e[i]-self.betas[i]
                    if abs(shift-mp.nint(shift.real)) > mp.mpf('1e-10') or shift.real < -1e-10:
                        raise ValueError(f"incompatible OPE valuation: {shift}")
        self._coefficient_cache = {}

    def value(self, x, y):
        with mp.workdps(self.precision):
            frame = frame_at(complex(x), complex(y), self.precision).pillow
            logs = tuple(mp.log(z) for z in (x, y, 1-x, 1-y, 1-x*y))
            cache = {}
            def block_value(block, anti):
                key = id(block), anti
                if key not in cache:
                    ll = tuple(mp.conj(l) for l in logs[:2]) if anti else logs[:2]
                    cache[key] = mp.exp(sum(a*l for a, l in zip(block.leading, ll))) * frame.evaluate(
                        block.table(), antiholomorphic=anti)/frame.regular_multiplier(
                        block.c, block.effective, block.internal, antiholomorphic=anti)
                return cache[key]
            return mp.fsum(pair.scalar*block_value(pair.holomorphic, False)*block_value(pair.antiholomorphic, True)
                *mp.exp(sum(a*l+b*mp.conj(l) for a, b, l in zip(pair.holomorphic_powers,
                    pair.antiholomorphic_powers, logs))) for pair in self.pairs)

    def _chiral_jet(self, block, powers, edges, maximum, tangent, anti):
        degree = maximum//2
        t = complex(tangent).conjugate() if anti and tangent is not None else tangent
        x, ratios, p, (Z, Y, theta) = layers().jets.pillow_jet_geometry(edges, t, degree, self.precision)
        units = (1-x[0], 1-x[1], 1-x[0]*x[1])
        shifts, prefactor = [0, 0], mp.mpc(1)
        for i in (0, 1):
            exponent = block.leading[i]+powers[i]+mp.mpf(block.parities[i])/2
            if i in edges:
                shifts[i] = int(mp.nint((2*exponent-self.betas[i]).real))
            else:
                prefactor *= mp.exp(exponent*mp.log(t))
        regular = x[0].lift(0)
        for (n, m), coefficient in block.table().items():
            regular += coefficient*p[0]**((n-block.parities[0])//2)*p[1]**((m-block.parities[1])//2)
        h, d, c = block.internal, block.effective, block.c
        regular *= Z**(-h[0]+c/24)*Y**(h[0]-h[1])
        regular *= (units[0]*units[1])**(-d[2]/2)*units[2]**(c/24-d[1]-d[3])
        regular *= theta**(c/2-4*(d[0]+d[1]+d[3]+d[4])-2*d[2])
        for i in (0, 1):
            regular *= ratios[i]**(mp.mpf(block.parities[i])/2)
        for unit, power in zip(units, powers[2:]):
            regular *= unit**power
        return {tuple(2*k[i]+shifts[i] if i in edges else 0 for i in (0, 1)): prefactor*v
            for k, v in regular.c.items() if all(0 <= 2*k[i]+shifts[i] <= maximum for i in edges)}

    def coefficients(self, active_edges, maximum_increment, tangent=None, *, weight_coefficients=None):
        edges = tuple(sorted(active_edges))
        if edges not in ((0,), (1,), (0, 1)):
            raise ValueError("select a face or the double corner")
        if type(maximum_increment) is not int or maximum_increment < 0:
            raise ValueError("nonnegative integer OPE depth required")
        if len(edges) == 1 and (tangent is None or not 0 < abs(tangent) < 1):
            raise ValueError("a face requires its noncolliding tangential coordinate")
        key = edges, maximum_increment, complex(tangent) if tangent is not None else None
        if weight_coefficients is None and key in self._coefficient_cache:
            return self._coefficient_cache[key]
        with mp.workdps(self.precision):
            weights = {(0, 0, 0, 0): 1} if weight_coefficients is None else weight_coefficients
            output = {k: mp.mpc(0) for k in product(*(
                range(maximum_increment+1) if i in edges else (0,) for i in (0, 1)))}
            cache = {}
            for pair in self.pairs:
                jets = []
                for block, powers, anti in ((pair.holomorphic, pair.holomorphic_powers, False),
                                            (pair.antiholomorphic, pair.antiholomorphic_powers, True)):
                    ck = id(block), powers, anti
                    if ck not in cache:
                        cache[ck] = self._chiral_jet(block, powers, edges, maximum_increment, tangent, anti)
                    jets.append(cache[ck])
                for k in output:
                    for wk, wv in weights.items():
                        kh = tuple(k[i]-2*wk[i] for i in (0, 1))
                        ka = tuple(k[i]-2*wk[i+2] for i in (0, 1))
                        output[k] += pair.scalar*wv*jets[0].get(kh, 0)*jets[1].get(ka, 0)
        if weight_coefficients is None:
            self._coefficient_cache[key] = output
        return output


class SToFourVectors:
    """One channel and spectral node; all three tensor densities share blocks."""
    def __init__(self, outgoing, momenta, *, ordering=range(5), maximum_twice_levels=(4, 4), precision=40,
                 maximum_total_twice_level=None, coefficient_store=None):
        self.ordering, self.precision = _ordering(ordering), precision
        with mp.workdps(precision):
            outgoing = tuple(map(mp.mpc, outgoing))
            if len(outgoing) != 4 or not all(mp.isfinite(x) for x in outgoing):
                raise ValueError("four finite outgoing energies required")
            self.energies = outgoing+(-sum(outgoing),)
            self.external_momenta = outgoing+(sum(outgoing),)
            self.internal = tuple(map(mp.mpf, momenta))
            if len(self.internal) != 2 or any(p <= 0 for p in self.internal):
                raise ValueError("two strictly positive real NS momenta required")
            external = tuple((1+self.external_momenta[i]**2)/2 for i in self.ordering)
            internal = tuple((1+p*p)/2 for p in self.internal)
            constants = ns_fivepoint_structure_weights(internal_momenta=self.internal,
                external_momenta=tuple(self.external_momenta[i] for i in self.ordering), precision=precision)
            self.blocks = {}
            densities = []
            self.ledger = []
            for tensor in range(3):
                pairs = []
                for selected, d, dbar, scalar, hol, anti in free_terms(self.energies, self.ordering, tensor):
                    for a in SECTORS:
                        for marks in (d, dbar):
                            if (marks, a) not in self.blocks:
                                self.blocks[marks, a] = NSFivePointCBlock(external_weights=external,
                                    internal_weights=internal, stars=marks, sectors=a,
                                    maximum_twice_levels=maximum_twice_levels, precision=precision,
                                    maximum_total_twice_level=maximum_total_twice_level,
                                    coefficient_store=coefficient_store)
                        H, A = self.blocks[d, a], self.blocks[dbar, a]
                        structure = mp.fprod(constants[j][a[j]] for j in range(3))
                        phase = (-1)**(d[-1]+dbar[-1])
                        pairs.append(DensityPair(H, A, scalar*structure*phase, hol, anti, selected, a))
                        if tensor == 0:
                            self.ledger.append(dict(time_fermion_labels=selected, hol_stars=d, anti_stars=dbar,
                                sectors=a, hol_parities=H.parities, anti_parities=A.parities,
                                separation_sign=graded_separation_sign(self.ordering, selected),
                                time_pair_phase=-1 if selected else 1,
                                infinity_conversion=phase))
                densities.append(EllipticDensity(pairs, precision=precision))
            self.densities = tuple(densities)

    def prefix(self, physical_total_level):
        result = copy(self)
        result.blocks = {k: b.prefix(physical_total_level) for k, b in self.blocks.items()}
        mapping = {id(b): result.blocks[k] for k, b in self.blocks.items()}
        result.densities = tuple(EllipticDensity([replace(p,
            holomorphic=mapping[id(p.holomorphic)], antiholomorphic=mapping[id(p.antiholomorphic)])
            for p in e.pairs], precision=self.precision) for e in self.densities)
        return result

    def values(self, x, y):
        # The Liouville/PCO bracket is tensor-independent. Do not evaluate
        # its twenty blocks three times for three elementary Pfaffian terms.
        with mp.workdps(self.precision):
            first = self.densities[0].value(x, y)
            logs = tuple(mp.conj(mp.log(z)) for z in (x, y, 1-x, 1-y, 1-x*y))
            factors = [spectator_factor(self.ordering, j) for j in range(3)]
            s0, p0 = factors[0]
            return tuple(first*(s/s0)*mp.exp(sum((a-b)*l for a, b, l in zip(p, p0, logs)))
                for s, p in factors)
