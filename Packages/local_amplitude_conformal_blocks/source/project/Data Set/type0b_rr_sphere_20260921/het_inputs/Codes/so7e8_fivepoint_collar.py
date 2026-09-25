"""Mixed NS/R adapter for the existing Type0B elliptic/OPE/collar layers.

The local engine implements Type0B's value/coefficients/betas protocol.
Coefficients are algebraic jets of the SAME elliptic approximant used in
value(), not fitted data or a different plane truncation. Physical mixed
PCO/GSO assembly remains a separate, unvalidated input. The supplied factory
assembles only the existing scalar-primary CFT test, in comb area measure.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
import cmath

import mpmath as mp

from so7e8_type0b_boundary import boundary_layers
from so7e8_fivepoint_blocks import EDGES


def _cutoff(source, maximum):
    cut = source.maximum_twice_levels if maximum is None else tuple(maximum)
    if len(cut) != 2 or any(type(n) is not int or n < 0 or n > old
                           for n, old in zip(cut, source.maximum_twice_levels)):
        raise ValueError("requested coefficients lie beyond the supplied rectangle")
    return tuple(cut)


@lru_cache(maxsize=1024)
def _frame(x, y, precision):
    return boundary_layers().elliptic.five_point_pillow_frame(
        x, y, working_precision=precision)


class JointEllipticBlock:
    """Use Type0B's JOINT p1,p2 pullback with this project's c-table.

    R=M*S is a representation change, not an extra physical prefactor.
    Effective NS weights include every supplied G_-1/2 insertion. R states
    use their actual h_R and integer descendant levels; no 1/16 is removed.
    """
    def __init__(self, source, *, precision=40):
        if type(precision) is not int or precision < 30:
            raise ValueError("precision must be an integer >=30")
        self.source, self.precision = source, precision
        self.c = 1.5 + 3*(source.b + 1/source.b)**2
        self.effective = tuple(h+s/2 for h, s in zip(source.external_weights, source.stars))
        self.parities = tuple(source.edge_parities[i] if sector == "NS" else 0
                              for i, sector in enumerate(EDGES[source.channel]))
        if any(any((k[i]-self.parities[i]) % 2 for i in (0, 1)) for k in source.coefficients):
            raise ValueError("coefficient parity disagrees with the physical edge sector")
        self._tables = {}

    def table(self, maximum=None):
        cut = _cutoff(self.source, maximum)
        if cut not in self._tables:
            with mp.workdps(self.precision):
                transform = boundary_layers().elliptic.cached_five_point_pillow_series(
                    cut, sum(cut), self.precision)
                self._tables[cut] = transform.pullback(
                    self.source.coefficients, self.c, self.effective, self.source.internal_weights)
        return self._tables[cut]

    def value(self, x, y, *, logarithms=None, maximum_twice_levels=None,
              antiholomorphic=False):
        coordinates = tuple(map(complex, (x, y)))
        if any(not 0 < abs(q) < 1 for q in coordinates):
            raise ValueError("select a noncolliding open sewing bidisk")
        logs = tuple(map(cmath.log, coordinates)) if logarithms is None else tuple(logarithms)
        if len(logs) != 2 or any(abs(cmath.exp(l)-q) > 1e-10 for l, q in zip(logs, coordinates)):
            raise ValueError("logarithms must lift the supplied coordinates")
        with mp.workdps(self.precision):
            frame = _frame(*coordinates, self.precision)
            log_y, log_z = mp.log(frame.y_unit), mp.log(frame.z_unit)
            log_p = (mp.mpc(logs[0])-mp.log(4)-log_z+log_y,
                     mp.mpc(logs[1])-mp.log(4)-log_y)
            if antiholomorphic:
                logs = tuple(z.conjugate() for z in logs)
                log_p = tuple(mp.conj(z) for z in log_p)
            regular = mp.fsum(v*mp.exp((i*log_p[0]+j*log_p[1])/2)
                             for (i, j), v in self.table(maximum_twice_levels).items())
            leading = mp.exp(sum(a*l for a, l in zip(self.source.leading_powers, logs)))
            return leading*regular/frame.regular_multiplier(
                self.c, self.effective, self.source.internal_weights,
                antiholomorphic=antiholomorphic)


@dataclass(frozen=True)
class DensityPair:
    """An ordered chiral/antichiral pair; scalar includes sewing signs ONCE.

    Factor powers refer to x,y,1-x,1-y,1-xy. Callers must supply the complete
    spectator/ghost/PCO/coordinate Jacobian factors for an amplitude. The
    primary factory supplies only the exact comb area Jacobian |y|^2.
    """
    holomorphic: JointEllipticBlock
    antiholomorphic: JointEllipticBlock
    scalar: complex
    holomorphic_powers: tuple = (0, 1, 0, 0, 0)
    antiholomorphic_powers: tuple = (0, 1, 0, 0, 0)


class MixedDensityJets:
    """Type0B integration protocol, with mixed-sector inputs and no NS guess."""
    def __init__(self, pairs, *, betas, precision=40, observable="primary_CFT_diagnostic"):
        self.pairs, self.betas = tuple(pairs), tuple(map(complex, betas))
        self.precision, self.observable = precision, observable
        if not self.pairs or len(self.betas) != 2:
            raise ValueError("nonempty ordered component pairs and two explicit radial exponents required")
        if type(precision) is not int or precision < 30 or any(not mp.isfinite(b) for b in self.betas):
            raise ValueError("finite radial exponents and integer precision >=30 required")
        for pair in self.pairs:
            if len(pair.holomorphic_powers) != 5 or len(pair.antiholomorphic_powers) != 5:
                raise ValueError("five explicit geometric powers per chirality required")
        self._cache = {}
        # A collar-depth refinement must not repeat the unchanged bulk CFT.
        self.value = lru_cache(maxsize=4096)(self.value)

    def value(self, x, y):
        with mp.workdps(self.precision):
            logs = tuple(mp.log(q) for q in (x, y, 1-x, 1-y, 1-x*y))
            return mp.fsum(
                pair.scalar*pair.holomorphic.value(x, y)*pair.antiholomorphic.value(x, y, antiholomorphic=True)
                * mp.exp(sum(a*l+b*mp.conj(l) for a, b, l in zip(
                    pair.holomorphic_powers, pair.antiholomorphic_powers, logs)))
                for pair in self.pairs)

    def _chiral_jet(self, block, powers, edges, maximum, tangent, anti):
        degree = maximum//2
        t = complex(tangent).conjugate() if anti and tangent is not None else tangent
        x, ratios, p, (Z, Y, theta) = boundary_layers().jets.pillow_jet_geometry(
            edges, t, degree, self.precision)
        units = (1-x[0], 1-x[1], 1-x[0]*x[1])
        shifts, prefactor = [0, 0], mp.mpc(1)
        for i in (0, 1):
            exponent = mp.mpc(block.source.leading_powers[i])+powers[i]+mp.mpf(block.parities[i])/2
            if i in edges:
                twice = 2*exponent-self.betas[i]
                shifts[i] = int(mp.nint(twice.real))
                if abs(twice-shifts[i]) > mp.mpf('1e-9') or shifts[i] < 0:
                    raise ValueError(f"incompatible common radial exponent on edge {i}: shift={twice}")
            else:
                prefactor *= mp.exp(exponent*mp.log(t))
        regular = x[0].lift(0)
        for (n, m), coefficient in block.table().items():
            regular += coefficient*p[0]**((n-block.parities[0])//2)*p[1]**((m-block.parities[1])//2)
        h, d, c = block.source.internal_weights, block.effective, mp.mpc(block.c)
        regular *= Z**(-mp.mpc(h[0])+c/24)*Y**(mp.mpc(h[0])-h[1])
        regular *= (units[0]*units[1])**(-mp.mpc(d[2])/2)*units[2]**(c/24-d[1]-d[3])
        regular *= theta**(c/2-4*(d[0]+d[1]+d[3]+d[4])-2*d[2])
        for i in (0, 1):
            regular *= ratios[i]**(mp.mpf(block.parities[i])/2)
        for unit, power in zip(units, powers[2:]):
            regular *= unit**power
        return {tuple(2*k[i]+shifts[i] if i in edges else 0 for i in (0, 1)): prefactor*v
                for k, v in regular.c.items()
                if all(2*k[i]+shifts[i] <= maximum for i in edges)}

    def coefficients(self, active_edges, maximum_increment, tangent=None, *, weight_coefficients=None):
        edges = tuple(sorted(active_edges))
        if edges not in ((0,), (1,), (0, 1)):
            raise ValueError("select one face or the common corner")
        if type(maximum_increment) is not int or maximum_increment < 0:
            raise ValueError("nonnegative integer OPE depth required")
        if len(edges) == 1 and (tangent is None or not 0 < abs(tangent) < 1):
            raise ValueError("a face requires a noncolliding tangential modulus")
        key = edges, maximum_increment, complex(tangent) if tangent is not None else None
        if weight_coefficients is None and key in self._cache:
            return self._cache[key]
        with mp.workdps(self.precision):
            weights = {(0, 0, 0, 0): 1} if weight_coefficients is None else weight_coefficients
            output = {k: mp.mpc(0) for k in product(*(
                range(maximum_increment+1) if i in edges else (0,) for i in (0, 1)))}
            for pair in self.pairs:
                jets = [self._chiral_jet(block, powers, edges, maximum_increment, tangent, anti)
                        for block, powers, anti in ((pair.holomorphic, pair.holomorphic_powers, False),
                                                   (pair.antiholomorphic, pair.antiholomorphic_powers, True))]
                for k in output:
                    for wk, wv in weights.items():
                        kh, ka = tuple(k[i]-2*wk[i] for i in (0, 1)), tuple(k[i]-2*wk[i+2] for i in (0, 1))
                        output[k] += pair.scalar*wv*jets[0].get(kh, 0)*jets[1].get(ka, 0)
        if weight_coefficients is None:
            self._cache[key] = output
        return output


def primary_collar_engine(bank, weights, *, metric_multiplier, precision=40):
    """Adapt a complete primary bank; never infer a Ramond pairing factor."""
    if not bank or set(bank) != set(weights):
        raise ValueError("weights must cover exactly the compiled primary bank")
    anchor = next(iter(bank.values()))
    from so7e8_fivepoint_primary import primary_labels
    if set(bank) != set(primary_labels(anchor.channel)):
        raise ValueError("the primary collar requires all ordered sewing components")
    betas = tuple(2*a+(2 if i == 1 else 0) for i, a in enumerate(anchor.leading_powers))
    pairs = []
    for label, source in bank.items():
        if source.channel != anchor.channel or source.leading_powers != anchor.leading_powers or any(source.stars):
            raise ValueError("this factory is only the unstarred scalar-primary observable")
        block = JointEllipticBlock(source, precision=precision)
        pairs.append(DensityPair(block, block, complex(metric_multiplier*weights[label])))
    return MixedDensityJets(pairs, betas=betas, precision=precision)


def integrate_primary_patch(engine, **options):
    """Reuse the Type0B disjoint bulk/faces/corner driver without alteration."""
    if engine.observable != "primary_CFT_diagnostic":
        raise ValueError("physical amplitude assembly has not passed its separate gate")
    result = boundary_layers().integration.integrate_ope_patch(engine, **options)
    return dict(result, observable=engine.observable, physical_amplitude=False,
                global_moduli_integral=False, convergence_certified=False)


def collar_forest(faces):
    """Use the upstream validated 10-face/15-corner ledger, not a new count."""
    layer = boundary_layers().forest
    faces = tuple(layer.canonical_divisor(pair) for pair in faces)
    corners = [layer.canonical_corner(a, b) for a, b in combinations(faces, 2)
               if layer.compatible_divisors(a, b)]
    return layer.FivePointCollarForestPlan.from_subtraction_audit(dict(
        divergent_boundary_pairs=[list(p) for p in faces],
        corner_overlap_subtractions=[[list(p) for p in corner] for corner in corners]))
