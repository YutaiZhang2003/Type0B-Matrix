"""L_1 Ward recursion for the radial Q Theta psi primary matrix element.

Production primary matrix elements are obtained recursively from the four
Ramond ground anchors.  The reusable L_{+/-1} action coefficients are computed
by the free-field descendant-span method already used by the paper.

No physical PBW matrix element is evaluated by this module.  Numeric precision
is inherited from ramond_branching_recursion.compute_target; ``dps`` explicitly
sets that module's shared precision when supplied.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys

_BRANCH_DIR = Path(__file__).resolve().parents[1] / "ramond_branching_recursion"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))
import compute_target as br


def _label(value):
    value = Fraction(value)
    if (4 * value).denominator != 1 or int(4 * value) % 2 == 0:
        raise ValueError("Ramond n must belong to Z/2 + 1/4")
    return value


class RamondActions:
    """Cached V coefficients, with negative labels evaluated by reflection."""

    def __init__(self, b, momentum):
        self.b, self.momentum = b, momentum
        self.diagnostics = {}

    @lru_cache(None)
    def module(self, reflected=False):
        return br.FreeFieldModule("R", self.b,
                                  -self.momentum if reflected else self.momentum)

    @lru_cache(None)
    def plus(self, label, parity):
        label = _label(label)
        if abs(label) == Fraction(1, 4):
            return ()
        sign = 1 if label > 0 else -1
        n = abs(label)
        module = self.module(sign < 0)
        high = module.r_branch(n, parity)
        low = module.r_branch(n - 1, parity)
        pairs = br.partition_pairs(int(4 * n - 3))
        columns = [module.descendant(low, first, second)
                   for first, second in pairs]
        fit = br.span_fit(module.apply_l(1, high), columns)
        self.diagnostics[("plus", label, parity)] = {
            key: value for key, value in fit.items() if key != "coefficients"
        }
        return tuple(br.ActionTerm(sign * (n - 1), first, second, value)
                     for (first, second), value in zip(pairs, fit["coefficients"]))

    @lru_cache(None)
    def minus(self, label, parity):
        label = _label(label)
        sign = 1 if label > 0 else -1
        terms, fit = br.solve_ramond_lminus(self.module(sign < 0), abs(label), parity)
        self.diagnostics[("minus", label, parity)] = {
            key: value for key, value in fit.items() if key != "coefficients"
        }
        return tuple(br.ActionTerm(sign * term.label, term.first, term.second,
                                   term.coefficient) for term in terms)


class MiddleBranching:
    """Radial matrix elements <v_out^a|Q Theta psi(1)|v_in^a>.

    ``actions`` can supply cached ``plus(n,a)`` and ``minus(n,a)`` ActionTerms.
    This allows reuse of the same V coefficients between blocks and momenta.
    No conjugation is taken: all matrix elements and norms are BPZ bilinear.
    """

    def __init__(self, b, momentum, *, dps=None, actions=None):
        if dps is not None:
            br.set_multiprecision(dps)
        self.b = br.real_number(b)
        self.momentum = br.complex_number(momentum)
        self.q = self.b + 1 / self.b
        self.weights = br.BranchWeights(self.b, (self.momentum,) * 3)
        self.external_weights = (
            -(1 + 2 * self.b ** 2) / (2 * (1 - self.b ** 2)),
            (self.b ** 2 + 2) / (2 * (1 - self.b ** 2)),
        )
        self.actions = actions or RamondActions(self.b, self.momentum)
        self.recursion_records = []

    def norm_squared(self, label, parity):
        return br.ramond_norm_squared(_label(label), parity,
                                      self.b, self.momentum)

    @lru_cache(None)
    def form(self, outgoing, incoming, copy):
        return br.VirasoroThreePoint(
            (self.weights.weight(0, outgoing, copy), self.external_weights[copy],
             self.weights.weight(0, incoming, copy)),
            self.weights.central_charges[copy])

    def _factor(self, outgoing, incoming, term, slot):
        first = self.form(outgoing, incoming, 0)
        second = self.form(outgoing, incoming, 1)
        words1 = [(), (), ()]
        words2 = [(), (), ()]
        words1[slot], words2[slot] = term.first, term.second
        return term.coefficient * first.value(*words1) * second.value(*words2)

    @lru_cache(None)
    def raw(self, outgoing, incoming, parity):
        outgoing, incoming = _label(outgoing), _label(incoming)
        parity = int(parity)
        if parity not in (0, 1):
            raise ValueError("parity must be 0 or 1")
        if abs(outgoing - incoming) != Fraction(1, 2):
            return br.complex_number(0)
        if abs(outgoing) == abs(incoming) == Fraction(1, 4):
            # Q Theta psi_0=(Q/sqrt(2))(-1)^auxiliary_ground.
            return self.q * br.scalar_sqrt(2) / (2 ** parity)

        denominator, numerator = br.complex_number(0), br.complex_number(0)
        if abs(outgoing) > abs(incoming):
            # <L1 out|O|in> = <out|O L_-1 in>.
            for term in self.actions.minus(incoming, parity):
                if term.label == incoming:
                    denominator += self._factor(outgoing, incoming, term, 2)
                elif abs(outgoing - term.label) == Fraction(1, 2):
                    raise AssertionError("A supposedly forbidden distant branch survived")
            for term in self.actions.plus(outgoing, parity):
                numerator += (self._factor(term.label, incoming, term, 0)
                              * self.raw(term.label, incoming, parity))
        else:
            # <L_-1 out|O|in> = <out|O L1 in>.
            for term in self.actions.minus(outgoing, parity):
                if term.label == outgoing:
                    denominator += self._factor(outgoing, incoming, term, 0)
                elif abs(term.label - incoming) == Fraction(1, 2):
                    raise AssertionError("A supposedly forbidden distant branch survived")
            for term in self.actions.plus(incoming, parity):
                numerator += (self._factor(outgoing, term.label, term, 2)
                              * self.raw(outgoing, term.label, parity))
        if abs(denominator) <= br.arithmetic_tolerance():
            raise ZeroDivisionError(
                f"L1 Ward pivot vanishes at n'= {outgoing}, n={incoming}, alpha={parity}; "
                "evaluate at generic momentum and analytically continue")
        value = numerator / denominator
        self.recursion_records.append(dict(outgoing=str(outgoing), incoming=str(incoming),
                                            parity=parity, denominator=denominator))
        return value

    def theta_coefficient(self, label, parity):
        """Theta v_n^a = returned scalar times v_n^{1-a}."""
        m = int(2 * abs(_label(label)) - Fraction(1, 2))
        return (-br.scalar_power_of_two(Fraction((-1) ** m, 2)) if parity == 0
                else br.scalar_power_of_two(Fraction(-(-1) ** m, 2)))

    def psi_raw(self, outgoing, incoming, outgoing_parity):
        """<v_out^a|V[v_1/2](1)|v_in^{1-a}>, without norm division."""
        return -self.raw(outgoing, incoming, outgoing_parity) / self.theta_coefficient(
            incoming, outgoing_parity)

    def branching(self, outgoing, incoming, outgoing_parity):
        """Radially ordered B(n',1/2,n;a,1-a), with principal norm roots.

        The external irreducible-vacuum primary is Q psi, of norm -Q^2
        in the notes' BPZ fermion convention.  No normalized square roots
        are used by the enlarged-block production contraction.
        """
        norm = (br.scalar_sqrt(self.norm_squared(outgoing, outgoing_parity))
                * br.scalar_sqrt(-self.q ** 2)
                * br.scalar_sqrt(self.norm_squared(incoming, 1-outgoing_parity)))
        return self.psi_raw(outgoing, incoming, outgoing_parity) / norm

    @staticmethod
    def required_pairs(total_q_level):
        """All primary pairs permitted by balanced split-edge total cutoff.

        qa=u*sqrt(q2), qb=sqrt(q2)/u, so the base q2 degree is the
        mean of the two branch levels.  Descendants only increase it.
        """
        cutoff = Fraction(total_q_level)
        bound = int((2 * float(cutoff) + 1) ** .5 * 2) + 5
        labels = [Fraction(k, 4) for k in range(-2*bound-1, 2*bound+2, 2)]
        return tuple((outgoing, incoming) for outgoing in labels for incoming in labels
                     if abs(outgoing-incoming) == Fraction(1, 2)
                     and outgoing*outgoing+incoming*incoming-Fraction(1, 8) <= cutoff)
