"""Ising decomposition of the ordered auxiliary ``Q Theta psi(1)`` block.

The parity decomposition is exact.  ``RegulatedIsingCCY`` deliberately does
not claim that a generic-Verma limit implements the irreducible Ising
quotient: a closed chain of null submodules can survive such a limit.
This candidate is exposed for the requested end-to-end PBW comparison,
never as a silent Fock/PBW replacement for central-charge recursion.

Indices use (2*l_NS, l_R_left, l_R_right, 2*l_R3), as in series_algebra.
"""

from __future__ import annotations

from functools import lru_cache
import mpmath

try:
    from .punctured_ccy import PuncturedCCY, downward_indices
except ImportError:
    from punctured_ccy import PuncturedCCY, downward_indices


class IrreducibleIsingLimitUnresolved(RuntimeError):
    """The CCY Verma recursion alone has not specified the Ising quotient."""


def parity_coefficients(ns_primary, scalar, Q=1):
    """Multiply a normalized Ising block by its exact primary sewing data.

    The common factor 1/sqrt(2) is supplied by the caller's own precision
    context.  For the vacuum the output is scalar*(1-eta2*eta3); for the
    fermion it is scalar*(eta1*eta2+eta1*eta3)/2.  ``scalar`` therefore
    already includes Q/sqrt(2).
    """
    out = [scalar * 0 for _ in range(8)]
    if int(ns_primary) == 0:
        out[0], out[6] = Q * scalar, -Q * scalar
    elif int(ns_primary) == 1:
        out[3] = out[5] = Q * scalar / 2
    else:
        raise ValueError("ns_primary must be 0 (identity) or 1 (fermion)")
    return tuple(out)


class RegulatedIsingCCY:
    """A complete-sum generic CCY limit, with its limitation made explicit.

    This class evaluates the genuine c-recursion at three nearby generic
    points and takes a quadratic Richardson constant.  It does not discard
    individually singular residues.  It also does not remove a null loop.
    The latter must be derived before this can be certified as the Ising
    block.  Instantiation requires ``allow_unproved_limit=True`` so the
    calling end-to-end run records this assumption rather than concealing it.

    ``regulator='linear'`` takes fixed Ising weights plus different first
    order perturbations, avoiding identically confluent internal poles.
    ``'kac_centered'`` continues Kac labels relative to the centre of the
    (3,4) table and adds smaller distinct second-order displacements.
    Neither prescription is asserted to quotient null loops.
    """

    irreducible_quotient_established = False

    def __init__(self, Q=1, *, dps=100, epsilon=None, regulator="linear",
                 allow_unproved_limit=False):
        if not allow_unproved_limit:
            raise IrreducibleIsingLimitUnresolved(
                "A generic CCY limit can retain closed Ising null-submodule "
                "loops. Supply a proved quotient prescription before using "
                "this as production; allow_unproved_limit=True is only for "
                "the explicitly requested end-to-end failure diagnosis."
            )
        if regulator not in ("linear", "kac_centered", "fixed_weights"):
            raise ValueError("regulator must be linear, kac_centered, or fixed_weights")
        self.mp = mpmath.mp.clone()
        self.mp.dps = int(dps)
        self.Q = self.mp.mpc(Q)
        self.epsilon = (self.mp.mpf(epsilon) if epsilon is not None
                        else self.mp.power(10, -max(8, int(dps)//8)))
        self.regulator = regulator
        self.engines = tuple(
            tuple(self._engine(ns, self.epsilon / (2 ** j)) for j in range(3))
            for ns in (0, 1)
        )
        self.diagnostics = {
            "backend": "CCY central-charge residues with global seed",
            "representation": "regulated generic Verma, quotient unproved",
            "regulator": regulator,
            "epsilon": self.mp.nstr(self.epsilon, 20),
            "dps": int(dps),
            "extrapolation": "(F(e)-6 F(e/2)+8 F(e/4))/3",
            "null_loop_subtraction": False,
        }

    def _engine(self, ns, epsilon):
        mp = self.mp
        if self.regulator == "linear":
            c = mp.mpf(1)/2 + epsilon
            hns = mp.mpf(ns)/2 + mp.mpf(3)*epsilon
            r = tuple(mp.mpf(1)/16 + slope*epsilon
                      for slope in (mp.mpf(5), mp.mpf(7), mp.mpf(11)))
            hext = mp.mpf(1)/2 + mp.mpf(13)*epsilon
        elif self.regulator == "fixed_weights":
            # c is the first deformation.  Vacuum and internal-weight
            # displacements are asymptotically smaller.  In particular,
            # the always-null L_-1 vacuum submodule is removed first.
            c = mp.mpf(1)/2 + epsilon
            hns = mp.mpf(ns)/2 + mp.mpf(3)*epsilon**3
            r = tuple(mp.mpf(1)/16+slope*epsilon**3
                      for slope in (mp.mpf(5),mp.mpf(7),mp.mpf(11)))
            hext = mp.mpf(1)/2 + mp.mpf(13)*epsilon**3
        else:
            # At b^2=-4/3, h_(1,2)=1/16, h_(2,1)=1/2.
            b = mp.sqrt(-mp.mpf(4)/3 + epsilon)
            q = b + 1/b
            c = 1 + 6*q*q

            def centred_h(r, s):
                momentum_twice = (r-mp.mpf(3)/2)*b + (s-2)/b
                return (q*q-momentum_twice*momentum_twice)/4

            hns = centred_h(1,1) if ns == 0 else centred_h(2,1)
            r = tuple(centred_h(1,2)+slope*epsilon**2
                      for slope in (mp.mpf(5),mp.mpf(7),mp.mpf(11)))
            hext = centred_h(2,1)+mp.mpf(13)*epsilon**2
        return PuncturedCCY(c, (hns,r[0],r[1],r[2]), hext, dps=mp.dps)

    @lru_cache(maxsize=None)
    def reduced_ising_coefficient(self, ns_primary, levels):
        values = [engine.reduced_coefficient(tuple(levels))
                  for engine in self.engines[int(ns_primary)]]
        return (values[0]-6*values[1]+8*values[2])/3

    def series(self, level, *, reduced=True):
        if not reduced:
            raise NotImplementedError(
                "Multiply the reduced series by the one common exact CCY "
                "vacuum seed using the root assembly's series algebra."
            )
        out = {}
        for ns in (0,1):
            for a,l,r,d in downward_indices(2*int(level)-ns, (2,1,1,2)):
                value = (self.Q/self.mp.sqrt(2)
                         *self.reduced_ising_coefficient(ns,(a,l,r,d)))
                out[(2*a+ns,l,r,2*d)] = parity_coefficients(ns,value)
        return out


class LevelFiveNullCycleIsingCCY(RegulatedIsingCCY):
    """First-null-loop subtraction, explicitly restricted to total level 5.

    This is an intermediate derivation, not a completed level-10 Ising
    implementation.  The sigma and fermion representations both have a
    level-two null vector with leading term L_-2.  At fixed highest weights
    its norm has c-derivative 1/2; so does a vertex with two such nulls.
    A closed level-two null loop thus has normalized primary coefficient
    one.  Subtract its shifted CCY block.  There are three simple cycles:
    (L,R,3), (NS,3), (NS,L,R).  Only the first exists in the NS vacuum branch.

    Before using this beyond a diagnostic, the requested physical PBW
    comparison must confirm the complete sewing/limit conventions.  Later
    primitive nulls and their intersections are deliberately not guessed.
    """

    def __init__(self, *args, **kwargs):
        kwargs["regulator"] = "fixed_weights"
        # Internal pole separations are O(epsilon**3).  Their cancellation
        # consumes far more guard digits than an ordinary first-order path.
        # Quadratic extrapolation still gives O(epsilon**3) truncation error.
        if kwargs.get("epsilon") is None:
            kwargs["epsilon"] = "1e-" + str(max(4,int(kwargs.get("dps",100))//16))
        super().__init__(*args, **kwargs)
        self.shifted_engines = {}
        self.diagnostics.update({
            "null_loop_subtraction": "all level-two simple cycles",
            "maximum_balanced_physical_level": 5,
            "higher_null_embeddings": "not yet implemented",
        })

    def _shifted(self, ns, cycle):
        key = (ns, cycle)
        if key not in self.shifted_engines:
            out = []
            for engine in self.engines[ns]:
                weights = tuple(h + (2 if edge in cycle else 0)
                                for edge,h in enumerate(engine.weights))
                out.append(PuncturedCCY(engine.c, weights,
                                        engine.external_weight,dps=self.mp.dps))
            self.shifted_engines[key] = tuple(out)
        return self.shifted_engines[key]

    @lru_cache(maxsize=None)
    def reduced_ising_coefficient(self, ns_primary, levels):
        ns = int(ns_primary)
        levels = tuple(levels)
        if 2*levels[0]+levels[1]+levels[2]+2*levels[3]+ns > 10:
            raise NotImplementedError("Only first-null cycles through level 5 are derived")
        answer = super().reduced_ising_coefficient(ns, levels)
        cycles = ((1,2,3),) if ns == 0 else ((1,2,3),(0,3),(0,1,2))
        for cycle in cycles:
            remaining = tuple(n-(2 if edge in cycle else 0)
                              for edge,n in enumerate(levels))
            if min(remaining) < 0:
                continue
            values = [engine.reduced_coefficient(remaining)
                      for engine in self._shifted(ns,cycle)]
            answer -= (values[0]-6*values[1]+8*values[2])/3
        return answer

    def series(self, level, *, reduced=True):
        if level > 5:
            raise NotImplementedError("Higher Ising null embeddings are not yet implemented")
        return super().series(level,reduced=reduced)


class IsingFermion(LevelFiveNullCycleIsingCCY):
    """Production auxiliary factor, established only through total level 5.

    The two requested integrated comparisons are recorded in
    results/validation_level5_mp_nullcycles.json.  The parent class enforces
    the level bound on both the series and individual coefficient APIs.
    No generic-Verma limit or level-ten extension is silently substituted.
    """

    established_through_level = 5
    irreducible_quotient_established = True  # Only within the enforced bound.

    def __init__(self, *args, **kwargs):
        # The lower-level generic evaluator is an ingredient of the proved
        # bounded subtraction, not an assertion about the uncorrected limit.
        kwargs["allow_unproved_limit"] = True
        super().__init__(*args, **kwargs)
        self.diagnostics.update({
            "representation": "irreducible Ising through balanced total level 5",
            "established_through_level": 5,
            "validation_report": "results/validation_level5_mp_nullcycles.json",
            "higher_null_embeddings": "not yet implemented",
        })
