"""Direct free-fermion sewing with the full ordered Q Theta psi(1).

This backend is intentionally not CCY recursion.  It was added after
the direct-definition auxiliary calculation was explicitly requested.
Every nonzero fermion mode is retained on the split Ramond edge.

``normalized_series(N)`` returns exact rational coefficients after
factoring out Q/sqrt(2).  Keys are (2*l_NS, l_L, l_R, 2*l_3), with
sum(key) <= 2*N.  ``series`` restores Q/sqrt(2) with mpmath arithmetic.
"""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from flint import fmpq
import mpmath


@lru_cache(maxsize=None)
def _strict_partitions(total, maximum, odd=False):
    """Descending positive distinct modes, optionally odd NS twice-modes."""
    if total == 0:
        return ((),)
    if total < 0 or maximum < 1:
        return ()
    answer = []
    for first in range(min(total, maximum), 0, -1):
        if odd and first % 2 == 0:
            continue
        for tail in _strict_partitions(total-first, first-1, odd):
            answer.append((first,) + tail)
    return tuple(answer)


@lru_cache(maxsize=None)
def _binomial_half(numerator, order):
    if order == 0:
        return fmpq(1)
    return (_binomial_half(numerator, order-1)
            * fmpq(numerator-2*(order-1), 2*order))


def _sign(exponent):
    return -1 if exponent & 1 else 1


def _parity(slot, state):
    return len(state) & 1 if slot == 0 else (len(state[0])+state[1]) & 1


def _act(slot, mode, state):
    """Fermion action in the rational basis e0=u0, e1=sqrt(2)*u1.

    NS modes are twice their physical mode numbers. Ramond modes are
    integers.  The zero mode has coefficients 1/2 and 1 in this basis.
    """
    modes = state if slot == 0 else state[0]
    if mode == 0:
        if slot == 0:
            raise ValueError("The NS sector has no zero mode")
        ground = state[1]
        return ((modes, 1-ground),
                fmpq(_sign(len(modes)), 1 if ground else 2))
    if mode > 0:
        if mode not in modes:
            return None, fmpq(0)
        position = modes.index(mode)
        final_modes = modes[:position] + modes[position+1:]
    else:
        created = -mode
        if created in modes:
            return None, fmpq(0)
        position = sum(existing > created for existing in modes)
        final_modes = modes[:position] + (created,) + modes[position:]
    final = final_modes if slot == 0 else (final_modes, state[1])
    return final, fmpq(_sign(position))


class RationalAuxiliaryThreePoint:
    """Exact Human-Note NSRR form from the fermion contour Ward identity.

    The returned rational r is defined by

      rho(NS, B*g, C*h) = i**parity(B*g) * r / sqrt(2)**(g+h).

    Equivalently r strips the known phase from the form in the rescaled
    rational Ramond basis. Its primary values are r(1,e0,e0)=1 and
    r(1,e1,e1)=2.  Dynamic mode support replaces the old fixed cutoff16.
    """

    def __init__(self):
        self.cache = {}
        self.calls = 0
        self.cache_hits = 0
        self.ward_terms = 0

    def value(self, states):
        self.calls += 1
        if states in self.cache:
            self.cache_hits += 1
            return self.cache[states]
        parities = tuple(_parity(slot, state)
                         for slot, state in enumerate(states))
        if sum(parities) & 1:
            return fmpq(0)
        ns, second, third = states
        if ns:
            target_slot = 0
            rest = (ns[1:], second, third)
        elif second[0]:
            target_slot = 1
            rest = (ns, (second[0][1:], second[1]), third)
        elif third[0]:
            target_slot = 2
            rest = (ns, second, (third[0][1:], third[1]))
        else:
            result = fmpq(1 << second[1])
            self.cache[states] = result
            return result

        rest_parity = tuple(_parity(slot, state)
                            for slot, state in enumerate(rest))
        koszul = (1, _sign(rest_parity[0]),
                  _sign(rest_parity[0]+rest_parity[1]))
        largest = (rest[0][0] if rest[0] else 0,
                   rest[1][0][0] if rest[1][0] else 0,
                   rest[2][0][0] if rest[2][0] else 0)
        target_phase = (1, 0, 3)[target_slot]
        target_coefficient = fmpq(0)
        remainder = fmpq(0)

        def add(slot, mode, coefficient, phase):
            nonlocal target_coefficient, remainder
            final, action = _act(slot, mode, rest[slot])
            if not action or not coefficient:
                return
            changed = rest[:slot] + (final,) + rest[slot+1:]
            exponent = (phase + _parity(1, changed[1])
                        - parities[1] - target_phase) % 4
            if exponent & 1:
                raise ArithmeticError("Inconsistent fermion Ward phase")
            weight = coefficient * action * _sign(exponent//2)
            if changed == states:
                target_coefficient += weight
            else:
                self.ward_terms += 1
                remainder += weight * self.value(changed)

        if target_slot == 0:
            first_mode = ns[0]
            k = (first_mode+1)//2
            cutoff = max((first_mode+largest[0])//2,
                         largest[1], largest[2]-k, 0)
            for j in range(cutoff+1):
                a = _sign(j)*_binomial_half(-1, j)
                add(0, 2*j-first_mode, koszul[0]*a, 1)
                add(1, j, koszul[1]*_binomial_half(2*k-1, j), 0)
                add(2, k+j, koszul[2]*a, 3)
        elif target_slot == 1:
            n = second[0][0]
            cutoff = max((largest[0]-2*n-1)//2,
                         n+largest[1], largest[2], 0)
            for j in range(cutoff+1):
                d = _binomial_half(-1, j)
                e = _sign(j)*_binomial_half(-2*n-1, j)
                add(0, 2*(n+j)+1, koszul[0]*e, 1)
                add(1, -n+j, koszul[1]*d, 0)
                add(2, j, koszul[2]*_sign(n)*e, 3)
        else:
            n = third[0][0]
            cutoff = max((largest[0]-2*n-1)//2,
                         largest[1], n+largest[2], 0)
            for j in range(cutoff+1):
                a = _sign(j)*_binomial_half(-1, j)
                add(0, 2*(n+j)+1, koszul[0]*a, 1)
                add(1, j, koszul[1]*_binomial_half(-2*n-1, j), 0)
                add(2, -n+j, koszul[2]*a, 3)
        if not target_coefficient:
            raise ArithmeticError("The fermion Ward identity missed its target")
        result = -remainder/target_coefficient
        self.cache[states] = result
        return result


class DirectFermion:
    """Sparse exact state sewing for the full split-edge fermion field.

    The local ingredients (Fock bases, mode Ward form and sparse fermion
    insertion) are independent of the genus. This method assembles the
    specific four-edge graph requested here, preserving all eight parity
    components and all off-diagonal split-edge coefficients.
    """

    def __init__(self, Q=1, *, arithmetic="exact", dps=80):
        if arithmetic != "exact":
            raise ValueError("This provider uses exact FLINT rational arithmetic")
        self.Q = Q
        self.dps = int(dps)
        self.form = RationalAuxiliaryThreePoint()
        self.diagnostics = {
            "backend": "direct fermion mode Ward identities and sparse state sewing",
            "arithmetic": "FLINT fmpq",
            "normalization": "normalized_series factors out Q/sqrt(2)",
            "insertion": "full ordered Q Theta psi(1), including nonzero modes",
            "key": "(2*l_NS,l_L,l_R,2*l_3); sum(key)<=2*N",
            "ramond_basis": "descending distinct positive modes, ground0 or1",
            "ward_basis": "e0=u0,e1=sqrt(2)*u1; known i parity phase stripped",
            "hardcoded_mode_cutoff": False,
            "validation": "no additional numerical comparisons performed",
        }

    def normalized_series(self, level, progress=None):
        level = int(level)
        if level < 0:
            raise ValueError("level must be nonnegative")
        started = perf_counter()
        cutoff = 2*level
        ns_basis = tuple(_strict_partitions(n, n, True)
                         for n in range(cutoff+1))
        r_basis = tuple(_strict_partitions(n, n)
                        for n in range(cutoff+1))
        basis_seconds = perf_counter()-started
        answer = {}
        contractions = zero_contractions = toggle_contractions = 0
        next_progress = started

        def accumulate(key, component, value):
            if not value:
                return
            if key not in answer:
                answer[key] = [fmpq(0) for _ in range(8)]
            answer[key][component] += value

        for ns_level, ns_states in enumerate(ns_basis):
            a = ns_level & 1
            for r3_level in range((cutoff-ns_level)//2+1):
                budget = cutoff-ns_level-2*r3_level
                for lower_level in range(budget//2+1):
                    for ns in ns_states:
                        for modes3 in r_basis[r3_level]:
                            for lower_modes in r_basis[lower_level]:
                                for ground in (0, 1):
                                    b = (len(lower_modes)+ground) & 1
                                    c = a ^ b
                                    ground3 = (c-len(modes3)) & 1
                                    component = a | (b << 1) | (c << 2)
                                    theta = _sign(a*b+a*c+b*c)
                                    # Two stripped outer phases multiply to i^(2*b).
                                    common = _sign(a+b)*theta
                                    third = (modes3, ground3)
                                    left = (lower_modes, ground)
                                    rho_left = self.form.value((ns, left, third))
                                    if not rho_left:
                                        continue
                                    # Same state: sqrt(2)*Theta*psi0=(-1)^ground.
                                    zero_value = (common*_sign(ground)*rho_left*rho_left
                                                  / (1 << (ground+ground3)))
                                    key = (ns_level, lower_level, lower_level, 2*r3_level)
                                    accumulate(key, component, zero_value)
                                    contractions += 1
                                    zero_contractions += 1

                                    # Enumerate an unordered state pair by creation.
                                    # Removal has the same action sign, so it supplies
                                    # the transposed split exponents without redoing rho.
                                    for mode in range(1, budget-2*lower_level+1):
                                        if mode in lower_modes:
                                            continue
                                        position = sum(m > mode for m in lower_modes)
                                        upper_modes = (lower_modes[:position]+(mode,)
                                                       +lower_modes[position:])
                                        right = (upper_modes, 1-ground)
                                        rho_right = self.form.value((ns, right, third))
                                        if not rho_right:
                                            continue
                                        # The sqrt(2) in K/(Q/sqrt(2)) cancels
                                        # the one odd ground-rescaling exponent.
                                        value = (common*_sign(position+b)
                                                 *rho_left*rho_right/(1 << ground3))
                                        upper_level = lower_level+mode
                                        key = (ns_level, lower_level, upper_level,
                                               2*r3_level)
                                        accumulate(key, component, value)
                                        transposed = (ns_level, upper_level, lower_level,
                                                      2*r3_level)
                                        accumulate(transposed, component, value)
                                        contractions += 2
                                        toggle_contractions += 2
                now = perf_counter()
                if progress is not None and now >= next_progress:
                    progress({"stage": "direct sewing", "ns_twice_level": ns_level,
                              "r3_level": r3_level, "elapsed_seconds": now-started,
                              "ward_cache_entries": len(self.form.cache),
                              "contractions": contractions})
                    next_progress = now+1.0
        result = {key: tuple(value) for key, value in sorted(answer.items(),
                  key=lambda item: (sum(item[0]), item[0])) if any(value)}
        self.diagnostics.update({
            "maximum_balanced_level": level,
            "maximum_split_ramond_level_allowed": cutoff,
            "basis_generation_seconds": basis_seconds,
            "series_computation_seconds": perf_counter()-started,
            "ns_mode_sets": sum(map(len, ns_basis)),
            "ramond_mode_sets": sum(map(len, r_basis)),
            "ward_cache_entries": len(self.form.cache),
            "ward_value_calls": self.form.calls,
            "ward_cache_hits": self.form.cache_hits,
            "ward_terms": self.form.ward_terms,
            "nonzero_state_contractions": contractions,
            "zero_mode_contractions": zero_contractions,
            "nonzero_mode_contractions": toggle_contractions,
            "series_monomials": len(result),
            "nonzero_parity_coefficients": sum(sum(bool(v) for v in row)
                                                for row in result.values()),
        })
        return result

    def series(self, level, progress=None):
        """Full un-reduced series with Q/sqrt(2) restored numerically."""
        normalized = self.normalized_series(level, progress=progress)
        mp = mpmath.mp.clone() if self.dps else mpmath.fp
        if self.dps:
            mp.dps = self.dps
        factor = mp.mpc(self.Q)/mp.sqrt(2)
        return {key: tuple(factor*mp.mpf(int(value.numerator))
                           /int(value.denominator) for value in row)
                for key, row in normalized.items()}
