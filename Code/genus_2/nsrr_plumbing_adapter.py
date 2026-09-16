"""Explicit geometry-to-package boundary for the new NSRR computation.

The protected package consumes (infinity, one, zero)=(NS,R,R).
Geometry routines consume (zero, one, infinity). Spin signs here are the
literal signs in the note's Rblock; an all-NS reference characteristic does
not determine them. This module neither infers a nonchiral spin projector
nor replaces it with a degeneracy factor.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Sequence


GEOMETRY_ORDER = ("zero", "one", "infinity")
HUMAN_SLOT_ORDER = ("infinity", "one", "zero")
GEOMETRY_SECTORS = ("R", "R", "NS")


def to_human_slots(values: Sequence):
    if len(values) != 3:
        raise ValueError("three geometric edges are required")
    return tuple(values[e] for e in (2, 1, 0))


@dataclass(frozen=True)
class NSRRPlumbingInputs:
    """A matched NS-at-infinity chart, not an arbitrary saved q tuple."""

    q_geometry: tuple[complex, complex, complex]
    literal_lifts_geometry: tuple[int, int, int]
    sectors_geometry: tuple[str, str, str]

    def __post_init__(self):
        q = tuple(complex(x) for x in self.q_geometry)
        lifts = tuple(self.literal_lifts_geometry)
        if tuple(self.sectors_geometry) != GEOMETRY_SECTORS:
            raise ValueError(
                "The package requires NS at infinity: geometric sectors must "
                "be (R,R,NS). Re-plumb/re-mark an old (NS,R,R) chart first; "
                "reversing its q tuple alone is not a frame transformation."
            )
        if len(q) != 3 or any(not math.isfinite(abs(x)) or not 0 < abs(x) < 1 for x in q):
            raise ValueError("q_geometry must contain three finite 0<|q|<1 values")
        if len(lifts) != 3 or any(x not in (-1, 1) for x in lifts):
            raise ValueError("literal_lifts_geometry must contain three +/-1 values")
        object.__setattr__(self, "q_geometry", q)
        object.__setattr__(self, "literal_lifts_geometry", tuple(int(x) for x in lifts))
        object.__setattr__(self, "sectors_geometry", GEOMETRY_SECTORS)

    @property
    def q_slots(self):
        return to_human_slots(self.q_geometry)

    @property
    def lifts_slots(self):
        return to_human_slots(self.literal_lifts_geometry)

    def momenta_slots(self, momenta_geometry):
        values = tuple(float(x) for x in momenta_geometry)
        if len(values) != 3 or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("geometric momenta must be finite and nonnegative")
        return to_human_slots(values)

    def weights_slots(self, b, momenta_geometry):
        """Primary weights in Human slots (NS infinity, R one, R zero).

        These are the weights of the SCA primaries. Descendant levels and
        double-Virasoro branching shifts stay in the reduced block.
        """
        if not math.isfinite(float(b)) or float(b) <= 0:
            raise ValueError("b must be finite and positive")
        p_ns, p_one, p_zero = self.momenta_slots(momenta_geometry)
        bg = float(b) + 1 / float(b)
        return (bg*bg/8+p_ns*p_ns/2,
                (1.5+3*bg*bg)/24+p_one*p_one/2,
                (1.5+3*bg*bg)/24+p_zero*p_zero/2)

    def primary(self, b, momenta_geometry):
        """Return exp(sum h_i Log(q_i)), separately from the reduced block.

        This pointwise adapter uses principal logarithms. Analytic
        continuation around a loop requires transported logarithms and
        compatible continuation of the fractional descendant powers.
        """
        weights = self.weights_slots(b, momenta_geometry)
        return cmath.exp(sum(h*cmath.log(q) for h, q in zip(weights, self.q_slots)))


def chiral_block_in_geometry(*, plumbing, b, momenta_geometry, cutoff,
                             form_parity, eta_left, eta_right,
                             primary_parity=0, completion="none",
                             recursion_order=None, global_tolerance=1e-13,
                             global_max_shell=64, global_method="resummed"):
    """Evaluate a literal chiral block with every edge-labelled input mapped.

    The result does NOT include primary powers. Use ``plumbing.primary``
    when constructing a separately specified nonchiral contraction.
    """
    from nsrr_double_virasoro_block import NSRRDoubleVirasoroTheta

    if not isinstance(plumbing, NSRRPlumbingInputs):
        raise TypeError("use NSRRPlumbingInputs, not an unlabelled q/lift tuple")
    evaluator = NSRRDoubleVirasoroTheta(
        b=b, physical_momenta=plumbing.momenta_slots(momenta_geometry),
        cutoff=cutoff, primary_parity=primary_parity, completion=completion,
        recursion_order=recursion_order, global_tolerance=global_tolerance,
        global_max_shell=global_max_shell, global_method=global_method)
    return evaluator.block(q_values=plumbing.q_slots, lifts=plumbing.lifts_slots,
                           form_parity=form_parity, eta_left=eta_left,
                           eta_right=eta_right)
