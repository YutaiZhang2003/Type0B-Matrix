#!/usr/bin/env python3
"""Five-node two-chart reconstruction of the hard Ramond polynomial.

This audit joins two independently testable pieces of the state-free
construction:

* :mod:`hard_two_chart_certificate` constructs the two hard charge-channel
  values from the one-leg fusion matrices and the zero-mode exchange.  The
  stored Ward polynomial is not an input to that construction.
* :mod:`two_chart_interpolation` samples each analytic chart on its own
  same-parity Coulomb-neutrality nodes, interpolates the two degree-four
  polynomials separately, and applies the constant ground-space change only
  after both interpolations are complete.

The last comparison imports the old expanded ``K,H`` pair solely as an
independent certificate.  In particular, none of the five interpolation
callbacks reads ``H`` from the Ward data.

Scope is important.  The node callback in this file is the finite
``(0,3/4,3/4)`` two-state Coulomb kernel.  It is not the still-missing
arbitrary-mode signed Pfaffian/Selberg callback.  Consequently this is a
noncircular hard-channel interpolation test, not an all-level production
evaluator.
"""

from __future__ import annotations

import sympy as sp

from ..two_chart_interpolation import (
    reconstruct_two_charts,
    resolve_eta_forms,
    same_parity_nodes,
)
from .hard_two_chart_certificate import (
    P1,
    P2,
    P3,
    Q,
    _reference_polynomials,
    canonical_chart_values,
    charge_channel_numerators,
    one_leg_coulomb_matrix,
)


I = sp.I
SQRT2 = sp.sqrt(2)
HARD_DEGREE = 4


def cleared_hard_chart_polynomials():
    """Return the two denominator-cleared finite-kernel chart polynomials."""

    positive, signed = canonical_chart_values()
    denominator = sp.expand(
        one_leg_coulomb_matrix(P2)[0, 0]
        * one_leg_coulomb_matrix(P3)[0, 0]
    )
    positive = sp.factor(sp.cancel(denominator * positive))
    signed = sp.factor(sp.cancel(denominator * signed))
    for name, value in (("positive", positive), ("signed", signed)):
        polynomial = sp.Poly(sp.expand(value), P1)
        if polynomial.degree() > HARD_DEGREE:
            raise AssertionError((name, polynomial.degree()))
    return denominator, positive, signed


def reconstruct_hard_at_sample():
    """Interpolate the two hard charts at five exact nodes apiece.

    The ordinary chart uses ``(+,+,+)`` in the neutrality equation.  The
    signed chart uses ``(+,+,-)``: its third Ramond representative carries
    momentum ``-P3`` and its endpoint ground index is acted on by ``Z``.
    Screening numbers ``3,5,7,9,11`` have one fixed parity, so the Ramond
    ground form does not alternate between the two chiral structures.
    """

    b = sp.Rational(3, 2)
    q = b + 1 / b
    p2 = sp.Rational(2, 5)
    p3 = sp.Rational(3, 7)
    positive_nodes = same_parity_nodes(
        HARD_DEGREE, q, b, p2, p3, 3, signs=(1, 1, 1)
    )
    signed_nodes = same_parity_nodes(
        HARD_DEGREE, q, b, p2, p3, 3, signs=(1, 1, -1)
    )
    denominator, positive, signed = cleared_hard_chart_polynomials()
    specialization = {Q: q, P2: p2, P3: p3}
    positive = sp.expand(positive.subs(specialization, simultaneous=True))
    signed = sp.expand(signed.subs(specialization, simultaneous=True))
    calls = {"positive": [], "signed": []}

    def evaluate_positive(node, screenings):
        calls["positive"].append((node, screenings))
        return positive.subs(P1, node)

    def evaluate_signed(node, screenings):
        calls["signed"].append((node, screenings))
        return signed.subs(P1, node)

    reconstruction = reconstruct_two_charts(
        P1,
        HARD_DEGREE,
        positive_nodes,
        signed_nodes,
        evaluate_positive,
        evaluate_signed,
    )
    if sp.expand(reconstruction.positive - positive) != 0:
        raise AssertionError("positive five-node interpolation failed")
    if sp.expand(reconstruction.signed - signed) != 0:
        raise AssertionError("signed five-node interpolation failed")
    if tuple(len(calls[name]) for name in ("positive", "signed")) != (5, 5):
        raise AssertionError("a charge-chart node was copied or skipped")

    resolved_zero = resolve_eta_forms(reconstruction, form_parity=0)

    # For the odd physical form the two native chart polynomials acquire
    # the common ground-frame factor 1+i before the odd ground map is used.
    odd_reconstruction = reconstruction.__class__(
        degree=reconstruction.degree,
        positive=(1 + I) * reconstruction.positive,
        signed=(1 + I) * reconstruction.signed,
        positive_nodes=reconstruction.positive_nodes,
        signed_nodes=reconstruction.signed_nodes,
    )
    resolved_one = resolve_eta_forms(odd_reconstruction, form_parity=1)

    ordinary, crossed = charge_channel_numerators()
    ordinary = sp.expand(ordinary.subs(specialization, simultaneous=True))
    crossed = sp.expand(crossed.subs(specialization, simultaneous=True))
    expected_zero = sp.Matrix(
        (-(1 + I) * ordinary, -(1 - I) * crossed)
    )
    if any(sp.expand(value) != 0 for value in resolved_zero - expected_zero):
        raise AssertionError("fixed-eta hard reconstruction failed")
    expected_one = sp.diag(I * SQRT2, -I * SQRT2) * expected_zero
    if any(sp.expand(value) != 0 for value in resolved_one - expected_one):
        raise AssertionError("odd-form hard phase reconstruction failed")

    # Only now load the independently stored expanded result.  This proves
    # that the crossed eigenchannel constructed before interpolation is H.
    reference_ordinary, reference_crossed = _reference_polynomials()
    reference_ordinary = sp.expand(
        reference_ordinary.subs(specialization, simultaneous=True)
    )
    reference_crossed = sp.expand(
        reference_crossed.subs(specialization, simultaneous=True)
    )
    if sp.expand(ordinary - reference_ordinary) != 0:
        raise AssertionError("interpolated ordinary channel disagrees with K")
    if sp.expand(crossed - reference_crossed) != 0:
        raise AssertionError("interpolated crossed channel disagrees with H")

    # A point outside both node sets guards against an interpolation-only
    # comparison at the sampling locations.
    holdout = sp.Rational(5, 11)
    for calculated, expected in zip(resolved_zero, expected_zero):
        if sp.factor((calculated - expected).subs(P1, holdout)) != 0:
            raise AssertionError("hard holdout check failed")
    return denominator.subs(specialization), positive_nodes, signed_nodes


def audit():
    _, positive_nodes, signed_nodes = reconstruct_hard_at_sample()
    print("hard two-chart interpolation: 5+5 independent finite-kernel nodes")
    print("fixed-eta output: exact K and irreducible H, including f=1 phases")
    print(
        "screening parities: ordinary and signed nodes use "
        f"{tuple(item[0] for item in positive_nodes)}"
    )
    if set(node for _, node in positive_nodes) & set(node for _, node in signed_nodes):
        raise AssertionError("the two chart node sets unexpectedly overlap")
    print("holdout: exact residual zero outside both neutrality node sets")
    print("scope: hard finite kernel; generic signed Pfaffian/Selberg callback missing")


if __name__ == "__main__":
    audit()
