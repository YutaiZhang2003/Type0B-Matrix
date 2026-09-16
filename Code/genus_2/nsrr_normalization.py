"""Explicit genus-two NSRR normalization hypotheses.

The user-requested factor four is a GLOBAL SEWING ASSUMPTION. It is not
part of the derived local BPZ matrix, the BRY coefficients, a chiral block,
or a primary power. Keep the local kernel available for independent tests.
"""
from __future__ import annotations

from nsrr_bilinear_sewing import nsrr_bilinear_matrix, contract_nsrr_bilinear

LOCAL = "local"
PROVISIONAL_FOUR = "provisional-times-four"
NORMALIZATIONS = (LOCAL, PROVISIONAL_FOUR)


def normalization_metadata(normalization):
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown NSRR normalization: {normalization!r}")
    assumed = normalization == PROVISIONAL_FOUR
    return dict(normalization=normalization, normalization_factor=4 if assumed else 1,
                normalization_status="assumption" if assumed else "local BPZ reference",
                normalization_scope="multiply the entire genus-two NSRR M once",
                global_normalization_verified=False,
                normalization_reason=("Fixed factor four requested by the user on 2026-09-16; "
                    "motivated by the converged ratios near one quarter; not derived or fitted"
                    if assumed else "Unscaled locally derived Human-Note BPZ pairing"))


def normalized_nsrr_matrix(left_bry, right_bry, *, physical_lifts_slots,
                           normalization=PROVISIONAL_FOUR):
    """Return a matrix and mandatory metadata identifying its assumption."""
    metadata = normalization_metadata(normalization)
    matrix = nsrr_bilinear_matrix(left_bry, right_bry,
                                 physical_lifts_slots=physical_lifts_slots)
    return dict(matrix=metadata["normalization_factor"] * matrix, **metadata)


def contract_normalized_nsrr(*, normalization=PROVISIONAL_FOUR, **kwargs):
    """Bilinear contraction, retaining independent antiholomorphic inputs."""
    metadata = normalization_metadata(normalization)
    result = contract_nsrr_bilinear(**kwargs)
    factor = metadata["normalization_factor"]
    for name in ("total", "diagonal", "interference"):
        result[name] *= factor
    for name in ("terms", "diagonal_terms"):
        result[name] = {key: factor * value for key, value in result[name].items()}
    result.update(metadata)
    return result
