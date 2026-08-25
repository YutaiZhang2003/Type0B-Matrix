"""Gaussian block for external ``chi=f-i psi`` insertions.

For a fixed choice of the two Ramond zero-mode channels, the auxiliary
fermion and the physical fermion are Gaussian.  The apparent ``2**K`` sum
over the two terms in each of ``K`` chi fields is therefore one Pfaffian.
The finite Ramond ground space requires only a fixed number of such blocks;
it does not change the level complexity.
"""

from __future__ import annotations

import sympy as sp

from .core import pfaffian


def chi_screen_matrix(auxiliary, physical_external, mixed, screening):
    """Assemble the covariance of chi fields and physical screenings.

    Parameters are the four ordered covariance blocks

    ``auxiliary``
        auxiliary-fermion covariance on the external chi contours;
    ``physical_external``
        physical-fermion covariance on the same contours;
    ``mixed``
        physical covariance from chi contours to screening positions;
    ``screening``
        physical covariance among screening positions.

    Since ``chi=f-i psi``, the blocks are ``A-B``, ``-i C``, and ``D``.
    """

    auxiliary = sp.Matrix(auxiliary)
    physical_external = sp.Matrix(physical_external)
    mixed = sp.Matrix(mixed)
    screening = sp.Matrix(screening)
    external_count = auxiliary.rows
    screening_count = screening.rows
    if auxiliary.cols != external_count or physical_external.shape != auxiliary.shape:
        raise ValueError("external covariance blocks have incompatible sizes")
    if mixed.shape != (external_count, screening_count):
        raise ValueError("mixed covariance block has incompatible size")
    if screening.cols != screening_count:
        raise ValueError("screening covariance must be square")

    answer = sp.zeros(external_count + screening_count)
    answer[:external_count, :external_count] = auxiliary - physical_external
    answer[:external_count, external_count:] = -sp.I * mixed
    answer[external_count:, :external_count] = sp.I * mixed.T
    answer[external_count:, external_count:] = screening
    return answer


def chi_screen_pfaffian(auxiliary, physical_external, mixed, screening):
    """Evaluate a fixed Ramond-channel chi/screening correlator."""

    return pfaffian(
        chi_screen_matrix(auxiliary, physical_external, mixed, screening)
    )


def operation_estimate(external_modes, screenings, ramond_channels=4):
    """Conservative cubic arithmetic count for all zero-mode channels."""

    size = int(external_modes) + int(screenings)
    return int(ramond_channels) * size**3 // 6

