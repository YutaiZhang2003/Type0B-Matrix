"""Ground-level Majorana-to-SCblock spin-frame dictionary.

This is a finite zero-mode calculation.  It does not identify either
Coulomb charge chart with a fixed ``eta`` at excited level.  In particular,
it must not be used as a substitute for the reflected nonzero-mode kernel.
"""

from __future__ import annotations

import sympy as sp


I = sp.I
SQRT2 = sp.sqrt(2)
FOCK_TO_SCBLOCK_MINUS = -(1 - I) / SQRT2


def scblock_ground_matrix(form_parity: int, eta: int) -> sp.Matrix:
    """Return ``Gamma_f^eta`` in the ordered ``(w^+,w^-)`` basis."""

    form_parity, eta = int(form_parity), int(eta)
    if eta not in (-1, 1):
        raise ValueError("eta must be +1 or -1")
    if form_parity == 0:
        return sp.diag(1, eta)
    if form_parity == 1:
        return sp.Matrix(((0, 1), (I * eta, 0)))
    raise ValueError("form_parity must be 0 or 1")


def normalized_majorana_ground_matrix(form_parity: int) -> sp.Matrix:
    """Native identity/fermion matrix in the normalized free ground basis."""

    form_parity = int(form_parity)
    if form_parity == 0:
        return sp.diag(1, -1)
    if form_parity == 1:
        return sp.Matrix(((0, 1), (-1, 0)))
    raise ValueError("form_parity must be 0 or 1")


def transported_native_ground_matrix(form_parity: int) -> sp.Matrix:
    """Transport the native matrix to the notes' ``w`` ground frame.

    The normalized free ground is ``(|+>,|->)=(w^+,C w^-)`` with
    ``C=-(1-i)/sqrt(2)``.  A bilinear form therefore transports with
    ``D^{-1}`` on both ground indices.
    """

    conversion = sp.diag(1, FOCK_TO_SCBLOCK_MINUS)
    inverse = conversion.inv()
    return sp.simplify(
        inverse * normalized_majorana_ground_matrix(form_parity) * inverse
    )


def native_eta_coefficients(form_parity: int) -> tuple[sp.Expr, sp.Expr]:
    """Coefficients on ``(Gamma_f^+,Gamma_f^-)`` at ground level."""

    form_parity = int(form_parity)
    if form_parity == 0:
        return ((1 - I) / 2, (1 + I) / 2)
    if form_parity == 1:
        return (-I / SQRT2, -1 / SQRT2)
    raise ValueError("form_parity must be 0 or 1")


def native_eta_coefficient_table() -> sp.Matrix:
    """Rows ``f=0,1`` and columns ``eta=+,-`` of the ground dictionary."""

    return sp.Matrix(
        (
            native_eta_coefficients(0),
            native_eta_coefficients(1),
        )
    )


def audit() -> None:
    for form_parity in (0, 1):
        plus, minus = native_eta_coefficients(form_parity)
        reconstructed = (
            plus * scblock_ground_matrix(form_parity, 1)
            + minus * scblock_ground_matrix(form_parity, -1)
        )
        residual = sp.simplify(
            transported_native_ground_matrix(form_parity) - reconstructed
        )
        if residual != sp.zeros(2):
            raise AssertionError((form_parity, residual))
    table = native_eta_coefficient_table()
    if sp.simplify(table.det() - (-1 + I) / SQRT2) != 0:
        raise AssertionError(table.det())
    print("native ground spin frame: both parity rows exact; det(A)!=0")


__all__ = (
    "FOCK_TO_SCBLOCK_MINUS",
    "native_eta_coefficient_table",
    "native_eta_coefficients",
    "normalized_majorana_ground_matrix",
    "scblock_ground_matrix",
    "transported_native_ground_matrix",
)


if __name__ == "__main__":
    audit()
