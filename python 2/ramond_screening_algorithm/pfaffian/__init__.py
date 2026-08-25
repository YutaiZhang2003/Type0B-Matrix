"""State-free fermionic kernels used by the Ramond screening oracle."""

from .core import pfaffian
from .finite_schur_reconstruction import reconstruct_schur_mod

__all__ = (
    "pfaffian",
    "projected_contour_laurent",
    "projected_determinant_constant",
    "projected_selberg_ratio",
    "reconstruct_schur_mod",
)


def __getattr__(name):
    """Load the heavier symbolic screening layer only when requested."""

    if name in {
        "projected_contour_laurent",
        "projected_determinant_constant",
        "projected_selberg_ratio",
    }:
        from . import boundary_zero_modes

        return getattr(boundary_zero_modes, name)
    raise AttributeError(name)
