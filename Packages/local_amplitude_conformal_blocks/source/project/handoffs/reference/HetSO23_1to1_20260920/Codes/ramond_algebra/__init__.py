"""Exact Ramond super-Virasoro primitives used by the heterotic blocks.

The package keeps three logically distinct ingredients separate:

* :mod:`ramond_sca` implements the integer-moded Ramond Verma module and
  its full two-state ground representation;
* :mod:`nrr_three_point_tensor` implements ordered R--NS--R descendant
  three-point forms; and
* :mod:`ramond_plumbing` implements the even and odd-modulus inverse-Gram
  sewing kernels.

All modules use the ordinary Virasoro central charge convention
``c = 3/2 + 3 Q**2``.  The polynomial ground basis is used internally so
that Gram matrices remain rational in ``h`` and ``c``.
"""

from .nrr_three_point_tensor import (
    hjs_ground_tensor,
    hjs_to_polynomial_ground_tensor,
    rr_three_point_from_ground_tensor,
    rr_three_point_hjs,
    rr_three_point_polynomial,
)
from .ns_rr_three_point_tensor import (
    hjs_ns_rr_polynomial_ground_tensor,
    ns_rr_three_point_from_ground_tensor,
)
from .ramond_plumbing import (
    RamondPlumbingKernels,
    normalized_ramond_edge_coefficients,
    ramond_plumbing_kernels,
    zero_mode_matrix,
)
from .ramond_sca import (
    G,
    L,
    Mode,
    PBWState,
    act_mode,
    gram_matrix,
    ground_action_matrix,
    ground_gram_matrix,
    ground_state,
    pbw_basis,
)

__all__ = [
    "hjs_ns_rr_polynomial_ground_tensor",
    "G",
    "L",
    "Mode",
    "PBWState",
    "RamondPlumbingKernels",
    "act_mode",
    "gram_matrix",
    "ground_action_matrix",
    "ground_gram_matrix",
    "ground_state",
    "hjs_ground_tensor",
    "hjs_to_polynomial_ground_tensor",
    "ns_rr_three_point_from_ground_tensor",
    "normalized_ramond_edge_coefficients",
    "pbw_basis",
    "ramond_plumbing_kernels",
    "rr_three_point_from_ground_tensor",
    "rr_three_point_hjs",
    "rr_three_point_polynomial",
    "zero_mode_matrix",
]
