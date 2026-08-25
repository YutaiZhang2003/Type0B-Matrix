"""Ground-resolved compression of literal Ramond chi strings.

This module exposes the state-free part of the Ramond screening problem.
For each Ramond leg the literal 2016 string contains one or two zero modes.
They are expanded explicitly, so the number of boundary sectors is at most
``4*4=16``.  Every nonzero chi assignment in one boundary sector is then
summed by one Pfaffian.

The compression is exact for both parity copies and for either sign of a
Ramond branch label.  It retains the auxiliary and physical ``2 by 2``
ground matrices and handles the four possibilities (even,even),
(even,odd), (odd,even), and (odd,odd).  In the last case the answer is

    Pf(K + mu_aux wedge mu_phys) - Pf(K),

not just the first Pfaffian: the subtraction removes the spurious term in
which two odd boundary functionals are treated as even.

There is an important scope boundary.  ``compressed_correlator`` is exact
for the supplied ordinary two-spin Majorana kernel.  It gives the
charge-preserving (factorized) screening channel.  A mixed-sign branch
requires the momentum-dependent reflected fermion ``psi^R``.  Its exact
level-one reflection already mixes ``psi_-1`` with the bosonic current
``c_-1``; higher blocks contain products of currents.  It is therefore not
in general a replacement scalar covariance.  The callback API accepts an
effective Gaussian kernel after current insertions have been reduced to
screening power sums; it does not perform that reduction.  The ordinary
kernel agrees on all ground states but fails already for the mixed
``(3/4,-3/4)`` excited pair.  Consequently the Gaussian Selberg candidate
exported here must not be advertised as the crossed Ramond branching
coefficient.

The implementation lives in the independent calibration harness so that
its literal-path oracle and compressed formula remain side by side.  The
public names below do not construct super-Virasoro PBW states.
"""

from .audit_ground_covariance import (
    PhysicalGroundContext,
    PhysicalInsertion,
    compressed_contour_polynomial,
    compressed_contour_polynomial_with_covariance,
    compressed_correlator,
    compressed_fixed_correlator,
    compressed_selberg_ratio as gaussian_selberg_candidate,
    compressed_selberg_ratio_with_covariance,
    compressed_zero_sector_data,
    literal_branch_sectors,
    ordinary_physical_covariance,
    ordinary_physical_mean,
)
from .reflected_current_multipliers import (
    complementary_charge_multiplier,
    ordinary_charge_multiplier,
    third_current_multiplier,
)
from .reflected_hard_oracle import (
    charge_plane_p1,
    hard_complementary_pair_multiplier,
    hard_mixed_sheet_value,
)
from .native_ground_change import (
    native_eta_coefficient_table,
    native_eta_coefficients,
    normalized_majorana_ground_matrix,
    scblock_ground_matrix,
    transported_native_ground_matrix,
)


MAXIMUM_BOUNDARY_SECTORS = 16


def operation_estimate(nonzero_chi_modes, screenings):
    """Conservative arithmetic count for the constant-sector Pfaffians."""

    size = int(nonzero_chi_modes) + int(screenings) + 2
    return MAXIMUM_BOUNDARY_SECTORS * size**3 // 6


__all__ = (
    "MAXIMUM_BOUNDARY_SECTORS",
    "PhysicalGroundContext",
    "PhysicalInsertion",
    "compressed_contour_polynomial",
    "compressed_contour_polynomial_with_covariance",
    "compressed_correlator",
    "compressed_fixed_correlator",
    "compressed_selberg_ratio_with_covariance",
    "compressed_zero_sector_data",
    "charge_plane_p1",
    "complementary_charge_multiplier",
    "gaussian_selberg_candidate",
    "hard_complementary_pair_multiplier",
    "hard_mixed_sheet_value",
    "literal_branch_sectors",
    "native_eta_coefficient_table",
    "native_eta_coefficients",
    "normalized_majorana_ground_matrix",
    "operation_estimate",
    "ordinary_charge_multiplier",
    "ordinary_physical_covariance",
    "ordinary_physical_mean",
    "scblock_ground_matrix",
    "third_current_multiplier",
    "transported_native_ground_matrix",
)
