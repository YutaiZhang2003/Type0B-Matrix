#!/usr/bin/env python3
"""Independent normalization layers for the genus-two vacuum amplitude.

Compact c=1 absolute normalization from lower-genus sewing (2026-07-23)
-----------------------------------------------------------------------

The coefficient ``alpha'/pi`` described below is inherited from the
*critical-string* sphere metric ``K_S2=8*pi/alpha'``.  For the compact c=1
string, the timelike sphere constant gives

    K_tilde_S2 = 2/sqrt(alpha'),
    K_res,S2^c1 = sqrt(alpha')*K_tilde_S2 = 2.

The explicit compact theta-graph density already contains the Fourier metric
``2*pi*r``.  Therefore its residual topology metric is the fixed number
``K_res=2``, not ``Khat=2*pi*K_res=4*pi``.  The critical-to-c=1 topology factor is

    Lambda_top = K_S2/K_res,S2^c1 = 4*pi/alpha'.

Before the fixed-puncture sewing conversion, the positive-real
``g_s^2``-stripped marked-plumbing multiplier is ``4``.  Xi's explicit genus-one
Hilbert-space trace supplies ``(2*pi)^2`` for each fixed-puncture torus
surface state relative to the bare torus cusp form.  Hence the complete
genus-two BRST multiplier is

    4 * (2*pi)^4 = 64*pi^4.

The production evaluator currently uses this complete multiplier.  Older v4
saved rows used ``32*pi^3`` and are migrated by the exact factor ``2*pi``
before reuse.  The multiplier is a working convention, not an absolute
normalization certificate: the 2026-08-22 critical-string control shows that
the existing maximal-theta comparison would also insert ``(2*pi)^4`` into
Xi's already unitarity-normalized critical formula.  That sewing/state-measure
bridge must be resolved before the compact genus-two coefficient is certified.

There are four distinct layers in the current calculation.

1. Polyakov gauge fixing in the string-note convention gives

       N_{h,n} = i^(3 h - 3 + n),

   when the sphere state metric is ``K_S2 = 8 pi / alpha'``.  Thus
   ``N_{2,0}=-i``.  Equation (4.105) of the notes supplies
   ``g_s^2 alpha'/(8 pi)``.  The printed equation (4.106) omits this external
   factor after the loop-momentum Gaussian; the omission is treated as a typo,
   and the corrected (4.106) retains it together with the displayed minus
   sign.  With

       alpha = dOmega_11 wedge dOmega_22 wedge dOmega_12,
       alpha wedge conjugate(alpha) = -8 i d^3X d^3Y,

   their product is the positive real coefficient

       (-i)(-1)(-8i) g_s^2 alpha'/(8 pi)
           = g_s^2 alpha'/pi.

   The Monte Carlo kernel strips only ``g_s^2`` and therefore carries the
   explicit multiplier ``alpha'/pi``.  This follows from (4.105) and the
   corrected (4.106), rather than from an inferred or fitted normalization.

2. The BRY-normalized sphere and torus amplitudes are often summarized by the relative topology
   relation

       C_Sigma2 * C_S2 = C_T2^2.

   With ``C_S2=2 pi/(g_s^BRY)^2`` and ``C_T2=1`` this gives
   ``C_Sigma2=(g_s^BRY)^2/(2 pi)``.  BRY's coupling is not Xi's coupling:

       g_s^BRY = 2 g_s^Xi,
       mu^-1 = 2 pi g_s^BRY = 4 pi g_s^Xi.

   The earlier ledger silently inserted ``g_s^Xi`` into the BRY formula.
   Correcting that algebra changes the BRY topology coefficient by four.
   This extrapolation is useful bookkeeping, but BRY do not write a genus-two
   vacuum formula.  A separate conversion of their ordinary ``d^2z`` measure,
   state normalization, and automorphism convention to the string-note
   differential form is required before this coefficient can be used.

3. The Mumford/cusp-form conversion fixes the bare period-coordinate density.

4. The fixed-puncture BRST trace converts that bare density by ``(2*pi)^4``.

Following Moore, the repository evaluates the raw weight-ten theta product

    Psi10 = product_even theta[delta](0 | Omega)^2.

For ``Omega_12 = epsilon`` and ``q = 2 pi i epsilon``, its separating
asymptotic is

    Psi10 = 2^12 q^2 eta(tau_1)^24 eta(tau_2)^24 + O(q^4).

The string notes instead call the Fourier-normalized form

    chi10_note = 2^-12 Psi10

and use ``Phi_2 = 2 pi i / chi10_note``.  This gives unit separating residue
relative to the bare torus cusp forms after converting ``d epsilon`` to
``d q``.  A fixed-puncture torus surface state carries the additional
``|B_tau|^2=(2*pi)^2`` coefficient.  When the raw Moore product is
used in the denominator, its nonchiral density therefore requires the
equivalent factor ``2^24``.  There is no minus sign in this string-note
conversion; any chiral orientation phase drops out of the nonchiral density.

The plumbing calculation naturally produces a dimensionless scalar with
``p=sqrt(alpha') k`` and measure ``d^2p``.  Xi instead uses physical momentum
and ``prod_I dk_I/(2 pi)``.  At genus two,

    Z_X^Xi = Z_X^p / (4 pi^2 alpha').

The compact target zero mode in the matching Xi convention is the ordinary
target length ``2 pi R_phys``, not the dimensionless radius by itself.  These
changes must be made simultaneously with the 26-scalar Gaussian already in
the critical seed.  Their net effect on the complete replacement integrand is
``1/(2 pi sqrt(alpha'))`` at fixed ``r=R_phys/sqrt(alpha')``.  In the
dimensionless sewing convention the critical genus-one Mumford form is
``d tau / eta(tau)^24``.  Multiplication by
``2^12`` chirally (``2^24`` nonchirally) makes the genus-two separating limit
exactly

    Phi_2 -> Phi_1(tau_1) Phi_1(tau_2) dq / q^2.

Together with Liouville's ``pi delta(P-P')`` two-point function and ``dP/pi``
completeness measure, this fixes the intrinsic CFT state convention.  Xi's notes
use the same normalized Liouville primary, the same DOZZ coefficient, the same
completeness measure, and the same literal plumbing relation ``u v=q``.
Consequently the BRY-to-Xi conversion of the intrinsic Liouville partition is
exactly one; there is no independent Liouville multiplier ``A_L``.

After the explicit Xi scalar conversion, the correlated normalization of the
*local* critical-boson replacement is fixed by the genus-one-anchored sewing
calculation.  In the older parameterization, if ``A_crit`` converts
the repository critical Mumford density, ``A_X`` converts one noncompact
scalar partition, and ``A_XR`` converts the compact scalar, then

    I_Xi / I_code = A_crit A_XR / A_X^26.

When compact and noncompact scalars use one common convention this is
``A_crit/A_X^25``.  They cannot be assigned independently, but the normalized
critical-boson and scalar identities give their correlated value
``Lambda_local=1`` after stripping the fixed-puncture antighost coefficient.
Xi (4.86)--(4.89) restore that coefficient explicitly:

    B_tau=2*pi*i*b_0,
    B_bar(tau)=-2*pi*i*bar(b)_0.

The Hilbert-space trace of ``bar(b)_0 b_0 c_0 bar(c)_0`` is normalized to
one.  Hence the completed genus-one surface-state form is ``(2*pi)^2`` times
the bare torus trace.  Two separating components give ``(2*pi)^4``.  The same
factor turns the maximal Mumford coefficient ``(2*pi)^-4`` into the unit
	two-sphere sewing coefficient.  Replacing the critical sphere metric by the
	residual compact topology metric independently supplies
	``Lambda_top=4*pi/alpha'`` above.

These are local sewing constants.  The numerical integral is over one
ordinary Siegel fundamental domain with weight one.  The ineffective center
``{+I,-I}`` of ``Sp(4,Z)`` does not create a second domain or an additional
orbifold factor, and the generic hyperelliptic involution is not divided out
again.

For comparison, D'Hoker--Phong use the same raw theta product and obtain the
chiral critical-bosonic measure ``pi^-12 d^3Omega / Psi10``.  Their
scalar determinant is quoted per ordinary target-space volume, whereas the
repository divides the connected scalar zero mode using ``dX/(2 pi)``.  Thus
``Z_X^code = 2 pi Z_X^DHP`` and the critical nonchiral coefficients differ by
exactly ``(2 pi)^26``.  The identical algebraic conversion occurs at genus
one.  This is a useful algebraic convention check.  The complete primitive
BRST audit additionally retains the torus antighost contours, the zero-mode
trace, both sphere ghost tensors, inverse BPZ metrics, and all differential
forms.  The separating and maximal channels independently derive the same
``(2*pi)^4`` conversion, which is applied in the v5 kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from free_boson_plumbing import dedekind_eta_abs_from_q, igusa_chi10_genus2
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.free_boson_plumbing import dedekind_eta_abs_from_q, igusa_chi10_genus2


RAW_THETA_PRODUCT = "product"
FACTORIZATION_NORMALIZED_CHI10 = "string_note_2^-12"
LEGACY_FACTORIZATION_NORMALIZED_CHI10 = "igusa_2^-12"
ALGEBRAIC_IGUSA_CHI10 = "igusa_2^-14"

RAW_PRODUCT_CHIRAL_RESIDUE = 2.0**-12
RAW_PRODUCT_NONCHIRAL_RESIDUE = 2.0**-24
RAW_PRODUCT_FACTORIZATION_NORMALIZATION = 2.0**24
# The numerical integral is over one ordinary Siegel fundamental domain.
# Since the central elements +/-I act identically on H_2, passing from Sp(4,Z)
# to PSp(4,Z) does not change that domain or its measure.  No extra global
# factor is applied.
GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT = 1.0
GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD = (
    "rqmc_fundamental_domain_integration_weight"
)
GENUS2_LOCAL_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD = (
    "fundamental_domain_integration_weight"
)
LEGACY_GENUS2_STACK_WEIGHT_FIELDS = (
    "rqmc_stack_integration_weight",
    "stack_integration_weight",
)
GENUS2_VACUUM_GAUGE_FIXING_NORMALIZATION = -1j
STRING_NOTE_GENUS2_GHOST_MATTER_FORM_SIGN = -1.0
STRING_NOTE_GENUS2_COMPLEX_FORM_JACOBIAN = -8j
STRING_NOTE_GENUS2_REAL_FORM_FACTOR = 8.0


def genus2_fundamental_domain_node_weight(row: dict[str, object]) -> float:
    """Reconstruct a node's full Siegel-domain integration weight.

    The weight is reconstructed from proposal data, which are authoritative.
    If a canonical saved weight is present, it must agree.  Obsolete fields
    containing the retracted half-weight are deliberately ignored.
    """

    proposal_count = int(row["rqmc_proposal_count"])
    if proposal_count <= 0:
        raise ValueError("RQMC proposal count must be positive")
    physical = str(row.get("rqmc_physical_measure_weight", "")).strip()
    coarse = str(row.get("rqmc_coarse_volume_weight", "")).strip()
    invariant = str(row.get("rqmc_invariant_weight", "")).strip()
    if physical:
        weight = float(physical) / proposal_count
    elif coarse:
        weight = float(coarse)
    elif invariant:
        weight = float(invariant) / proposal_count
    else:
        raise ValueError(
            "RQMC row lacks the proposal data needed to reconstruct the full "
            "fundamental-domain integration weight"
        )
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("RQMC integration weight must be positive and finite")
    for field in (
        GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD,
        GENUS2_LOCAL_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD,
    ):
        saved = str(row.get(field, "")).strip()
        if saved and not math.isclose(
            float(saved),
            weight,
            rel_tol=2.0e-13,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"saved {field} disagrees with the reconstructed "
                "fundamental-domain weight"
            )
    return weight


def canonicalize_genus2_fundamental_domain_weight_fields(
    source: dict[str, object],
) -> dict[str, object]:
    """Return a row with only canonical full-domain weight bookkeeping.

    This is the explicit migration boundary for historical tables.  It drops
    both obsolete ``*stack_integration_weight`` fields, writes the full
    fundamental-domain weight under the canonical names, and rebuilds saved
    node contributions when transformed integrands are available.
    """

    row = dict(source)
    weight = genus2_fundamental_domain_node_weight(row)
    for field in LEGACY_GENUS2_STACK_WEIGHT_FIELDS:
        row.pop(field, None)
    row[GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD] = weight
    row[GENUS2_LOCAL_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD] = weight
    for order in ("low", "high"):
        transformed_field = f"transformed_integrand_{order}"
        contribution_field = f"node_contribution_{order}"
        transformed = str(row.get(transformed_field, "")).strip()
        if transformed:
            row[contribution_field] = weight * float(transformed)
    return row


LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION = (
    "string-note-4.97-4.105-positive-real-measure-gs2-stripped"
)
PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION = (
    "string-note-4.97-4.105-positive-real-measure-"
    "xi-scalar-dk-over-2pi-gs2-stripped-v2"
)
PRE_BRST_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION = (
    "c1-sphere-normalized-positive-real-measure-"
    "xi-scalar-dk-over-2pi-gs2-stripped-v3"
)
LEGACY_V4_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION = (
    "c1-brst-sewing-normalized-positive-real-measure-"
    "xi-scalar-dk-over-2pi-gs2-stripped-v4"
)
STRING_NOTE_INTEGRATION_KERNEL_CONVENTION = (
    "c1-g0-g1-factorization-normalized-positive-real-measure-"
    "xi-scalar-dk-over-2pi-gs2-stripped-v5"
)
DHP_RAW_PRODUCT_CHIRAL_PREFACTOR = math.pi**-12
DHP_RAW_PRODUCT_NONCHIRAL_PREFACTOR = math.pi**-24
CODE_SCALAR_OVER_DHP_PER_VOLUME = 2.0 * math.pi
CODE_GENUS1_CRITICAL_AMPLITUDE_PREFACTOR = 0.5
DHP_GENUS1_CRITICAL_AMPLITUDE_PREFACTOR = 0.5 * (2.0 * math.pi) ** -26

# These certifications refer to different layers and must remain separate.
POLYAKOV_N_HN_RECURRENCE_CERTIFIED = True
WORLDSHEET_GAUGE_FIXING_NORMALIZATION_CERTIFIED = True
LOCAL_CFT_MEASURE_CONVENTIONS_RECONCILED = True
GENUS1_FIXED_PUNCTURE_BRST_TRACE_CERTIFIED = True
GENUS2_BRST_PERIOD_TO_PUNCTURED_SEWING_MULTIPLIER = (2.0 * math.pi) ** 4
# The implemented theta sewing is internally consistent, but its comparison
# object fails the critical-string normalization control.  Keep the numerical
# coefficient unchanged while distinguishing internal consistency from an
# absolute certification.
MAXIMAL_FULL_BRST_SEWING_INTERNAL_CONSISTENCY_CHECK_PASSED = True
MAXIMAL_FULL_BRST_FACTORIZATION_CERTIFIED = False
# Backward-compatible name: this is the multiplier missing from v3 and
# first applied in v4, not an unresolved factor.
MAXIMAL_FULL_BRST_MISSING_MULTIPLIER = (
    GENUS2_BRST_PERIOD_TO_PUNCTURED_SEWING_MULTIPLIER
)

# The BRY and Xi local Liouville modular-functor data agree literally.  Their
# coupling dictionary follows from their respective tree amplitudes.  The
# local critical/scalar bridge to one.  The completed fixed-puncture torus
# surface state carries |B_tau|^2=(2*pi)^2 relative to the bare torus cusp
# form.  Two separating components therefore derive the global
# (2*pi)^4 period-form-to-BRST-sewing bridge.  The same multiplier turns the
# maximal period/Mumford coefficient (2*pi)^-4 into the unit two-sphere sewing
# coefficient.  The compact c=1 sphere metric decomposes as
# K_res*(2*pi*r); because the explicit sewn density already contains
# 1/(2*pi*r), the remaining topology factor is K_crit/K_res=4*pi/alpha'.
# The v5 production kernel applies both factors.
BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED = True
BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED = True
C1_SPHERE_TOPOLOGY_NORMALIZATION_AUDITED = True
C1_SPHERE_TOPOLOGY_CORRECTION_APPLIED_TO_PRODUCTION_KERNEL = True
# BRY do not state a genus-two vacuum formula, so a literature-level
# full-amplitude BRY/Xi dictionary remains unavailable.  This is separate
# from the first-principles Xi-convention BRST certification above.
BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED = False

# Backward-compatible name used by existing result metadata.
BRY_XI_GENUS2_TOPOLOGY_DICTIONARY_RECONCILED = (
    BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED
)
GENUS2_C1_ABSOLUTE_NORMALIZATION_CERTIFIED = False

# Backward-compatible aggregate used in result metadata.
FULL_CFT_FACTORIZATION_CERTIFIED = (
    LOCAL_CFT_MEASURE_CONVENTIONS_RECONCILED
    and WORLDSHEET_GAUGE_FIXING_NORMALIZATION_CERTIFIED
    and GENUS1_FIXED_PUNCTURE_BRST_TRACE_CERTIFIED
    and MAXIMAL_FULL_BRST_FACTORIZATION_CERTIFIED
    and GENUS2_C1_ABSOLUTE_NORMALIZATION_CERTIFIED
)


def xi_genus2_scalar_over_dimensionless(alpha_prime: float = 1.0) -> float:
    r"""Return ``Z_X^Xi/Z_X^p`` for one scalar at genus two.

    ``Z_X^p`` uses ``p=sqrt(alpha') k`` and ``d^2p``.  Xi's partition per
    ordinary target-space volume uses ``prod_I dk_I/(2 pi)``.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 1.0 / (4.0 * math.pi * math.pi * alpha_prime)


def xi_compact_target_zero_mode(
    dimensionless_radius: float,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return Xi's compact zero-mode length ``2 pi R_phys``.

    The public radius is ``r=R_phys/sqrt(alpha')``.
    """

    dimensionless_radius = float(dimensionless_radius)
    alpha_prime = float(alpha_prime)
    if not math.isfinite(dimensionless_radius) or dimensionless_radius <= 0.0:
        raise ValueError("dimensionless_radius must be positive and finite")
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 2.0 * math.pi * math.sqrt(alpha_prime) * dimensionless_radius


def xi_full_replacement_over_dimensionless(alpha_prime: float = 1.0) -> float:
    r"""Return the net Xi/dimensionless factor in the genus-two replacement.

    This is the exact cancellation

    ``(4 pi^2 alpha')^-26 * (4 pi^2 alpha')^25
       * (2 pi sqrt(alpha')) = 1/(2 pi sqrt(alpha'))``.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 1.0 / (2.0 * math.pi * math.sqrt(alpha_prime))


@dataclass(frozen=True)
class SeparatingMumfordNormalization:
    """Numerical separating residue for one cusp-form normalization."""

    tau_left: complex
    tau_right: complex
    epsilon: complex
    q_bridge: complex
    chi10_normalization: str
    chi10: complex
    chiral_residue: complex
    nonchiral_residue: float
    factorization_normalization: float
    normalized_nonchiral_residue: float


@dataclass(frozen=True)
class BryXiBareConventionMap:
    """Known BRY-to-Xi factors before topology/state normalization.

    BRY write ordinary ``d^2z`` integrals, while Xi's differential forms use
    ``i dz wedge dbar(z)=2 d^2z``.  The string coupling obeys
    ``g_s^BRY=2 g_s^Xi``.  This dataclass intentionally does not infer the
    remaining topology, state-metric, or automorphism factor.
    """

    genus: int
    punctures: int
    complex_moduli_dimension: int
    string_coupling_power: int
    xi_over_bry_real_measure_factor: float
    xi_over_bry_coupling_weight: float
    xi_over_bry_known_product: float


def mumford_factorization_normalization(chi10_normalization: str) -> float:
    r"""Return the multiplier giving unit nonchiral Mumford-form residue.

    The values follow directly from the constant relating the selected
    ``chi10`` convention to the raw even-theta product.  They do not depend
    on either torus modulus or on the plumbing parameter.
    """

    if chi10_normalization == RAW_THETA_PRODUCT:
        return RAW_PRODUCT_FACTORIZATION_NORMALIZATION
    if chi10_normalization in {
        FACTORIZATION_NORMALIZED_CHI10,
        LEGACY_FACTORIZATION_NORMALIZED_CHI10,
    }:
        return 1.0
    if chi10_normalization == ALGEBRAIC_IGUSA_CHI10:
        return 1.0 / 16.0
    raise ValueError(f"unsupported chi10 normalization {chi10_normalization!r}")


def critical_prefactor_ratio_to_dhp(
    *,
    mumford_prefactor: complex = 2j * math.pi,
    chi10_normalization: str = RAW_THETA_PRODUCT,
) -> float:
    r"""Compare the repository critical prefactor with D'Hoker--Phong.

    D'Hoker--Phong's raw-product chiral bosonic measure is

    ``pi^-12 d^3Omega / Psi10``.

    The returned ratio is nonchiral and includes the repository's selected
    cusp-form residue conversion.  For the defaults it is ``(2 pi)^26``.
    In the repository convention this ratio is exactly the 26th power of the
    one-scalar zero-mode conversion ``Z_X^code/Z_X^DHP = 2 pi``.
    """

    repository_prefactor = (
        mumford_factorization_normalization(chi10_normalization)
        * abs(complex(mumford_prefactor)) ** 2
    )
    return float(repository_prefactor / DHP_RAW_PRODUCT_NONCHIRAL_PREFACTOR)


def genus1_critical_prefactor_ratio_to_dhp() -> float:
    r"""Return the code/D'Hoker--Phong genus-one critical-measure ratio.

    D'Hoker--Phong's standard torus vacuum measure can be written as the
    repository's Xi-amplitude-normalized critical density multiplied by
    ``(2 pi)^-26``.  The returned code/DHP ratio therefore equals
    ``(2 pi)^26``, exactly as at genus two.
    """

    return float(
        CODE_GENUS1_CRITICAL_AMPLITUDE_PREFACTOR
        / DHP_GENUS1_CRITICAL_AMPLITUDE_PREFACTOR
    )


def worldsheet_gauge_fixing_normalization(genus: int, punctures: int) -> complex:
    r"""Return the formal Polyakov factor ``N_{h,n}=i^(3h-3+n)``.

    This is the differential-form convention derived by plumbing unitarity
    with ``K_S2=8 pi/alpha'``.  It is not automatically a real multiplicative
    factor for the positive coordinate density used by the Monte Carlo code.
    """

    if isinstance(genus, bool) or int(genus) != genus or int(genus) < 0:
        raise ValueError("genus must be a nonnegative integer")
    if isinstance(punctures, bool) or int(punctures) != punctures or int(punctures) < 0:
        raise ValueError("punctures must be a nonnegative integer")
    genus = int(genus)
    punctures = int(punctures)
    return complex((1j) ** (3 * genus - 3 + punctures))


def bry_xi_bare_convention_map(genus: int, punctures: int) -> BryXiBareConventionMap:
    r"""Return the convention factors that can be read directly from BRY and Xi.

    For a stable amplitude, let ``d=3g-3+n`` and ``k=2g-2+n``.  Relative to
    BRY's ordinary area elements, Xi's differential form contributes ``2^d``.
    Since ``g_s^BRY=2 g_s^Xi``, the genus-counting weight in Xi's coupling is
    ``2^-k`` times the BRY weight.  Their product is only a *bare* conversion:
    it excludes sphere/state metrics, critical-to-c=1 replacement constants,
    and automorphism quotients.

    In particular, ``(g,n)=(1,2)`` gives ``4 * 1/4 = 1``, reproducing the
    lower-genus measure explanation in Xi's footnote.  For ``(2,0)`` it gives
    ``8 * 1/4 = 2``; this is not a factor to apply to the present kernel,
    which already uses Xi's measure and coupling.
    """

    if isinstance(genus, bool) or int(genus) != genus or int(genus) < 0:
        raise ValueError("genus must be a nonnegative integer")
    if isinstance(punctures, bool) or int(punctures) != punctures or int(punctures) < 0:
        raise ValueError("punctures must be a nonnegative integer")
    genus = int(genus)
    punctures = int(punctures)
    dimension = 3 * genus - 3 + punctures
    coupling_power = 2 * genus - 2 + punctures
    if dimension < 0 or coupling_power < 0:
        raise ValueError("the BRY/Xi map requires a stable amplitude")
    measure_factor = float(2.0**dimension)
    coupling_factor = float(2.0**(-coupling_power))
    return BryXiBareConventionMap(
        genus=genus,
        punctures=punctures,
        complex_moduli_dimension=dimension,
        string_coupling_power=coupling_power,
        xi_over_bry_real_measure_factor=measure_factor,
        xi_over_bry_coupling_weight=coupling_factor,
        xi_over_bry_known_product=measure_factor * coupling_factor,
    )


def sphere_state_metric_normalization(alpha_prime: float = 1.0) -> float:
    r"""Return the plumbing-unitarity convention ``K_S2=8 pi/alpha'``."""

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 8.0 * math.pi / alpha_prime


def c1_timelike_sphere_constant(alpha_prime: float = 1.0) -> float:
    r"""Return the c=1 timelike sphere constant ``K_tilde_S2=2/sqrt(alpha')``."""

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 2.0 / math.sqrt(alpha_prime)


def c1_reduced_sphere_metric(alpha_prime: float = 1.0) -> float:
    r"""Return the energy-delta sphere metric ``Khat=4*pi``.

    Converting the timelike zero-mode delta function from physical energy
    ``k^0`` to the dimensionless energy ``omega`` contributes
    ``sqrt(alpha')``.  Including the Fourier ``2*pi`` gives

    ``Khat_S2^c1 = 2*pi*sqrt(alpha')*K_tilde_S2 = 4*pi``.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return (
        2.0
        * math.pi
        * math.sqrt(alpha_prime)
        * c1_timelike_sphere_constant(alpha_prime)
    )


def c1_residual_topology_metric(alpha_prime: float = 1.0) -> float:
    r"""Return the fixed ``K_res=sqrt(alpha')*K_tilde_S2=2``.

    In the compact sewing basis the full timelike sphere metric is
    ``K_res*(2*pi*r)``.  The explicit theta-graph density already contains
    the inverse Fourier metric ``1/(2*pi*r)``, so ``K_res=2`` is the residual
    topology metric that remains outside that density.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return (
        math.sqrt(alpha_prime)
        * c1_timelike_sphere_constant(alpha_prime)
    )


def c1_genus_topology_correction(
    genus: int,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the critical-to-c=1 vacuum topology factor at genus ``g``.

    With ``C_T2=1`` and the compact Fourier metric displayed explicitly,
    separating sewing gives

    ``C_g=(K_S2)^(1-g)``.

    Consequently the replacement of the critical sphere metric by the
    residual compact c=1 topology metric is
    ``(4*pi/alpha')^(g-1)``.  In particular, it is unity at genus one and
    ``4*pi/alpha'`` at genus two.  The Fourier ``2*pi`` is not part of this
    topology metric because it is already present in the local compact
    state metric ``2*pi*r``.
    """

    if isinstance(genus, bool) or int(genus) != genus or int(genus) < 0:
        raise ValueError("genus must be a nonnegative integer")
    sphere_ratio = (
        sphere_state_metric_normalization(alpha_prime)
        / c1_residual_topology_metric(alpha_prime)
    )
    return sphere_ratio ** (int(genus) - 1)


def c1_genus2_topology_correction(alpha_prime: float = 1.0) -> float:
    r"""Return the compact c=1 genus-two topology factor ``4*pi/alpha'``.

    The local matter/ghost sewing density is held fixed.  The factor is the
    ratio of the critical sphere metric to the residual compact topology
    metric, ``K_S2/K_res,S2^c1``.  It is independent of the subsequent
    unit-weight fundamental-domain integration.
    """

    return c1_genus_topology_correction(2, alpha_prime)


def c1_sphere_normalized_genus2_kernel_multiplier(
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the complete v5 c=1 multiplier ``64*pi^4``.

    The three independently derived factors are

    ``alpha'/pi * (4*pi/alpha') * (2*pi)^4``.

    The middle factor replaces the critical sphere metric by the c=1 sphere
    residual compact topology metric.  The final factor converts the bare
    period/Mumford form to the
    completed fixed-puncture BRST sewing convention.  It is derived both from
    Xi's genus-one Hilbert-space trace and from maximal two-sphere sewing.
    """

    return (
        string_note_genus2_kernel_multiplier(alpha_prime)
        * c1_genus2_topology_correction(alpha_prime)
        * GENUS2_BRST_PERIOD_TO_PUNCTURED_SEWING_MULTIPLIER
    )


def integration_kernel_scale_to_current(
    source_convention: str,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the exact scale from a saved kernel convention to v5.

    The legacy dimensionless-scalar kernel already contains the inherited
    critical-string multiplier ``alpha'/pi``.  The v2 kernel additionally
    contains Xi's scalar conversion.  The v3 kernel contains both Xi's scalar
    conversion and the old energy-delta sphere factorization.  The v4 kernel
    additionally contains the completed BRST sewing multiplier ``(2*pi)^4``.
    The current v5 kernel corrects the double-counted compact Fourier metric
    and is larger than v4 by exactly ``2*pi``.
    """

    convention = str(source_convention).strip()
    if convention == STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
        return 1.0
    if convention == LEGACY_V4_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
        return 2.0 * math.pi
    if convention == PRE_BRST_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
        return (
            2.0
            * math.pi
            * GENUS2_BRST_PERIOD_TO_PUNCTURED_SEWING_MULTIPLIER
        )
    if convention == PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
        return (
            c1_genus2_topology_correction(alpha_prime)
            * GENUS2_BRST_PERIOD_TO_PUNCTURED_SEWING_MULTIPLIER
        )
    if convention == LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION:
        return (
            xi_full_replacement_over_dimensionless(alpha_prime)
            * c1_genus2_topology_correction(alpha_prime)
            * GENUS2_BRST_PERIOD_TO_PUNCTURED_SEWING_MULTIPLIER
        )
    raise ValueError(f"unsupported saved integration-kernel convention {convention!r}")


def genus2_relative_topology_normalization(
    *,
    sphere_normalization: float,
    torus_normalization: float = 1.0,
) -> float:
    r"""Return the relative BRY/CFT topology coefficient ``C_Sigma2``.

    Sewing two once-punctured tori uses the inverse sphere two-point metric,
    so the topology constants obey

    ``C_Sigma2 * C_S2 = C_T2^2``.

    This positive coefficient belongs to the normalization ledger for full
    amplitudes.  It is not the pure Polyakov differential-form phase
    ``N_{2,0}=-i``.
    """

    sphere = float(sphere_normalization)
    torus = float(torus_normalization)
    if not math.isfinite(sphere) or sphere <= 0.0:
        raise ValueError("sphere_normalization must be positive and finite")
    if not math.isfinite(torus) or torus <= 0.0:
        raise ValueError("torus_normalization must be positive and finite")
    return torus * torus / sphere


def genus2_topology_normalization(
    *,
    sphere_normalization: float,
    torus_normalization: float = 1.0,
) -> float:
    r"""Backward-compatible alias for the relative topology coefficient."""

    return genus2_relative_topology_normalization(
        sphere_normalization=sphere_normalization,
        torus_normalization=torus_normalization,
    )


def genus2_worldsheet_gauge_fixing_normalization() -> complex:
    r"""Return ``N_{2,0}=-i`` in the string-note gauge-fixing convention.

    The zero-argument signature is intentional.  Earlier development versions
    used this name for the unrelated positive ``C_Sigma2`` coefficient; use
    :func:`genus2_relative_topology_normalization` for that quantity.
    """

    return worldsheet_gauge_fixing_normalization(2, 0)


def string_note_genus2_complex_form_real_factor() -> float:
    r"""Return the positive factor converting the note's six-form to ``d^6Omega``.

    In the ordering used in the string notes,

    ``d^3Omega wedge d^3 conjugate(Omega) = -8 i d^3X d^3Y``.

    The corrected equation (4.106) contributes a minus sign and (4.97)
    contributes ``N_(2,0)=-i``.  Their product is the positive real number
    eight; the external coefficient from (4.105) is retained separately.
    """

    value = (
        GENUS2_VACUUM_GAUGE_FIXING_NORMALIZATION
        * STRING_NOTE_GENUS2_GHOST_MATTER_FORM_SIGN
        * STRING_NOTE_GENUS2_COMPLEX_FORM_JACOBIAN
    )
    if abs(value.imag) > 1.0e-15 or value.real <= 0.0:
        raise RuntimeError("string-note genus-two form conversion is not positive real")
    return float(value.real)


def string_note_genus2_kernel_multiplier(alpha_prime: float = 1.0) -> float:
    r"""Return the string-note integration-kernel multiplier with ``g_s^2`` stripped.

    Equation (4.105) gives ``g_s^2 alpha'/(8 pi)``.  The corrected (4.106)
    retains it after the Gaussian integration.  Multiplying by the complex-
    form conversion factor eight and removing ``g_s^2`` leaves ``alpha'/pi``.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return (
        string_note_genus2_complex_form_real_factor()
        * alpha_prime
        / (8.0 * math.pi)
    )


def string_note_genus2_full_kernel_multiplier(
    string_coupling: float,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return ``g_s^2 alpha'/pi`` multiplying the ordinary moduli integral."""

    string_coupling = float(string_coupling)
    if not math.isfinite(string_coupling) or string_coupling <= 0.0:
        raise ValueError("string_coupling must be positive and finite")
    return string_coupling**2 * string_note_genus2_kernel_multiplier(alpha_prime)


def bry_genus2_worldsheet_normalization(bry_string_coupling: float) -> float:
    r"""Return ``C_Sigma2=(g_s^BRY)^2/(2 pi)`` in BRY conventions.

    BRY's sphere three-tachyon normalization obeys
    ``(g_s^BRY)^3 C_S2 = g`` and their resonance calculation gives
    ``g = 2 pi g_s^BRY``.  Hence
    ``C_S2 = 2 pi/(g_s^BRY)^2``.  Together with ``C_T2 = 1``, separating
    factorization gives the result returned here.

    Xi's string notes instead use ``mu=1/(4 pi g_s^Xi)``.  Since BRY use
    ``g=mu^-1``, the two string couplings obey
    ``g_s^BRY=2 g_s^Xi``.  Callers must perform that conversion explicitly.
    """

    bry_string_coupling = float(bry_string_coupling)
    if not math.isfinite(bry_string_coupling) or bry_string_coupling <= 0.0:
        raise ValueError("bry_string_coupling must be positive and finite")
    return genus2_relative_topology_normalization(
        sphere_normalization=2.0 * math.pi / bry_string_coupling**2,
        torus_normalization=1.0,
    )


def bry_genus2_relative_topology_normalization(bry_string_coupling: float) -> float:
    r"""Unambiguous name for the BRY ``C_Sigma2`` coefficient."""

    return bry_genus2_worldsheet_normalization(bry_string_coupling)


def bry_genus2_topology_normalization(bry_string_coupling: float) -> float:
    r"""Backward-compatible alias for ``bry_genus2_worldsheet_normalization``."""

    return bry_genus2_worldsheet_normalization(bry_string_coupling)


def raw_product_bry_combined_coefficient(bry_string_coupling: float) -> float:
    r"""Return ``2^24 C_Sigma2`` for the raw-product representation.

    This is a useful coefficient when the local integrand is written with the
    raw theta product.  It is not the overall worldsheet normalization:
    ``C_Sigma2`` is only ``g_s^2/(2 pi)``, while ``2^24`` converts between two
    representations of the local CFT density.
    """

    return (
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION
        * bry_genus2_worldsheet_normalization(bry_string_coupling)
    )


def raw_product_bry_overall_normalization(bry_string_coupling: float) -> float:
    r"""Backward-compatible alias for the raw-product combined coefficient."""

    return raw_product_bry_combined_coefficient(bry_string_coupling)


def xi_string_coupling_from_mqm_fermi_level(fermi_level: float) -> float:
    r"""Return ``g_s^Xi=1/(4 pi mu)`` in the string-note convention."""

    fermi_level = float(fermi_level)
    if not math.isfinite(fermi_level) or fermi_level <= 0.0:
        raise ValueError("fermi_level must be positive and finite")
    return 1.0 / (4.0 * math.pi * fermi_level)


def mqm_fermi_level_from_xi_string_coupling(xi_string_coupling: float) -> float:
    r"""Return ``mu=1/(4 pi g_s^Xi)``."""

    xi_string_coupling = float(xi_string_coupling)
    if not math.isfinite(xi_string_coupling) or xi_string_coupling <= 0.0:
        raise ValueError("xi_string_coupling must be positive and finite")
    return 1.0 / (4.0 * math.pi * xi_string_coupling)


def bry_string_coupling_from_xi_string_coupling(xi_string_coupling: float) -> float:
    r"""Convert Xi's coupling to BRY's: ``g_s^BRY=2 g_s^Xi``."""

    xi_string_coupling = float(xi_string_coupling)
    if not math.isfinite(xi_string_coupling) or xi_string_coupling <= 0.0:
        raise ValueError("xi_string_coupling must be positive and finite")
    return 2.0 * xi_string_coupling


def bry_string_coupling_from_mqm_fermi_level(fermi_level: float) -> float:
    r"""Return BRY's ``g_s^BRY=1/(2 pi mu)``."""

    fermi_level = float(fermi_level)
    if not math.isfinite(fermi_level) or fermi_level <= 0.0:
        raise ValueError("fermi_level must be positive and finite")
    return 1.0 / (2.0 * math.pi * fermi_level)


def mqm_fermi_level_from_bry_string_coupling(bry_string_coupling: float) -> float:
    r"""Return ``mu=1/(2 pi g_s^BRY)``."""

    bry_string_coupling = float(bry_string_coupling)
    if not math.isfinite(bry_string_coupling) or bry_string_coupling <= 0.0:
        raise ValueError("bry_string_coupling must be positive and finite")
    return 1.0 / (2.0 * math.pi * bry_string_coupling)


def bry_genus2_worldsheet_normalization_from_mqm(fermi_level: float) -> float:
    r"""Return BRY's ``C_Sigma2=1/(8 pi^3 mu^2)``."""

    return bry_genus2_worldsheet_normalization(
        bry_string_coupling_from_mqm_fermi_level(fermi_level)
    )


def bry_genus2_topology_normalization_from_mqm(fermi_level: float) -> float:
    r"""Backward-compatible alias for the worldsheet normalization."""

    return bry_genus2_worldsheet_normalization_from_mqm(fermi_level)


def raw_product_bry_normalization_from_mqm(fermi_level: float) -> float:
    r"""Return ``2^21/(pi^3 mu^2)`` for the raw theta-product density."""

    return RAW_PRODUCT_FACTORIZATION_NORMALIZATION * (
        bry_genus2_worldsheet_normalization_from_mqm(fermi_level)
    )


def c1_matrix_model_genus2_coefficient(radius: float) -> float:
    r"""Return ``f_2(R)`` in ``F_2 = f_2(R) mu^-2``."""

    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    return (7.0 * radius**2 + 10.0 + 7.0 / radius**2) / (5760.0 * radius)


def factorization_normalized_moduli_integral_target(radius: float) -> float:
    r"""Return the bare period/Mumford fundamental-domain target from MQM.

    If ``J_2(R)`` is the bare period/Mumford integral over one Siegel
    fundamental domain, the completed v5 c=1 convention gives

    ``F_2 = g_s^2 64*pi^4 J_2(R)``.

    With ``mu=1/(4 pi g_s)``, matching ``F_2=f_2(R) mu^-2`` fixes
    ``J_2(R)=f_2(R)/(4*pi^2)``.  There is no additional global factor from the
    ineffective center ``{+/-I}`` of ``Sp(4,Z)``.

    This is an algebraic normalization target, not a fit to the numerical
    plumbing calculation.
    """

    return c1_matrix_model_genus2_coefficient(radius) / (4.0 * math.pi**2)


def coarse_genus2_fundamental_domain_target(radius: float) -> float:
    r"""Return the target over one ordinary Siegel fundamental domain.

    ``Sp(4,Z)`` and ``PSp(4,Z)`` have the same orbits and the same ordinary
    fundamental domain on ``H_2`` because ``+/-I`` acts trivially.
    """

    return factorization_normalized_moduli_integral_target(radius)


def raw_product_moduli_integral_target(radius: float) -> float:
    r"""Return the orbifold target before applying the ``2^24`` multiplier."""

    return (
        factorization_normalized_moduli_integral_target(radius)
        / RAW_PRODUCT_FACTORIZATION_NORMALIZATION
    )


def string_note_integration_kernel_target(
    radius: float,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the external MQM target expressed in the current c=1 convention.

    The sampled v5 kernel includes the complete multiplier ``64*pi^4``.
    Its target is ``16 pi^2 f_2(R)`` and equals ``pi^2/15`` at ``R=1``.
    """

    return (
        c1_sphere_normalized_genus2_kernel_multiplier(alpha_prime)
        * factorization_normalized_moduli_integral_target(radius)
    )


def coarse_string_note_integration_kernel_target(
    radius: float,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the string-note kernel target on the Siegel fundamental domain."""

    return string_note_integration_kernel_target(radius, alpha_prime)


def separating_mumford_normalization(
    tau_left: complex,
    tau_right: complex,
    epsilon: complex,
    *,
    chi10_normalization: str = RAW_THETA_PRODUCT,
    mumford_prefactor: complex = 2j * math.pi,
    theta_nmax: int | None = None,
    theta_tolerance: float = 1.0e-13,
    eta_max_mode: int = 400,
    eta_tolerance: float = 1.0e-16,
) -> SeparatingMumfordNormalization:
    r"""Evaluate the dimensionless Mumford residue near separation.

    We use ``q_bridge = 2 pi i epsilon``.  The dimensionless chiral residue
    is

    ``(2 pi i / chi10) (d epsilon / d q_bridge)
       q_bridge^2 eta(tau_left)^24 eta(tau_right)^24``.

    Its absolute square is the coefficient left after stripping the two
    genus-one Mumford factors and the universal ``|dq/q^2|^2`` pole.
    """

    tau_left = complex(tau_left)
    tau_right = complex(tau_right)
    epsilon = complex(epsilon)
    if tau_left.imag <= 0.0 or tau_right.imag <= 0.0:
        raise ValueError("both torus moduli must lie in the upper half-plane")
    if epsilon == 0.0:
        raise ValueError("epsilon must be nonzero for a numerical residue")

    omega = np.asarray(
        [[tau_left, epsilon], [epsilon, tau_right]],
        dtype=np.complex128,
    )
    if float(np.min(np.linalg.eigvalsh(np.imag(omega)))) <= 0.0:
        raise ValueError("the genus-two period matrix must have positive imaginary part")

    chi10 = complex(
        igusa_chi10_genus2(
            omega,
            nmax=theta_nmax,
            tol=theta_tolerance,
            normalization=chi10_normalization,
        )
    )
    q_bridge = 2j * math.pi * epsilon
    q_left = np.exp(2j * math.pi * tau_left)
    q_right = np.exp(2j * math.pi * tau_right)
    eta_left_abs = dedekind_eta_abs_from_q(
        q_left,
        max_mode=eta_max_mode,
        tolerance=eta_tolerance,
    )
    eta_right_abs = dedekind_eta_abs_from_q(
        q_right,
        max_mode=eta_max_mode,
        tolerance=eta_tolerance,
    )

    # Only the absolute square enters the nonchiral string measure.  The
    # eta phases therefore cancel and their positive absolute values suffice.
    eta_product_abs = (eta_left_abs * eta_right_abs) ** 24
    d_epsilon_d_q = 1.0 / (2j * math.pi)
    chiral_residue = (
        complex(mumford_prefactor)
        * d_epsilon_d_q
        * q_bridge**2
        * eta_product_abs
        / chi10
    )
    nonchiral_residue = float(abs(chiral_residue) ** 2)
    normalization = mumford_factorization_normalization(chi10_normalization)
    return SeparatingMumfordNormalization(
        tau_left=tau_left,
        tau_right=tau_right,
        epsilon=epsilon,
        q_bridge=q_bridge,
        chi10_normalization=chi10_normalization,
        chi10=chi10,
        chiral_residue=chiral_residue,
        nonchiral_residue=nonchiral_residue,
        factorization_normalization=normalization,
        normalized_nonchiral_residue=float(normalization * nonchiral_residue),
    )
