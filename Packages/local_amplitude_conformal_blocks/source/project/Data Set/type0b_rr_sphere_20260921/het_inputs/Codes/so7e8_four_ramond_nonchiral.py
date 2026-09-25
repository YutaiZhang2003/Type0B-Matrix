"""Four-R sewing in the nonchiral spin/disorder basis.

The previous chiral-ground Fourier prescription is retained in
so7e8_four_ramond_assembly for comparison.  It is not used here.  This
construction first contracts the full Ramond fields, then reduces to
canonical chiral blocks.  See SO7E8_FOUR_RAMOND_REPAIR_20260911.md for the
conventions, the independent crossing tests, and the remaining physical
external-state normalization check.
"""
from functools import lru_cache
from itertools import product
import math

from so7e8_four_ramond_elliptic_recursion import (
    four_ramond_ground_map, general_four_ramond_sld_elliptic_h_series,
)
from so7e8_four_ramond_assembly import (
    FixedPFourRamondKernel, FixedPFourRamondKernelTerm,
    FourRamondPolarizationComponent, allowed_four_ramond_parity_sewings,
    general_four_ramond_sld_chiral_block, spin7_spectator_block_series,
)
from spin23_super_liouville_data import rr_ns_chiral_structure_constant


def _small_ground(r):
    s = 1/math.sqrt(2)
    return ((0,0,s),(1,1,-1j*s)) if r == 0 else ((0,1,s),(1,0,s))


def _ramond_operator(r, hol_parity, anti_parity, sign):
    if hol_parity ^ anti_parity != r:
        return 0
    if r == 1:
        return sign
    return 1 if hol_parity == 0 else -1j


@lru_cache(maxsize=None)
def nonchiral_ramond_weight(r, qh, qa, sl, sr):
    """Coefficient of canonical blocks for four full R+/R- fields.

    HJS 0810.1203v2 (3.6), (4.13), (4.15), in the ordinary graded tensor
    operator convention: (A tensor B)(v tensor w)=(-1)^(|B||v|)Av tensor Bw.
    Endpoint coefficients are dualized only at infinity.  The final sign
    converts the printed HJS odd block to this repository's odd block.
    """
    total = 0j
    for a1,b1,k1 in _small_ground(r[0]):
        for a4,b4,k4 in _small_ground(r[3]):
            phr, par = qh ^ a1, qa ^ b1
            phl, pal = qh ^ a4, qa ^ b4
            op = (_ramond_operator(r[1],phr,par,sr)
                  * _ramond_operator(r[2],phl,pal,sl)/2)
            grade = (-1)**(par*a1 + pal*qh)
            hol = four_ramond_ground_map((a1,0,0,a4),
                component=("even","odd")[qh],
                left_structure_sign=sl,right_structure_sign=sr).phase
            anti = four_ramond_ground_map((b1,0,0,b4),
                component=("even","odd")[qa],
                left_structure_sign=sl,right_structure_sign=sr,
                chirality="antiholomorphic").phase
            total += k1*complex(k4).conjugate()*op*grade*hol*anti
    return total*(sl*sr)**(qh+qa)


@lru_cache(maxsize=None)
def nonchiral_heterotic_sewing_weight(families, qh, qt, qa, q7, sl, sr):
    """Contract the spin/disorder fields before applying chiral recursion.

    The Spin(7) map reverses the auxiliary Ising channel: qb=1-q7.
    For unit-leading blocks it maps the Ising identity to +K7_vector/2,
    and the Ising fermion to -2*K7_vacuum.  Including the time/Ising OPE
    factors gives magnitudes (1,1/2,1/4), not an adjustable phase.
    """
    if len(families) != 4 or any(f not in ("Psi","Psi_tilde") for f in families):
        raise ValueError("families must contain four Psi/Psi_tilde labels")
    if any(q not in (0,1) for q in (qh,qt,qa,q7)) or sl not in (-1,1) or sr not in (-1,1):
        raise ValueError("parities must be bits and HJS signs must be +/-1")
    if qh ^ qt ^ qa ^ q7 != 1:
        return 0j
    eta = tuple(1 if f == "Psi" else -1 for f in families)
    qb = 1-q7
    total = 0j
    for r in product((0,1), repeat=4):
        if sum(r) % 2:
            continue
        external = math.prod(e**bit for e,bit in zip(eta,r))/4
        sld = nonchiral_ramond_weight(r,qh,qa,sl,sr)
        # Relative order/disorder-line cocycle for the Spin(7) channel
        # reversal.  The endpoint pair (1,4) and middle pair (2,3) each
        # contribute a minus when both fields are disorder fields.  This
        # discrete sign is fixed by the second crossing generator; omitting
        # it passes inversion but fails z -> 1-z by about 30 percent.
        line_cocycle = (-1)**(r[0]*r[3] + r[1]*r[2])
        spectator = (line_cocycle*nonchiral_ramond_weight(r,qt,qb,1,1)
                     * .5**(qt+qb) * (.5 if qb == 0 else -2))
        total += external*sld*spectator
    return 0j if abs(total) < 1e-14 else total


def build_nonchiral_fixed_p_four_ramond_kernel(
    internal_momentum, *, external_liouville_momenta, time_momenta, families,
    maximum_twice_level=1, spin7_maximum_order=2, fermion_turning_phase=None,
    sld_block_backend="inverse_gram", digits=40, condition_limit=1e13,
):
    """Build the contracted kernel; the public integral uses this path."""
    m = tuple(complex(p) for p in external_liouville_momenta)
    times = tuple(complex(p) for p in time_momenta)
    families = tuple(families)
    if len(m) != 4 or len(times) != 4 or len(families) != 4:
        raise ValueError("momenta, times, and families must each have length four")
    if abs(sum(times)) > 1e-10:
        raise ValueError("signed time momenta must sum to zero")
    tau = 1 if fermion_turning_phase is None else complex(fermion_turning_phase)
    if abs(abs(tau)-1) > 1e-12:
        raise ValueError("fermion_turning_phase must have unit magnitude")
    if sld_block_backend not in ("inverse_gram","elliptic_recursion"):
        raise ValueError("sld_block_backend must be 'inverse_gram' or 'elliptic_recursion'")
    builder = (general_four_ramond_sld_chiral_block if sld_block_backend == "inverse_gram"
               else general_four_ramond_sld_elliptic_h_series)
    structures = {(sl,sr): .25*rr_ns_chiral_structure_constant(
        m[3],m[2],internal_momentum,structure_sign=sl,precision=digits
    )*rr_ns_chiral_structure_constant(m[1],m[0],-complex(internal_momentum),
        structure_sign=sr,precision=digits) for sl,sr in product((-1,1),repeat=2)}
    spin = {c:spin7_spectator_block_series(c,maximum_order=spin7_maximum_order)
            for c in ("vacuum","vector")}
    # Component contraction is complete: these are canonical block grounds,
    # not a new uncontracted physical vertex polarization.
    pol = FourRamondPolarizationComponent(families,(0,0,0,0),(0,0,0,0),(0,0,0,0),1)
    terms, cache = [], {}
    for q in allowed_four_ramond_parity_sewings():
        for sl,sr in product((-1,1),repeat=2):
            weight = nonchiral_heterotic_sewing_weight(families,
                q.holomorphic_sld_parity,q.time_ising_parity,
                q.antiholomorphic_sld_parity,q.spin7_fermion_parity,sl,sr)
            if not weight:
                continue
            blocks = []
            for chirality,parity in (("holomorphic",q.holomorphic_sld_parity),
                                    ("antiholomorphic",q.antiholomorphic_sld_parity)):
                key = (chirality,parity,sl,sr)
                if key not in cache:
                    cache[key] = builder(internal_momentum,external_momenta=m,
                        external_ground_parities=(0,0,0,0),
                        maximum_twice_level=maximum_twice_level,
                        component=("even","odd")[parity],left_structure_sign=sl,
                        right_structure_sign=sr,chirality=chirality,
                        digits=digits,condition_limit=condition_limit)
                blocks.append(cache[key])
            terms.append(FixedPFourRamondKernelTerm(pol,q,sl,sr,
                structures[sl,sr]*weight*tau**q.time_ising_parity,
                blocks[0],blocks[1],q.time_channel,spin[q.spin7_channel]))
    return FixedPFourRamondKernel(families,complex(internal_momentum),times,
        tau,sld_block_backend,tuple(terms))
