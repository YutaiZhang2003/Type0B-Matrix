"""Heterotic amplitude *integrands* on the frozen literature CCY blocks.

This is a pre-production assembly layer, not a certified moduli integral.
The mixed full-field candidate obeys the literature supercurrent Ward
identity; its microscopic Ramond cocycle dictionary is still a separate
check. The four-R reference retains the previously crossing-checked finite
GSO tensor, including its documented bootstrap-fixed cocycle. Neither
construction acquires an absolute physical normalization by passing a
numerical crossing test. See SO7E8_LITERATURE_AMPLITUDE_ASSEMBLY_20260921.md.

Mixed coordinates are (R0,Rz,NS1,NSinfinity), as in the bank manifest.
The two letters of ``species`` consequently refer to NS1 and NSinfinity,
in that order (reverse the original scan's NSz,NS1 labels).
Every returned value omits the common external Liouville factor, sphere
constant, coupling, energy delta function and external reflection phases.
The spectral measure dP/pi is applied by the integration caller, once.
"""
from functools import lru_cache
from itertools import product
import cmath
import math

import numpy as np

from literature_component_blocks import crossing_phase, SECTORS, ramond_state, ramond_vertex
from so7e8_direct_domain import spin7_global_value
from so7e8_four_ramond_assembly import (
    SPIN7_BILINEAR_EXCHANGE_SIGNS, SPIN7_FIERZ_02_TO_01,
)
from so7e8_four_ramond_nonchiral import nonchiral_heterotic_sewing_weight


FAMILIES = ("Psi_tilde", "Psi_tilde", "Psi_tilde", "Psi")
_X = np.array([[0, 1], [1, 0]], complex)
_I = np.eye(2, dtype=complex)
_THETA = cmath.exp(-1j*math.pi/4)


def _point(z):
    """A single slit-plane lift; a real cut needs an explicit nonreal lip."""
    z = complex(z)
    if not np.isfinite(z) or z in (0, 1):
        raise ValueError("a finite coordinate away from the punctures is required")
    if z.imag == 0 and not 0 < z.real < 1:
        raise ValueError("specify the cut lip in z before forming 1-z")
    return z


def _times(values):
    values = tuple(map(complex, values))
    if len(values) != 4 or not np.isfinite(values).all():
        raise ValueError("four finite signed time momenta required")
    if abs(sum(values)) > 1e-11*max(1., sum(map(abs, values))):
        raise ValueError("signed time momenta must sum to zero")
    return values


def timelike_factor(z, time_momenta):
    z, k = _point(z), _times(time_momenta)
    # Conjugate coordinates only. Complex physical energies are held fixed.
    return cmath.exp(-2*k[0]*k[1]*math.log(abs(z))
                     -2*k[1]*k[2]*math.log(abs(1-z)))


def g0_phase(parity):
    """G0 R^r = p*g0_phase(r)*R^(1-r), Suchanek's small representation."""
    if parity not in (0, 1):
        raise ValueError("Ramond parity must be a bit")
    return cmath.exp(1j*math.pi*(2*parity-1)/4)/math.sqrt(2)


def mixed_candidate_tensors(z, species, *, eta=(-1, 1)):
    r"""Finite Clifford/BRST candidate, separately exposing A and D.

    A multiplies a single G descendant. D multiplies the primary term
    associated with the time fermion. It is derived from the literal
    literature G0 relation, not determined from a crossing ratio:

      D_rs = eta_z (-1)^b3 A_(r,1-s) g0_phase(1-s).

    The second endpoint must satisfy the independent algebraic condition
      eta_z i (-1)^(b3+s) A_(1-r,s) g0_phase(1-r)
        = eta_0 eta_z D_rs.

    This establishes BRST compatibility of this finite candidate, not its
    equivalence to the microscopic heterotic spin-field cocycles. In
    particular it must not silently replace the historical GSO tensor.
    """
    z = _point(z)
    if species not in ("SS", "SV", "VS", "VV"):
        raise ValueError("species must be SS, SV, VS, or VV")
    if len(eta) != 2 or any(e not in (-1, 1) for e in eta):
        raise ValueError("two Ramond family signs required")
    zb = z.conjugate()
    logz, log1 = cmath.log(z), cmath.log(1-z)
    root_bar = cmath.exp(.5*logz.conjugate())
    root_one_bar = cmath.exp(.5*log1.conjugate())
    common = cmath.exp(-logz/8-7*logz.conjugate()/8)
    v = tuple(s == "V" for s in species)
    if not any(v):
        free = np.array([_X])
    elif sum(v) == 1:
        free = np.array([_I*root_bar/math.sqrt(2)
                         /(root_one_bar if v[0] else 1)])
    else:
        # Coefficients of delta_ab C and C gamma_ab, respectively.
        free = np.array([_X*(2-zb)/(2*root_one_bar),
                         -_X*zb/(2*root_one_bar)])
    free *= common
    a = np.zeros((len(free), 2, 2), complex)
    d = np.zeros_like(a)
    for r, s in product((0, 1), repeat=2):
        a[:, r, s] = (eta[0]**r*eta[1]**s*_THETA**(r-s)/2
                      * (-1)**(v[0]*(r+s))*free[:, s, r])
    b3 = int(species[0] == "S")
    for r, s in product((0, 1), repeat=2):
        d[:, r, s] = eta[1]*(-1)**b3*a[:, r, 1-s]*g0_phase(1-s)
    return a, d


def mixed_candidate_integrand(z, species, time_momenta, correlator, *,
                              picture="one", eta=(-1, 1)):
    """Assemble one candidate from a full, already P-integrated SL tensor.

    ``correlator(external)`` must use (R,R,NS,NS) at this same z and in
    the literal literature frame. For the R-internal channel, transport
    the complete component with ``crossing_phase`` first. This function
    does not conjugate the callback or its external physical momenta.

    The sphere ghost has weight 3/8 on each R and weight 1/2 on the NS
    leg left in picture -1. G shifts its own external SL weight by 1/2
    in the block layer. The root at the Ramond zero in the contour Ward
    relation is +i*sqrt(z), not an independently evaluated sqrt(-z).
    """
    z, k = _point(z), _times(time_momenta)
    if picture not in ("one", "infinity"):
        raise ValueError("the raised NS leg must be one or infinity")
    a, d = mixed_candidate_tensors(z, species, eta=eta)
    rootz, root1 = cmath.exp(.5*cmath.log(z)), cmath.exp(.5*cmath.log(1-z))
    psi_coefficient = k[2]*rootz/root1 if picture == "one" else -k[3]*rootz
    answer = np.zeros(len(a), complex)
    for r, s in product((0, 1), repeat=2):
        anti = tuple(int(x == "S") for x in species)
        primary = (r, s, (0, anti[0]), (0, anti[1]))
        descendant = (r, s, (int(picture == "one"), anti[0]),
                      (int(picture == "infinity"), anti[1]))
        answer += (a[:, r, s]*correlator(descendant)
                   + psi_coefficient*d[:, r, s]*correlator(primary))
    ghost = cmath.exp(-.25*cmath.log(z))/(root1 if picture == "infinity" else 1)
    return timelike_factor(z, k)*ghost*answer


def transport_mixed_r_component(z, external, crossed_correlator):
    """Explicit clockwise 0<->2 frame transport, including NS spin shifts."""
    _point(z)
    changed = (external[2], external[1], external[0], external[3])
    return crossing_phase(SECTORS["mixed_ns"], external)*crossed_correlator(1-z, changed)


@lru_cache(maxsize=1024)
def four_ramond_spectators(z):
    """Unit-leading time-Ising and fixed-channel Spin(7) blocks.

    Keep Log(z) and Log(1-z) separate; Log(z*(1-z)) can lose a sheet.
    The rationalized fermion root also avoids cancellation near z=0.
    """
    z = _point(z)
    lz, l1 = cmath.log(z), cmath.log(1-z)
    root1 = cmath.exp(.5*l1)
    plus = cmath.sqrt((1+root1)/2)
    common = cmath.exp(-(lz+l1)/8)
    time = np.array([common*plus, common*cmath.exp(.5*lz)/plus])
    spin = np.array([spin7_global_value(c, z.conjugate())
                     for c in ("vacuum", "vector")])
    time.setflags(write=False)
    spin.setflags(write=False)
    return time, spin


@lru_cache(None)
def four_ramond_reference_terms(families=FAMILIES):
    """Retain the complete historical finite tensor in canonical block space.

    Do NOT replace it by a diagonal product of physical R+/R- SL
    correlators and auxiliary Ising correlators. Its odd-block convention
    factor (sL*sR)^(qh+qa) is part of the coefficient tensor. Dropping it
    changes the observable. Its bootstrap-fixed provenance is retained.
    """
    terms = []
    for qh, qt, qa, q7, sl, sr in product(*([(0, 1)]*4+[(-1, 1)]*2)):
        weight = nonchiral_heterotic_sewing_weight(tuple(families), qh, qt, qa, q7, sl, sr)
        if weight:
            terms.append((qh, qt, qa, q7, sl, sr, complex(weight)))
    return tuple(terms)


def native_four_ramond_weight(external, qh, qa, sl, sr):
    """Literal literature full-field coefficient of canonical block products.

    Endpoint Ward transport is i**(q*(a_infinity-a_zero)). This is separate
    from the historical GSO tensor's odd-block conversion. Keeping these
    two objects distinct prevents a double application or loss of that
    conversion when a physical component sum is simplified.
    """
    if len(external) != 4 or any(r not in (0, 1) for r in external):
        raise ValueError("four Ramond parity bits required")
    if qh not in (0, 1) or qa not in (0, 1) or sl not in (-1, 1) or sr not in (-1, 1):
        raise ValueError("invalid internal parity or structure sign")
    total = 0j
    for ((a0, b0), ket), ((a4, b4), bra) in product(
            ramond_state(external[0]), ramond_state(external[3])):
        rf, brf = (qh+a0) % 2, (qa+b0) % 2
        lf, blf = (qh+a4) % 2, (qa+b4) % 2
        total += (ket*complex(bra).conjugate()
                  * ramond_vertex(external[1], sr, rf, brf)
                  * ramond_vertex(external[2], sl, lf, blf)
                  * (-1)**(brf*a0+blf*qh)
                  * 1j**(qh*(a4-a0)) * (-1j)**(qa*(b4-b0)))
    return complex(total)


def four_ramond_reference_integrand(z, time_momenta, blocks, anti_value,
                                    constants, P, *, families=FAMILIES):
    """Canonical GSO reference tensor evaluated by literature CCY blocks.

    Returns four Spin(7) tensor ranks, at fixed internal NS momentum.
    This reuses a finite coefficient tensor, not the old block recursion.
    Absolute normalization and microscopic cocycle certification remain
    outstanding; this API therefore deliberately says ``reference``.
    """
    z, k = _point(z), _times(time_momenta)
    if blocks.family != "rrrr":
        raise ValueError("four-R literature blocks required")
    time, spin = four_ramond_spectators(z)
    result = np.zeros(4, complex)
    for qh, qt, qa, q7, sl, sr, weight in four_ramond_reference_terms(tuple(families)):
        scalar = (weight*constants.density("rrrr", blocks.p, P, 0, sl, sr)
                  * blocks.value(z, (0, 0, 0, 0), qh, sl, sr)
                  * anti_value(z, (0, 0, 0, 0), qa, sl, sr)*time[qt])
        result += scalar*spin[q7]
    ghost = cmath.exp(-.25*(cmath.log(z)+cmath.log(1-z)))
    return timelike_factor(z, k)*ghost*result


def reflect_four_ramond_tensor(value):
    """Fermion exchange (02), in the existing oriented Spin(7) rank basis."""
    d = np.diag(SPIN7_BILINEAR_EXCHANGE_SIGNS)
    return -np.asarray(value)@(d@np.asarray(SPIN7_FIERZ_02_TO_01)@d).T


def require_physical_amplitude_ready(report):
    """Check the regional numerical atlas and independent physical inputs.

    A disagreement with an unused, unresolved channel is not a production
    veto. Compare overlaps only where both retained CFT expansions resolve.
    The historical fixed-z ``assembled_crossing_pass`` is diagnostic data,
    not the numerical release condition for a regional moduli calculation.
    """
    needed = ("mixed_physical_dictionary_verified", "four_ramond_physical_dictionary_verified",
              "absolute_normalization_verified", "moduli_atlas_complete",
              "moduli_grid_adjacent_orders_pass", "resolved_overlaps_consistent",
              "analytic_disks_checked")
    missing = [key for key in needed if report.get(key) is not True]
    if missing:
        raise ValueError("amplitude production held: " + ", ".join(missing))
