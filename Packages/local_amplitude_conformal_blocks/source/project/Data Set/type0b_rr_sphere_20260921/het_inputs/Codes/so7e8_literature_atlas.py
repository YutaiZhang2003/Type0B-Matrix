"""Regional NS/R channel selection for the literature SO(7) sphere layer.

Geometry ranks channels by |q|, never by which value makes crossing pass.
Actual block-order checks then determine eligibility. A channel which is
unresolved at a point is not required to agree there with a resolved one.
The fixed 32-node momentum rule and its accuracy are a separate issue.

Mixed base slots are (R0,Rz,NS1,NSinfinity). The u chart is obtained by
exchanging the two NS fields and applying the already defined reflection:
v=z/(z-1), w=1-v=1/(1-z). This preserves the R/NS variant labels.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import elliprf

from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3_fast as fast
from so7e8_literature_campaign import chiral_components, sign_pairs
from sphere_block_uniformization import elliptic_nome
from so7e8_literature_amplitude import (
    mixed_candidate_integrand, transport_mixed_r_component,
    reflect_four_ramond_tensor,
)
from so7e8_four_ramond_assembly import SPIN7_FIERZ_02_TO_01


CHARTS = {
    "mixed": {
        "s": ("mixed_ns", "NS", (0, 1, 2, 3)),
        "t": ("mixed_r", "R", (2, 1, 0, 3)),
        "u": ("mixed_r", "R", (3, 1, 0, 2)),
    },
    "rrrr": {
        "s": ("rrrr", "NS", (0, 1, 2, 3)),
        "t": ("rrrr", "NS", (2, 1, 0, 3)),
        "u": ("rrrr", "NS", (0, 2, 1, 3)),
    },
}


def _points(z):
    z = np.asarray(z, complex)
    if np.any(~np.isfinite(z)) or np.any((z == 0) | (z == 1)):
        raise ValueError("finite points away from the punctures required")
    if np.any((z.imag == 0) & ((z.real <= 0) | (z.real >= 1))):
        raise ValueError("choose a nonreal cut lip in the original coordinate")
    return z


def channel_coordinates(z, observable):
    z = _points(z)
    if observable not in CHARTS:
        raise ValueError("observable must be mixed or rrrr")
    return dict(s=z, t=1-z, u=1/(1-z) if observable == "mixed" else 1/z)


def geometric_assignment(z, observable):
    """Return one owner per point and all nome sizes on inherited sheets."""
    coordinates = channel_coordinates(z, observable)
    sizes = np.array([abs(elliptic_nome(coordinates[c])) for c in ("s", "t", "u")])
    return np.argmin(sizes, axis=0), sizes


def select_resolved_channel(z, observable, convergence):
    """Prefer the smallest nome among independently order-resolved channels.

    ``convergence`` maps chart names to {block_converged, order, ...}.
    No result is called converged from |q| alone. If all tested channels
    fail, retain an explicitly unresolved result; never omit the point.
    """
    coordinates = channel_coordinates(z, observable)
    if np.ndim(z):
        raise ValueError("the resolution selector takes one point")
    ranking = sorted(coordinates, key=lambda c: (abs(elliptic_nome(coordinates[c])), c))
    eligible = [c for c in ranking if convergence.get(c, {}).get("block_converged") is True]
    tested = [c for c in ranking if c in convergence]
    chosen = (eligible or tested or ranking)[0]
    return dict(channel=chosen, coordinate=complex(coordinates[chosen]),
                nome_abs=abs(elliptic_nome(coordinates[chosen])),
                block_converged=bool(eligible),
                status="order_resolved" if eligible else "unresolved",
                diagnostics=convergence.get(chosen),
                momentum_resolution="fixed 32-node rule; not certified by a block-order test")


def overlap_diagnostic(left, right, *, relative_tolerance=.02):
    """Interpret an overlap only after checking both CFT order diagnostics."""
    if not left["block_converged"] or not right["block_converged"]:
        return dict(status="unresolved_channel_not_a_crossing_failure", compared=False,
                    reason="both channel expansions must be order-resolved here")
    a, b = np.asarray(left["value"], complex), np.asarray(right["value"], complex)
    if a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("matching finite component tensors required")
    scale = max(float(np.max(abs(a))), float(np.max(abs(b))), 1e-300)
    error = float(np.max(abs(a-b)/np.maximum(np.maximum(abs(a), abs(b)), 1e-12*scale)))
    return dict(status="consistent" if error <= relative_tolerance else "needs_spectral_or_convention_diagnosis",
                compared=True, relative_error=error, tolerance=relative_tolerance,
                inference="an order-resolved disagreement alone does not identify its cause")


def exchange_mixed_ns_tensor(value, species):
    """Oriented Spin(7) coefficients under exchange of the two NS legs."""
    if species not in ("SS", "SV", "VS", "VV"):
        raise ValueError("two NS species required")
    value = np.asarray(value, complex)
    expected = 2 if species == "VV" else 1
    if value.shape[-1] != expected:
        raise ValueError("wrong number of Spin(7) tensor coefficients")
    signs = (1, -1) if species == "VV" else ((1,) if species == "SS" else (-1,))
    return value*np.asarray(signs)


def mixed_u_to_base(value, z, species):
    """Transport the FULL integrand from v=z/(z-1), not a bare SL block.

    The local amplitude has swapped NS momenta/species and swapped PCO
    placement. Its SL t chart is at 1-v=1/(1-z). Complete vertex weights
    (1,1) supply |1-z|^-4. Apply this once, outside component sewing.
    """
    return exchange_mixed_ns_tensor(value, species)/abs(1-complex(_points(z)))**4


def mixed_channel_integrand(z, species, time_momenta, channel, correlator, *,
                            picture="one", eta=(-1, 1)):
    """Assemble a chosen chart in the common mixed amplitude frame.

    ``correlator(w, external)`` integrates the native SL tensor using the
    chart's external momenta, sectors and 32-node internal-momentum rule.
    This is the same candidate coefficient tensor as the s-chart API;
    selecting a chart does not change its physical-certification status.
    """
    z = complex(_points(z))
    if channel not in CHARTS["mixed"]:
        raise ValueError("mixed channel must be s, t, or u")
    if picture not in ("one", "infinity"):
        raise ValueError("the raised NS leg must be one or infinity")
    if channel == "s":
        return mixed_candidate_integrand(z, species, time_momenta,
            lambda e: correlator(z, e), picture=picture, eta=eta)
    if channel == "t":
        return mixed_candidate_integrand(z, species, time_momenta,
            lambda e: transport_mixed_r_component(z, e, correlator), picture=picture, eta=eta)
    # Exchange the NS legs before using the established mixed reflection.
    # Form the small coordinate directly, avoiding 1-z/(z-1) cancellation.
    w = 1/(1-z)
    v = 1-w
    k = tuple(time_momenta)
    swapped_times = (k[0], k[1], k[3], k[2])
    def component(e):
        return transport_mixed_r_component(v, e, lambda unused, changed: correlator(w, changed))
    value = mixed_candidate_integrand(v, species[::-1], swapped_times, component,
        picture="infinity" if picture == "one" else "one", eta=eta)
    return mixed_u_to_base(value, z, species)


def four_ramond_channel_to_base(value, z, channel):
    """Transport full four-R integrands in the oriented Spin(7) rank basis.

    The t chart exchanges slots 0,2. The u chart exchanges slots 1,2
    with w=1/z. These exchange the identical three outgoing families in
    the existing scan. General family choices require a separate dictionary.
    """
    z = complex(_points(z))
    value = np.asarray(value, complex)
    if value.shape[-1] != 4:
        raise ValueError("four Spin(7) tensor ranks required")
    if channel == "s":
        return value
    if channel == "t":
        return reflect_four_ramond_tensor(value)
    if channel == "u":
        return -value@np.asarray(SPIN7_FIERZ_02_TO_01).T/abs(z)**4
    raise ValueError("four-R channel must be s, t, or u")


@dataclass(frozen=True)
class BlockGeometry:
    points: np.ndarray
    logz: np.ndarray
    log1: np.ndarray
    logq: np.ndarray
    t: np.ndarray
    logtheta: np.ndarray

    @classmethod
    def build(cls, points):
        z = _points(points).reshape(-1)
        # Evaluate the modular parameter before exponentiation. Never take
        # another square root of q, which discards its modular-sheet lift.
        logq = -math.pi*elliprf(0, z, 1)/elliprf(0, 1-z, 1)
        q = np.exp(logq)
        qmax = float(np.max(abs(q))) if len(q) else 0.
        if not 0 <= qmax < 1:
            raise ArithmeticError("nome outside the convergent disk")
        n = max(2, int(math.ceil(math.sqrt(math.log(1e-18)/math.log(qmax))))) if qmax else 2
        if n > 10000:
            raise ArithmeticError("choose a locally convergent channel")
        theta = 1+2*sum(q**(j*j) for j in range(1, n+1))
        return cls(z, np.log(z), np.log(1-z), logq,
                   np.exp(logq/2), np.log(theta))


def block_grid_values(blocks, geometry):
    """Vectorized native values, identical conventions to blocks.value().

    Output axes: external chiral component, internal parity, sign pair,
    coordinate. No momentum conjugation, variant inference or new recursion.
    """
    g = geometry
    result = np.empty((len(chiral_components(blocks.family)), 2,
                       len(sign_pairs(blocks.family)), len(g.points)), complex)
    p_part = blocks.internal.p**2/2*(math.log(16)+g.logq)
    for a, external in enumerate(chiral_components(blocks.family)):
        aa, bb, kk = blocks.prefactor_exponents(external)
        prefactor = np.exp(p_part+aa*g.logz+bb*g.log1+kk*g.logtheta)
        for parity in (0, 1):
            for b, (sl, sr) in enumerate(sign_pairs(blocks.family)):
                coefficients = blocks.elliptic_coefficients(external, parity, sl, sr)
                result[a, parity, b] = prefactor*np.polynomial.polynomial.polyval(g.t, coefficients)
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite block values on the selected atlas")
    return result


def pointwise_adjacent_change(previous, current, *, zero_floor=1e-12):
    """Never let a large block at a different modulus hide a local change."""
    a, b = np.asarray(previous, complex), np.asarray(current, complex)
    if a.shape != b.shape or a.ndim < 2 or not np.isfinite([a, b]).all():
        raise ValueError("matching finite arrays with coordinates on the last axis required")
    axes = tuple(range(a.ndim-1))
    scales = np.maximum(np.maximum(np.max(abs(a), axis=axes), np.max(abs(b), axis=axes)), 1e-300)
    relative = abs(a-b)/np.maximum(np.maximum(abs(a), abs(b)), zero_floor*scales)
    return np.max(relative, axis=axes)


def four_ns_moduli_layout(settings):
    """Preserve the four-NS bulk/lens rules and analytic order-17 disks.

    Coordinates here are the original moving puncture x. Mixed canonical
    z=1-x; four-R canonical z=x. The exterior is x=1/u and carries its
    area Jacobian explicitly. Disks are OPE patches, not omitted points or
    a replacement ordinary radial quadrature.
    """
    angular = int(settings["z_angular_nodes"])
    bulk, bw = fast.sewing_annulus_grid(
        settings["ope_radius"], settings["crossed_ope_radius"],
        (max(12, angular//4), max(12, angular//4), angular),
        settings["z_radial_nodes"])
    w, lw = fast.ref._lens_grid(settings["crossed_ope_radius"],
        settings["lens_radial_nodes"], settings["lens_angular_nodes"],
        settings.get("lens_power", 3.))
    lens = 1-w
    grids = {}
    for name, x, weights in (("bulk", bulk, bw), ("lens", lens, lw)):
        grids["original_"+name] = (x, weights)
        grids["exterior_"+name] = (1/x, weights/abs(x)**4)
    disks = [dict(name="original_zero_disk", coordinate="x", radius=settings["ope_radius"],
                  total_order=settings["disk_total_order"], integration="analytic OPE"),
             dict(name="exterior_zero_disk", coordinate="u=1/x", radius=settings["ope_radius"],
                  total_order=settings["disk_total_order"], integration="analytic OPE with inversion measure")]
    return grids, disks


def atlas_point_sets(settings):
    """Selected bulk/lens points plus labelled OPE-boundary order probes."""
    grids, disks = four_ns_moduli_layout(settings)
    out, receipt = {}, dict(grids={}, disks=disks)
    # A boundary probe does not certify an integrated analytic disk remainder.
    n = 2*int(settings["disk_total_order"])+4
    theta = 2*math.pi*(np.arange(n)+.5)/n
    boundary = settings["ope_radius"]*np.exp(1j*theta)
    probes = dict(original_disk_boundary=boundary, exterior_disk_boundary=1/boundary)
    for observable in CHARTS:
        selected = {c: [] for c in ("s", "t", "u")}
        counts = {c: 0 for c in selected}
        qmax = 0.
        for name, x in [(n, g[0]) for n, g in grids.items()]+list(probes.items()):
            z = 1-x if observable == "mixed" else x
            owner, sizes = geometric_assignment(z, observable)
            local = channel_coordinates(z, observable)
            for j, chart in enumerate(("s", "t", "u")):
                mask = owner == j
                selected[chart].extend(local[chart][mask])
                counts[chart] += int(np.sum(mask))
            qmax = max(qmax, float(np.max(np.min(sizes, axis=0))))
            receipt["grids"][observable+"/"+name] = dict(points=len(z), owners={c: int(np.sum(owner == j)) for j, c in enumerate(("s", "t", "u"))})
        out.update({observable+"/"+chart: np.array(points, complex) for chart, points in selected.items()})
        receipt[observable] = dict(channel_point_counts=counts, maximum_selected_nome=qmax)
    receipt["disk_probe_limitation"] = "boundary order probes; integrated disk remainder check remains separate"
    return out, receipt
