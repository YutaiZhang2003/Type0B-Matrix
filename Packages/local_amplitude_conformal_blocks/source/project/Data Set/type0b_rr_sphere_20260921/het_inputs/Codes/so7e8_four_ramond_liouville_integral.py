#!/usr/bin/env python3
r"""Generic-energy four-R quadrature using the full spin/disorder contraction.

The public path uses so7e8_four_ramond_nonchiral, HJS elliptic recursion,
Spin(7) KZ continuation, and the six-piece sphere atlas.  Both independent
crossing generators now pass numerical tests for the ordered physical
four-fermion target.  ``lens_channel='reordered'`` uses the tested local
z=1 channel and avoids the slow same-channel nome expansion on the lens.

One discrete spectator cocycle was fixed by crossing consistency; its
microscopic vertex derivation and the independent cubic normalization audit
remain open.  Individual quadratures still need order, spectral, and modulus
refinement.  Results therefore retain production_certified=False.  See
SO7E8_FOUR_RAMOND_REPAIR_20260911.md.  The older failure report describes
the superseded chiral-ground Fourier assembly, retained only for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal, Mapping, Sequence

import numpy as np

from spin23_singlet_amplitudes import _momentum_quadrature, _stable_complex_fsum
from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3_fast as fast
from ramond_sphere_uniformization import elliptically_prefactor_four_ramond_ns_series
from so7e8_ramond_fourpoint import FourRamondNSBlockSeries
from so7e8_four_ramond_assembly import (
    FixedPFourRamondKernel,
    SLDBlockBackend,
    SPIN7_BILINEAR_EXCHANGE_SIGNS,
    normalized_time_ising_scalar_block,
    spin7_reexpress_pairing_coefficients,
)
from so7e8_four_ramond_nonchiral import (
    build_nonchiral_fixed_p_four_ramond_kernel as build_fixed_p_four_ramond_kernel,
)
from so7e8_four_ramond_elliptic_recursion import FourRamondEllipticBlockSeries
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_ramond_cocycles import RamondFamily
from so7e8_two_ramond_liouville_integral import collision_convergence_margins
from so7e8_spin7_kz_continuation import continue_spin7_spectator_block_many


Spin7Backend = Literal["kz_ode", "local_series"]


@dataclass(frozen=True)
class FourRamondLiouvilleEvaluation:
    coefficients: tuple[complex, complex, complex, complex]
    pieces: Mapping[str, tuple[complex, complex, complex, complex]]
    maximum_twice_level: int
    lower_maximum_twice_level: int | None
    spin7_maximum_order: int
    lower_spin7_order: int | None
    lower_level_coefficients: tuple[complex, complex, complex, complex] | None
    # A two-step comparison adds one coefficient on each NS parity lattice.
    two_step_level_difference: tuple[complex, complex, complex, complex] | None
    lower_spin7_coefficients: tuple[complex, complex, complex, complex] | None
    spin7_order_difference: tuple[complex, complex, complex, complex] | None
    # Coarse balance between the two folded halves.  It need not vanish.
    folded_half_difference: tuple[complex, complex, complex, complex]
    momentum_nodes: int
    p_max: float
    p_cut: float
    momentum_scheme: str
    momentum_tail_certified: bool
    epsilon0: float
    epsilon1: float
    theta_orders: tuple[int, int, int]
    radial_order: int
    disk_grid: tuple[int, int, float]
    lens_grid: tuple[int, int, float]
    sld_block_backend: str
    spin7_backend: str
    lens_channel: str
    # ``same`` uses no channel braid and is certified in principle, although
    # its present finite local-series backend can be very poorly converged.
    z1_channel_certified: bool
    # This remains false until both the high-q SLD truncation and every
    # spectator continuation diagnostic are controlled on the whole atlas.
    production_certified: bool
    exterior_transport_certified: bool = False
    sewing_basis: str = "nonchiral_spin_fields"
    sewing_cocycle_status: str = "bootstrap_fixed"


@dataclass(frozen=True)
class _UniformTerm:
    source: object
    hol: object
    anti: object


def _prefactored_block(series: object) -> object:
    """Apply the universal HJS four-R elliptic prefactor to a general block."""

    if isinstance(series, FourRamondEllipticBlockSeries):
        # The residue-recursion backend constructs H(t) itself.  Applying
        # the inverse-Gram re-expansion here would factor the universal HJS
        # asymptotic a second time.
        return series

    # The assembly block carries additional external-ground/chirality
    # metadata, but its conformal weights and NS level lattice are exactly
    # those of the public FourRamondNSBlockSeries adapter.
    surrogate = FourRamondNSBlockSeries(
        coefficients=series.coefficients,
        component=series.component,
        c=series.c,
        h_internal=series.h_internal,
        external_betas=series.external_betas,
        external_weights=series.external_weights,
        left_structure_sign=series.left_structure_sign,
        right_structure_sign=series.right_structure_sign,
        maximum_twice_level=series.maximum_twice_level,
        gram_condition_numbers=series.gram_condition_numbers,
    )
    return elliptically_prefactor_four_ramond_ns_series(surrogate)


def _block_has_nonzero_coefficient(series: object) -> bool:
    if isinstance(series, FourRamondEllipticBlockSeries):
        return any(series.h_coefficients)
    return bool(series.coefficients)


def _evaluate_uniform_kernel(
    kernel: FixedPFourRamondKernel,
    z: np.ndarray,
    block_cache: dict[int, object] | None = None,
    spin7_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    points = np.asarray(z, dtype=np.complex128)
    zb = np.conjugate(points)
    result = np.zeros((points.size, 4), dtype=np.complex128)
    cache = {} if block_cache is None else block_cache
    for term in kernel.terms:
        if not _block_has_nonzero_coefficient(
            term.holomorphic_sld
        ) or not _block_has_nonzero_coefficient(term.antiholomorphic_sld):
            continue
        def value(series: object, coordinates: np.ndarray, side: str) -> np.ndarray:
            key = id(series)
            if key not in cache:
                cache[key] = _prefactored_block(series)
            return np.asarray(cache[key].value(coordinates, cut_side=side))
        hol = value(term.holomorphic_sld, points, "upper")
        anti = value(term.antiholomorphic_sld, zb, "lower")
        time = np.asarray([
            normalized_time_ising_scalar_block(term.time_channel, complex(x))
            for x in points
        ])
        if spin7_values is None:
            spin = np.stack([term.spin7_block.value(complex(x)) for x in zb])
        else:
            spin = np.asarray(spin7_values[term.spin7_block.channel])
            if spin.shape != (points.size, 4):
                raise ValueError("cached Spin(7) values have the wrong shape")
        result += term.coefficient * np.asarray(hol)[:, None] * np.asarray(anti)[:, None] * time[:, None] * spin
    k1, k2, k3, _ = kernel.time_momenta
    tx = np.exp(-2*k1*k2*np.log(np.abs(points))-2*k2*k3*np.log(np.abs(1-points)))
    superghost = np.exp(-0.25*(np.log(points)+np.log(1-points)))
    return tx[:, None] * superghost[:, None] * result


def _grid(
    kernel: FixedPFourRamondKernel,
    z: np.ndarray,
    weights: np.ndarray,
    block_cache: dict[int, object] | None = None,
    spin7_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    return np.sum(
        weights[:, None]
        * _evaluate_uniform_kernel(
            kernel,
            z,
            block_cache,
            spin7_values=spin7_values,
        ),
        axis=0,
    )


@lru_cache(maxsize=32)
def _cached_continued_spin7_values(
    shape: tuple[int, ...],
    flattened_coordinates: tuple[complex, ...],
) -> Mapping[str, np.ndarray]:
    coordinates = np.asarray(flattened_coordinates, dtype=np.complex128).reshape(shape)
    result = {
        channel: np.asarray(
            continue_spin7_spectator_block_many(channel, coordinates),
            dtype=np.complex128,
        )
        for channel in ("vacuum", "vector")
    }
    for values in result.values():
        values.setflags(write=False)
    return result


def _continued_spin7_values(z: np.ndarray) -> Mapping[str, np.ndarray]:
    """Continue both affine blocks once per immutable modulus grid.

    The same cached arrays are reused for every spectral ``P`` node and for
    adjacent SLD-order evaluations, for which the Spin(7) problem is
    identical.
    """

    coordinates = np.conjugate(np.asarray(z, dtype=np.complex128))
    return _cached_continued_spin7_values(
        tuple(coordinates.shape),
        tuple(complex(value) for value in coordinates.reshape(-1)),
    )


def _z1_local_to_base_pairing(values: Sequence[complex]) -> np.ndarray:
    r"""Return the ``w=1-z`` local tensor in the base ``01|23`` ordering.

    The local order is ``[old2,old1,old0,old3]`` in zero-based labels.
    Its coefficient map is ``D F D``: the permutation (02) is (01)(12)(01).
    The repository's oriented ``03|12`` map already equals ``D F D``.
    Multiplying by another bilinear-exchange ``D`` double-counts that sign.
    Exchanging the two complete fermion vertices supplies the common minus.
    """

    return -spin7_reexpress_pairing_coefficients(
        values,
        source_pairing="03|12",
        target_pairing="01|23",
    )


def _evaluate_order(
    momenta: tuple[complex, complex, complex, complex],
    families: tuple[RamondFamily, RamondFamily, RamondFamily, RamondFamily],
    *, maximum_twice_level: int,
    p_nodes: int | str | Sequence[int],
    p_max: float,
    p_cut: float,
    momentum_scheme: str,
    infinite_gauss_scale: float,
    epsilon0: float,
    epsilon1: float,
    theta_orders: Sequence[int],
    radial_order: int,
    disk_radial_order: int,
    disk_angular_order: int,
    disk_radial_power: float,
    lens_radial_order: int,
    lens_angular_order: int,
    lens_radial_power: float,
    digits: int,
    spin7_maximum_order: int,
    sld_block_backend: SLDBlockBackend,
    spin7_backend: Spin7Backend,
    lens_channel: Literal["same", "reordered", "reordered_experimental"],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    times = (momenta[0], momenta[1], momenta[2], -momenta[3])
    swapped_momenta = (momenta[0], momenta[2], momenta[1], momenta[3])
    swapped_families = (families[0], families[2], families[1], families[3])
    swapped_times = (times[0], times[2], times[1], times[3])
    ps, pweights = _momentum_quadrature(
        p_nodes,
        p_max,
        p_cut,
        scheme=momentum_scheme,
        infinite_gauss_scale=infinite_gauss_scale,
    )
    bulk, bw, _, _ = fast.annulus_excluding_lens(
        epsilon0,
        epsilon1,
        theta_orders,
        radial_order,
    )
    disk, dw = fast.disk_grid(
        epsilon0,
        disk_radial_order,
        disk_angular_order,
        disk_radial_power,
    )
    lensw, lw = fast.ref._lens_grid(
        epsilon1,
        lens_radial_order,
        lens_angular_order,
        lens_radial_power,
    )
    grids = (("bulk",bulk,bw),("z0",disk,dw))
    spin7_grids: dict[str, Mapping[str, np.ndarray] | None]
    if spin7_backend == "kz_ode":
        lens_key = "same_z1" if lens_channel == "same" else "reordered_z1"
        spin7_grids = {
            "bulk": _continued_spin7_values(bulk),
            "z0": _continued_spin7_values(disk),
            lens_key: _continued_spin7_values(
                1.0 - lensw if lens_channel == "same" else lensw
            ),
        }
    else:
        spin7_grids = {
            "bulk": None,
            "z0": None,
            "same_z1": None,
            "reordered_z1": None,
        }
    pieces = {
        f"{side}_{name}": []
        for side in ("original", "swapped")
        for name in ("bulk", "z0", "z1")
    }
    for p, pw in zip(ps, pweights, strict=True):
        original = build_fixed_p_four_ramond_kernel(
            float(p), external_liouville_momenta=momenta, time_momenta=times,
            families=families, maximum_twice_level=maximum_twice_level,
            spin7_maximum_order=spin7_maximum_order, digits=digits,
            sld_block_backend=sld_block_backend,
        )
        swapped = build_fixed_p_four_ramond_kernel(
            float(p), external_liouville_momenta=swapped_momenta,
            time_momenta=swapped_times, families=swapped_families,
            maximum_twice_level=maximum_twice_level,
            spin7_maximum_order=spin7_maximum_order, digits=digits,
            sld_block_backend=sld_block_backend,
        )
        if lens_channel != "same":
            # The local w=1-z order is [old2,old1,old0,old3].  The
            # nonchiral contraction passes this independent crossing test.
            original_lens = build_fixed_p_four_ramond_kernel(
                float(p),
                external_liouville_momenta=(
                    momenta[2], momenta[1], momenta[0], momenta[3]
                ),
                time_momenta=(times[2], times[1], times[0], times[3]),
                families=(families[2], families[1], families[0], families[3]),
                maximum_twice_level=maximum_twice_level,
                spin7_maximum_order=spin7_maximum_order,
                sld_block_backend=sld_block_backend,
                digits=digits,
            )
            swapped_lens = build_fixed_p_four_ramond_kernel(
                float(p),
                external_liouville_momenta=(
                    swapped_momenta[2],
                    swapped_momenta[1],
                    swapped_momenta[0],
                    swapped_momenta[3],
                ),
                time_momenta=(
                    swapped_times[2],
                    swapped_times[1],
                    swapped_times[0],
                    swapped_times[3],
                ),
                families=(
                    swapped_families[2],
                    swapped_families[1],
                    swapped_families[0],
                    swapped_families[3],
                ),
                maximum_twice_level=maximum_twice_level,
                spin7_maximum_order=spin7_maximum_order,
                sld_block_backend=sld_block_backend,
                digits=digits,
            )
        original_cache: dict[int, object] = {}
        swapped_cache: dict[int, object] = {}
        for name,z,w in grids:
            pieces[f"original_{name}"].append(
                pw / math.pi
                * _grid(
                    original,
                    z,
                    w,
                    original_cache,
                    spin7_values=spin7_grids[name],
                )
            )
            crossed = _grid(
                swapped,
                z,
                w,
                swapped_cache,
                spin7_values=spin7_grids[name],
            )
            # Fierz only reorders spinor indices.  The exchanged complete
            # Ramond vertices are spacetime-fermion odd and contribute one
            # additional Grassmann/Klein-factor minus.  Möbius, local-frame,
            # and bc-ghost Jacobians cancel against the integrated measure.
            crossed = -spin7_reexpress_pairing_coefficients(
                crossed, source_pairing="02|13", target_pairing="01|23"
            )
            pieces[f"swapped_{name}"].append(pw/math.pi*crossed)
        if lens_channel == "same":
            original_lens_value = _grid(
                original,
                1.0 - lensw,
                lw,
                original_cache,
                spin7_values=spin7_grids["same_z1"],
            )
            swapped_lens_value = _grid(
                swapped,
                1.0 - lensw,
                lw,
                swapped_cache,
                spin7_values=spin7_grids["same_z1"],
            )
        else:
            original_lens_value = _z1_local_to_base_pairing(
                _grid(
                    original_lens,
                    lensw,
                    lw,
                    {},
                    spin7_values=spin7_grids["reordered_z1"],
                )
            )
            swapped_lens_value = _z1_local_to_base_pairing(
                _grid(
                    swapped_lens,
                    lensw,
                    lw,
                    {},
                    spin7_values=spin7_grids["reordered_z1"],
                )
            )
        swapped_lens_value = -spin7_reexpress_pairing_coefficients(
            swapped_lens_value,
            source_pairing="02|13",
            target_pairing="01|23",
        )
        pieces["original_z1"].append(pw / math.pi * original_lens_value)
        pieces["swapped_z1"].append(pw / math.pi * swapped_lens_value)
    reduced = {name: np.asarray([_stable_complex_fsum(x[i] for x in vals) for i in range(4)]) for name,vals in pieces.items()}
    total = sum(reduced.values(), np.zeros(4,dtype=complex))
    return total, reduced


def evaluate_four_ramond_liouville_convergent(
    external_liouville_momenta: Sequence[complex], *,
    families: Sequence[RamondFamily], maximum_twice_level: int = 3,
    p_nodes: int | str | Sequence[int] = 2,
    p_max: float = 2.0,
    p_cut: float = 0.03,
    momentum_scheme: str = "cutoff",
    infinite_gauss_scale: float = 1.0,
    epsilon0: float = .1,
    epsilon1: float = .08,
    theta_orders: Sequence[int] = (4, 4, 8),
    radial_order: int = 4,
    disk_radial_order: int = 4,
    disk_angular_order: int = 12,
    disk_radial_power: float = 3.0,
    lens_radial_order: int = 4,
    lens_angular_order: int = 12,
    lens_radial_power: float = 3.0,
    digits: int = 35,
    spin7_maximum_order: int = 40,
    adjacent_level: bool = True,
    spin7_order_diagnostic: bool = True,
    sld_block_backend: SLDBlockBackend = "elliptic_recursion",
    spin7_backend: Spin7Backend = "kz_ode",
    lens_channel: Literal["same", "reordered", "reordered_experimental"] = "same",
) -> FourRamondLiouvilleEvaluation:
    momenta = tuple(complex(x) for x in external_liouville_momenta)
    fam = tuple(families)
    if len(momenta)!=4 or len(fam)!=4:
        raise ValueError("momenta and families must have length four")
    if lens_channel not in ("same", "reordered", "reordered_experimental"):
        raise ValueError("lens_channel must be 'same', 'reordered', or 'reordered_experimental'")
    if spin7_backend not in ("kz_ode", "local_series"):
        raise ValueError("spin7_backend must be 'kz_ode' or 'local_series'")
    if sld_block_backend not in ("inverse_gram", "elliptic_recursion"):
        raise ValueError(
            "sld_block_backend must be 'inverse_gram' or 'elliptic_recursion'"
        )
    require_conservative_real_p_chamber(momenta[:3], momenta[3])
    if not collision_convergence_margins(momenta).convergent:
        raise ValueError("all three collision margins must be positive")
    if not isinstance(spin7_maximum_order, int) or spin7_maximum_order < 0:
        raise ValueError("spin7_maximum_order must be a nonnegative integer")
    if not 0 < epsilon0 < 1 or not 0 < epsilon1 < 1:
        raise ValueError("epsilon0 and epsilon1 must lie in (0,1)")
    if epsilon0 + epsilon1 >= 1:
        raise ValueError("the zero disk and one lens must not overlap")
    if momentum_scheme == "cutoff" and p_max <= 0:
        raise ValueError("p_max must be positive for cutoff quadrature")
    theta_order_tuple = tuple(int(value) for value in theta_orders)
    if len(theta_order_tuple) != 3 or any(value < 1 for value in theta_order_tuple):
        raise ValueError("theta_orders must contain three positive integers")
    quadrature_options = dict(
        p_nodes=p_nodes,
        p_max=p_max,
        p_cut=p_cut,
        momentum_scheme=momentum_scheme,
        infinite_gauss_scale=infinite_gauss_scale,
        epsilon0=epsilon0,
        epsilon1=epsilon1,
        theta_orders=theta_order_tuple,
        radial_order=radial_order,
        disk_radial_order=disk_radial_order,
        disk_angular_order=disk_angular_order,
        disk_radial_power=disk_radial_power,
        lens_radial_order=lens_radial_order,
        lens_angular_order=lens_angular_order,
        lens_radial_power=lens_radial_power,
    )
    total, pieces = _evaluate_order(
        momenta,
        fam,
        maximum_twice_level=maximum_twice_level,
        digits=digits,
        spin7_maximum_order=spin7_maximum_order,
        sld_block_backend=sld_block_backend,
        spin7_backend=spin7_backend,
        lens_channel=lens_channel,
        **quadrature_options,
    )
    lower = None
    difference = None
    if adjacent_level and maximum_twice_level >= 2:
        lower, _ = _evaluate_order(
            momenta,
            fam,
            maximum_twice_level=max(0, maximum_twice_level - 2),
            digits=digits,
            spin7_maximum_order=spin7_maximum_order,
            sld_block_backend=sld_block_backend,
            spin7_backend=spin7_backend,
            lens_channel=lens_channel,
            **quadrature_options,
        )
        difference = total-lower
    lower_spin = None
    spin_difference = None
    lower_spin_order = None
    if (
        spin7_backend == "local_series"
        and spin7_order_diagnostic
        and spin7_maximum_order >= 2
    ):
        lower_spin_order = spin7_maximum_order // 2
        lower_spin, _ = _evaluate_order(
            momenta,
            fam,
            maximum_twice_level=maximum_twice_level,
            digits=digits,
            spin7_maximum_order=lower_spin_order,
            sld_block_backend=sld_block_backend,
            spin7_backend=spin7_backend,
            lens_channel=lens_channel,
            **quadrature_options,
        )
        spin_difference = total - lower_spin
    original = sum((v for k,v in pieces.items() if k.startswith("original")),np.zeros(4,dtype=complex))
    swapped = sum((v for k,v in pieces.items() if k.startswith("swapped")),np.zeros(4,dtype=complex))
    sampled_momenta, _ = _momentum_quadrature(
        p_nodes,
        p_max,
        p_cut,
        scheme=momentum_scheme,
        infinite_gauss_scale=infinite_gauss_scale,
    )
    return FourRamondLiouvilleEvaluation(
        coefficients=tuple(total),
        pieces={k: tuple(v) for k, v in pieces.items()},
        maximum_twice_level=maximum_twice_level,
        lower_maximum_twice_level=(
            None if lower is None else max(0, maximum_twice_level - 2)
        ),
        spin7_maximum_order=spin7_maximum_order,
        lower_spin7_order=lower_spin_order,
        lower_level_coefficients=None if lower is None else tuple(lower),
        two_step_level_difference=(
            None if difference is None else tuple(difference)
        ),
        lower_spin7_coefficients=(
            None if lower_spin is None else tuple(lower_spin)
        ),
        spin7_order_difference=(
            None if spin_difference is None else tuple(spin_difference)
        ),
        folded_half_difference=tuple(original - swapped),
        momentum_nodes=len(sampled_momenta),
        p_max=float(p_max),
        p_cut=float(p_cut),
        momentum_scheme=momentum_scheme,
        # Covering the half-line is not the same as demonstrating numerical
        # convergence of its quadrature.  Certification requires a separate
        # node/scale or cutoff/tail comparison and is never inferred here.
        momentum_tail_certified=False,
        epsilon0=float(epsilon0),
        epsilon1=float(epsilon1),
        theta_orders=theta_order_tuple,
        radial_order=int(radial_order),
        disk_grid=(
            int(disk_radial_order),
            int(disk_angular_order),
            float(disk_radial_power),
        ),
        lens_grid=(
            int(lens_radial_order),
            int(lens_angular_order),
            float(lens_radial_power),
        ),
        sld_block_backend=sld_block_backend,
        spin7_backend=spin7_backend,
        lens_channel=lens_channel,
        z1_channel_certified=(lens_channel == "same"),
        production_certified=False,
    )


__all__ = [
    "FourRamondLiouvilleEvaluation",
    "SLDBlockBackend",
    "Spin7Backend",
    "evaluate_four_ramond_liouville_convergent",
]
