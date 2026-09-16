"""Physical NSRR bilinear contraction of pointwise resummed blocks.

All geometry, transported lifts, structure constants and quadrature weights
are supplied by the caller. This module introduces no momentum sampling.
The default uses the derived Human-Note physical BPZ matrix. Full-state
tube signs are supplied separately from the two-lift block projection.
The previous scalar is available only by explicit legacy selection.
"""
from __future__ import annotations

import cmath
import math
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[1] / "full_ramond_block_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
from nsrr_resummed_backend import ResummedNSRR
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
from physical_nsrr_sewing import contract_physical_blocks
from nsrr_bilinear_sewing import (BASE_PROJECTION_LIFTS, CONVENTION,
                                physical_signs)
from nsrr_normalization import (LOCAL, contract_normalized_nsrr,
                                normalization_metadata)


def resummed_integrand(*, b, momenta_geometry, q_geometry, lifts_geometry, bry_constants,
                       branch_level, recursion_order, branch_truncation="per-edge",
                       global_tolerance=1e-13, global_max_shell=64, dps=40,
                       output_directory=None, use_parity_symmetry=True,
                       physical_lifts_slots=None, sewing_convention="human-bilinear",
                       normalization=LOCAL):
    # Explicit opt-in keeps historical/local calculations reproducible.
    # The provisional factor multiplies M once, never either chiral block.
    norm_metadata = normalization_metadata(normalization)
    lifts = tuple(tuple(l) for l in lifts_geometry)
    if len(lifts) != 2:
        raise ValueError("supply the saved two-lift projection explicitly")
    if sewing_convention not in ("human-bilinear", "legacy-times-four"):
        raise ValueError("Unknown NSRR sewing convention")
    if sewing_convention != "human-bilinear" and normalization != LOCAL:
        raise ValueError("The normalization hypothesis applies only to the Human-Note bilinear pairing")
    if sewing_convention == "human-bilinear":
        if physical_lifts_slots is None:
            raise ValueError("Supply physical_lifts_slots in NS,R1,R0 order; the two-lift basis projection does not determine these signs")
        physical_signs(physical_lifts_slots)
        if set(lifts) != set(BASE_PROJECTION_LIFTS):
            raise ValueError("The derived matrix uses the base two-lift projection (+++),(+-+)")
    plumbing = NSRRPlumbingInputs(tuple(q_geometry), lifts[0], GEOMETRY_SECTORS)
    runtime = ResummedNSRR(b, plumbing.momenta_slots(momenta_geometry),
                          branch_level=branch_level, recursion_order=recursion_order,
                          branch_truncation=branch_truncation, global_tolerance=global_tolerance,
                          global_max_shell=global_max_shell, dps=dps,
                          output_directory=output_directory)
    values = runtime.channels(plumbing.q_slots, use_parity_symmetry=use_parity_symmetry)
    primary = plumbing.primary(b, momenta_geometry)
    blocks = {channel: primary*sum(runtime.project(vector, lift[::-1]) for lift in lifts)/math.sqrt(2)
              for channel,vector in values.items()}
    descendant_blocks = {
        channel: sum(runtime.project(vector, lift[::-1]) for lift in lifts)/math.sqrt(2)
        for channel, vector in values.items()
    }
    if sewing_convention == "human-bilinear":
        # This wrapper is a real-b, real-momentum evaluator. Only on this
        # slice do the independent HJS anti blocks at qbar equal F*.
        if complex(b).imag or any(complex(p).imag for p in momenta_geometry):
            raise ValueError("The resummed anti specialization requires real b and momenta")
        result = contract_normalized_nsrr(normalization=normalization, descendant_blocks=descendant_blocks,
            antiholomorphic_blocks={k: z.conjugate() for k, z in descendant_blocks.items()},
            left_bry=bry_constants, right_bry=bry_constants,
            primary=primary, antiholomorphic_primary=primary.conjugate(),
            physical_lifts_slots=physical_lifts_slots)
        integrand = {}
        for name in ("total", "diagonal", "interference"):
            z = result[name]
            if abs(z.imag) > 1e-11*max(abs(z), 1e-300):
                raise ArithmeticError("Real-slice BPZ integrand has a nonzero imaginary part")
            integrand[name] = z.real
        convention, verified = CONVENTION, normalization == LOCAL
    else:
        result = contract_physical_blocks(blocks, bry_constants)
        integrand = {name: 4*result[name] for name in ("total", "diagonal", "interference")}
        convention, verified = "legacy Hermitian candidate times four", False
    return dict(integrand=integrand,
                sewing_convention=convention,
                human_bilinear_pairing_verified=verified,
                local_bilinear_kernel_verified=sewing_convention == "human-bilinear",
                **(norm_metadata if sewing_convention == "human-bilinear" else
                   dict(normalization="legacy-times-four", normalization_factor=4,
                        normalization_status="legacy candidate", global_normalization_verified=False)),
                physical_lifts_slots=physical_lifts_slots,
                marked_spin_transport_verified=False,
                # Preserve the historical API: `blocks` contains propagated
                # amplitudes. The Human-Note blocks are `descendant_blocks`.
                blocks=blocks, descendant_blocks=descendant_blocks,
                primary_prefactor=primary,
                primary_weights_slots=plumbing.weights_slots(b, momenta_geometry),
                log_q_slots=tuple(cmath.log(q) for q in plumbing.q_slots),
                primary_log_branch="principal, pointwise",
                diagnostics=runtime.diagnostics())
