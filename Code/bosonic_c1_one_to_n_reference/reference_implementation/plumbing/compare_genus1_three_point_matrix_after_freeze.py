#!/usr/bin/env python3
"""Compare a frozen torus three-point worldsheet result with the matrix model.

This program is deliberately separate from the worldsheet integrand and RQMC
driver. It refuses inputs without ``blind_freeze: true`` and records the
SHA-256 digest of every blind artifact used in the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MATRIX_SOURCE_URL = "https://arxiv.org/pdf/2604.06301"


def _complex(record: dict[str, float]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def _record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_blind_artifact(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("blind_freeze") is not True:
        raise ValueError(f"{path} is not marked blind_freeze: true")
    if data.get("calculation") != "direct c=1 genus-one three-point worldsheet integral":
        raise ValueError(f"{path} is not a torus three-point worldsheet artifact")
    return data


def matrix_stripped_genus1_three_point(
    omega_out_1: complex,
    omega_out_2: complex,
) -> complex:
    r"""Return the continued stripped matrix polynomial ``S-hat_{1,3}``.

    This is Eq. (2.55) of Collier--Eberhardt--Rodriguez, first written in the
    2->1 chamber and then continued to the 1->2 chamber by simultaneous energy
    reflection. The arguments here are the two positive-energy magnitudes on
    the two-particle side, analytically continued as complex variables.
    """
    omega_out_1 = complex(omega_out_1)
    omega_out_2 = complex(omega_out_2)
    total = omega_out_1 + omega_out_2
    return (
        -(total - 1.0j)
        * (total - 2.0j)
        * (
            omega_out_1**2
            - 1.0j * omega_out_1
            + omega_out_2**2
            - 1.0j * omega_out_2
            + 1.0
        )
        / 48.0
    )


def matrix_f1_bry_normalization(
    omega_in: complex,
    omega_out_1: complex,
    omega_out_2: complex,
) -> complex:
    r"""Return ``F_1^MM`` in the BRY/``mu`` expansion.

    We use ``g_mu=mu^-1=2*pi*g_s^BRY`` and
    ``A=i*g_mu*F0+i*g_mu^3*F1+...``.  The stable three-point normalization is
    fixed at genus zero, where the stripped CER answer is one and
    ``F0=omega*omega1*omega2``.  CER instead expand their amplitudes in
    ``g_s^CER=sqrt(2)/mu=sqrt(2)*g_mu`` (their Eq. (A.2)).  Hence the
    genus-one/tree ratio carries the additional factor
    ``(g_s^CER/g_mu)^2=2``, and ``F1=2*F0*S-hat_{1,3}``.
    """
    f0 = complex(omega_in) * complex(omega_out_1) * complex(omega_out_2)
    return 2.0 * f0 * matrix_stripped_genus1_three_point(
        omega_out_1,
        omega_out_2,
    )


def compare(
    blind_path: Path,
    *,
    momentum_comparison_path: Path | None = None,
) -> dict[str, object]:
    blind = load_blind_artifact(blind_path)
    kinematics = blind["kinematics"]
    tail = blind["tail_completion"]
    omega_in = _complex(kinematics["omega_in"])
    omega_out_1 = _complex(kinematics["omega_out_1"])
    omega_out_2 = _complex(kinematics["omega_out_2"])
    worldsheet_i = _complex(tail["final_I"])
    worldsheet_i_se = _complex(tail["final_rqmc_standard_error"])

    f0 = omega_in * omega_out_1 * omega_out_2
    stripped = matrix_stripped_genus1_three_point(omega_out_1, omega_out_2)
    matrix_f1 = matrix_f1_bry_normalization(
        omega_in,
        omega_out_1,
        omega_out_2,
    )
    matrix_i = 4.0 * math.pi * matrix_f1
    worldsheet_f1 = worldsheet_i / (4.0 * math.pi)
    worldsheet_f1_se = worldsheet_i_se / (4.0 * math.pi)
    residual_f1 = worldsheet_f1 - matrix_f1
    residual_i = worldsheet_i - matrix_i

    momentum_shift = None
    momentum_comparison = None
    if momentum_comparison_path is not None:
        comparison_blind = load_blind_artifact(momentum_comparison_path)
        comparison_i = _complex(comparison_blind["tail_completion"]["final_I"])
        momentum_shift = worldsheet_i - comparison_i
        momentum_comparison = {
            "path": str(momentum_comparison_path),
            "sha256": _sha256(momentum_comparison_path),
            "order_per_edge": int(
                comparison_blind["momentum_rule"]["order_per_edge"]
            ),
            "final_I": _record(comparison_i),
            "shift_primary_minus_comparison": _record(momentum_shift),
        }

    bulk_block_shift = _complex(blind["cusp_fit"]["last_retained_order_shift"])
    rqmc_scale = abs(worldsheet_i_se.imag)
    numerical_scales = [rqmc_scale, abs(bulk_block_shift.imag)]
    if momentum_shift is not None:
        numerical_scales.append(abs(momentum_shift.imag))
    smoke_combined_scale = math.sqrt(sum(value * value for value in numerical_scales))
    imaginary_pull_rqmc = (
        residual_i.imag / rqmc_scale if rqmc_scale > 0.0 else float("nan")
    )
    imaginary_pull_smoke = (
        residual_i.imag / smoke_combined_scale
        if smoke_combined_scale > 0.0
        else float("nan")
    )

    result: dict[str, object] = {
        "calculation": "post-freeze matrix comparison for c=1 genus-one three-point amplitude",
        "worldsheet_was_frozen_before_comparison": True,
        "blind_input": {
            "path": str(blind_path),
            "sha256": _sha256(blind_path),
            "momentum_order_per_edge": int(blind["momentum_rule"]["order_per_edge"]),
        },
        "kinematics": kinematics,
        "normalization": {
            "amplitude_expansion": (
                "A_1->2=i*g_mu*F0+i*g_mu^3*F1+O(g_mu^5)"
            ),
            "coupling_convention": "g_s means the BRY string coupling",
            "g_mu": "mu^-1=2*pi*g_s",
            "worldsheet_prefactor": "A1_ws=2*pi^2*i*g_s^3*I_1,3",
            "F0": "omega_in*omega_out_1*omega_out_2",
            "worldsheet_conversion": "F1_ws=I_1,3/(4*pi)",
            "cer_coupling": "g_s_CER=sqrt(2)/mu=sqrt(2)*g_mu",
            "matrix_conversion": "F1_MM=2*F0*S_hat_1,3",
            "factor_two_origin": (
                "(g_s_CER/g_mu)^2=2 between the genus-one and tree terms"
            ),
            "normalization_anchor": (
                "stable three-tachyon sphere amplitude; no fit to the genus-one data"
            ),
        },
        "matrix_model": {
            "source": "Collier, Eberhardt and Rodriguez, c=1 strings as a matrix integral",
            "source_url": MATRIX_SOURCE_URL,
            "source_equation": "Eq. (2.55)",
            "analytic_chamber": (
                "2->1 polynomial, continued to 1->2 by simultaneous energy reflection"
            ),
            "stripped_formula": (
                "-(w1+w2-i)*(w1+w2-2i)*"
                "(w1^2-i*w1+w2^2-i*w2+1)/48"
            ),
            "S_hat_1,3": _record(stripped),
            "F0": _record(f0),
            "F1_MM": _record(matrix_f1),
            "equivalent_I_1,3": _record(matrix_i),
        },
        "worldsheet": {
            "I_1,3": _record(worldsheet_i),
            "I_1,3_rqmc_standard_error": _record(worldsheet_i_se),
            "F1_ws": _record(worldsheet_f1),
            "F1_ws_rqmc_standard_error": _record(worldsheet_f1_se),
            "cusp_tail_fraction": float(
                abs(_complex(tail["mean_integrated_tail"])) / abs(worldsheet_i)
            ),
            "cusp_fit_relative_residual": float(tail["mean_relative_fit_residual"]),
        },
        "comparison": {
            "worldsheet_over_matrix_I": _record(worldsheet_i / matrix_i),
            "I_residual": _record(residual_i),
            "F1_residual": _record(residual_f1),
            "rho_candidate_at_this_single_point": _record(residual_f1 / f0),
            "imaginary_pull_rqmc_only": float(imaginary_pull_rqmc),
            "imaginary_pull_smoke_numerical_scale": float(imaginary_pull_smoke),
            "interpretation": (
                "single local-smoke point; the cusp model is not assigned a calibrated "
                "systematic error, so this is not a discrepancy claim"
            ),
        },
        "numerical_stability": {
            "bulk_last_retained_block_order_shift": _record(bulk_block_shift),
            "momentum_comparison": momentum_comparison,
            "smoke_combined_scale": float(smoke_combined_scale),
            "smoke_combined_scale_definition": (
                "quadrature sum of RQMC SE, p16-p12 shift, and bulk block-order shift; "
                "does not include a calibrated cusp-model systematic"
            ),
        },
    }
    return result


def parser() -> argparse.ArgumentParser:
    result_dir = Path(
        "plumbing/results/genus1_three_point_worldsheet/"
        "equal_split_t060_local_smoke_n256_v1"
    )
    out = argparse.ArgumentParser()
    out.add_argument(
        "--blind-input",
        type=Path,
        default=result_dir / "worldsheet_blind_exact_p16_tail.json",
    )
    out.add_argument(
        "--momentum-comparison",
        type=Path,
        default=result_dir / "worldsheet_blind_exact_p12_tail.json",
    )
    out.add_argument(
        "--output",
        type=Path,
        default=result_dir / "matrix_comparison_after_freeze.json",
    )
    return out


def main() -> None:
    args = parser().parse_args()
    result = compare(
        args.blind_input,
        momentum_comparison_path=args.momentum_comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        "I_ws={:+.12e}, I_MM={:+.12e}, ratio={:+.6f}".format(
            _complex(result["worldsheet"]["I_1,3"]).imag,
            _complex(result["matrix_model"]["equivalent_I_1,3"]).imag,
            _complex(result["comparison"]["worldsheet_over_matrix_I"]).real,
        )
    )


if __name__ == "__main__":
    main()
