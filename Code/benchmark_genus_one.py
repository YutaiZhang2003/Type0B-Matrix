"""Evaluate the fluxless genus-one target from the modular spin sum."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


BASELINE_PATH = Path(__file__).with_name("baseline.json")


def _require_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def dimensionless_radius(radius: float, alpha_prime: float) -> float:
    """Return x = R / sqrt(2 alpha')."""
    radius = _require_positive_finite("radius", radius)
    alpha_prime = _require_positive_finite("alpha_prime", alpha_prime)
    return radius / math.sqrt(2.0 * alpha_prime)


def bry_circle_modular_integral(radius: float) -> float:
    """Convergent orbit integral of the compact-circle lattice.

    The input is rho=R_phys/sqrt(alpha'/2). The result follows by separating
    the zero orbit and unfolding the nonzero orbits to the strip.
    """
    radius = _require_positive_finite("radius", radius)
    return math.pi / 3.0 * (1.0 + 2.0 / radius**2)


def bry_momentum_density(radius: float) -> float:
    """Two BRY momentum towers per regulated Liouville coordinate length.

    The input is rho=R_phys/sqrt(alpha'/2), not a dimensionful radius.
    The tachyon and axion each contribute 1/(12 rho).
    """
    radius = _require_positive_finite("radius", radius)
    return 1.0 / (6.0 * radius)


def bry_winding_density(radius: float) -> float:
    """The single 0B winding tower as a function of dimensionless rho."""
    radius = _require_positive_finite("radius", radius)
    return radius / 24.0


def bry_genus_one_density(radius: float) -> float:
    """0B density from the even modular integral plus odd supermodulus."""
    return bry_even_spin_density(radius) + bry_odd_spin_density(radius)


def bry_even_spin_density(radius: float) -> float:
    """Three even structures evaluated by the modular orbit integral."""
    radius = _require_positive_finite("radius", radius)
    return 3.0 * radius / (16.0 * math.pi) * bry_circle_modular_integral(
        radius,
    )


def bry_odd_spin_density(radius: float) -> float:
    """Odd-spin 0B contribution as a function of dimensionless rho."""
    radius = _require_positive_finite("radius", radius)
    return -radius / 48.0 + 1.0 / (24.0 * radius)


def bry_0a_momentum_density(radius: float) -> float:
    """The single 0A momentum tower as a function of dimensionless rho_A."""
    radius = _require_positive_finite("radius", radius)
    return 1.0 / (12.0 * radius)


def bry_0a_winding_density(radius: float) -> float:
    """The two 0A winding towers as a function of dimensionless rho_A."""
    radius = _require_positive_finite("radius", radius)
    return radius / 12.0


def bry_0a_genus_one_density(radius: float) -> float:
    """0A density from the even modular integral plus opposite odd sign."""
    return bry_even_spin_density(radius) + bry_0a_odd_spin_density(radius)


def bry_0a_odd_spin_density(radius: float) -> float:
    """Odd-spin 0A term, with the Arf sign opposite to type 0B."""
    radius = _require_positive_finite("radius", radius)
    return radius / 48.0 - 1.0 / (24.0 * radius)


def bry_0a_genus_one_free_energy(
    radius: float,
    liouville_mu_over_cutoff: float,
    *,
    b: float = 1.0,
) -> float:
    """Universal connected fluxless 0A genus-one free energy."""
    return bry_0a_genus_one_density(radius) * bry_liouville_volume_log(
        liouville_mu_over_cutoff,
        b=b,
    )


def bry_0a_radius_from_0b(radius: float) -> float:
    """Return rho_A=2/rho_B for physical T-duality R_A=alpha'/R_B."""
    radius = _require_positive_finite("radius", radius)
    return 2.0 / radius


def bry_liouville_volume_log(
    liouville_mu_over_cutoff: float,
    *,
    b: float = 1.0,
) -> float:
    """Universal coordinate-volume term from the BRY Liouville wall.

    For the interaction mu_L exp(b phi), shifting the wall gives
    V_phi|universal = -log(mu_L/Lambda_L)/b. Additive regulator constants
    are deliberately omitted.
    """
    ratio = _require_positive_finite(
        "liouville_mu_over_cutoff",
        liouville_mu_over_cutoff,
    )
    b = _require_positive_finite("b", b)
    return -math.log(ratio) / b


def translated_liouville_volume_log(
    parameter_over_cutoff: float,
    *,
    field_scale: float,
    liouville_power: float,
    b: float = 1.0,
) -> float:
    """Translate a source convention Phi=a phi, mu_L=kappa lambda^p.

    The constant kappa affects only the omitted additive volume constant.
    """
    ratio = _require_positive_finite(
        "parameter_over_cutoff",
        parameter_over_cutoff,
    )
    field_scale = _require_positive_finite("field_scale", field_scale)
    liouville_power = _require_positive_finite(
        "liouville_power",
        liouville_power,
    )
    b = _require_positive_finite("b", b)
    return -(field_scale * liouville_power / b) * math.log(ratio)


def bry_genus_one_free_energy(
    radius: float,
    liouville_mu_over_cutoff: float,
    *,
    b: float = 1.0,
) -> float:
    """Universal connected genus-one free energy in BRY variables."""
    return bry_genus_one_density(radius) * bry_liouville_volume_log(
        liouville_mu_over_cutoff,
        b=b,
    )


def dual_radius(radius: float, alpha_prime: float) -> float:
    """Return the radius 2 alpha' / R preserving the genus-one shape."""
    radius = _require_positive_finite("radius", radius)
    alpha_prime = _require_positive_finite("alpha_prime", alpha_prime)
    return 2.0 * alpha_prime / radius


def torus_log_coefficient(radius: float, alpha_prime: float) -> float:
    """Coefficient C1 in Z1|log = C1 log(|mu|/Lambda)."""
    x = dimensionless_radius(radius, alpha_prime)
    return -(x + 1.0 / x) / 12.0


def torus_log_term(
    radius: float,
    alpha_prime: float,
    liouville_mu_over_cutoff: float,
) -> float:
    """Universal logarithmic part of the genus-one amplitude."""
    ratio = _require_positive_finite(
        "liouville_mu_over_cutoff",
        liouville_mu_over_cutoff,
    )
    return torus_log_coefficient(radius, alpha_prime) * math.log(ratio)


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _table(config: dict[str, object]) -> str:
    alpha_prime = float(config["alpha_prime"])
    liouville_mu_over_cutoff = float(config["liouville_mu_over_cutoff"])
    radii = [float(radius) for radius in config["radii_bry"]]  # type: ignore[index]
    rows = [
        "rho=R_phys/sqrt(alpha'/2),x=rho/2,bry_density,momentum,winding,"
        "even_spin,odd_spin,F1_log_mu_L",
    ]
    for radius in radii:
        x = dimensionless_radius(radius, alpha_prime)
        density = bry_genus_one_density(radius)
        momentum = bry_momentum_density(radius)
        winding = bry_winding_density(radius)
        even_spin = bry_even_spin_density(radius)
        odd_spin = bry_odd_spin_density(radius)
        term = bry_genus_one_free_energy(radius, liouville_mu_over_cutoff)
        rows.append(
            f"{radius:.12g},{x:.12g},{density:.12g},"
            f"{momentum:.12g},{winding:.12g},"
            f"{even_spin:.12g},{odd_spin:.12g},{term:.12g}"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BASELINE_PATH,
        help="JSON configuration (default: baseline.json beside this module)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_baseline(args.config)
    print(_table(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
