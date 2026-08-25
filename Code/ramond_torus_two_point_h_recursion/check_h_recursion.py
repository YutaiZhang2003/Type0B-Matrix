#!/usr/bin/env python3
"""Ramond torus two-point h-recursion in the theta plumbing frame.

This is an independent check of the q1 -> 0 limit of the enlarged Ramond
theta block.  It uses only

* Ramond null-vector residues in the conventions of SCblock.tex;
* the simultaneous-large-weight argument of Cho--Collier--Yin; and
* the closed L_-1 global theta block to convert the CCY character seed to
  the theta plumbing frame.

No SCA PBW block coefficient is used.  The comparison data are the q1=0
restriction of the double-Virasoro q-expansion.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import platform
import time
from functools import lru_cache
from pathlib import Path

import mpmath as mp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

Exponent = tuple[int, int]
Series = dict[Exponent, complex]


def add_series(left: dict, right: dict, cutoff: int) -> dict:
    answer = dict(left)
    for exponent, coefficient in right.items():
        if sum(exponent) <= cutoff:
            answer[exponent] = answer.get(exponent, 0) + coefficient
    return answer


def multiply_series(left: dict, right: dict, cutoff: int) -> dict:
    answer: dict = {}
    for (i, j), first in left.items():
        for (k, ell), second in right.items():
            exponent = (i + k, j + ell)
            if sum(exponent) <= cutoff:
                answer[exponent] = answer.get(exponent, 0) + first * second
    return answer


def scale_series(series: dict, factor) -> dict:
    return {exponent: factor * coefficient for exponent, coefficient in series.items()}


def logarithm_series(series: dict, cutoff: int) -> dict:
    """Formal logarithm of a series with constant coefficient one."""

    shifted = dict(series)
    shifted[(0, 0)] = shifted.get((0, 0), 0) - 1
    answer: dict = {}
    unit = next(iter(series.values())) * 0 + 1
    power = {(0, 0): unit}
    for order in range(1, cutoff + 1):
        power = multiply_series(power, shifted, cutoff)
        answer = add_series(
            answer,
            scale_series(
                power, (1 if order % 2 else -1) * unit / order
            ),
            cutoff,
        )
    return answer


def exponential_series(logarithm: dict, cutoff: int) -> dict:
    """Formal exponential of a series with zero constant coefficient."""

    sample = next(iter(logarithm.values()), 1.0)
    unit = sample * 0 + 1
    answer = {(0, 0): unit}
    power = dict(answer)
    factorial = 1
    for order in range(1, cutoff + 1):
        power = multiply_series(power, logarithm, cutoff)
        factorial *= order
        answer = add_series(
            answer, scale_series(power, unit / factorial), cutoff
        )
    return answer


def rising(value, order: int):
    answer = type(value)(1)
    for index in range(order):
        answer *= value + index
    return answer


def global_theta_logarithm(H, difference, external_weight, cutoff: int) -> dict:
    """Log of the explicit q1=0 global theta block.

    The internal weights are H and H+difference.  For descendant levels
    (m,n), the three-point function is

      (d-H-m+1-H-a-n)_m (2H+a-d)_n.
    """

    series: dict = {}
    for m in range(cutoff + 1):
        norm2 = math.factorial(m) * rising(2 * H, m)
        for n in range(cutoff + 1 - m):
            rho = rising(
                external_weight - H - m + 1 - H - difference - n,
                m,
            ) * rising(2 * H + difference - external_weight, n)
            norm3 = math.factorial(n) * rising(2 * (H + difference), n)
            series[(m, n)] = rho * rho / (norm2 * norm3)
    return logarithm_series(series, cutoff)


def asymptotic_constant(values, scale, degree: int, divide_by_H: bool = False):
    """Extrapolate the constant at 1/H=0 with high-precision arithmetic."""

    matrix = mp.matrix(
        [[(scale / H) ** power for power in range(degree + 1)] for H, _ in values]
    )
    vector = mp.matrix(
        [value / H if divide_by_H else value for H, value in values]
    )
    return mp.lu_solve(matrix, vector)


def universal_theta_seed(cutoff: int) -> tuple[dict[str, Series], dict[str, float]]:
    """Extract log X, log Y, log D and the theta-frame multiplier Q.

    If d is the external weight and the internal weights are H,H+a, the
    simultaneous-large-H global block has the form

      X^H Y^a D^d (1-Q)^(-1).

    The four universal series are extracted from the closed global block,
    never from an SCA descendant calculation.
    """

    started = time.perf_counter()
    old_precision = mp.mp.dps
    mp.mp.dps = max(old_precision, 100)
    extrapolation_degree = 10
    scale = mp.mpf("1e5")
    samples = [
        scale * (1 + mp.mpf(index) / 5)
        for index in range(extrapolation_degree + 1)
    ]
    choices = ((0, 0), (1, 0), (0, 1))
    logarithms = {
        (difference, external, H): global_theta_logarithm(
            H, mp.mpf(difference), mp.mpf(external), cutoff
        )
        for difference, external in choices
        for H in samples
    }

    log_x: dict = {}
    log_y: dict = {}
    log_d: dict = {}
    log_character: dict = {}
    maximum_spurious = mp.mpf(0)
    for total in range(1, cutoff + 1):
        for first in range(total + 1):
            exponent = (first, total - first)
            base = [
                (H, logarithms[(0, 0, H)].get(exponent, mp.mpf(0)))
                for H in samples
            ]
            base_fit = asymptotic_constant(
                base, scale, extrapolation_degree, divide_by_H=True
            )
            log_x[exponent] = base_fit[0]
            log_character[exponent] = base_fit[1] * scale

            y_values = [
                (
                    H,
                    logarithms[(1, 0, H)].get(exponent, mp.mpf(0))
                    - logarithms[(0, 0, H)].get(exponent, mp.mpf(0)),
                )
                for H in samples
            ]
            d_values = [
                (
                    H,
                    logarithms[(0, 1, H)].get(exponent, mp.mpf(0))
                    - logarithms[(0, 0, H)].get(exponent, mp.mpf(0)),
                )
                for H in samples
            ]
            log_y[exponent] = asymptotic_constant(
                y_values, scale, extrapolation_degree
            )[0]
            log_d[exponent] = asymptotic_constant(
                d_values, scale, extrapolation_degree
            )[0]

            for value in (
                log_x[exponent],
                log_y[exponent],
                log_d[exponent],
                log_character[exponent],
            ):
                if abs(value) < mp.mpf("1e-45"):
                    maximum_spurious = max(maximum_spurious, abs(value))

    # The simultaneous-large-H limit of the global (L_-1-only) necklace
    # block is (1-Q)^(-1), not the full Virasoro character.  Hence
    # log_character=-log(1-Q) and Q=1-exp(-log_character).
    minus_character = scale_series(log_character, -1)
    exp_minus_character = exponential_series(minus_character, cutoff)
    multiplier = scale_series(exp_minus_character, -1)
    multiplier[(0, 0)] = multiplier.get((0, 0), 0) + 1

    x_series = exponential_series(log_x, cutoff)
    expected_multiplier = {
        (first + 1, second + 1): coefficient
        for (first, second), coefficient in x_series.items()
        if first + second + 2 <= cutoff
    }
    multiplier_error = max(
        (
            abs(multiplier.get(exponent, 0) - expected_multiplier.get(exponent, 0))
            for exponent in set(multiplier) | set(expected_multiplier)
        ),
        default=mp.mpf(0),
    )

    result = {
        "log_x": {key: complex(value) for key, value in log_x.items()},
        "log_y": {key: complex(value) for key, value in log_y.items()},
        "log_d": {key: complex(value) for key, value in log_d.items()},
        "multiplier": {key: complex(value) for key, value in multiplier.items()},
    }
    diagnostic = {
        "seconds": time.perf_counter() - started,
        "mpmath_decimal_digits": mp.mp.dps,
        "large_H_scale": float(scale),
        "extrapolation_degree": extrapolation_degree,
        "maximum_discarded_spurious_coefficient": float(maximum_spurious),
        "maximum_error_in_Q_equals_q2_q3_X": float(multiplier_error),
    }
    mp.mp.dps = old_precision
    return result, diagnostic


def beta_rs(r: int, s: int, b: float) -> float:
    return (r * b + s / b) / (2 * mp.sqrt(2))


def beta_prime_rs(r: int, s: int, b: float) -> float:
    return (-1) ** s * (r * b - s / b) / (2 * mp.sqrt(2))


def inverse_null_norm(r: int, s: int, b: float) -> complex:
    """Main-note BPZ inverse null norm at fixed c.

    The even-lattice condition is essential in the two-ground-state,
    no-G0 basis used in SCblock.tex.
    """

    answer = mp.mpc(2 ** (r * s - 2) * (r * b + s / b))
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2:
                continue
            if (p, q) == (0, 0):
                continue
            answer /= p * b + q / b
    return answer


def fusion_polynomial(
    external_momentum: float,
    other_beta: complex,
    eta: int,
    r: int,
    s: int,
    b: float,
) -> complex:
    """P_{rs}^{R,eta}(h_external,beta_other;c) of SCblock.tex."""

    lambda_external = 2 * external_momentum
    answer = mp.mpc(1)
    for k in range(r):
        for ell in range(s):
            lattice = (1 - r + 2 * k) * b + (1 - s + 2 * ell) / b
            if (k + ell) % 2 == 0:
                numerator = (
                    lambda_external
                    - 2 * mp.sqrt(2) * eta * other_beta
                    - lattice
                )
            else:
                numerator = (
                    lambda_external
                    + 2 * mp.sqrt(2) * eta * other_beta
                    - lattice
                )
            answer *= numerator / (2 * mp.sqrt(2))
    return answer


def degenerate_pairs(maximum_level: int) -> tuple[tuple[int, int, int], ...]:
    answer = []
    for r in range(1, 2 * maximum_level + 1):
        for s in range(1, 2 * maximum_level + 1):
            if (r + s) % 2 == 0:
                continue
            if r * s % 2:
                continue
            level = r * s // 2
            if level <= maximum_level:
                answer.append((r, s, level))
    return tuple(answer)


class RamondTwoPointHRecursion:
    def __init__(
        self,
        b: float,
        momenta: tuple[float, float, float],
        cutoff: int,
        seed_data: dict[str, Series],
    ):
        self.b = mp.mpf(b)
        self.momenta = tuple(mp.mpf(value) for value in momenta)
        self.cutoff = int(cutoff)
        self.q = self.b + 1 / self.b
        self.central_charge = 1.5 + 3 * self.q * self.q
        self.external_weight = self.q * self.q / 8 - self.momenta[0] ** 2 / 2
        self.seed_data = seed_data
        self.pairs = tuple(
            (
                r,
                s,
                level,
                beta_rs(r, s, self.b),
                beta_prime_rs(r, s, self.b),
                inverse_null_norm(r, s, self.b),
            )
            for r, s, level in degenerate_pairs(self.cutoff)
        )
        self.character = self._enlarged_character_coefficients(self.cutoff)
        self.calls = 0
        self.maximum_denominator_inverse = mp.mpf(0)

    @staticmethod
    def _enlarged_character_coefficients(cutoff: int) -> tuple[complex, ...]:
        """Coefficients of 4 prod_n (1+Q^n)^2/(1-Q^n)."""

        maximum = cutoff // 2
        values = [0] * (maximum + 1)
        values[0] = 4
        for mode in range(1, maximum + 1):
            following = [0] * (maximum + 1)
            for level, coefficient in enumerate(values):
                if not coefficient:
                    continue
                following[level] += coefficient
                power = 1
                while level + power * mode <= maximum:
                    following[level + power * mode] += coefficient * (3 if power == 1 else 4)
                    power += 1
            values = following
        return tuple(mp.mpc(value) for value in values)

    @staticmethod
    def _state_key(value: complex) -> tuple[str, str]:
        value = mp.mpc(value)
        digits = max(25, mp.mp.dps // 2)
        return (mp.nstr(value.real, digits), mp.nstr(value.imag, digits))

    @lru_cache(maxsize=None)
    def pole_terms_for_pair(
        self,
        beta2_key,
        beta3_key,
        eta: int,
        edge: int,
        pair_index: int,
    ):
        """The four sheet-resolved poles for one (r,s) and one edge."""

        beta2 = mp.mpc(beta2_key[0], beta2_key[1])
        beta3 = mp.mpc(beta3_key[0], beta3_key[1])
        parameter = beta2 + beta3
        difference = beta2 * beta2 - beta3 * beta3
        answer = []
        r, s, null_level, degenerate_beta, shifted_beta, null_norm = self.pairs[
            pair_index
        ]
        other_root = mp.sqrt(
            degenerate_beta**2 - difference
            if edge == 2
            else degenerate_beta**2 + difference
        )
        for degenerate_sign in (1, -1):
            for other_sign in (1, -1):
                if edge == 2:
                    pole_beta2 = degenerate_sign * degenerate_beta
                    pole_beta3 = other_sign * other_root
                    derivative = pole_beta3 / (pole_beta2 + pole_beta3)
                    other_beta = pole_beta3
                    next_beta2 = degenerate_sign * shifted_beta
                    next_beta3 = pole_beta3
                else:
                    pole_beta3 = degenerate_sign * degenerate_beta
                    pole_beta2 = other_sign * other_root
                    derivative = pole_beta2 / (pole_beta2 + pole_beta3)
                    other_beta = pole_beta2
                    next_beta2 = pole_beta2
                    next_beta3 = degenerate_sign * shifted_beta
                polynomial = fusion_polynomial(
                    self.momenta[0],
                    other_beta,
                    degenerate_sign * eta,
                    r,
                    s,
                    self.b,
                )
                residue_beta = (
                    -degenerate_sign
                    * null_norm
                    * polynomial
                    * polynomial
                    / (2 * degenerate_beta)
                )
                denominator = parameter - pole_beta2 - pole_beta3
                self.maximum_denominator_inverse = max(
                    self.maximum_denominator_inverse, 1 / abs(denominator)
                )
                answer.append(
                    (
                        residue_beta / (derivative * denominator),
                        self._state_key(next_beta2),
                        self._state_key(next_beta3),
                    )
                )
        return tuple(answer)

    def seed_coefficient(self, level2: int, level3: int) -> complex:
        if level2 != level3 or level2 >= len(self.character):
            return 0.0j
        return self.character[level2]

    def coefficient(
        self,
        beta2: complex,
        beta3: complex,
        level2: int,
        level3: int,
        eta: int = 1,
    ) -> complex:
        return self._coefficient(
            self._state_key(beta2),
            self._state_key(beta3),
            int(level2),
            int(level3),
            int(eta),
        )

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        beta2_key,
        beta3_key,
        level2: int,
        level3: int,
        eta: int,
    ) -> complex:
        self.calls += 1
        beta2 = mp.mpc(beta2_key[0], beta2_key[1])
        beta3 = mp.mpc(beta3_key[0], beta3_key[1])
        answer = self.seed_coefficient(level2, level3)
        if level2:
            for pair_index, pair in enumerate(self.pairs):
                null_level = pair[2]
                if null_level > level2:
                    continue
                for factor, next_beta2, next_beta3 in self.pole_terms_for_pair(
                    beta2_key, beta3_key, eta, 2, pair_index
                ):
                    answer += factor * self._coefficient(
                        next_beta2,
                        next_beta3,
                        level2 - null_level,
                        level3,
                        eta,
                    )
        if level3:
            for pair_index, pair in enumerate(self.pairs):
                null_level = pair[2]
                if null_level > level3:
                    continue
                for factor, next_beta2, next_beta3 in self.pole_terms_for_pair(
                    beta2_key, beta3_key, eta, 3, pair_index
                ):
                    answer += factor * self._coefficient(
                        next_beta2,
                        next_beta3,
                        level2,
                        level3 - null_level,
                        eta,
                    )
        return answer

    def series(self, beta2: complex, beta3: complex, eta: int = 1) -> Series:
        answer = {}
        for total in range(self.cutoff + 1):
            for level2 in range(total + 1):
                level3 = total - level2
                answer[(level2, level3)] = self.coefficient(
                    beta2, beta3, level2, level3, eta
                )
        return answer


def shift_variable(series: Series, edge: int, cutoff: int) -> Series:
    answer = {}
    for exponent, coefficient in series.items():
        changed = list(exponent)
        changed[edge] += 1
        changed = tuple(changed)
        if sum(changed) <= cutoff:
            answer[changed] = coefficient
    return answer


def powers_of_series(series: Series, cutoff: int) -> list[Series]:
    powers = [{(0, 0): 1.0 + 0.0j}]
    for _ in range(cutoff):
        powers.append(multiply_series(powers[-1], series, cutoff))
    return powers


def necklace_to_theta(
    necklace: Series,
    beta2: complex,
    beta3: complex,
    central_charge: float,
    external_weight: float,
    seed_data: dict[str, Series],
    cutoff: int,
) -> Series:
    """Apply the exact formal change from CCY necklace to theta plumbing."""

    x_factor = exponential_series(
        add_series(seed_data["log_x"], scale_series(seed_data["log_y"], -1), cutoff),
        cutoff,
    )
    y_factor = exponential_series(seed_data["log_y"], cutoff)
    necklace_q2 = shift_variable(x_factor, 0, cutoff)
    necklace_q3 = shift_variable(y_factor, 1, cutoff)
    q2_powers = powers_of_series(necklace_q2, cutoff)
    q3_powers = powers_of_series(necklace_q3, cutoff)

    substituted: Series = {}
    for (level2, level3), coefficient in necklace.items():
        term = multiply_series(q2_powers[level2], q3_powers[level3], cutoff)
        substituted = add_series(
            substituted, scale_series(term, coefficient), cutoff
        )

    H = central_charge / 24 - beta2 * beta2
    difference = beta2 * beta2 - beta3 * beta3
    prefactor_log: Series = {}
    for exponent in seed_data["log_x"]:
        prefactor_log[exponent] = (
            (H + 1 / 16) * seed_data["log_x"].get(exponent, 0)
            + difference * seed_data["log_y"].get(exponent, 0)
            + external_weight * seed_data["log_d"].get(exponent, 0)
        )
    prefactor = exponential_series(prefactor_log, cutoff)
    return multiply_series(prefactor, substituted, cutoff)


def double_virasoro_target(path: Path, cutoff: int) -> Series:
    payload = json.loads(path.read_text())
    answer: Series = {}
    for row in payload["coefficients"]:
        level1, twice_level2, twice_level3 = row["twice_levels"]
        if level1 != 0 or twice_level2 % 2 or twice_level3 % 2:
            continue
        exponent = (twice_level2 // 2, twice_level3 // 2)
        if sum(exponent) > cutoff:
            continue
        value = complex(
            row["coefficient"]["real"], row["coefficient"]["imag"]
        )
        # eta2=eta3=+1: all stored parity components enter with sign +.
        answer[exponent] = answer.get(exponent, 0.0j) + value
    return answer


def encode(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def run(cutoff: int, target_path: Path, output_path: Path, precision: int) -> dict:
    old_precision = mp.mp.dps
    mp.mp.dps = int(precision)
    b = mp.mpf(7) / 5
    momenta = (mp.mpf(11) / 23, mp.mpf(13) / 29, mp.mpf(17) / 31)
    total_started = time.perf_counter()

    seed_data, seed_diagnostic = universal_theta_seed(cutoff)
    recursion = RamondTwoPointHRecursion(b, momenta, cutoff, seed_data)
    beta2 = momenta[1] / mp.sqrt(2)
    beta3 = momenta[2] / mp.sqrt(2)
    recursion_started = time.perf_counter()
    necklace = recursion.series(beta2, beta3, eta=1)
    recursion_seconds = time.perf_counter() - recursion_started
    transform_started = time.perf_counter()
    recursive = necklace_to_theta(
        necklace,
        beta2,
        beta3,
        recursion.central_charge,
        recursion.external_weight,
        seed_data,
        cutoff,
    )
    transform_seconds = time.perf_counter() - transform_started
    target = double_virasoro_target(target_path, cutoff)

    comparisons = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst = (0, 0)
    for exponent in sorted(set(recursive) | set(target), key=lambda x: (sum(x), x)):
        first = recursive.get(exponent, 0.0j)
        second = target.get(exponent, 0.0j)
        absolute = abs(first - second)
        relative = absolute / max(1.0, abs(first), abs(second))
        if relative > maximum_relative:
            maximum_relative = relative
            worst = exponent
        maximum_absolute = max(maximum_absolute, absolute)
        comparisons.append(
            {
                "levels": list(exponent),
                "h_recursion": encode(first),
                "double_virasoro": encode(second),
                "absolute_difference": float(absolute),
                "relative_difference": float(relative),
            }
        )

    payload = {
        "description": "q1=0 enlarged Ramond theta block: Ramond h-recursion versus double Virasoro",
        "cutoff": cutoff,
        "parameters": {
            "b": float(b),
            "momenta": [float(value) for value in momenta],
            "central_charge": float(recursion.central_charge),
            "external_weight": float(recursion.external_weight),
            "beta2": float(beta2),
            "beta3": float(beta3),
            "mpmath_decimal_digits": int(precision),
        },
        "method": {
            "pbw_used": False,
            "ramond_momentum_uniformizer": "t=beta2+beta3 at fixed beta2^2-beta3^2",
            "negative_pole_rule": "eta -> -eta and beta_prime -> -beta_prime",
            "null_norm_lattice": "p+q even",
            "necklace_regular_seed": "4 prod_n (1+Q^n)^2/(1-Q^n)",
            "theta_frame_map": "q2_tilde=q2 X/Y, q3_tilde=q3 Y, prefactor=X^(H+1/16)Y^aD^h1",
        },
        "seed_diagnostic": seed_diagnostic,
        "recursion_diagnostic": {
            "coefficient_calls": recursion.calls,
            "coefficient_cache": recursion._coefficient.cache_info()._asdict(),
            "maximum_inverse_pole_separation": float(recursion.maximum_denominator_inverse),
        },
        "comparison": {
            "coefficient_count": len(comparisons),
            "maximum_absolute_difference": float(maximum_absolute),
            "maximum_relative_difference": float(maximum_relative),
            "worst_levels": list(worst),
            "passed_at_2e-6": maximum_relative < 2e-6,
        },
        "timing_seconds": {
            "seed_extraction": seed_diagnostic["seconds"],
            "h_recursion": recursion_seconds,
            "necklace_to_theta": transform_seconds,
            "total": time.perf_counter() - total_started,
        },
        "coefficients": comparisons,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    mp.mp.dps = old_precision
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "python" / "full_ramond_block_runtime" / "level10_q_expansion.json",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--precision", type=int, default=50)
    arguments = parser.parse_args()
    result = run(
        arguments.cutoff,
        arguments.target,
        arguments.output,
        arguments.precision,
    )
    print(json.dumps(result["comparison"], indent=2))
    print(json.dumps(result["timing_seconds"], indent=2))


if __name__ == "__main__":
    main()
