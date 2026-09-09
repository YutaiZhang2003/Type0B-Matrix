"""Direct level-five test of the fixed-beta Ramond large-c block.

This script does not use the c-recursion or its regular seed.  It constructs
the even Ramond PBW basis, Gram matrices, and N-R-R Ward vectors directly,
then compares the stripped local coefficients

    [z^N] (1-z)^(c/12) B_beta(c; z)

with [z^N] (1-z)^A through N=5.
"""

from __future__ import annotations

from dataclasses import dataclass

from ramond_descendant_blocks import (
    RamondThreePointWardVector,
    RamondVermaModule,
)
from ramond_fixed_beta_c_recursion import (
    normalize_hjs_series,
    ramond_large_c_seed_series,
)
from superconformal_blocks import (
    NSSphereFourPointBlock,
    _series_compose,
    _series_mul,
    _series_pow,
)


@dataclass(frozen=True)
class FixedData:
    beta: float
    beta2: float
    beta3: float
    h1: float
    h4: float

    @property
    def exponent(self) -> float:
        return self.h1 + self.h4 + self.beta2**2 + self.beta3**2


DATA = FixedData(
    beta=0.70,
    beta2=0.41,
    beta3=0.23,
    h1=0.31,
    h4=0.37,
)
CENTRAL_CHARGES = (120.0, 240.0, 480.0, 960.0)
MAX_LEVEL = 5


def generalized_binomial(value: complex, order: int) -> complex:
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= (value - offset) / (offset + 1)
    return result


def target_coefficients(data: FixedData) -> tuple[complex, ...]:
    return tuple(
        (-1) ** level * generalized_binomial(data.exponent, level)
        for level in range(MAX_LEVEL + 1)
    )


def direct_local_coefficients(
    c: float,
    data: FixedData,
    sign3: int,
    sign2: int,
) -> tuple[complex, ...]:
    module = RamondVermaModule(
        c=c,
        weight=c / 24.0 - data.beta**2,
    )
    left = RamondThreePointWardVector(
        module=module,
        external_beta=data.beta3,
        external_ramond_weight=c / 24.0 - data.beta3**2,
        external_ns_weight=data.h4,
        sign=sign3,
    )
    right = RamondThreePointWardVector(
        module=module,
        external_beta=data.beta2,
        external_ramond_weight=c / 24.0 - data.beta2**2,
        external_ns_weight=data.h1,
        sign=sign2,
    )

    coefficients = []
    for level in range(MAX_LEVEL + 1):
        gram = module.gram_matrix(level, 0)
        solved_right = module._solve(gram, right.vector(level, 0))
        coefficients.append(
            sum(
                left_value * right_value
                for left_value, right_value in zip(
                    left.vector(level, 0), solved_right
                )
            )
        )
    return tuple(coefficients)


def strip_universal_power(
    coefficients: tuple[complex, ...], c: float
) -> tuple[complex, ...]:
    stripping = tuple(
        (-1) ** level * generalized_binomial(c / 12.0, level)
        for level in range(MAX_LEVEL + 1)
    )
    return tuple(
        sum(
            stripping[offset] * coefficients[level - offset]
            for offset in range(level + 1)
        )
        for level in range(MAX_LEVEL + 1)
    )


def direct_normalized_elliptic_coefficients(
    local: tuple[complex, ...], c: float, data: FixedData
) -> tuple[complex, ...]:
    """Convert direct local sewing to the oscillator-normalized HJS series."""

    theta3_full, _, z_full = NSSphereFourPointBlock._elliptic_series_data(
        MAX_LEVEL + 1
    )
    theta3 = theta3_full[: MAX_LEVEL + 1]
    z_series = z_full[: MAX_LEVEL + 1]
    z_over_16q = [
        z_full[power + 1] / 16.0 for power in range(MAX_LEVEL + 1)
    ]
    one_minus_z = [-value for value in z_series]
    one_minus_z[0] += 1.0

    internal_weight = c / 24.0 - data.beta**2
    h2 = c / 24.0 - data.beta2**2
    h3 = c / 24.0 - data.beta3**2
    vacuum_shift = (c - 1.5) / 24.0
    q_exponent = internal_weight - vacuum_shift - 1.0 / 16.0
    one_minus_exponent = vacuum_shift - h2 - h3
    theta_exponent = (
        (c - 1.5) / 2.0
        - 4.0 * (data.h1 + h2 + h3 + data.h4)
        + 0.5
    )

    reduced_prefactor = _series_mul(
        _series_mul(
            _series_pow(z_over_16q, -q_exponent, MAX_LEVEL),
            _series_pow(one_minus_z, one_minus_exponent, MAX_LEVEL),
            MAX_LEVEL,
        ),
        _series_pow(theta3, theta_exponent, MAX_LEVEL),
        MAX_LEVEL,
    )
    local_in_q = _series_compose(local, z_series, MAX_LEVEL)
    raw_hjs = _series_mul(
        local_in_q,
        _series_pow(reduced_prefactor, -1.0, MAX_LEVEL),
        MAX_LEVEL,
    )
    return normalize_hjs_series(raw_hjs, c)


def stable_normalized_elliptic_coefficients(
    stripped_local: tuple[complex, ...], data: FixedData
) -> tuple[complex, ...]:
    """Use the exact HJS prefactor identity after stripping in z-space."""

    theta3_full, _, z_full = NSSphereFourPointBlock._elliptic_series_data(
        MAX_LEVEL + 1
    )
    theta3 = theta3_full[: MAX_LEVEL + 1]
    z_series = z_full[: MAX_LEVEL + 1]
    z_over_16q = [
        z_full[power + 1] / 16.0 for power in range(MAX_LEVEL + 1)
    ]
    one_minus_z = [-value for value in z_series]
    one_minus_z[0] += 1.0
    theta_exponent = (
        4.0
        * (data.h1 + data.h4 - data.beta2**2 - data.beta3**2)
        + 0.25
    )
    conversion = _series_mul(
        _series_mul(
            _series_pow(z_over_16q, -data.beta**2, MAX_LEVEL),
            _series_pow(
                one_minus_z,
                -data.beta2**2 - data.beta3**2 + 1.0 / 16.0,
                MAX_LEVEL,
            ),
            MAX_LEVEL,
        ),
        _series_pow(theta3, theta_exponent, MAX_LEVEL),
        MAX_LEVEL,
    )
    return tuple(
        _series_mul(
            conversion,
            _series_compose(stripped_local, z_series, MAX_LEVEL),
            MAX_LEVEL,
        )
    )


def comparison_ledger() -> tuple[
    dict[float, dict[tuple[int, int], tuple[complex, ...]]],
    dict[float, dict[tuple[int, int], tuple[complex, ...]]],
]:
    local_ledger = {}
    elliptic_ledger = {}
    for c in CENTRAL_CHARGES:
        local_ledger[c] = {}
        elliptic_ledger[c] = {}
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                signs = (sign3, sign2)
                local = direct_local_coefficients(c, DATA, sign3, sign2)
                stripped_local = strip_universal_power(local, c)
                local_ledger[c][signs] = stripped_local
                elliptic_ledger[c][signs] = (
                    stable_normalized_elliptic_coefficients(
                        stripped_local, DATA
                    )
                )
    return local_ledger, elliptic_ledger


def extrapolate_to_infinite_c(
    values: tuple[tuple[complex, ...], ...]
) -> tuple[complex, ...]:
    """Interpolate cubically in 1/c and evaluate at 1/c=0."""

    x_values = tuple(1.0 / c for c in CENTRAL_CHARGES)
    weights = []
    for index, x_value in enumerate(x_values):
        weight = 1.0
        for other_index, other_x in enumerate(x_values):
            if other_index == index:
                continue
            weight *= -other_x / (x_value - other_x)
        weights.append(weight)
    return tuple(
        sum(
            weights[index] * values[index][power]
            for index in range(len(values))
        )
        for power in range(MAX_LEVEL + 1)
    )


def main() -> None:
    target = target_coefficients(DATA)
    ledger, elliptic_ledger = comparison_ledger()
    print("Direct long-R descendant sewing through level 5")
    print(
        "fixed data: "
        f"beta={DATA.beta}, beta2={DATA.beta2}, beta3={DATA.beta3}, "
        f"h1={DATA.h1}, h4={DATA.h4}, A={DATA.exponent}"
    )
    print("PBW dimensions: 1, 2, 4, 8, 14, 24")
    print()
    print("Target [z^N](1-z)^A")
    for level, coefficient in enumerate(target):
        print(f"N={level}: {coefficient.real:+.15e}")

    print()
    print("Maximum error over the four (eta3,eta2) components")
    print("       c", *(f"N={level}" for level in range(MAX_LEVEL + 1)))
    for c in CENTRAL_CHARGES:
        errors = [
            max(
                abs(coefficients[level] - target[level])
                for coefficients in ledger[c].values()
            )
            for level in range(MAX_LEVEL + 1)
        ]
        print(
            f"{c:8.1f}",
            *(f"{error:.6e}" for error in errors),
        )

    elliptic_target = ramond_large_c_seed_series(
        max_power=MAX_LEVEL,
        internal_beta=DATA.beta,
        beta2_r=DATA.beta2,
        beta3_r=DATA.beta3,
        h1_ns=DATA.h1,
        h4_ns=DATA.h4,
    )
    print()
    print("Target theta-function seed S_beta(q)")
    for power, coefficient in enumerate(elliptic_target):
        print(f"q^{power}: {coefficient.real:+.15e}")

    print()
    print("Maximum elliptic-seed error over the four components")
    print("       c", *(f"q^{power}" for power in range(MAX_LEVEL + 1)))
    for c in CENTRAL_CHARGES:
        errors = [
            max(
                abs(coefficients[power] - elliptic_target[power])
                for coefficients in elliptic_ledger[c].values()
            )
            for power in range(MAX_LEVEL + 1)
        ]
        print(
            f"{c:8.1f}",
            *(f"{error:.6e}" for error in errors),
        )

    lower_c = CENTRAL_CHARGES[-2]
    upper_c = CENTRAL_CHARGES[-1]
    print()
    print(
        "Richardson estimate 2*C_N(2c)-C_N(c), "
        f"using c={lower_c:g},{upper_c:g}"
    )
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        extrapolated = tuple(
            2.0 * ledger[upper_c][signs][level]
            - ledger[lower_c][signs][level]
            for level in range(MAX_LEVEL + 1)
        )
        errors = tuple(
            abs(extrapolated[level] - target[level])
            for level in range(MAX_LEVEL + 1)
        )
        print(
            f"signs={signs}: ",
            ", ".join(
                f"N={level} err={errors[level]:.3e}"
                for level in range(MAX_LEVEL + 1)
            ),
        )

    print()
    print("Cubic 1/c extrapolation using all four central charges")
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        local_extrapolated = extrapolate_to_infinite_c(
            tuple(ledger[c][signs] for c in CENTRAL_CHARGES)
        )
        elliptic_extrapolated = extrapolate_to_infinite_c(
            tuple(elliptic_ledger[c][signs] for c in CENTRAL_CHARGES)
        )
        local_errors = tuple(
            abs(local_extrapolated[level] - target[level])
            for level in range(MAX_LEVEL + 1)
        )
        elliptic_errors = tuple(
            abs(elliptic_extrapolated[power] - elliptic_target[power])
            for power in range(MAX_LEVEL + 1)
        )
        print(
            f"signs={signs}: "
            f"max local err={max(local_errors):.3e}, "
            f"max elliptic err={max(elliptic_errors):.3e}, "
            f"q^5 err={elliptic_errors[5]:.3e}"
        )

    print()
    print(
        "Elliptic Richardson estimate 2*Hhat_n(2c)-Hhat_n(c), "
        f"using c={lower_c:g},{upper_c:g}"
    )
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        extrapolated = tuple(
            2.0 * elliptic_ledger[upper_c][signs][power]
            - elliptic_ledger[lower_c][signs][power]
            for power in range(MAX_LEVEL + 1)
        )
        errors = tuple(
            abs(extrapolated[power] - elliptic_target[power])
            for power in range(MAX_LEVEL + 1)
        )
        print(
            f"signs={signs}: ",
            ", ".join(
                f"q^{power} err={errors[power]:.3e}"
                for power in range(MAX_LEVEL + 1)
            ),
        )


if __name__ == "__main__":
    main()
