#!/usr/bin/env python3
"""Fixed-difference NS h-recursion for sphere linear-channel blocks.

This is the production numerical implementation of the sphere recursion in
the machine h-recursion note. It is independent of the central-charge
recursion and supports any number of bottom-component NS external primaries
in the standard linear channel.

The recursion uses one correlated base weight H=h_1. All other internal
weights and the two endpoint external weights are represented by fixed
differences from H. A residue on any edge moves within this correlated
family; the root call nevertheless has exactly the physical weights supplied
by the caller.

At the self-dual point b=1 individual Kac residues are confluent even though
the complete block is finite. Public coefficient and functional evaluation
therefore take an even-in-log(b) extrapolated limit of the complete generic-b
answer. Away from the self-dual point no detuning or extrapolation is used.
"""

from __future__ import annotations

import fcntl
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import mpmath

from ns_global_osp_block import osp_sector_vertex
from ns_recursion_recipe import (
    ns_fusion_polynomial_mp,
    ns_inverse_null_slope_mp,
)


def _validate_twice_levels(
    values: Sequence[int], *, expected_length: int, name: str
) -> tuple[int, ...]:
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} entries")
    result = tuple(values)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in result
    ):
        raise ValueError(f"{name} must contain non-negative integers")
    return result


def _validate_sectors(
    values: Sequence[int], *, expected_length: int, required_parity: int
) -> tuple[int, ...]:
    if len(values) != expected_length or any(value not in (0, 1) for value in values):
        raise ValueError(
            f"vertex_sectors must contain exactly {expected_length} zero/one entries"
        )
    result = tuple(int(value) for value in values)
    if sum(result) % 2 != required_parity:
        raise ValueError(
            "sphere vertex-sector parity does not match the external components"
        )
    return result


def _iter_ns_h_poles(maximum_product: int) -> Iterable[tuple[int, int]]:
    for r in range(1, int(maximum_product) + 1):
        for s in range(1, int(maximum_product) // r + 1):
            if (r + s) % 2 == 0:
                yield r, s


def ns_central_charge_from_b(b):
    """Return the ordinary NS central charge for Q=b+1/b."""

    b = mpmath.mpc(b)
    return mpmath.mpf("1.5") + 3 * (b + 1 / b) ** 2


def ns_b_from_c(central_charge):
    """Return the principal b >= 1 sheet determined by the central charge."""

    c_value = mpmath.mpc(central_charge)
    background = mpmath.sqrt((c_value - mpmath.mpf("1.5")) / 3)
    return (background + mpmath.sqrt(background * background - 4)) / 2


def ns_degenerate_weight_mp(b, r: int, s: int):
    """Return the NS Kac weight h_(r,s)(b)."""

    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS h-poles require r,s >= 1 and r+s even")
    b = mpmath.mpc(b)
    return (
        -mpmath.mpf(r * s - 1) / 4
        + mpmath.mpf(1 - r * r) * b * b / 8
        + mpmath.mpf(1 - s * s) / (8 * b * b)
    )


def _lagrange_value_at_zero(nodes, values):
    """Polynomial extrapolation through (nodes, values) evaluated at zero."""

    result = mpmath.mpc(0)
    for index, (node, value) in enumerate(zip(nodes, values)):
        weight = mpmath.mpf(1)
        for other_index, other in enumerate(nodes):
            if other_index == index:
                continue
            weight *= -other / (node - other)
        result += weight * value
    return result


def _polynomial_fit_value_at_zero(nodes, values, degree: int):
    """Least-squares polynomial fit evaluated at the origin.

    The self-dual regulator uses more eta samples than fit coefficients, just
    as in the attached c=1 implementation.  Keeping this solve in mpmath
    avoids throwing away the extra precision used to resolve confluent Kac
    residues before the compact coefficient table is converted to ordinary
    complex numbers by the moduli integrator.
    """

    selected_degree = int(degree)
    if selected_degree < 0 or selected_degree >= len(nodes):
        raise ValueError("polynomial degree must lie in [0,len(nodes)-1]")
    design = mpmath.matrix(
        [[node**power for power in range(selected_degree + 1)] for node in nodes]
    )
    target = mpmath.matrix(values)
    coefficients, _residual = mpmath.qr_solve(design, target)
    return coefficients[0]


class NSSphereLinearHRecursion:
    """NS sphere N-point block in the standard linear channel.

    External weights are ordered from zero to infinity. For five points the
    vertices are rho(h1,d2,d1), rho(h2,d3,h1), rho(d5,d4,h2).
    Levels are represented by non-negative integers equal to twice the
    physical level.
    """

    _SELF_DUAL_LOG_B_NODES = ("0.12", "0.09", "0.065", "0.045", "0.03")

    def __init__(
        self,
        *,
        external_weights: Sequence[object],
        internal_weights: Sequence[object],
        external_descendants: Sequence[int] | None = None,
        vertex_sectors: Sequence[int] | None = None,
        central_charge=None,
        b=None,
        working_precision: int = 60,
        pole_tolerance: float = 1.0e-30,
        self_dual_strategy: str = "extrapolate",
        self_dual_log_b_nodes: Sequence[object] | None = None,
        self_dual_polynomial_degree: int | None = None,
        self_dual_comparison_degree: int | None = None,
        coefficient_cache_directory: str | os.PathLike[str] | None = None,
    ) -> None:
        if len(external_weights) < 4:
            raise ValueError("a sphere linear block requires at least four points")
        if not isinstance(working_precision, int) or working_precision < 30:
            raise ValueError("working_precision must be an integer at least 30")
        if not math.isfinite(pole_tolerance) or pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")
        if self_dual_strategy not in ("extrapolate", "raise"):
            raise ValueError("self_dual_strategy must be 'extrapolate' or 'raise'")
        if (central_charge is None) == (b is None):
            raise ValueError("supply exactly one of central_charge or b")

        self.edge_count = len(external_weights) - 3
        self.vertex_count = self.edge_count + 1
        if external_descendants is None:
            external_descendants = (0,) * len(external_weights)
        if len(external_descendants) != len(external_weights) or any(
            value not in (0, 1) for value in external_descendants
        ):
            raise ValueError(
                "external_descendants must contain one zero/one marking per "
                "external weight"
            )
        self.external_descendants = tuple(
            int(value) for value in external_descendants
        )
        if len(internal_weights) != self.edge_count:
            raise ValueError(
                f"internal_weights must contain exactly {self.edge_count} entries"
            )
        if vertex_sectors is None:
            vertex_sectors = (
                sum(self.external_descendants) % 2,
            ) + (0,) * (self.vertex_count - 1)
        self.vertex_sectors = _validate_sectors(
            vertex_sectors,
            expected_length=self.vertex_count,
            required_parity=sum(self.external_descendants) % 2,
        )
        self.working_precision = int(working_precision)
        self.pole_tolerance = float(pole_tolerance)
        self.self_dual_strategy = self_dual_strategy

        node_values = (
            self._SELF_DUAL_LOG_B_NODES
            if self_dual_log_b_nodes is None
            else tuple(self_dual_log_b_nodes)
        )
        if len(node_values) < 3:
            raise ValueError("self_dual_log_b_nodes must contain at least three values")
        with mpmath.workdps(self.working_precision):
            normalized_nodes = tuple(mpmath.mpf(value) for value in node_values)
        if any(value <= 0 or not mpmath.isfinite(value) for value in normalized_nodes):
            raise ValueError("self-dual log(b) nodes must be positive and finite")
        if len(set(normalized_nodes)) != len(normalized_nodes):
            raise ValueError("self-dual log(b) nodes must be distinct")
        production_degree = (
            len(normalized_nodes) - 1
            if self_dual_polynomial_degree is None
            else int(self_dual_polynomial_degree)
        )
        comparison_degree = (
            production_degree - 1
            if self_dual_comparison_degree is None
            else int(self_dual_comparison_degree)
        )
        if not 1 <= production_degree < len(normalized_nodes):
            raise ValueError(
                "self_dual_polynomial_degree must lie in [1,len(nodes)-1]"
            )
        if not 0 <= comparison_degree < production_degree:
            raise ValueError(
                "self_dual_comparison_degree must be below the production degree"
            )
        self.self_dual_log_b_nodes = normalized_nodes
        self.self_dual_polynomial_degree = production_degree
        self.self_dual_comparison_degree = comparison_degree
        self.coefficient_cache_directory = (
            None
            if coefficient_cache_directory is None
            else Path(coefficient_cache_directory).expanduser().resolve()
        )

        with mpmath.workdps(self.working_precision):
            self.external_weights = tuple(
                mpmath.mpc(weight) for weight in external_weights
            )
            self.internal_weights = tuple(
                mpmath.mpc(weight) for weight in internal_weights
            )
            if b is None:
                self.central_charge = mpmath.mpc(central_charge)
                self.b = ns_b_from_c(self.central_charge)
            else:
                self.b = mpmath.mpc(b)
                self.central_charge = ns_central_charge_from_b(self.b)

            self.base_weight = self.internal_weights[0]
            self.weight_differences = tuple(
                weight - self.base_weight for weight in self.internal_weights
            )
            self.e_left = self.external_weights[0] - self.base_weight
            self.e_right = self.external_weights[-1] - self.base_weight
            self.fixed_middle_weights = self.external_weights[1:-1]

        self._coefficient_cache: dict[tuple[object, ...], object] = {}
        self._limit_backends: tuple[NSSphereLinearHRecursion, ...] | None = None
        self._self_dual_coefficient_cache: dict[
            tuple[int, tuple[int, ...]], object
        ] = {}
        self._coefficient_cache_artifacts: set[str] = set()

    @property
    def is_self_dual(self) -> bool:
        with mpmath.workdps(self.working_precision):
            return abs(self.b - 1) <= self.pole_tolerance

    def compatible_level_parities(
        self, vertex_sectors: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        sectors = (
            self.vertex_sectors
            if vertex_sectors is None
            else _validate_sectors(
                vertex_sectors,
                expected_length=self.vertex_count,
                required_parity=sum(self.external_descendants) % 2,
            )
        )
        parities = [
            sectors[0]
            ^ self.external_descendants[0]
            ^ self.external_descendants[1]
        ]
        for vertex in range(1, self.edge_count):
            parities.append(
                parities[-1]
                ^ sectors[vertex]
                ^ self.external_descendants[vertex + 1]
            )
        expected_last = (
            sectors[-1]
            ^ self.external_descendants[-2]
            ^ self.external_descendants[-1]
        )
        if parities[-1] != expected_last:
            raise ValueError("inconsistent sphere vertex-sector assignment")
        return tuple(parities)

    def _regular_seed(
        self,
        *,
        base_weight,
        differences: tuple[object, ...],
        e_left,
        e_right,
        vertex_sectors: tuple[int, ...],
    ):
        """Return the level-zero correlated large-weight boundary value."""

        internal, external = self._actual_weights(
            base_weight, differences, e_left, e_right
        )
        result = osp_sector_vertex(
            sector=vertex_sectors[0],
            n1=0,
            n2=0,
            n3=0,
            epsilon1=0,
            epsilon2=self.external_descendants[1],
            epsilon3=self.external_descendants[0],
            d1=internal[0],
            d2=external[1],
            d3=external[0],
        )
        for vertex in range(1, self.edge_count):
            result *= osp_sector_vertex(
                sector=vertex_sectors[vertex],
                n1=0,
                n2=0,
                n3=0,
                epsilon1=0,
                epsilon2=self.external_descendants[vertex + 1],
                epsilon3=0,
                d1=internal[vertex],
                d2=external[vertex + 1],
                d3=internal[vertex - 1],
            )
        result *= osp_sector_vertex(
            sector=vertex_sectors[-1],
            n1=0,
            n2=0,
            n3=0,
            epsilon1=self.external_descendants[-1],
            epsilon2=self.external_descendants[-2],
            epsilon3=0,
            d1=external[-1],
            d2=external[-2],
            d3=internal[-1],
        )
        return mpmath.mpc(result)

    def _actual_weights(
        self,
        base_weight,
        differences: tuple[object, ...],
        e_left,
        e_right,
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        internal = tuple(
            base_weight + difference for difference in differences
        )
        external = (
            base_weight + e_left,
            *self.fixed_middle_weights,
            base_weight + e_right,
        )
        return internal, external

    def _residue_data(
        self,
        *,
        edge: int,
        r: int,
        s: int,
        differences: tuple[object, ...],
        e_left,
        e_right,
        vertex_sectors: tuple[int, ...],
    ):
        product_rs = r * s
        level_shift = mpmath.mpf(product_rs) / 2
        degenerate = ns_degenerate_weight_mp(self.b, r, s)
        pole_base = degenerate - differences[edge]
        pole_internal, pole_external = self._actual_weights(
            pole_base, differences, e_left, e_right
        )

        if edge == 0:
            first_pair = (pole_external[0], pole_external[1])
        else:
            first_pair = (
                pole_internal[edge - 1],
                pole_external[edge + 1],
            )
        if edge == self.edge_count - 1:
            second_pair = (pole_external[-1], pole_external[-2])
        else:
            second_pair = (
                pole_internal[edge + 1],
                pole_external[edge + 2],
            )

        residue = (
            (-1 if product_rs % 2 else 1)
            * ns_inverse_null_slope_mp(r, s, self.b)
            * ns_fusion_polynomial_mp(
                r=r,
                s=s,
                a=vertex_sectors[edge],
                first_weight=first_pair[0],
                second_weight=first_pair[1],
                b=self.b,
            )
            * ns_fusion_polynomial_mp(
                r=r,
                s=s,
                a=vertex_sectors[edge + 1],
                first_weight=second_pair[0],
                second_weight=second_pair[1],
                b=self.b,
            )
        )

        child_sectors = list(vertex_sectors)
        if product_rs % 2:
            child_sectors[edge] ^= 1
            child_sectors[edge + 1] ^= 1

        child_differences = list(differences)
        if edge == 0:
            child_base = degenerate + level_shift
            for index in range(1, self.edge_count):
                child_differences[index] -= level_shift
            child_e_left = e_left - level_shift
            child_e_right = e_right - level_shift
        else:
            child_base = pole_base
            child_differences[edge] += level_shift
            child_e_left = e_left
            child_e_right = e_right

        return (
            degenerate,
            residue,
            child_base,
            tuple(child_differences),
            child_e_left,
            child_e_right,
            tuple(child_sectors),
        )

    def _coefficient_on_line(
        self,
        twice_levels: tuple[int, ...],
        vertex_sectors: tuple[int, ...],
        base_weight,
        differences: tuple[object, ...],
        e_left,
        e_right,
    ):
        key = (
            twice_levels,
            vertex_sectors,
            base_weight,
            differences,
            e_left,
            e_right,
        )
        if key in self._coefficient_cache:
            return self._coefficient_cache[key]

        result = (
            self._regular_seed(
                base_weight=base_weight,
                differences=differences,
                e_left=e_left,
                e_right=e_right,
                vertex_sectors=vertex_sectors,
            )
            if all(level == 0 for level in twice_levels)
            else mpmath.mpc(0)
        )
        for edge, available_level in enumerate(twice_levels):
            for r, s in _iter_ns_h_poles(available_level):
                product_rs = r * s
                (
                    degenerate,
                    residue,
                    child_base,
                    child_differences,
                    child_e_left,
                    child_e_right,
                    child_sectors,
                ) = self._residue_data(
                    edge=edge,
                    r=r,
                    s=s,
                    differences=differences,
                    e_left=e_left,
                    e_right=e_right,
                    vertex_sectors=vertex_sectors,
                )
                denominator = base_weight + differences[edge] - degenerate
                scale = max(mpmath.mpf(1), abs(base_weight), abs(degenerate))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise ZeroDivisionError(
                        f"h-recursion encountered the ({r},{s}) pole on edge "
                        f"{edge}: h={base_weight + differences[edge]!r}"
                    )
                child_levels = list(twice_levels)
                child_levels[edge] -= product_rs
                result += residue / denominator * self._coefficient_on_line(
                    tuple(child_levels),
                    child_sectors,
                    child_base,
                    child_differences,
                    child_e_left,
                    child_e_right,
                )

        self._coefficient_cache[key] = result
        return result

    def _generic_coefficient(self, levels: tuple[int, ...]):
        parities = self.compatible_level_parities()
        if any(level % 2 != parity for level, parity in zip(levels, parities)):
            return mpmath.mpc(0)
        return self._coefficient_on_line(
            levels,
            self.vertex_sectors,
            self.base_weight,
            self.weight_differences,
            self.e_left,
            self.e_right,
        )

    def _get_limit_backends(self) -> tuple["NSSphereLinearHRecursion", ...]:
        if self.self_dual_strategy == "raise":
            raise ZeroDivisionError(
                "the self-dual h-recursion has confluent Kac residues; "
                "select self_dual_strategy='extrapolate'"
            )
        if self._limit_backends is None:
            backends = []
            for log_b_value in self.self_dual_log_b_nodes:
                with mpmath.workdps(self.working_precision + 20):
                    detuned_b = mpmath.exp(log_b_value)
                backends.append(
                    NSSphereLinearHRecursion(
                        b=detuned_b,
                        external_weights=self.external_weights,
                        internal_weights=self.internal_weights,
                        external_descendants=self.external_descendants,
                        vertex_sectors=self.vertex_sectors,
                        working_precision=self.working_precision + 20,
                        pole_tolerance=self.pole_tolerance,
                        self_dual_strategy="raise",
                    )
                )
            self._limit_backends = tuple(backends)
        return self._limit_backends

    def _self_dual_limit(self, evaluator, *, degree: int | None = None):
        backends = self._get_limit_backends()
        with mpmath.workdps(self.working_precision):
            nodes = tuple(value**2 for value in self.self_dual_log_b_nodes)
            values = tuple(evaluator(backend) for backend in backends)
            selected_degree = (
                self.self_dual_polynomial_degree if degree is None else int(degree)
            )
            if selected_degree == len(nodes) - 1:
                return _lagrange_value_at_zero(nodes, values)
            return _polynomial_fit_value_at_zero(nodes, values, selected_degree)

    def _self_dual_coefficient(self, levels: tuple[int, ...], degree: int):
        key = (int(degree), levels)
        cached = self._self_dual_coefficient_cache.get(key)
        if cached is None:
            cached = self._self_dual_limit(
                lambda backend: backend._generic_coefficient(levels),
                degree=degree,
            )
            self._self_dual_coefficient_cache[key] = cached
        return cached

    def coefficient(
        self,
        twice_levels: Sequence[int],
        *,
        fit_variant: str = "production",
    ):
        """Return one reduced plumbing coefficient."""

        levels = _validate_twice_levels(
            twice_levels,
            expected_length=self.edge_count,
            name="twice_levels",
        )
        if fit_variant not in ("production", "comparison"):
            raise ValueError("fit_variant must be 'production' or 'comparison'")
        with mpmath.workdps(self.working_precision):
            if self.is_self_dual:
                degree = (
                    self.self_dual_polynomial_degree
                    if fit_variant == "production"
                    else self.self_dual_comparison_degree
                )
                return self._self_dual_coefficient(levels, degree)
            return self._generic_coefficient(levels)

    def coefficient_fit_error(self, twice_levels: Sequence[int]):
        """Return production minus comparison self-dual coefficient fits."""

        if not self.is_self_dual:
            return mpmath.mpc(0)
        return self.coefficient(
            twice_levels, fit_variant="production"
        ) - self.coefficient(twice_levels, fit_variant="comparison")

    def coefficient_table(
        self,
        max_twice_levels: Sequence[int],
        *,
        max_total_twice_level: int | None = None,
        fit_variant: str = "production",
    ) -> dict[tuple[int, ...], object]:
        maxima = _validate_twice_levels(
            max_twice_levels,
            expected_length=self.edge_count,
            name="max_twice_levels",
        )
        parities = self.compatible_level_parities()
        if max_total_twice_level is not None and (
            not isinstance(max_total_twice_level, int)
            or isinstance(max_total_twice_level, bool)
            or max_total_twice_level < 0
        ):
            raise ValueError(
                "max_total_twice_level must be a non-negative integer or None"
            )
        return {
            levels: self.coefficient(levels, fit_variant=fit_variant)
            for levels in product(
                *(
                    range(parity, maximum + 1, 2)
                    for parity, maximum in zip(parities, maxima)
                )
            )
            if max_total_twice_level is None
            or sum(levels) <= max_total_twice_level
        }

    @staticmethod
    def _encoded_mpc(value, digits: int) -> list[str]:
        selected = mpmath.mpc(value)
        return [
            mpmath.nstr(selected.real, n=digits, strip_zeros=False),
            mpmath.nstr(selected.imag, n=digits, strip_zeros=False),
        ]

    def _coefficient_artifact_manifest(
        self, maxima: tuple[int, ...], max_total_twice_level: int | None
    ) -> dict[str, object]:
        digits = self.working_precision + 10
        return {
            "schema": "ns-sphere-linear-h-self-dual-coefficients-v1",
            "algorithm": "fixed-difference-h-recursion-eta2-qr-fit-v1",
            "external_weights": [
                self._encoded_mpc(value, digits) for value in self.external_weights
            ],
            "internal_weights": [
                self._encoded_mpc(value, digits) for value in self.internal_weights
            ],
            "external_descendants": list(self.external_descendants),
            "vertex_sectors": list(self.vertex_sectors),
            "log_b_nodes": [
                mpmath.nstr(value, n=digits, strip_zeros=False)
                for value in self.self_dual_log_b_nodes
            ],
            "polynomial_degree": self.self_dual_polynomial_degree,
            "comparison_degree": self.self_dual_comparison_degree,
            "working_precision": self.working_precision,
            "pole_tolerance": repr(self.pole_tolerance),
            "max_twice_levels": list(maxima),
            "max_total_twice_level": max_total_twice_level,
        }

    def _coefficient_artifact_path(
        self, manifest: dict[str, object]
    ) -> Path | None:
        if self.coefficient_cache_directory is None:
            return None
        encoded = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.coefficient_cache_directory / digest[:2] / f"{digest}.json"

    def _load_coefficient_artifact(
        self, path: Path, manifest: dict[str, object]
    ) -> bool:
        if not path.exists():
            return False
        payload = json.loads(path.read_text())
        if payload.get("manifest") != manifest:
            raise ArithmeticError(f"coefficient-cache manifest mismatch in {path}")
        production_degree = self.self_dual_polynomial_degree
        comparison_degree = self.self_dual_comparison_degree
        with mpmath.workdps(self.working_precision):
            for entry in payload["coefficients"]:
                levels = tuple(int(value) for value in entry["levels"])
                production = mpmath.mpc(*entry["production"])
                comparison = mpmath.mpc(*entry["comparison"])
                self._self_dual_coefficient_cache[
                    (production_degree, levels)
                ] = production
                self._self_dual_coefficient_cache[
                    (comparison_degree, levels)
                ] = comparison
        self._coefficient_cache_artifacts.add(str(path))
        return True

    def _write_coefficient_artifact(
        self,
        path: Path,
        manifest: dict[str, object],
        maxima: tuple[int, ...],
        max_total_twice_level: int | None,
    ) -> None:
        digits = self.working_precision + 5
        production = self.coefficient_table(
            maxima,
            max_total_twice_level=max_total_twice_level,
            fit_variant="production",
        )
        comparison = self.coefficient_table(
            maxima,
            max_total_twice_level=max_total_twice_level,
            fit_variant="comparison",
        )
        payload = {
            "manifest": manifest,
            "coefficients": [
                {
                    "levels": list(levels),
                    "production": self._encoded_mpc(value, digits),
                    "comparison": self._encoded_mpc(comparison[levels], digits),
                }
                for levels, value in sorted(production.items())
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            path.suffix + f".{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(path)
        self._coefficient_cache_artifacts.add(str(path))

    def _prepare_self_dual_rectangle(
        self, maxima: tuple[int, ...], max_total_twice_level: int | None
    ) -> None:
        """Materialize both fitted tables, then release generic-b caches."""

        if not self.is_self_dual:
            return
        manifest = self._coefficient_artifact_manifest(
            maxima, max_total_twice_level
        )
        artifact = self._coefficient_artifact_path(manifest)
        if artifact is not None and str(artifact) in self._coefficient_cache_artifacts:
            return
        if artifact is not None and self._load_coefficient_artifact(
            artifact, manifest
        ):
            if self._limit_backends is not None:
                for backend in self._limit_backends:
                    backend.clear_cache()
                self._limit_backends = None
            return
        if artifact is None:
            for variant in ("production", "comparison"):
                self.coefficient_table(
                    maxima,
                    max_total_twice_level=max_total_twice_level,
                    fit_variant=variant,
                )
        else:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            lock_path = artifact.with_suffix(artifact.suffix + ".lock")
            with lock_path.open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                if not self._load_coefficient_artifact(artifact, manifest):
                    self._write_coefficient_artifact(
                        artifact, manifest, maxima, max_total_twice_level
                    )
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        if self._limit_backends is not None:
            for backend in self._limit_backends:
                backend.clear_cache()
            self._limit_backends = None

    def _series_value_generic(
        self,
        log_tuple,
        maxima,
        max_total_twice_level,
        minima,
    ):
        parities = self.compatible_level_parities()
        result = mpmath.mpc(0)
        for levels in product(
            *(
                range(parity, maximum + 1, 2)
                for parity, maximum in zip(parities, maxima)
            )
        ):
            if any(level < minimum for level, minimum in zip(levels, minima)):
                continue
            if (
                max_total_twice_level is not None
                and sum(levels) > max_total_twice_level
            ):
                continue
            monomial = mpmath.fprod(
                mpmath.exp(log_q * (mpmath.mpf(level) / 2))
                for log_q, level in zip(log_tuple, levels)
            )
            result += self._generic_coefficient(levels) * monomial
        return result

    def _prepare_series_arguments(
        self,
        q_values: Sequence[object],
        max_twice_levels: Sequence[int],
        max_total_twice_level: int | None,
        q_log_values: Sequence[object] | None,
        minimum_twice_levels: Sequence[int] | None,
    ):
        maxima = _validate_twice_levels(
            max_twice_levels,
            expected_length=self.edge_count,
            name="max_twice_levels",
        )
        minima = (
            (0,) * self.edge_count
            if minimum_twice_levels is None
            else _validate_twice_levels(
                minimum_twice_levels,
                expected_length=self.edge_count,
                name="minimum_twice_levels",
            )
        )
        if len(q_values) != self.edge_count:
            raise ValueError(
                f"q_values must contain exactly {self.edge_count} entries"
            )
        if max_total_twice_level is not None and (
            not isinstance(max_total_twice_level, int)
            or isinstance(max_total_twice_level, bool)
            or max_total_twice_level < 0
        ):
            raise ValueError(
                "max_total_twice_level must be a non-negative integer or None"
            )
        q_tuple = tuple(mpmath.mpc(value) for value in q_values)
        log_tuple = (
            tuple(mpmath.log(value) for value in q_tuple)
            if q_log_values is None
            else tuple(mpmath.mpc(value) for value in q_log_values)
        )
        if len(log_tuple) != self.edge_count:
            raise ValueError("q_log_values must contain one logarithm per edge")
        if any(
            not (mpmath.isfinite(value.real) and mpmath.isfinite(value.imag))
            for value in q_tuple + log_tuple
        ):
            raise ValueError("plumbing parameters and logarithms must be finite")
        return log_tuple, maxima, minima

    def series_value(
        self,
        q_values: Sequence[object],
        max_twice_levels: Sequence[int],
        *,
        max_total_twice_level: int | None = None,
        q_log_values: Sequence[object] | None = None,
        minimum_twice_levels: Sequence[int] | None = None,
        fit_variant: str = "production",
    ):
        """Evaluate the reduced coefficient series in a rectangular cutoff."""

        if fit_variant not in ("production", "comparison"):
            raise ValueError("fit_variant must be 'production' or 'comparison'")

        with mpmath.workdps(self.working_precision):
            log_tuple, maxima, minima = self._prepare_series_arguments(
                q_values,
                max_twice_levels,
                max_total_twice_level,
                q_log_values,
                minimum_twice_levels,
            )
            if self.is_self_dual:
                self._prepare_self_dual_rectangle(maxima, max_total_twice_level)
                parities = self.compatible_level_parities()
                result = mpmath.mpc(0)
                for levels in product(
                    *(
                        range(parity, maximum + 1, 2)
                        for parity, maximum in zip(parities, maxima)
                    )
                ):
                    if any(
                        level < minimum
                        for level, minimum in zip(levels, minima)
                    ):
                        continue
                    if (
                        max_total_twice_level is not None
                        and sum(levels) > max_total_twice_level
                    ):
                        continue
                    monomial = mpmath.fprod(
                        mpmath.exp(log_q * (mpmath.mpf(level) / 2))
                        for log_q, level in zip(log_tuple, levels)
                    )
                    result += self.coefficient(
                        levels, fit_variant=fit_variant
                    ) * monomial
                return result
            return self._series_value_generic(
                log_tuple, maxima, max_total_twice_level, minima
            )

    def self_dual_fit_diagnostics(self) -> dict[str, object]:
        """Return compact diagnostics for coefficients materialized so far."""

        production_degree = self.self_dual_polynomial_degree
        comparison_degree = self.self_dual_comparison_degree
        common_levels = sorted(
            levels
            for degree, levels in self._self_dual_coefficient_cache
            if degree == production_degree
            and (comparison_degree, levels) in self._self_dual_coefficient_cache
        )
        absolute_errors = []
        relative_errors = []
        for levels in common_levels:
            production = self._self_dual_coefficient_cache[
                (production_degree, levels)
            ]
            comparison = self._self_dual_coefficient_cache[
                (comparison_degree, levels)
            ]
            error = abs(production - comparison)
            absolute_errors.append(error)
            relative_errors.append(error / max(mpmath.mpf(1), abs(production)))
        return {
            "self_dual": bool(self.is_self_dual),
            "log_b_nodes": [float(value) for value in self.self_dual_log_b_nodes],
            "polynomial_degree": production_degree,
            "comparison_degree": comparison_degree,
            "coefficient_count": len(common_levels),
            "coefficient_cache_artifact_count": len(
                self._coefficient_cache_artifacts
            ),
            "maximum_absolute_fit_shift": float(max(absolute_errors, default=0)),
            "maximum_scaled_fit_shift": float(max(relative_errors, default=0)),
        }

    def _recursive_series_value_generic(
        self,
        *,
        log_tuple,
        budget: int,
        accumulated_maxima: tuple[int | None, ...],
        minimum_levels: tuple[int, ...],
    ):
        cache: dict[tuple[object, ...], object] = {}

        def recurse(
            remaining_budget,
            accumulated_levels,
            vertex_sectors,
            base_weight,
            differences,
            e_left,
            e_right,
        ):
            key = (
                remaining_budget,
                accumulated_levels,
                vertex_sectors,
                base_weight,
                differences,
                e_left,
                e_right,
            )
            if key in cache:
                return cache[key]

            result = (
                self._regular_seed(
                    base_weight=base_weight,
                    differences=differences,
                    e_left=e_left,
                    e_right=e_right,
                    vertex_sectors=vertex_sectors,
                )
                if all(
                    level >= minimum
                    for level, minimum in zip(
                        accumulated_levels, minimum_levels
                    )
                )
                else mpmath.mpc(0)
            )
            for edge, maximum in enumerate(accumulated_maxima):
                remaining_on_edge = (
                    remaining_budget
                    if maximum is None
                    else maximum - accumulated_levels[edge]
                )
                allowed = min(remaining_budget, remaining_on_edge)
                for r, s in _iter_ns_h_poles(allowed):
                    shift = r * s
                    (
                        degenerate,
                        residue,
                        child_base,
                        child_differences,
                        child_e_left,
                        child_e_right,
                        child_sectors,
                    ) = self._residue_data(
                        edge=edge,
                        r=r,
                        s=s,
                        differences=differences,
                        e_left=e_left,
                        e_right=e_right,
                        vertex_sectors=vertex_sectors,
                    )
                    denominator = base_weight + differences[edge] - degenerate
                    scale = max(
                        mpmath.mpf(1), abs(base_weight), abs(degenerate)
                    )
                    if abs(denominator) <= self.pole_tolerance * scale:
                        raise ZeroDivisionError(
                            f"functional h-recursion encountered the ({r},{s}) "
                            f"pole on edge {edge}"
                        )
                    child_accumulated = list(accumulated_levels)
                    child_accumulated[edge] += shift
                    result += (
                        mpmath.exp(
                            log_tuple[edge] * (mpmath.mpf(shift) / 2)
                        )
                        * residue
                        / denominator
                        * recurse(
                            remaining_budget - shift,
                            tuple(child_accumulated),
                            child_sectors,
                            child_base,
                            child_differences,
                            child_e_left,
                            child_e_right,
                        )
                    )
            cache[key] = result
            return result

        return recurse(
            budget,
            (0,) * self.edge_count,
            self.vertex_sectors,
            self.base_weight,
            self.weight_differences,
            self.e_left,
            self.e_right,
        )

    def recursive_series_value(
        self,
        q_values: Sequence[object],
        recursion_max_twice_level: int,
        *,
        maximum_accumulated_twice_levels: Sequence[int | None] | None = None,
        q_log_values: Sequence[object] | None = None,
        minimum_twice_levels: Sequence[int] | None = None,
    ):
        """Evaluate the functional h-recursion with its constant seed."""

        if (
            not isinstance(recursion_max_twice_level, int)
            or isinstance(recursion_max_twice_level, bool)
            or recursion_max_twice_level < 0
        ):
            raise ValueError(
                "recursion_max_twice_level must be a non-negative integer"
            )
        if maximum_accumulated_twice_levels is None:
            maxima: tuple[int | None, ...] = (
                recursion_max_twice_level,
            ) * self.edge_count
        else:
            if len(maximum_accumulated_twice_levels) != self.edge_count:
                raise ValueError(
                    "maximum_accumulated_twice_levels must contain one entry per edge"
                )
            maxima = tuple(maximum_accumulated_twice_levels)
            if any(
                value is not None
                and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                )
                for value in maxima
            ):
                raise ValueError(
                    "maximum accumulated levels must be non-negative integers or None"
                )
        minima = (
            (0,) * self.edge_count
            if minimum_twice_levels is None
            else _validate_twice_levels(
                minimum_twice_levels,
                expected_length=self.edge_count,
                name="minimum_twice_levels",
            )
        )
        with mpmath.workdps(self.working_precision):
            if len(q_values) != self.edge_count:
                raise ValueError(
                    f"q_values must contain exactly {self.edge_count} entries"
                )
            q_tuple = tuple(mpmath.mpc(value) for value in q_values)
            log_tuple = (
                tuple(mpmath.log(value) for value in q_tuple)
                if q_log_values is None
                else tuple(mpmath.mpc(value) for value in q_log_values)
            )
            if len(log_tuple) != self.edge_count:
                raise ValueError("q_log_values must contain one logarithm per edge")
            if self.is_self_dual:
                finite_maxima = tuple(
                    recursion_max_twice_level
                    if maximum is None
                    else min(maximum, recursion_max_twice_level)
                    for maximum in maxima
                )
                return self.series_value(
                    q_tuple,
                    finite_maxima,
                    max_total_twice_level=recursion_max_twice_level,
                    q_log_values=log_tuple,
                    minimum_twice_levels=minima,
                )
            return self._recursive_series_value_generic(
                log_tuple=log_tuple,
                budget=recursion_max_twice_level,
                accumulated_maxima=maxima,
                minimum_levels=minima,
            )

    def clear_cache(self) -> None:
        self._coefficient_cache.clear()
        self._self_dual_coefficient_cache.clear()
        self._coefficient_cache_artifacts.clear()
        if self._limit_backends is not None:
            for backend in self._limit_backends:
                backend.clear_cache()


__all__ = [
    "NSSphereLinearHRecursion",
    "ns_b_from_c",
    "ns_central_charge_from_b",
    "ns_degenerate_weight_mp",
]
