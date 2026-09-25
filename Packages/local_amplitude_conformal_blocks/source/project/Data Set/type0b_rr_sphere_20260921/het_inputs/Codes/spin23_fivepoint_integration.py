"""Exploratory global Sobol/collar/threshold driver with a fail-closed 3% gate.

Only the S -> VVVV adapter is implemented here. A chart subtotal, fixed-P
integral or incomplete run must never be labelled a physical amplitude.
All 120 oriented charts are summed; their partition weights include the
overlap multiplicities, with no 120 factor or guessed symmetry quotient.
"""
from dataclasses import dataclass, replace
from itertools import combinations, product
import math
from time import perf_counter

import mpmath as mp
import numpy as np
from scipy.stats import qmc

from spin23_fivepoint_density import SToFourVectors, TENSOR_NAMES
from spin23_fivepoint_type0b_reference import layers
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import (
    ThresholdConfig, threshold_weighted_rule)


REFINEMENTS = ("block_left", "block_right", "momentum_left", "momentum_right",
    "threshold_envelope", "collar_radius", "ope_depth", "precision", "sobol_prefix")


def accuracy_gate(values, standard_errors, refinements, *, complete, target=.03,
                  physical_adapter_verified=False, absolute_scales=None,
                  successive_stable_refinements=0):
    """Observed shifts are diagnostics, not a rigorous or 95%-coverage bound.

    Every refinement is an absolute shift vector for the three tensors.
    None is missing data, never zero error. Near-zero absolute scales must
    be explicitly supplied and are returned with the resulting error mode.
    """
    values, se = np.asarray(values, complex), np.asarray(standard_errors, float)
    if values.shape != (3,) or se.shape != (3,) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(se)) or np.any(se < 0):
        raise ValueError("three finite values and nonnegative standard errors required")
    if not 0 < target < 1:
        raise ValueError("relative target must be between zero and one")
    unknown = set(refinements)-set(REFINEMENTS)
    if unknown:
        raise ValueError(f"unknown refinement labels: {unknown}")
    missing = [name for name in REFINEMENTS if refinements.get(name) is None]
    shifts = np.zeros(3)
    for name, shift in refinements.items():
        if shift is not None:
            shift = np.asarray(shift, float)
            if shift.shape != (3,) or not np.all(np.isfinite(shift)) or np.any(shift < 0):
                raise ValueError(f"invalid absolute refinement shift: {name}")
            shifts += shift
    scale = np.abs(values)
    modes = ["relative"]*3
    if absolute_scales is not None:
        floor = np.asarray(absolute_scales, float)
        if floor.shape != (3,) or not np.all(np.isfinite(floor)) or np.any(floor <= 0):
            raise ValueError("three positive declared absolute scales required")
        modes = ["declared_absolute_scale" if f > v else "relative" for f, v in zip(floor, scale)]
        scale = np.maximum(scale, floor)
    observed = 2*se+shifts
    if type(successive_stable_refinements) is not int or successive_stable_refinements < 0:
        raise ValueError("nonnegative count of independently checked stable refinement steps required")
    eligible = bool(complete and physical_adapter_verified and not missing and np.all(scale > 0)
        and successive_stable_refinements >= 2)
    passed = eligible and bool(np.all(observed <= target*scale))
    return dict(target=target, passed=passed, eligible=eligible, complete=bool(complete),
        physical_adapter_verified=bool(physical_adapter_verified), missing_refinements=missing,
        successive_stable_refinements=successive_stable_refinements,
        twice_standard_error=(2*se).tolist(), systematic_shift_sum=None if missing else shifts.tolist(),
        measured_error_lower_bound=observed.tolist(),
        observed_error_diagnostic=None if missing else observed.tolist(), acceptance_scales=scale.tolist(),
        error_modes=modes,
        relative_diagnostic=None if missing else np.divide(observed, scale, out=np.full(3, np.inf), where=scale > 0).tolist(),
        relative_statistical_2se=np.divide(2*se, scale, out=np.full(3, np.inf), where=scale > 0).tolist(),
        rigorous_bound=False)


def continuation_audit(outgoing):
    """Sufficient no-DOZZ-wall test on the straight external-momentum path.

    Internal contours remain positive real. In every outer trinion,
    0 < Re(1+i(+-p_i+-p_j+-P)) < 2 if |Im p_i|+|Im p_j| < 1.
    This is a zero-free NS-Upsilon strip and also safe for R-Upsilon in
    Ctilde. The middle trinion has just one external imaginary part.
    This does NOT assert moduli integrability or fix real-energy limits.
    """
    outgoing = tuple(map(complex, outgoing))
    if len(outgoing) != 4 or not all(np.isfinite(z) for z in outgoing):
        raise ValueError("four finite external energies required")
    ps = outgoing+(sum(outgoing),)
    largest = max(abs(a.imag)+abs(b.imag) for a, b in combinations(ps, 2))
    return dict(path="keep real parts fixed; scale all imaginary parts from 0 to 1",
        largest_outer_trinion_imaginary_sum=largest, zero_free_strip_margin=1-largest,
        straight_real_spectral_contours_safe=bool(largest < 1-1e-12),
        certifies_moduli_convergence=False)


def spectral_rules(counts=(12, 13), configs=None):
    if len(counts) != 2:
        raise ValueError("two momentum-axis node budgets required")
    if configs is None:
        a = ThresholdConfig()
        # Independently chosen thresholds, including bulk panel edges, so
        # differing total node counts do not accidentally share pole nodes.
        b = replace(a, endpoint=.187, tail=3.27,
            bulk_breakpoints=tuple(1.017*x for x in a.bulk_breakpoints))
        configs = (a, b)
    if len(configs) != 2:
        raise ValueError("two threshold configurations required")
    rules = tuple(threshold_weighted_rule(n, c) for n, c in zip(counts, configs))
    gap = np.abs(rules[0].momenta[:, None]-rules[1].momenta[None, :])
    if np.any(gap < 1e-12*np.maximum(1, rules[0].momenta[:, None])):
        raise ValueError("quadrature grids contain coincident recursion poles; no nodes may be dropped")
    return rules


def spectral_density(outgoing, x, y, *, ordering=range(5), counts=(12, 13), configs=None,
                     maximum_twice_levels=(4, 4), precision=40, progress=None):
    """Pointwise CORRELATOR density after both continuum integrals.

    Suitable for bounded physical-adapter/channel checks, NOT the amplitude.
    All nine endpoint/bulk/tail rectangles are retained separately.
    """
    audit = continuation_audit(outgoing)
    if not audit['straight_real_spectral_contours_safe']:
        raise ValueError("external continuation requires a separate contour/residue audit")
    rules = spectral_rules(counts, configs)
    regions = np.zeros((3, 3, 3), complex)
    started = perf_counter()
    for i, j in product(range(len(rules[0].momenta)), range(len(rules[1].momenta))):
        momenta = float(rules[0].momenta[i]), float(rules[1].momenta[j])
        bank = SToFourVectors(outgoing, momenta, ordering=ordering,
            maximum_twice_levels=maximum_twice_levels, precision=precision)
        values = np.asarray(bank.values(x, y), complex)
        if not np.all(np.isfinite(values)):
            raise ArithmeticError(f"nonfinite dressed spectral node {momenta}; estimate rejected")
        weight = rules[0].weights[i]*rules[1].weights[j]/math.pi**2
        regions[rules[0].regions[i], rules[1].regions[j]] += weight*values
        if progress is not None:
            progress(dict(completed=i*len(rules[1].momenta)+j+1,
                required=len(rules[0].momenta)*len(rules[1].momenta), elapsed=perf_counter()-started))
    return dict(observable="momentum_integrated_S_to_VVVV_density_not_amplitude",
        values=regions.sum(axis=(0, 1)), regions=regions, ordering=tuple(ordering),
        coordinates=(x, y), counts=counts, maximum_twice_levels=maximum_twice_levels,
        elapsed_seconds=perf_counter()-started, continuation=audit,
        quadrature=[r.metadata() for r in rules])


@dataclass(frozen=True)
class PatchSettings:
    sample_power: int = 6
    replicates: int = 4
    collar: float = .08
    radius: float = 1.
    maximum_increment: int = 4
    atlas_power: int = 16
    seed: int = 230916

    def __post_init__(self):
        if type(self.sample_power) is not int or self.sample_power < 0:
            raise ValueError("nonnegative integer Sobol power required")
        if type(self.replicates) is not int or self.replicates < 2:
            raise ValueError("at least two independent scrambles required")
        if not 0 < self.collar < self.radius <= 1:
            raise ValueError("require 0 < collar < radius <= 1")
        if type(self.maximum_increment) is not int or self.maximum_increment < 0:
            raise ValueError("nonnegative OPE increment required")
        if type(self.atlas_power) is not int or self.atlas_power <= 4*(self.maximum_increment//2):
            raise ValueError("atlas cutoff must exceed the retained corner-jet degree")


def _annulus(u, v, settings):
    span = math.log(settings.radius/settings.collar)
    r = settings.collar*np.exp(span*np.asarray(u))
    return r*np.exp(2j*math.pi*np.asarray(v)), 2*math.pi*span*r*r


def integrate_patch_qmc(engine, settings=PatchSettings(), *, use_atlas=True):
    """Disjoint bulk, faces and corner; normal collars restored analytically.

    Interior collar regular remainders are approximated with the same finite
    OPE. The radius/depth refinements are consequently mandatory. Independent
    face scrambles are separate from bulk, while node/chart sums preserve
    each replicate index before computing its standard error.
    """
    generic = layers()
    density = generic.atlas.WeightedEllipticDensity(engine, settings.atlas_power) if use_atlas else engine
    generic.integration.require_complete_depth(engine.betas, settings.maximum_increment)
    with mp.workdps(engine.precision):
        moments = tuple(tuple(generic.integration.radial_moment(beta, j, settings.collar)
            for j in range(settings.maximum_increment+1)) for beta in engine.betas)
        corner = density.coefficients((0, 1), settings.maximum_increment)
        corner_value = sum(c*moments[0][k[0]]*moments[1][k[1]] for k, c in corner.items())
    parts = np.zeros((settings.replicates, 4), complex)
    parts[:, 3] = complex(corner_value)
    for replicate in range(settings.replicates):
        seed = settings.seed+104729*replicate
        points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(settings.sample_power)
        x, wx = _annulus(points[:, 0], points[:, 1], settings)
        y, wy = _annulus(points[:, 2], points[:, 3], settings)
        parts[replicate, 0] = np.mean([w1*w2*complex(density.value(a, b)) for a, b, w1, w2 in zip(x, y, wx, wy)])
        for edge in (0, 1):
            points = qmc.Sobol(2, scramble=True, seed=seed+1009*(edge+1)).random_base2(settings.sample_power)
            t, wt = _annulus(points[:, 0], points[:, 1], settings)
            vals = []
            with mp.workdps(engine.precision):
                for z, weight in zip(t, wt):
                    coefficients = density.coefficients((edge,), settings.maximum_increment, z)
                    vals.append(weight*complex(sum(c*moments[edge][k[edge]] for k, c in coefficients.items())))
            parts[replicate, 1+edge] = np.mean(vals)
    if not np.all(np.isfinite(parts)):
        raise ArithmeticError("nonfinite patch contribution; estimate rejected")
    return parts


class PilotBudgetExceeded(RuntimeError):
    """Bounded pilot ended before completing an estimator; partial is NOT an amplitude."""
    def __init__(self, partial):
        super().__init__("pilot wall-time budget reached; no complete global estimate")
        self.partial = partial


def global_pilot(outgoing, *, counts=(12, 13), configs=None, maximum_twice_levels=(4, 4),
                 precision=40, settings=PatchSettings(), wall_seconds=120, progress=None):
    """Exact labelled-chart sum with both threshold momentum integrals.

    This is intentionally cost-bounded. A complete run is still unqualified
    until independent numerical refinements and physical-adapter tests pass.
    No failed node or unfinished chart is converted to a zero contribution.
    """
    if settings.radius != 1:
        raise ValueError("global atlas requires the complete unit bidisk; use patch integration for a smaller radius")
    audit = continuation_audit(outgoing)
    if not audit['straight_real_spectral_contours_safe']:
        raise ValueError("external continuation exits the audited strip; residue treatment required")
    if wall_seconds <= 0 or not math.isfinite(wall_seconds):
        raise ValueError("finite positive pilot wall budget required")
    rules = spectral_rules(counts, configs)
    total = np.zeros((settings.replicates, 3, 4), complex)
    regions = np.zeros((3, 3, settings.replicates, 3, 4), complex)
    started = perf_counter()
    completed, work = 0, 120*len(rules[0].momenta)*len(rules[1].momenta)
    base = dict(continuation=audit, source_manifest=layers().manifest,
        tensor_names=TENSOR_NAMES, required_chart_nodes=work, spectral_rules=[r.metadata() for r in rules],
        maximum_twice_levels=maximum_twice_levels, precision=precision, settings=vars(settings),
        normalization="reduced raw-S CFT integral; dPa dPb/pi^2 included; outer string factors excluded")
    def snapshot(complete):
        samples = total.sum(axis=-1)
        mean = samples.mean(axis=0)
        se = np.sqrt(np.sum(np.abs(samples-mean)**2, axis=0)/(settings.replicates*(settings.replicates-1)))
        return dict(base, complete=complete, amplitude_qualified=False, completed_chart_nodes=completed,
            elapsed_seconds=perf_counter()-started, subtotal=mean, subtotal_standard_errors=se,
            replicate_parts=total.copy(), spectral_region_parts=regions.copy(),
            accuracy=accuracy_gate(mean, se, {}, complete=complete, physical_adapter_verified=False))
    for i, j in product(range(len(rules[0].momenta)), range(len(rules[1].momenta))):
        p = (float(rules[0].momenta[i]), float(rules[1].momenta[j]))
        weight = rules[0].weights[i]*rules[1].weights[j]/math.pi**2
        for ordering in layers().atlas.ORDERINGS:
            if perf_counter()-started >= wall_seconds:
                raise PilotBudgetExceeded(snapshot(False))
            try:
                bank = SToFourVectors(outgoing, p, ordering=ordering,
                    maximum_twice_levels=maximum_twice_levels, precision=precision)
                parts = np.stack([integrate_patch_qmc(e, settings) for e in bank.densities], axis=1)*weight
            except Exception as exc:
                raise RuntimeError(f"failed chart {ordering}, momenta {p}: estimate rejected") from exc
            total += parts
            regions[rules[0].regions[i], rules[1].regions[j]] += parts
            completed += 1
            if progress is not None:
                progress(dict(completed_chart_nodes=completed, required_chart_nodes=work,
                    ordering=ordering, momenta=p, elapsed_seconds=perf_counter()-started))
    return snapshot(True)


def spectral_proposal(rule, uniform_fraction=.05):
    """Full-support proposal for the existing finite quadrature SUM.

    Sampling is not a new quadrature: w_i/prob_i restores each original
    dP weight. A 5% uniform mixture prevents tiny tail-node probabilities.
    Its variance is included in the independent replicate statistics.
    """
    if not 0 < uniform_fraction < 1:
        raise ValueError("strictly between zero and one uniform proposal fraction required")
    logs = np.log(rule.weights)+rule.config.log_weight(rule.momenta)
    probabilities = np.exp(logs-logs.max())
    probabilities /= probabilities.sum()
    probabilities = (1-uniform_fraction)*probabilities+uniform_fraction/len(probabilities)
    if not np.all(probabilities > 0) or not np.isclose(probabilities.sum(), 1):
        raise ArithmeticError("proposal lost support or normalization")
    return probabilities


def _stratum_draw(bank, u, settings):
    """One shared spectral/chart draw; bulk, two faces and analytic corner."""
    generic = layers()
    result = np.zeros((3, 4), complex)
    x, wx = _annulus(u[0], u[1], settings)
    y, wy = _annulus(u[2], u[3], settings)
    result[:, 0] = np.asarray(bank.values(x, y), complex)*wx*wy*generic.atlas.atlas_weight(
        complex(x), complex(y), settings.atlas_power)
    tangents = [_annulus(u[4+2*edge], u[5+2*edge], settings) for edge in (0, 1)]
    with mp.workdps(bank.precision):
        for j, engine in enumerate(bank.densities):
            generic.integration.require_complete_depth(engine.betas, settings.maximum_increment)
            density = generic.atlas.WeightedEllipticDensity(engine, settings.atlas_power)
            moments = tuple(tuple(generic.integration.radial_moment(beta, k, settings.collar)
                for k in range(settings.maximum_increment+1)) for beta in engine.betas)
            for edge, (t, w) in enumerate(tangents):
                coefficients = density.coefficients((edge,), settings.maximum_increment, t)
                result[j, 1+edge] = w*complex(sum(v*moments[edge][key[edge]] for key, v in coefficients.items()))
            coefficients = density.coefficients((0, 1), settings.maximum_increment)
            result[j, 3] = complex(sum(v*moments[0][key[0]]*moments[1][key[1]] for key, v in coefficients.items()))
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("nonfinite stratum draw; reject the entire estimator")
    return result


def global_importance_pilot(outgoing, *, counts=(12, 13), configs=None, maximum_twice_levels=(4, 4),
                            precision=40, settings=PatchSettings(sample_power=4), wall_seconds=180,
                            progress=None):
    """Unbiased randomized-QMC estimate of the same finite global approximant.

    Three Sobol coordinates select one of ALL 120 charts and the two
    full-support quadrature proposals. Eight coordinates sample bulk and
    face moduli. Corner normal integrals are analytic. The original positive
    spectral weights and every atlas/Jacobian factor are unchanged.

    This is a pilot estimator, not permission to skip deterministic momentum,
    block, collar or precision refinements. No physical 3% claim is made here.
    """
    if settings.radius != 1:
        raise ValueError("global atlas requires the complete unit bidisk; use patch integration for a smaller radius")
    audit = continuation_audit(outgoing)
    if not audit['straight_real_spectral_contours_safe']:
        raise ValueError("straight contours leave the audited zero-free strip")
    if wall_seconds <= 0 or not math.isfinite(wall_seconds):
        raise ValueError("finite positive pilot wall budget required")
    rules = spectral_rules(counts, configs)
    proposals = tuple(spectral_proposal(r) for r in rules)
    cdfs = tuple(np.cumsum(p) for p in proposals)
    charts = layers().atlas.ORDERINGS
    sums = np.zeros((settings.replicates, 3, 4), complex)
    regions = np.zeros((3, 3, settings.replicates, 3, 4), complex)
    done = np.zeros(settings.replicates, int)
    hits = np.zeros((3, 3), int)
    chart_hits = np.zeros(120, int)
    started = perf_counter()
    expected = settings.replicates*2**settings.sample_power
    def report(complete):
        denominator = np.maximum(done, 1)[:, None, None]
        parts = sums/denominator
        samples = parts.sum(axis=-1)
        mean = samples.mean(axis=0)
        se = np.sqrt(np.sum(np.abs(samples-mean)**2, axis=0)/(settings.replicates*(settings.replicates-1)))
        return dict(observable="stochastic_global_finite_elliptic_approximant_NOT_qualified_amplitude",
            complete=complete, amplitude_qualified=False, settings=vars(settings),
            maximum_twice_levels=maximum_twice_levels, counts=counts, precision=precision,
            values=mean if complete else None, standard_errors=se if complete else None,
            replicate_parts=parts if complete else None, completed_draws_per_replicate=done.copy(),
            required_draws=expected, spectral_region_parts=regions/denominator[None, None, :],
            spectral_region_draw_counts=hits.copy(), chart_draw_counts=chart_hits.copy(),
            charts_in_proposal_support=120, spectral_rules=[r.metadata() for r in rules],
            discrete_spectral_probabilities=proposals, spectral_proposal_uniform_fraction=.05,
            continuation=audit, elapsed_seconds=perf_counter()-started,
            statistical_error_includes="moduli, chart selection and discrete spectral-node sampling",
            normalization="dPa dPb/pi^2 included; raw singlet; outer string factors excluded",
            accuracy=accuracy_gate(mean, se, {}, complete=complete, physical_adapter_verified=False))
    # Each scramble estimates the WHOLE integral, not separate chart subsets.
    for replicate in range(settings.replicates):
        samples = qmc.Sobol(11, scramble=True, seed=settings.seed+104729*replicate).random_base2(settings.sample_power)
        for u in samples:
            if perf_counter()-started >= wall_seconds:
                raise PilotBudgetExceeded(report(False))
            chart = min(119, int(120*u[0]))
            i, j = (min(len(cdf)-1, int(np.searchsorted(cdf, a, side='right')))
                for cdf, a in zip(cdfs, u[1:3]))
            momenta = float(rules[0].momenta[i]), float(rules[1].momenta[j])
            bank = SToFourVectors(outgoing, momenta, ordering=charts[chart],
                maximum_twice_levels=maximum_twice_levels, precision=precision)
            importance = 120*rules[0].weights[i]*rules[1].weights[j]/(
                math.pi**2*proposals[0][i]*proposals[1][j])
            parts = _stratum_draw(bank, u[3:], settings)*importance
            sums[replicate] += parts
            region = int(rules[0].regions[i]), int(rules[1].regions[j])
            regions[region[0], region[1], replicate] += parts
            hits[region] += 1
            chart_hits[chart] += 1
            done[replicate] += 1
            if progress is not None:
                progress(dict(completed_draws=int(done.sum()), required_draws=expected,
                    elapsed_seconds=perf_counter()-started))
    return report(True)
