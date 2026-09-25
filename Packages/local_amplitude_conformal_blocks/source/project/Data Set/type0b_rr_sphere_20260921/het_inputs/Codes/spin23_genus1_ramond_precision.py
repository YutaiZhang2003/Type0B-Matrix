"""Evaluate the existing two-Virasoro Ramond recursion with mpmath scalars.

The finite branch states, Ward tensors, Virasoro recurrence and auxiliary
character division are the same as in the binary64 implementation. Function
clones have private numeric globals; no production module is monkey-patched.
"""
from functools import lru_cache
from types import FunctionType, SimpleNamespace
import inspect
import math

import mpmath as mp
import numpy as np
import sympy as sp


def _number(value=0, imaginary=None):
    if imaginary is not None:
        return mp.mpc(value, imaginary)
    if isinstance(value, sp.Basic):
        real, imag = value.as_real_imag()
        return mp.mpc(str(sp.N(real, mp.mp.dps)), str(sp.N(imag, mp.mp.dps)))
    return mp.mpc(value)


class _Arrays:
    """Small object arrays; only branch eigenvectors require linear algebra."""
    complex128 = object
    ndarray = np.ndarray

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def array(values, dtype=None, **kwargs):
        return np.array(values, dtype=object, **kwargs)

    asarray = array

    @staticmethod
    def zeros(shape, dtype=None):
        return np.full(shape, mp.mpc(0), dtype=object)

    @staticmethod
    def eye(size, dtype=None):
        return np.array(mp.eye(size).tolist(), dtype=object)

    @staticmethod
    def _svd(array):
        # This caller only uses the isolated null vector and the next singular
        # value. Binary64 SVD supplies a gauge and a rank diagnostic; solve ALL
        # eigenvector equations again in the requested precision. The border
        # fixes normalization without dropping a potentially important row.
        u, s, v = np.linalg.svd(np.asarray(array, dtype=complex))
        n = len(array)
        bordered = mp.matrix(n+1)
        for i in range(n):
            for j in range(n):
                bordered[i, j] = array[i, j]
            bordered[i, n] = _number(u[i, -1])
            bordered[n, i] = _number(v[-1, i])
        rhs = mp.matrix(n+1, 1)
        rhs[n] = 1
        solution = mp.lu_solve(bordered, rhs)
        precise_v = np.asarray(v, dtype=object)
        precise_v[-1, :] = [mp.conj(solution[i]) for i in range(n)]
        return u, s, precise_v

    @staticmethod
    def _solve(a, b):
        x = mp.lu_solve(mp.matrix(a.tolist()), mp.matrix(b.tolist()))
        return np.array(list(x), dtype=object)

    @staticmethod
    def _norm(a, ord=None):
        if a.ndim == 2 and ord == 2:
            return np.linalg.norm(np.asarray(a, dtype=complex), ord=2)
        return mp.norm(mp.matrix(a.tolist()))

    linalg = property(lambda self: SimpleNamespace(
        svd=self._svd, solve=self._solve, norm=self._norm))


class _Sympy:
    def __getattr__(self, name):
        return getattr(sp, name)

    @staticmethod
    def N(value, digits=15):
        return sp.N(value, max(digits, mp.mp.dps))

    @staticmethod
    def lambdify(*args, **kwargs):
        if kwargs.get("modules") == "numpy":
            kwargs["modules"] = "mpmath"
        return sp.lambdify(*args, **kwargs)


def _clone(function, scope):
    original = inspect.unwrap(function)
    clone = FunctionType(original.__code__, scope, original.__name__,
                         original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    if hasattr(function, "cache_parameters"):
        clone = lru_cache(**function.cache_parameters())(clone)
    return clone


@lru_cache(maxsize=None)
def _ward_template(infinity, middle, zero):
    # These compiled symbolic polynomials contain no numerical inputs or
    # precision-dependent values, so they can be shared by all contours.
    import spin23_two_virasoro_ramond as embedded
    builder = _clone(embedded._rr_three_point_numeric_template,
                     dict(vars(embedded), sp=_Sympy()))
    return builder(infinity, middle, zero)


def _engine():
    import spin23_two_virasoro_ramond as embedded
    import spin23_genus1_branch_recursion as branch
    import spin23_genus1_recursion as conventions
    from spin23_genus1_virasoro_collision import high_precision_engine

    arithmetic = dict(complex=_number, np=_Arrays(), cmath=mp,
                      math=SimpleNamespace(**dict(vars(math), sqrt=mp.sqrt)),
                      sp=_Sympy())
    scope = dict(vars(embedded), **arithmetic)
    for name, function in vars(embedded).items():
        if inspect.isfunction(inspect.unwrap(function)) and getattr(
                function, "__module__", None) == embedded.__name__:
            scope[name] = _clone(function, scope)
    cs = dict(vars(conventions), complex=_number)
    for name in ("super_liouville_central_charge", "ns_liouville_weight"):
        scope[name] = _clone(getattr(conventions, name), cs)
    scope["_rr_three_point_numeric_template"] = _ward_template

    outer = dict(vars(branch), **arithmetic)
    outer["embedded_branch_state"] = scope["embedded_branch_state"]
    outer["oriented_branching_coefficient"] = scope["oriented_branching_coefficient"]
    for name in ("external_expansion", "vertex", "_convolve", "generic_ramond_tables"):
        outer[name] = _clone(getattr(branch, name), outer)
    ordinary = high_precision_engine()

    @lru_cache(maxsize=16384)
    def coefficients(c, internal, external, cutoffs, total_cutoff=None):
        return tuple(ordinary(c=c, internal_weights=internal,
            external_weights=external, maximum_levels=cutoffs,
            maximum_total_level=total_cutoff).items())

    outer["ordinary_coefficients"] = coefficients
    return outer["generic_ramond_tables"]


def contour_tables(*, radius, samples, digits=40, **options):
    """Return every form/sign on one log(b) contour without early rounding."""
    if type(digits) is not int or digits < 30:
        raise ValueError("at least 30 decimal digits required")
    if type(samples) is not int or samples < 8 or samples % 2:
        raise ValueError("even sample count of at least eight required")
    with mp.workdps(digits):
        engine = _engine()
        momenta = tuple(map(_number, options["internal_momenta"]))
        coincident = any(abs(p-q) < mp.mpf("1e-8")*max(1, abs(p), abs(q))
            for i, p in enumerate(momenta) for q in momenta[i+1:])
        scale = min(mp.mpf(1), min(map(abs, momenta)))
        count = samples if coincident else samples//2
        total = {}
        for j in range(count):
            t = mp.mpf(str(radius))*mp.exp(2j*mp.pi*(mp.mpf(j)+mp.mpf(".5"))/samples)
            current = options
            if coincident:
                current = dict(options, internal_momenta=tuple(
                    p+mp.mpf(".1")*(i+1)*scale*t for i, p in enumerate(momenta)))
            table, forms, signs, _ = engine(b=mp.exp(t), **current)
            for key, value in table.items():
                total[key] = total.get(key, 0) + value/count
        levels = tuple(total)
        values = np.asarray(np.stack([total[k] for k in levels], axis=-1), dtype=complex)
    if not np.isfinite(values).all():
        raise ArithmeticError("non-finite high-precision Ramond coefficients")
    return np.array(levels, dtype=int), values, forms, signs, coincident


def checked_tables(*, radius=.15, check_radius=.20, tolerance=2e-7,
                   sample_counts=(16, 32, 64), digits=40, check_digits=60,
                   **options):
    """Require radius, angular-resolution and precision agreement separately.

    Higher precision permits smaller contours, whose angular quadrature
    converges faster; cancellation at intermediate b remains fully resolved.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite tolerance required")
    if check_digits <= digits:
        raise ValueError("the confirming precision must be higher")
    if not sample_counts or tuple(sorted(set(sample_counts))) != tuple(sample_counts):
        raise ValueError("strictly increasing continuation sample counts required")
    if any(type(n) is not int or n < 8 or n % 2 for n in sample_counts):
        raise ValueError("even continuation sample counts of at least eight required")
    if any(not math.isfinite(r) or r <= 0 for r in (radius, check_radius)) or radius == check_radius:
        raise ValueError("distinct positive finite radii required")
    cache = {}
    history = []

    def evaluate(r, n, d):
        key = (r, n, d)
        if key not in cache:
            cache[key] = contour_tables(radius=r, samples=n, digits=d, **options)
        return cache[key]

    def compare(a, b):
        if not np.array_equal(a[0], b[0]) or a[2:4] != b[2:4]:
            raise ArithmeticError("precision continuation changed coefficient support or labels")
        errors = np.max(abs(a[1]-b[1]), axis=-1)
        scale = np.maximum(1., np.max(abs(a[1]), axis=-1))
        return float(np.max(errors)), float(np.max(errors/scale))

    for n in sample_counts:
        primary = evaluate(radius, n, digits)
        check = evaluate(check_radius, n, digits)
        absolute, radial = compare(primary, check)
        attempt = dict(samples=n, digits=digits, scaled_finite_part_error=radial)
        history.append(attempt)
        if radial > tolerance:
            continue
        refined = evaluate(radius, 2*n, digits)
        _, angular = compare(refined, primary)
        attempt["angular_refinement_error"] = angular
        if angular > tolerance:
            continue
        precise = evaluate(radius, 2*n, check_digits)
        _, rounding = compare(precise, refined)
        attempt["precision_refinement_error"] = rounding
        if rounding > tolerance:
            continue
        return precise[:4] + (dict(
            backend="two_virasoro_recursion_mpmath", digits=check_digits,
            check_digits=digits, samples=2*n, radius_check_samples=n,
            radius=radius, check_radius=check_radius,
            maximum_absolute_finite_part_error=absolute,
            scaled_finite_part_error=radial, angular_refinement_error=angular,
            precision_refinement_error=rounding,
            joint_momentum_continuation=precise[4], precision_attempts=history),)
    raise ArithmeticError("high-precision Ramond continuation failed: " + str(history))
