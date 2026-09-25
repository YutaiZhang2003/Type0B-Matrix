#!/usr/bin/env python3
"""Check a relative inverse-square inverted oscillator, not heterotic scattering.

Run with a Python environment containing mpmath and SymPy.
Incoming/outgoing WKB waves are fixed explicitly in wkb_wave.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps = 65
RESULTS = []


def regular_wave(r, energy, exponent, frequency):
    a = exponent / 2 + mp.mpf("0.25") - 1j * energy / (2 * frequency)
    b = exponent + mp.mpf("0.5")
    z = -1j * frequency * r * r / 2
    return r**exponent * mp.exp(1j * frequency * r * r / 4) * mp.hyp1f1(a, b, z)


def reflection(energy, exponent, frequency):
    a0 = exponent / 2 + mp.mpf("0.25")
    eta = energy / (2 * frequency)
    return (mp.exp(-1j * mp.pi * a0)
            * mp.exp(1j * energy / frequency * mp.log(frequency / 2))
            * mp.gamma(a0 - 1j * eta) * mp.rgamma(a0 + 1j * eta))


def wkb_wave(r, energy, frequency, sign):
    wave = r**(-mp.mpf("0.5")) * mp.exp(
        sign * 1j * (frequency * r * r / 4 + energy / frequency * mp.log(r)))
    derivative = wave * (-1 / (2 * r) + sign * 1j * (
        frequency * r / 2 + energy / (frequency * r)))
    return wave, derivative


def fitted_reflection(r, energy, exponent, frequency):
    wave = lambda x: regular_wave(x, energy, exponent, frequency)
    value, derivative = wave(r), mp.diff(wave, r)
    outgoing, outgoing_derivative = wkb_wave(r, energy, frequency, +1)
    incoming, incoming_derivative = wkb_wave(r, energy, frequency, -1)
    # The common determinant cancels in the ratio of fitted coefficients.
    return ((value * incoming_derivative - derivative * incoming)
            / (outgoing * derivative - outgoing_derivative * value))


def record(name, error, tolerance):
    assert error < tolerance, (name, error, tolerance)
    RESULTS.append({"check": name, "passed": True, "absolute_error": float(error)})


frequency = mp.mpf("0.8")
for exponent in map(mp.mpf, ("0.1", "1.1", "2.1", "0.9")):
    for energy in (mp.mpf("-3.2"), mp.mpf("0.7"), mp.mpc("1.2", "0.3")):
        for r in map(mp.mpf, ("0.3", "1.1", "3.2")):
            wave = lambda x: regular_wave(x, energy, exponent, frequency)
            residual = (-mp.diff(wave, r, 2)
                        - frequency**2 * r**2 / 4 * wave(r)
                        + exponent * (exponent - 1) / r**2 * wave(r)
                        - energy * wave(r))
            record(f"ODE s={exponent}, E={energy}, r={r}", abs(residual), mp.mpf("1e-52"))
    for energy in map(mp.mpf, ("-3.2", "0.7", "2.5")):
        record(f"real-energy unitarity s={exponent}, E={energy}",
               abs(abs(reflection(energy, exponent, frequency)) - 1), mp.mpf("1e-55"))
        a_scale = frequency * (exponent + mp.mpf("0.5"))
        ratio = reflection(energy, exponent + 2, frequency) / reflection(energy, exponent, frequency)
        expected = -(a_scale - 1j * energy) / (a_scale + 1j * energy)
        record(f"channel shift s={exponent}, E={energy}", abs(ratio - expected), mp.mpf("1e-54"))

# The WKB extraction is independent of the Kummer connection formula used for R.
for exponent in map(mp.mpf, ("0.1", "0.9", "2.1")):
    energy = mp.mpf("0.7")
    errors = []
    for r in map(mp.mpf, ("40", "80", "160")):
        errors.append(abs(fitted_reflection(r, energy, exponent, frequency)
                          - reflection(energy, exponent, frequency)))
    assert errors[-1] < errors[0] / 5, errors
    record(f"fixed-WKB phase s={exponent}, r=160", errors[-1], mp.mpf("0.001"))

# At the upper-half-plane ratio pole the denominator channel has a zero.
exponent = mp.mpf("0.1")
a_scale = frequency * (exponent + mp.mpf("0.5"))
pole_energy = 1j * a_scale
record("upper ratio pole divides by a zero of R_s", abs(reflection(pole_energy, exponent, frequency)), mp.mpf("1e-55"))
shifted_at_pole = reflection(pole_energy, exponent + 2, frequency)
assert mp.isfinite(shifted_at_pole) and abs(shifted_at_pole) > 0
RESULTS.append({"check": "R_(s+2) finite and nonzero at apparent upper ratio pole", "passed": True})

mu, eps, aa, zz = sp.symbols("mu eps aa zz", real=True)
ratio = -(aa - sp.I * (-mu + eps)) / (aa + sp.I * (-mu + eps))
large_mu = sp.series(ratio.subs(mu, 1 / zz), zz, 0, 3).removeO().expand()
expected_series = 1 - 2 * sp.I * aa * zz - (2 * aa**2 + 2 * sp.I * aa * eps) * zz**2
assert sp.simplify(large_mu - expected_series) == 0
RESULTS.append({"check": "deep-relative-energy expansion through mu^(-2)", "passed": True})

destination = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_radial_reflection_results.json')
destination.write_text(json.dumps(RESULTS, indent=2) + "\n")
print(f"{len(RESULTS)} radial-reflection checks passed; wrote {destination.name}")
