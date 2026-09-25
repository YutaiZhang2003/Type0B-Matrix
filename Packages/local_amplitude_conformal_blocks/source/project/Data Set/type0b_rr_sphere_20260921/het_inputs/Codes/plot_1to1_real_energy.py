"""Plot the analytically continued vector 1->1 amplitude at g=1.

All momentum factors are retained.  We compute the exact Taylor coefficients
at omega=0 by the integral hierarchy and reexpand in a conformal coordinate
for the two-cut energy plane.  Pade approximants accelerate this conformal
series at high energy; three rational orders and the direct integral at
anchor points check the evaluation beyond omega=3/2.

Dependencies: mpmath, numpy, scipy and matplotlib.  The Tricomi/Volterra
anchor checks import check_1to1_omega_analyticity from this directory.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Machine Note/data_exports/heterotic_1to1_real_omega_g1_20260916"
FIGURES = ROOT / "Machine Note/figures"


def taylor_coefficients(degree, lam):
    """Coefficients of S(a)=i F(-ia,lambda), a=i*omega.

    J_r(a)=i I_r(-ia), V_r=r! J_r, W_r=V_r'.  The exact momentum
    hierarchy gives (a+r) W_r'=r^2 W_(r-1).  We sum B_r*V_r with
    B_r=lambda^r ((a)_r/r!)^2.  Terms r>degree-3 cannot contribute.
    """
    zero = mp.mpf(0)
    coefficients = [zero] * (degree + 1)
    coefficients[1] = mp.mpf(1)
    previous_w = [mp.mpf(1)] + [zero] * (degree - 1)
    previous_b = [mp.mpf(1)]
    for r in range(1, degree - 2):
        w = [zero] * degree
        rr = mp.mpf(r)
        for n in range(r - 1, degree - 1):
            w[n + 1] = (rr * previous_w[n] - (mp.mpf(n) / rr) * w[n]) / (n + 1)
        top = min(2 * r, degree - r - 1)
        # Retain B through degree, since later updates need its low powers.
        b = [zero] * min(2 * r + 1, degree + 1)
        factor = lam / (r * r)
        for k, value in enumerate(previous_b):
            if k < len(b):
                b[k] += factor * (r - 1) ** 2 * value
            if k + 1 < len(b):
                b[k + 1] += factor * 2 * (r - 1) * value
            if k + 2 < len(b):
                b[k + 2] += factor * value
        v = [zero] + [w[n - 1] / n for n in range(1, degree + 1)]
        for k in range(2, top + 1):
            if not b[k]:
                continue
            for j in range(r + 1, degree - k + 1):
                coefficients[k + j] += b[k] * v[j]
        previous_w, previous_b = w, b
        if r % 80 == 0:
            print(f"Momentum hierarchy: {r}/{degree - 3} contractions", flush=True)
    return coefficients


def conformal_coefficients(coefficients, rho):
    """Compose with a(u)=4*rho*u/[(rho+1)(1+u^2)+2(1-rho)u]."""
    c = 4 * rho / (rho + 1)
    d = 2 * (1 - rho) / (rho + 1)
    polynomial = [coefficients[-1]]
    for coefficient in reversed(coefficients[:-1]):
        product = [mp.mpf(0)] * (len(polynomial) + 1)
        for n in range(1, len(product)):
            product[n] = c * polynomial[n - 1] - d * product[n - 1]
            if n >= 2:
                product[n] -= product[n - 2]
        product[0] = coefficient
        polynomial = product
    return polynomial


def evaluate(omega, coefficients, rho=mp.mpf('1.5')):
    a = mp.j * omega
    root = mp.sqrt((1 + a) / (1 - a / rho))
    coordinate = (root - 1) / (root + 1)
    S = mp.polyval(coefficients[::-1], coordinate)
    amplitude = -mp.j * mp.exp(-a * mp.log(mp.mpf('1.5'))) * S
    return amplitude, coordinate


def evaluate_pade(omega, rational):
    a = mp.j * omega
    root = mp.sqrt((1 + a) / (1 - a / mp.mpf('1.5')))
    coordinate = (root - 1) / (root + 1)
    numerator, denominator = rational
    S = mp.polyval(numerator[::-1], coordinate) / mp.polyval(denominator[::-1], coordinate)
    amplitude = -mp.j * mp.exp(-a * mp.log(mp.mpf('1.5'))) * S
    return amplitude, coordinate


def anchor_checks(rational):
    from check_1to1_omega_analyticity import checked_amplitude

    records = []
    for omega in (0.75, 1.5, 3., 5.):
        reference, _ = evaluate_pade(mp.mpf(omega), rational)
        row = {"omega": omega, "continued_amplitude": [mp.nstr(mp.re(reference), 35), mp.nstr(mp.im(reference), 35)]}
        for order in (192, 256):
            with mp.workdps(40):
                value, diagnostics = checked_amplitude(omega, 1., 0, order=order)
            row[str(order)] = {"amplitude": [value.real, value.imag],
                               "absolute_difference": float(abs(value - reference)),
                               "momentum_refinement_change": diagnostics['richardson_refinement_change']}
        records.append(row)
        print("Anchor:", json.dumps(row), flush=True)
    return records


def render(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    omega = np.asarray([row['omega'] for row in rows])
    amplitude = np.asarray([complex(row['amplitude_re'], row['amplitude_im']) for row in rows])
    normalized = np.asarray([row['normalized_magnitude'] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.15})
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.105, right=0.975)
    fig.suptitle(r"Analytically continued $1\!\to\!1$ amplitude at $g=1$", y=0.965, fontsize=19)
    fig.text(0.5, 0.908, r"$\mathcal{A}(\omega)=(3/2)^{-i\omega}F(\omega,2/3)$  ·  coefficient of $\delta^{ab}$",
             ha='center', color='#4b5563', fontsize=12)
    axes[0].plot(omega, amplitude.real, color='#156b9a', linewidth=2, label=r'$\mathrm{Re}\,\mathcal{A}$')
    axes[0].plot(omega, amplitude.imag, color='#d66536', linewidth=2, label=r'$\mathrm{Im}\,\mathcal{A}$')
    axes[0].plot(omega, abs(amplitude), color='#273341', linewidth=1.7, linestyle='--', label=r'$|\mathcal{A}|$')
    axes[0].axhline(0, color='#b7bec6', linewidth=.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=.96, ncol=3)
    axes[1].plot(omega, normalized, color='#6f4ba8', linewidth=2,
                 label=r'$|\mathcal{A}|/\omega$')
    axes[1].axhline(1, color='#9aa3ac', linestyle=':', linewidth=1)
    axes[1].set_ylabel(r'$|\mathcal{A}|/\omega$')
    axes[1].set_xlabel(r'Positive real energy $\omega$')
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.axvspan(1.5, omega[-1], color='#e6eef4', alpha=.45, zorder=-10)
        ax.axvline(1.5, color='#7b8794', linestyle=(0, (4, 3)), linewidth=1)
        ax.grid(axis='y', color='#e1e6eb', linewidth=.6)
        ax.set_xlim(0, omega[-1])
    axes[1].text(1.5 + .12, .94, r'$\omega=1.5$: original series boundary',
                 transform=axes[1].get_xaxis_transform(), ha='left', va='top',
                 fontsize=10, color='#526273')
    fig.text(.105, .035, 'Shading: beyond the contraction-series convergence radius.  The endpoint at zero is the analytic limit.',
             fontsize=9, color='#596572')
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ('png', 'svg', 'pdf'):
        path = FIGURES / f'heterotic_1to1_g1_real_omega.{suffix}'
        fig.savefig(path, dpi=210)
        paths.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int, default=400)
    parser.add_argument('--dps', type=int, default=300)
    parser.add_argument('--omega-max', type=float, default=10.)
    parser.add_argument('--samples', type=int, default=501)
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--reuse-coefficients', action='store_true')
    args = parser.parse_args()
    if args.degree < 120 or args.degree % 2:
        raise ValueError('Choose an even coefficient degree of at least 120.')
    mp.mp.dps = args.dps
    DATA.mkdir(parents=True, exist_ok=True)
    coefficient_path = DATA / 'conformal_coefficients.json'
    started = time.monotonic()
    if args.reuse_coefficients:
        cached = json.loads(coefficient_path.read_text())
        if cached['degree'] != args.degree or cached['dps'] < args.dps:
            raise ValueError('Cached coefficients have incompatible order or precision.')
        coefficients = [mp.mpf(value) for value in cached['coefficients']]
    else:
        series = taylor_coefficients(args.degree, mp.mpf(2) / 3)
        assert abs(series[4] - mp.mpf(1)/3) < mp.mpf(10)**(-args.dps+10)
        assert abs(series[5] + mp.mpf(2)/27) < mp.mpf(10)**(-args.dps+10)
        coefficients = conformal_coefficients(series, mp.mpf('1.5'))
        coefficient_path.write_text(json.dumps({
            "degree": args.degree, "dps": args.dps,
            "function": "S(a)=i F(-ia,2/3)",
            "inverse_map": "a=(12/5)*u/(1-(2/5)*u+u^2)",
            "coefficients": [str(v) for v in coefficients]}, indent=2) + '\n')
    print(f"Coefficients ready in {time.monotonic()-started:.1f} seconds", flush=True)
    pade_orders = [args.degree//2-40, args.degree//2-8, args.degree//2]
    rationals = {}
    for order in pade_orders:
        print(f"Constructing Pade [{order}/{order}]", flush=True)
        numerator, denominator = mp.pade(coefficients, order, order)
        rationals[order] = (numerator, denominator)
        (DATA / f'pade_{order}.json').write_text(json.dumps({
            "dps": args.dps, "p": [str(v) for v in numerator],
            "q": [str(v) for v in denominator]}, indent=2) + '\n')
    rows = []
    series_comparisons = []
    for value in np.linspace(0, args.omega_max, args.samples):
        omega = mp.mpf(float(value))
        amplitude, coordinate = evaluate_pade(omega, rationals[pade_orders[-1]])
        coarser, _ = evaluate_pade(omega, rationals[pade_orders[-2]])
        coarsest, _ = evaluate_pade(omega, rationals[pade_orders[0]])
        if omega <= 5:
            direct_series, _ = evaluate(omega, coefficients)
            series_comparisons.append(float(abs(amplitude-direct_series)))
        rows.append({"omega": float(omega),
                     "amplitude_re": float(mp.re(amplitude)), "amplitude_im": float(mp.im(amplitude)),
                     "magnitude": float(abs(amplitude)),
                     "normalized_magnitude": float(abs(amplitude)/omega) if omega else 1.,
                     "conformal_modulus": float(abs(coordinate)),
                     "coarse_order_refinement_change": float(abs(coarser-coarsest)),
                     "order_refinement_change": float(abs(amplitude-coarser))})
    maximum_difference = max(row['order_refinement_change'] for row in rows)
    if not np.isfinite(maximum_difference) or maximum_difference > 1e-7:
        raise ArithmeticError(f'Pade order refinement is insufficient: {maximum_difference}')
    with (DATA / 'amplitude_g1.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"g": 1, "lambda": "2/3", "omega_interval": [0,args.omega_max],
               "zero_endpoint": "analytic limit", "sample_count": args.samples,
               "method": "Pade-accelerated conformal Taylor expansion with exact momentum-integral coefficients",
               "coefficient_degree": args.degree, "pade_orders": pade_orders, "working_dps": args.dps,
               "max_order_refinement_change": maximum_difference,
               "max_coarse_order_refinement_change": max(row['coarse_order_refinement_change'] for row in rows),
               "max_difference_from_unaccelerated_conformal_series_through_omega_5": max(series_comparisons),
               "error_interpretation": "Empirical comparisons, not interval-certified error bounds.",
               "anchors": anchor_checks(rationals[pade_orders[-1]])}
    if not args.data_only:
        summary['figures'] = render(rows)
    (DATA / 'plot_checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='anchors'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
