#!/usr/bin/env python3
"""Inspect saved four-R RC integrals without recomputing or rescaling them.

Polynomial diagnostics fit the RAW reduced integrals.  Only the exact
crossing-polynomial basis is reused from the older contact-fit module;
its physical pole subtraction and cubic normalization are not used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
from so7e8_four_ramond_contact_fit import (  # noqa: E402
    four_ramond_contact_basis_block,
    four_ramond_crossing_polynomial_basis,
)


def carray(pairs):
    return np.asarray([complex(*pair) for pair in pairs])


def pairs(values):
    return [[float(z.real), float(z.imag)] for z in values]


def ctext(z, digits=7):
    return f"{z.real:.{digits}g}{z.imag:+.{digits}g}i"


def scaled_fit(matrix, values):
    scale = np.linalg.norm(matrix, axis=0)
    assert np.all(scale > 0)
    coef, _, rank, singular = np.linalg.lstsq(matrix / scale, values, rcond=None)
    return coef / scale, int(rank), float(singular[0] / singular[-1])


def polynomial_diagnostics(momenta, values, degree):
    basis = four_ramond_crossing_polynomial_basis(degree)
    blocks = np.stack([
        four_ramond_contact_basis_block(*p[:3], basis=basis) for p in momenta
    ])
    matrix = blocks.reshape(-1, basis.column_count)
    coef, rank, condition = scaled_fit(matrix, values.ravel())
    fitted = (matrix @ coef).reshape(values.shape)
    predictions, ranks, conditions = [], [], []
    for i in range(len(values)):
        keep = np.arange(len(values)) != i
        fit, loo_rank, loo_condition = scaled_fit(
            blocks[keep].reshape(-1, basis.column_count), values[keep].ravel()
        )
        predictions.append(blocks[i] @ fit)
        ranks.append(loo_rank)
        conditions.append(loo_condition)
    predictions = np.asarray(predictions)
    return {
        "degree": degree,
        "complex_parameter_count": basis.column_count,
        "rank": rank,
        "minimum_leave_one_energy_out_rank": min(ranks),
        "scaled_condition_number": condition,
        "maximum_leave_one_energy_out_condition_number": max(conditions),
        "training_relative_residual": float(np.linalg.norm(fitted-values)/np.linalg.norm(values)),
        "leave_one_energy_out_relative_residual": float(np.linalg.norm(predictions-values)/np.linalg.norm(values)),
        "per_energy_leave_one_out_relative_residual": [
            float(np.linalg.norm(a-b)/np.linalg.norm(b))
            for a, b in zip(predictions, values)
        ],
        "feature_labels": list(basis.labels),
        "coefficients": pairs(coef),
        "leave_one_energy_out_predictions": [pairs(row) for row in predictions],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "data_exports/so7e8_fourpoint_rc_complete_20260911")
    args = parser.parse_args()
    out = args.batch / "fourr_amplitude_inspection"
    out.mkdir(exist_ok=True)
    grouped = {}
    for path in sorted((args.batch / "rc_snapshot/results").glob("task_*.json")):
        data = json.loads(path.read_text())
        task = data["task"]
        if task["kind"] == "fourr_integral":
            grouped.setdefault(task["energy"], {})[task["setting"]] = data
    settings = ("base", "block", "momentum", "moduli", "endpoint", "p_scale")
    assert len(grouped) == 8
    assert all(set(rows) == set(settings) for rows in grouped.values())
    names = list(grouped)
    momenta, arrays = [], {key: [] for key in settings}
    for name in names:
        for setting in settings:
            data = grouped[name][setting]
            p = carray(data["task"]["momenta"])
            assert abs(sum(p[:3])-p[3]) < 1e-12
            assert np.array_equal(p, carray(grouped[name]["endpoint"]["task"]["momenta"]))
            value = carray(data["result"]["coefficients"])
            assert value.shape == (4,) and np.all(np.isfinite(value))
            pieces = sum((carray(v) for v in data["result"]["pieces"].values()), np.zeros(4, complex))
            assert np.allclose(pieces, value, atol=1e-13, rtol=1e-11)
            arrays[setting].append(value)
        momenta.append(p)
    momenta = np.asarray(momenta)
    arrays = {key: np.asarray(values) for key, values in arrays.items()}
    values = arrays["endpoint"]
    changes = {
        f"{a}_to_{b}": np.linalg.norm(arrays[a]-arrays[b], axis=1)/np.linalg.norm(arrays[b], axis=1)
        for a, b in zip(settings[:-1], settings[1:])
    }
    component_change = abs(values-arrays["moduli"])/abs(values)
    cayley = np.asarray((0, 3, 1, 0))
    projection = (values @ cayley) / np.dot(cayley, cayley)
    cayley_residual = np.linalg.norm(values-projection[:, None]*cayley, axis=1)/np.linalg.norm(values, axis=1)
    fits = {
        setting: [polynomial_diagnostics(momenta, arrays[setting], d) for d in range(4)]
        for setting in ("moduli", "endpoint")
    }
    report = {
        "source": str(args.batch / "rc_snapshot/results"),
        "source_manifest_sha256": grouped[names[0]]["endpoint"].get("source_manifest_sha256"),
        "process": "Psi(p4) -> Psi_tilde(p1) Psi_tilde(p2) Psi_tilde(p3), p4=p1+p2+p3",
        "tensor_basis": "T_r=(C gamma^{a1...ar})_12 (C gamma_{a1...ar})_34 / r!, r=0,1,2,3",
        "normalization": "Raw reduced dP/pi d2z integral; no pole subtraction or external-leg rescaling",
        "status": "provisional; bootstrap-fixed sewing cocycle; physical cubic/BPZ audit pending",
        "variation_note": "Finite grid/remapping differences, not statistical errors or certified error bounds. All vector norms here are Euclidean coefficient norms in the stated basis.",
        "energies": [],
        "aggregate_endpoint_remapping_change": float(np.linalg.norm(values-arrays['moduli'])/np.linalg.norm(values)),
        "polynomial_diagnostics": fits,
    }
    csv_fields = ["energy"]
    csv_fields += [f"p{k}_{part}" for k in range(1, 5) for part in ("re", "im")]
    csv_fields += [f"I{r}_{part}" for r in range(4) for part in ("re", "im")]
    csv_fields += ["endpoint_vector_change", "cayley_projection_residual"]
    csv_fields += [f"I{r}_endpoint_relative_change" for r in range(4)]
    csv_rows = []
    for i, name in enumerate(names):
        report["energies"].append({
            "name": name, "momenta": pairs(momenta[i]), "coefficients": pairs(values[i]),
            "convergence_relative_changes": {key: float(v[i]) for key, v in changes.items()},
            "endpoint_component_relative_changes": component_change[i].tolist(),
            "cayley_projection_coefficient": pairs([projection[i]])[0],
            "cayley_projection_relative_residual": float(cayley_residual[i]),
            "rank0_over_7rank3": pairs([values[i, 0]/(7*values[i, 3])])[0],
        })
        row = {"energy": name}
        row.update({f"p{k}_{part}": getattr(p, attr) for k, p in enumerate(momenta[i], 1) for part, attr in (("re", "real"), ("im", "imag"))})
        row.update({f"I{r}_{part}": getattr(v, attr) for r, v in enumerate(values[i]) for part, attr in (("re", "real"), ("im", "imag"))})
        row.update(endpoint_vector_change=changes["moduli_to_endpoint"][i], cayley_projection_residual=cayley_residual[i])
        row.update({f"I{r}_endpoint_relative_change": component_change[i, r] for r in range(4)})
        csv_rows.append(row)
    with (out / "fourr_coefficients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (out / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for rank, ax in enumerate(axes.flat):
        for offset, attr, color in ((-.10, "real", "#2463a5"), (.10, "imag", "#c36a12")):
            low = getattr(arrays["moduli"][:, rank], attr)*1e3
            high = getattr(values[:, rank], attr)*1e3
            for xx, a, b in zip(x+offset, low, high):
                ax.plot([xx, xx], [a, b], color=color, alpha=.7, linewidth=2)
            ax.scatter(x+offset, low, s=62, facecolors="none", edgecolors=color, linewidths=1.2)
            ax.scatter(x+offset, high, s=22, color=color, zorder=3)
        ax.axhline(0, color=".7", lw=.7)
        ax.set_title(rf"$I_{rank}$", loc="left", fontweight="bold")
        ax.set_ylabel("Coefficient × 1,000")
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], marker="o", ls="none", color=c, label=lab) for c, lab in (("#2463a5", "Real part"), ("#c36a12", "Imaginary part"))]
    handles += [Line2D([], [], marker="o", ls="none", color=".3", markerfacecolor="none", markersize=8, label="ultra grid"), Line2D([], [], marker="o", ls="none", color=".3", markersize=4, label="ultra grid, endpoint power 5")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("SO(7) × E₈ four-fermion amplitude: completed reduced integrals\nEight distinct complex energy triples; horizontal labels are sample IDs, not an energy axis", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fourr_coefficients.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    labels = ("Block truncation", "Momentum nodes", "Modulus grid", "Endpoint remapping", "Momentum remapping")
    for key, label in zip(changes, labels):
        axes[0].scatter(x, changes[key]*100, label=label, s=32)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, names, rotation=30, ha="right")
    axes[0].set_ylabel("Change in coefficient vector (%)")
    axes[0].set_title("Convergence comparisons", loc="left")
    axes[0].grid(axis="y", alpha=.2)
    convergence_handles, convergence_labels = axes[0].get_legend_handles_labels()
    fig.legend(convergence_handles, convergence_labels, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    axes[1].bar(x-.18, cayley_residual*100, width=.36, label="Distance from f(p) × (0,3,1,0)", color="#2463a5")
    axes[1].bar(x+.18, changes["moduli_to_endpoint"]*100, width=.36, label="Endpoint remapping change", color="#c36a12")
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    axes[1].set_ylabel("Relative coefficient-vector norm (%)")
    axes[1].set_title("Additional tensor structure remains visible", loc="left")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Numerical variations are diagnostics, not certified error bounds", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fourr_convergence.{ext}", dpi=180)
    plt.close(fig)

    lines = [
        "# Completed four-fermion amplitudes: inspection", "",
        "All 48 four-R integrals (eight energy choices times six settings) are available.", "",
        "The ordered process is `Psi(p4) -> Psi_tilde(p1) Psi_tilde(p2) Psi_tilde(p3)`, with `p4=p1+p2+p3`.", "",
        "The reduced amplitude is `A = sum_{r=0}^3 I_r T_r`, where `T_r = (C gamma^{a1...ar})_12 (C gamma_{a1...ar})_34/r!`.",
        "The displayed numbers are the full spectral and sphere integrals, not just pole terms or fixed-modulus kernels. The measure is `dP/pi d2z`.",
        "The common sphere normalization, string coupling, energy delta function, and asymptotic external reflection phases are not included. No energy-dependent rescaling or pole subtraction was applied.", "",
        "These remain provisional physical amplitudes: the discrete sewing cocycle was fixed by crossing, and the independent physical cubic/BPZ normalization audit is unfinished.", "",
        "## Data", "", "Reference values use order 19, 64 momentum nodes, and the `ultra_p5` grid. All momenta are complex; these are not a one-dimensional energy scan.", "",
        "| Sample | p1 | p2 | p3 | p4 |", "|---|---|---|---|---|",
    ]
    lines += ["| "+" | ".join([name]+[ctext(p) for p in momenta[i]])+" |" for i, name in enumerate(names)]
    lines += ["", "| Sample | I0 | I1 | I2 | I3 |", "|---|---|---|---|---|"]
    lines += ["| "+" | ".join([name]+[ctext(v) for v in values[i]])+" |" for i, name in enumerate(names)]
    lines += ["", "![Raw coefficients](fourr_coefficients.png)", "", "## Stability and tensor structure", "",
        "All norms in this report are Euclidean norms of the four coefficients in the explicitly stated basis, not spin-summed probabilities.", "",
        "The order 15-to-19 and 48-to-64 momentum-node comparisons change the coefficient vector by at most `3.8e-10` and `2.5e-8` fractionally. The remaining visible dependence is on the integration grid near vertex collisions.", "",
        "The ultra-to-ultra_p5 comparison changes the radial endpoint mapping at fixed node counts. It is a sensitivity check, not a rigorous error bar; even the highest setting is not proven converged.", "",
        "| Sample | Fine to ultra change | Endpoint remapping change | Largest component endpoint change | Distance from any scalar × Cayley |", "|---|---:|---:|---:|---:|"]
    for i, name in enumerate(names):
        lines.append(f"| {name} | {100*changes['momentum_to_moduli'][i]:.3f}% | {100*changes['moduli_to_endpoint'][i]:.3f}% | {100*max(component_change[i]):.2f}% | {100*cayley_residual[i]:.2f}% |")
    lines += ["", "The amplitude is nonzero. It is not well described by a single arbitrary energy function multiplying the constant Cayley tensor `3T1+T2`: rank-zero/rank-three and the orthogonal rank-one/rank-two combination remain visible under remapping. This is evidence within the current prescription, not a certified exclusion of a physical model.", "",
        "The approximate relation `I0 ≈ 7 I3` is not exact in the data: deviations are resolved relative to their remapping changes. It should not be imposed as an identity.", "", "![Convergence](fourr_convergence.png)", "",
        "## Exploratory energy fits", "",
        "We fit the RAW reduced coefficient vectors to complete crossing-covariant polynomials of total degree 0–3 in the three outgoing momenta. Each polynomial satisfies the existing exact outgoing-fermion crossing convention. This uses only the algebraic basis from `Codes/so7e8_four_ramond_contact_fit.py`, bypassing that module's older physical pole subtraction.", "",
        "Validation removes all four tensor coefficients of one complete energy point, fits the other seven points, and predicts the omitted point. No algebraically generated crossing images count as extra data. Fit coefficients are unrestricted complex numbers. Residuals are pooled relative norms, so larger coefficients carry more weight.", "",
        "| Maximum degree | Complex fit coefficients | Training residual (endpoint grid) | Omitted-energy prediction residual (endpoint grid) | Omitted-energy prediction residual (ultra grid) |", "|---:|---:|---:|---:|---:|"]
    for d in range(4):
        a, b = fits["endpoint"][d], fits["moduli"][d]
        lines.append(f"| {d} | {a['complex_parameter_count']} | {100*a['training_relative_residual']:.3f}% | {100*a['leave_one_energy_out_relative_residual']:.3f}% | {100*b['leave_one_energy_out_relative_residual']:.3f}% |")
    lines += ["", "The quadratic and cubic fits are promising local descriptions. They do not identify an exact amplitude: only eight generic complex kinematics are sampled, the degree-three model has 14 complex parameters, numerical errors are correlated, and a meromorphic function can look polynomial over a restricted region. The scaled design condition numbers are about 166 (quadratic) and 3,624 (cubic); individual fitted coefficients should not be interpreted as exact constants.", "",
        "A reliable analytic reconstruction needs improved collision-region quadrature, independently computed validation momenta (including scans that vary one energy or scale at a time), and the physical pole-residue audit before interpreting a pole-subtracted remainder. No new cluster jobs were submitted by this inspection.", "",
        "## Reproduction", "", "Run `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python inspect_so7e8_fourr_rc_results.py` from the repository root. CSV contains full-precision reduced coefficients; JSON includes all convergence and fit diagnostics. The script checks momentum conservation, all 48 expected inputs, and agreement between summed integration pieces and stored totals.", "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(out), "fourr_results": 48, "energies": names, "pooled_endpoint_change": report["aggregate_endpoint_remapping_change"], "fits": [{key: v for key, v in d.items() if key in ("degree", "complex_parameter_count", "training_relative_residual", "leave_one_energy_out_relative_residual")} for d in fits["endpoint"]]}, indent=2))


if __name__ == "__main__":
    main()
