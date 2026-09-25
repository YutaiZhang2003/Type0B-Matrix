#!/usr/bin/env python3
"""Compare the supplied four-fermion formula to saved RC integrals.

This does not assert a physical Ramond external-state/LSZ normalization.
The screenshot uses incoming-first indices alpha,beta,gamma,delta.  Its
Cayley tensor is taken to map to +(3*T1+T2)/4 in the evaluator's ordering
beta,gamma,delta,alpha; both possible Cayley signs are checked explicitly.
"""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
sys.path.insert(0, str(ROOT / "tests"))
from so7e8_four_ramond_assembly import spin7_pairing_change_matrix
from test_so7e8_four_ramond_assembly import SO7E8FourRamondAssemblyTests


def complex_pairs(values):
    return [[float(v.real), float(v.imag)] for v in values]


def read_complex(pairs):
    return np.asarray([complex(*v) for v in pairs])


def main():
    batch = ROOT / "data_exports/so7e8_fourpoint_rc_complete_20260911"
    output = batch / "fourr_screenshot_comparison"
    output.mkdir(exist_ok=True)
    # Each column represents an INCOMING-FIRST tensor in the fixed
    # OUTGOING-FIRST gamma-rank basis. Rank-zero C is a symmetric metric.
    d1 = spin7_pairing_change_matrix("03|12")[:, 0]
    d2 = spin7_pairing_change_matrix("02|13")[:, 0]
    d3 = np.eye(4)[:, 0]
    omega = np.asarray([0, 3, 1, 0]) / 4
    transform = np.column_stack([d1, d2, d3, omega])
    weight = np.asarray([1, 7, 21, 35])

    # Independently check the two delta rearrangements with explicit
    # Clifford tensors, including the finite normalization of Omega.
    tensors = np.asarray(SO7E8FourRamondAssemblyTests._clifford_tensor_basis())
    for column, permutation in ((d1, (0, 2, 3, 1)), (d2, (0, 2, 1, 3))):
        assert np.allclose(np.einsum("r,rabcd->abcd", column, tensors), tensors[0].transpose(permutation))
    cayley = np.einsum("r,rabcd->abcd", omega, tensors)
    assert np.isclose(np.vdot(cayley, cayley), 336)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        assert np.allclose(cayley, -cayley.swapaxes(a, b))

    def norm(v):
        return float(np.sqrt(np.sum(weight * abs(v)**2)))

    def fit_scale(prediction, numerical):
        return np.vdot(prediction*weight, numerical)/np.vdot(prediction*weight, prediction)

    records, numerical_all, h_brackets, roots = [], [], [], []
    for energy_index in range(8):
        results = [json.loads((batch / f"rc_snapshot/results/task_{6*energy_index+j:04d}.json").read_text()) for j in range(6)]
        p = read_complex(results[4]["task"]["momenta"])
        x, y, z, E = p
        assert abs(x+y+z-E) < 1e-12
        a = (y-z)/(1+1j*(y+z))
        b = -(x-z)/(1+1j*(x+z))
        c = (x-y)/(1+1j*(x+y))
        bracket = transform @ np.asarray([a, b, c, -1j])
        reversed_omega = transform @ np.asarray([a, b, c, 1j])
        H = 1+2j*E
        root = np.prod(np.sqrt(p))  # Continue each positive-real leg root.
        prediction = -1j*math.pi/32*H*bracket
        values = np.asarray([read_complex(r["result"]["coefficients"]) for r in results])
        final = values[4]
        # An independent scalar at each kinematic point eliminates every
        # common leg factor and tests only relative tensor coefficients.
        local_scale = fit_scale(bracket, final)
        wrong_scale = fit_scale(reversed_omega, final)
        record = {
            "energy": results[4]["task"]["energy"],
            "momenta": complex_pairs(p),
            "coefficients_raw": complex_pairs(final),
            "coefficients_candidate": complex_pairs(prediction),
            "incoming_delta_delta_Omega_coefficients_raw": complex_pairs(np.linalg.solve(transform, final)),
            "sqrt_W_branch": complex_pairs([root])[0],
            "raw_candidate_relative_residual": norm(final-prediction)/norm(final),
            "raw_candidate_component_relative_residual": (abs(final-prediction)/abs(final)).tolist(),
            "tensor_shape_relative_residual": norm(final-local_scale*bracket)/norm(final),
            "opposite_Omega_tensor_shape_relative_residual": norm(final-wrong_scale*reversed_omega)/norm(final),
            "endpoint_grid_relative_change": norm(final-values[3])/norm(final),
            "relative_residual_by_setting": {r["task"]["setting"]: norm(v-prediction)/norm(v) for r, v in zip(results, values)},
        }
        records.append(record)
        numerical_all.append(final)
        h_brackets.append(H*bracket)
        roots.append(root)
    numerical_all, h_brackets, roots = map(np.asarray, (numerical_all, h_brackets, roots))
    pooled = {}
    for name, model in (("H_times_bracket", h_brackets), ("H_times_sqrtW_times_bracket", roots[:, None]*h_brackets)):
        scale = fit_scale(model, numerical_all)
        pooled[name] = {
            "best_common_complex_scale": complex_pairs([scale])[0],
            "relative_residual_after_one_common_scale": norm(numerical_all-scale*model)/norm(numerical_all),
        }
    data = {
        "process": "Psi(E) -> Psi_tilde(x) Psi_tilde(y) Psi_tilde(z)",
        "tensor_column_order": ["delta_alpha_beta delta_gamma_delta", "delta_alpha_gamma delta_beta_delta", "delta_alpha_delta delta_beta_gamma", "Omega_alpha_beta_gamma_delta"],
        "incoming_tensor_to_outgoing_gamma_rank_matrix": transform.tolist(),
        "Cayley_convention": "Omega_incoming-first = +(3*T1+T2)/4; equivalently Omega_outgoing-first = -(3*T1+T2)/4. The opposite sign was also tested; screenshot does not define Omega independently.",
        "metric": "Full tensor Frobenius norm, proportional to sum_r binomial(7,r)*abs(I_r)^2; common 64 omitted.",
        "raw_candidate": "M_raw = -i*pi/32 * H * screenshot_bracket",
        "normalization_status": "Raw data match H*bracket. Full screenshot agreement requires A_target = (8*i*g^2/pi)*sqrt(W)*M_raw. A fermionic external-state/LSZ derivation of this map remains open; the prior NS-only map cannot be imported.",
        "absolute_constant_status": "-i*pi/32 is a simple numerical candidate identified in this inspection, not an independently certified sphere-normalization derivation.",
        "physical_status": "Current four-R assembly uses a bootstrap-fixed cocycle; physical cubic/BPZ audit remains open.",
        "pooled_normalization_diagnostics": pooled,
        "energies": records,
    }
    (output / "comparison.json").write_text(json.dumps(data, indent=2)+"\n")
    fields = ["energy", "raw_candidate_relative_residual", "tensor_shape_relative_residual", "opposite_Omega_tensor_shape_relative_residual", "endpoint_grid_relative_change"]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    lines = [
        "# Four-fermion screenshot comparison", "",
        "The new screenshot predicts `g^2 H sqrt(W)/4` times the bracket", "",
        "`B = (y-z)/D(y+z) delta_ab delta_cd - (x-z)/D(x+z) delta_ac delta_bd + (x-y)/D(x+y) delta_ad delta_bc - i Omega_abcd`,", "",
        "where `E=x+y+z`, `W=Exyz`, `H=1+2iE`, and `D(q)=1+iq`.", "",
        "All eight raw integrated RC amplitudes are close to the compact candidate `M_raw = -i pi H B / 32`. The endpoint refinement moves every point closer to this candidate. This tests the full bracket, including the relative contact coefficient; there is no fitted energy polynomial.", "",
        "The numerical constant `-i*pi/32` was identified during this comparison. It is not an independent determination of the absolute physical sphere coupling.", "",
        "## Tensor conventions", "",
        "The screenshot orders the spinor indices `(incoming, outgoing1, outgoing2, outgoing3)=(alpha,beta,gamma,delta)`. The evaluator orders them `(beta,gamma,delta,alpha)`. In its factorial-normalized gamma-rank basis `(T0,T1,T2,T3)`, the incoming-first delta tensors have columns", "",
        "`delta_ab delta_cd -> (1,1,-1,-1)/8`,", "`delta_ac delta_bd -> (1,-1,1,-1)/8`,", "`delta_ad delta_bc -> (1,0,0,0)`.", "",
        "We take `Omega_abcd -> (0,3,1,0)/4`. Equivalently the Cayley tensor in outgoing-first order is `-(3T1+T2)/4`; the cyclic reordering contributes a minus sign. The tensor has squared norm 336. Its sign must be compared with the definition used to obtain the screenshot. The opposite sign is tested separately below.", "",
        "The analysis independently verifies the delta rearrangements, complete antisymmetry, and Cayley norm with explicit Clifford matrices. Residuals use the full tensor Frobenius norm: `||I||^2 = 64 sum_r binomial(7,r)|I_r|^2`, not an unweighted coefficient norm.", "",
        "## Numerical comparison", "",
        "The candidate column includes the specified `-i*pi/32`, with no fitted scale. The shape-only column permits a separate overall complex scalar per energy, testing tensor ratios independently of all external leg normalizations and phases. Grid changes are sensitivity diagnostics, not rigorous error bounds.", "",
        "| Sample | Raw candidate discrepancy | Shape-only discrepancy | Endpoint grid change | Opposite Cayley sign, shape-only |", "|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append("| {} | {:.5f}% | {:.5f}% | {:.3f}% | {:.3f}% |".format(row["energy"],100*row[fields[1]],100*row[fields[2]],100*row[fields[4]],100*row[fields[3]]))
    lines += ["", "## External normalization: unresolved part", "",
        "The raw amplitudes are CFT-vertex integrals. Their relative tensor dependence can be tested without an external-state dictionary, but the screenshot's common `sqrt(W)` cannot.", "",
        "Literal comparison to `H sqrt(W) B`, allowing one common complex scale for all eight points, leaves a pooled residual of {:.3f}%. Comparison to `H B` gives {:.3f}%. This establishes which dependence is present in the raw data; it does not by itself exclude the screenshot in a different, independently specified state normalization.".format(100*pooled['H_times_sqrtW_times_bracket']['relative_residual_after_one_common_scale'],100*pooled['H_times_bracket']['relative_residual_after_one_common_scale']), "",
        "The exact external conversion required for agreement is `A_target = (8 i g^2/pi) sqrt(W) M_raw`, in this Cayley and phase convention. Its energy-dependent part requires one factor `sqrt(omega)` for every external Ramond leg. Such factors must follow from the fermionic BRST/kinetic pairing; multiplying the data by them to obtain agreement would not establish that pairing. The earlier NS scalar `1/sqrt(omega)` derivation is not a fermionic normalization derivation.", "",
        "The relevant distinction is explicit in Sen, [arXiv:1501.00988](https://arxiv.org/pdf/1501.00988), section 3: Ramond sewing includes a picture-changing zero mode. The existing three-point implementation also explicitly omits target-space spinor wavefunctions and LSZ factors. No completed physical Ramond normalization audit is asserted here.", "",
        "The assembly's bootstrap-fixed cocycle and independent physical cubic/BPZ audit remain additional limitations. These results support a compact raw amplitude very strongly, but are not an analytic proof or a certified delta-normalized spacetime amplitude.", "",
        "Reproduce with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python compare_so7e8_fourr_screenshot.py`. No cluster jobs or paper TEX changes are made by this analysis.", "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "median_raw_discrepancy": float(np.median([r[fields[1]] for r in records])), "max_raw_discrepancy": max(r[fields[1]] for r in records), "normalization_comparisons": pooled}, indent=2))


if __name__ == "__main__":
    main()
