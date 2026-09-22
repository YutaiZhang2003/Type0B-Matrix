# Quadrature convergence of the new NSRR proposal

**Both momentum integrals and their full complex spin matrices meet the two-step 0.01% stability criterion.** This statement is for the central saved surface t=0.60 at b=1.4, with source total descendant cutoff L=5 and target recursion twice-level R=12 (resummed global blocks).

The reported ratio is $\mathcal R=(Z_{\rm NSRR}/Z_{\rm NSNSNS})(Z_{{\rm free},t}/Z_{{\rm free},s})^{1+2(b+b^{-1})^2}$. Agreement requires one.

The source converges at N=7 and is then held fixed while the target is refined. N is the number of nodes per momentum direction, so each grid has N^3 nodes.

| Source N | Target N | Ratio s=+ | Ratio s=- | Spin-sum ratio |
|---:|---:|---:|---:|---:|
| 3 | 3 | 0.252925745 | 0.252836380 | 0.252879225 |
| 4 | 4 | 0.252306619 | 0.252134476 | 0.252217026 |
| 5 | 5 | 0.250782696 | 0.250562482 | 0.250668084 |
| 6 | 6 | 0.250255801 | 0.250023212 | 0.250134747 |
| 7 | 7 | 0.250108170 | 0.249871433 | 0.249984956 |
| 7 | 8 | 0.250068188 | 0.249829678 | 0.249944051 |
| 7 | 9 | 0.250056418 | 0.249817280 | 0.249931954 |
| 7 | 10 | 0.250052595 | 0.249813244 | 0.249928019 |

## Convergence evidence

- Last two maximum relative source changes: 0.000912%, 0.000918%.
- Last two maximum relative target changes: 0.004963%, 0.001616%.
- These checks cover each sign, each resolved spin, and their sum, with each integral checked separately.
- The full complex 2x2 integrated matrices also pass two successive relative Frobenius-norm changes below 1e-4; `matrix_convergence.json` records their norms and off-diagonal phases.
- All accepted target global sums converge. Input hashes, complete grids, primary powers, and momentum measures are checked.

These are empirical convergence observations at fixed block cutoffs, not rigorous remainder bounds. Only t=0.60 has been refined to this criterion.

## Normalization and phase

The original coefficients, pairing, free-field frame factor, and primary factors remain fixed. No multiplicative factor is fitted or applied. For comparison with the earlier normalization discussion, the diagnostic 4R-1 at the final grids is:

| Combination | 4R-1 |
|---|---:|
| s=+1 | +0.021038% |
| s=-1 | -0.074703% |
| spin_sum | -0.028792% |

Thus the coarse quadrature caused much of the smaller residual seen after a hypothetical factor four. The near-quarter value does not itself derive a correction to M.

The integrated off-diagonal phase difference is `-4.802614953e-04` radians in the tested spin basis. The Frobenius distance between the trace-normalized matrices is `5.706858874e-04`. These describe the integrated spin matrix K, not the three-point coefficient matrix M. Their remaining differences must still be assessed against block truncation and the proposed interacting spin transport.

All channel-dependent q^h factors remain outside descendant blocks and M. Complex lift data are retained; the spin sum removes relative-phase interference.

## Reproduction and data

```sh
python Code/genus_2/reproduce_nsrr_bilinear_quadrature.py --workers 6
python Code/genus_2/audit_nsrr_bilinear_quadrature.py
python Code/genus_2/check_nsrr_quadrature_matrices.py
python Code/genus_2/report_nsrr_quadrature_limit.py
```

- `final_result.json`: final values, convergence steps, and limitations.
- `target_refinement.csv`: both channel integrals and all ratios at every completed order.
- `matrix_convergence.json`: full complex matrix convergence.
- `final_provenance.json`: verified input and node hashes.
- `source/`, `target/`: numerical nodes. The unused partial source N8 grid is explicitly excluded.
