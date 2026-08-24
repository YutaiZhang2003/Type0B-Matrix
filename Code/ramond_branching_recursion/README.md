# Configurable Ramond branching recursion

This folder contains a self-contained numerical implementation of the branching-coefficient recursion in `SCblock.tex`. It constructs every branch state and decomposition needed by the requested target; it does not import another project module or use any stored decomposition or branching coefficient.

## Convention status

The numerical convention is **only** the one written in `Human Notes/SCblock.tex`: its (w^\pm) basis, (G_0) action, ground three-forms, BPZ pairing, reflection rule, and Section 8 hatted-form sign. No external Ramond-frame conversion is permitted. A successful CLI run certifies the internal branch decompositions and the finite (L_{\pm1}) Ward solve only. It does **not** certify agreement with the independent PBW/double-Virasoro computation; the JSON field `pbw_double_virasoro_match_certified` remains `false` while the first descendant mismatch is unresolved.

## Algorithm

The script constructs the NS and Ramond branch primaries from the free-field \(\chi\)-strings in the main notes. It constructs \(L_n^{(1)}\) and \(L_n^{(2)}\) from \(L_n\), \(L_n^{\mathsf F}\), and \(U_n\), and then solves the required finite-dimensional decompositions of \(L_1v_{n_1}\) and \(L_{-1}v_{n_2,n_3}^{\alpha}\). The default path uses column-scaled double-precision least squares. With `--mp-dps`, a pivoted QR selects a full-rank square set of oscillator rows, the solution is improved using multiprecision residuals and mixed-precision iterative refinement, and the final residual is certified in multiprecision on every oscillator row. No closed-form decomposition coefficient is inserted.

Only the first Ward identity is used:

\[
\widehat\rho(L_1v_1,v_2,v_3)
=\widehat\rho(v_1,L_{-1}v_2,v_3)
+\widehat\rho(v_1,v_2,L_{-1}v_3).
\]

For a requested target \((n_1,n_2,n_3)\), the code generates the NS chain from \(0\) to \(n_1\) and the connected Ramond reflection component containing each target label. The corresponding tensor-ground value, with \(n_1=0\) and \(n_2,n_3=\pm1/4\), is evaluated directly from the ground-state normalization and the factorization sign in the main notes. Restricting to the connected component avoids numerical leakage between exactly decoupled Ward blocks. The resulting finite first-Ward system supplies every boundary coefficient reached by the normalized three-term recursion. As an independent assembly check, the recursive answer is compared with the target entry of the full first-Ward solution for every \(\alpha_2,\alpha_3\), and \(\eta\).

The correlated square-root branch for the two Virasoro momenta is fixed by the branch states themselves. For every generated label, the script checks

\[
L_1^{(i)}L_{-1}^{(i)}v_n=2h_n^{(i)}v_n.
\]

This check is important: the opposite correlated branch replaces \(h_n^{(i)}\) by \(h_{-n}^{(i)}\) and makes the Ward system inconsistent.

## Supported targets

The configurable recursion currently supports

\[
n_1\in\mathbb Z_{\geq0},\qquad n_2,n_3\in\tfrac12\mathbb Z+\tfrac14.
\]

Positive and negative Ramond reflection labels are accepted. Targets already on a recursion boundary are returned directly from the finite Ward solution. The other NS congruence class, \(n_1\in\mathbb Z+\tfrac12\), requires independent low-level anchors at \(n_1=\pm\tfrac12\); those anchors are not yet part of this self-contained driver. The CLI rejects that class explicitly rather than silently applying the integer-chain normalization. Runtime and conditioning deteriorate rapidly with the highest requested label because the Virasoro descendant spaces grow by partitions. In double precision, the \(n_1=3\) stress test constructs the result but fails the strict finite-Ward residual certificate. The multiprecision path below resolves this conditioning problem.

## Generic-point run

The default smoke-test point is

\[
b=\frac75,\qquad P_1=\frac{11}{23},\qquad
P_2=\frac{13}{29},\qquad P_3=\frac{17}{31}.
\]

Reproduce the original target with

```sh
python3 -B Code/ramond_branching_recursion/compute_target.py
```

Choose another target using exact rational labels:

```sh
python3 -B Code/ramond_branching_recursion/compute_target.py \
  --n1 1 --n2 9/4 --n3 5/4 \
  --json Code/ramond_branching_recursion/results_1_9o4_5o4.json
```

The generic point remains configurable through `--b`, `--p1`, `--p2`, and `--p3`.

## Multiprecision \(n_1=3\) run

Run the level-three stress target with 60 decimal digits using

```sh
python3 -u -B Code/ramond_branching_recursion/compute_target.py \
  --n1 3 --n2 3/4 --n3 3/4 --mp-dps 60 \
  --json Code/ramond_branching_recursion/results_n1_3_mp60.json
```

At the generic point above, the largest NS system has 13,132 oscillator rows and 300 descendant columns. Pivoted row selection gives a scaled condition number $3.46\times10^8$, and four refinement steps reduce its all-row relative residual to $5.03\times10^{-52}$. Across every NS and Ramond decomposition, the worst relative residual is $9.16\times10^{-47}$. The full recursion then has worst finite-Ward residual $1.54\times10^{-36}$ and worst recursion-versus-Ward disagreement $5.81\times10^{-46}$, so all four internal-parity assignments pass. The recorded local runtime was 577.5 seconds. The complete output is written to `results_n1_3_mp60.json`.

The numerical branching table previously printed here used a ground-frame adapter that is not part of the Human Note. It is intentionally withdrawn. Regenerate internal recursion diagnostics with the current code, but do not use the resulting branching values as physical data until the independent PBW/double-Virasoro comparison is certified.

The complete decomposition diagnostics, generated label sets, boundary values, ground anchors, linear reductions, results, and timings are written to the requested JSON file. The checked-in `results.json` is historical and must not be treated as Human-convention physical data.

Run the generalized-target regression suite with

```sh
python3 -B -m unittest Code/ramond_branching_recursion/test_compute_target.py
```
