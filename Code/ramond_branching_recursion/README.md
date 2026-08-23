# Ramond branching recursion at \((2,7/4,5/4)\)

This folder contains a self-contained implementation of the recursive algorithm in `SCblock.tex`. It does not import another project module or use any stored decomposition or branching coefficient.

## Algorithm

The script constructs the NS and Ramond branch primaries from the free-field \(\chi\)-strings in the main notes. It constructs \(L_n^{(1)}\) and \(L_n^{(2)}\) from \(L_n\), \(L_n^{\mathsf F}\), and \(U_n\), and then solves the required finite-dimensional decompositions of \(L_1v_{n_1}\) and \(L_{-1}v_{n_2,n_3}^{\alpha}\) by column-scaled least squares. No closed-form decomposition coefficient is inserted.

Only the first Ward identity is used:

\[
\widehat\rho(L_1v_1,v_2,v_3)
=\widehat\rho(v_1,L_{-1}v_2,v_3)
+\widehat\rho(v_1,v_2,L_{-1}v_3).
\]

The boundary subsystem uses

\[
n_1\in\{0,1,2\},\qquad
n_2\in\{-3/4,-1/4,1/4,3/4,7/4\},\qquad
n_3\in\{-3/4,-1/4,1/4,3/4,5/4\}.
\]

Its four tensor-ground values, with \(n_1=0\) and \(n_2,n_3=\pm1/4\), are evaluated directly from the ground-state normalization and the factorization sign in the main notes. The resulting first-Ward system fixes all 75 unnormalized three-point functions. The normalized three-term recurrence is then applied recursively to \((2,7/4,5/4)\). It visits four interior nodes and reduces the target to eight boundary coefficients. As an independent assembly check, the recursive answer is compared with the target entry of the full first-Ward solution.

The correlated square-root branch for the two Virasoro momenta is fixed by the branch states themselves. For every label used by the calculation, the script checks

\[
L_1^{(i)}L_{-1}^{(i)}v_n=2h_n^{(i)}v_n.
\]

This check is important: the opposite correlated branch replaces \(h_n^{(i)}\) by \(h_{-n}^{(i)}\) and makes the Ward system inconsistent.

## Generic-point run

The default smoke-test point is

\[
b=\frac75,\qquad P_1=\frac{11}{23},\qquad
P_2=\frac{13}{29},\qquad P_3=\frac{17}{31}.
\]

Run it with

```sh
python3 python/ramond_branching_recursion/compute_target.py
```

Using principal square roots for the displayed norms, the recursive values are:

| \(\alpha_2\) | \(\alpha_3\) | \(\eta\) | \(\mathbb B_f^{(\eta)}(2,7/4,5/4)\) |
|---:|---:|---:|---:|
| 0 | 0 | \(+1\) | \(-102.612336391-9.78\times10^{-9}i\) |
| 0 | 0 | \(-1\) | \(1.17702\times10^{-5}\) |
| 0 | 1 | \(+1\) | \(51.306142235+51.306206886i\) |
| 0 | 1 | \(-1\) | \(51.306142247-51.306206877i\) |
| 1 | 0 | \(+1\) | \(51.305829330-51.306514995i\) |
| 1 | 0 | \(-1\) | \(51.305829319+51.306515005i\) |
| 1 | 1 | \(+1\) | \(-0.00204019660\) |
| 1 | 1 | \(-1\) | \(-102.612376520-8.24\times10^{-9}i\) |

The small imaginary parts in nominally real entries are numerical noise. The two small real answers are more sensitive to conditioning than the order-\(10^2\) answers, so `results.json` records both the recursive value and the independently assembled first-Ward value.

For the final run, the worst decomposition residual was \(3.30\times10^{-13}\), the largest branch-weight error was \(3.91\times10^{-14}\), the smallest recursion denominator had magnitude \(0.576\), the worst first-Ward residual was \(1.45\times10^{-9}\), and the largest recursion-versus-Ward relative disagreement was \(1.61\times10^{-8}\). The script reported \(0.955\) seconds internally; `/usr/bin/time` reported \(1.08\) seconds of wall time.

The complete decomposition diagnostics, boundary values, ground anchors, linear reductions, results, and timings are in `results.json`.
