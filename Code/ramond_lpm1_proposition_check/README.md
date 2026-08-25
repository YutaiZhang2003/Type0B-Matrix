# Numerical check of the proposition around Eq. (7.13)

## Result

The support statement in the proposition passes at

\[
n=\frac34,\frac54,\frac74,\frac94
\]

for both Ramond parity copies \(\alpha=0,1\). More precisely, the calculation
finds

\[
L_1v_n^\alpha\in
\operatorname{span}\left\{
L_{-A}^{(1)}L_{-B}^{(2)}v_{n-1}^\alpha:
|A|+|B|=4n-3
\right\},
\]

and

\[
L_{-1}v_n^\alpha\in
\operatorname{span}\left\{
L_{-1}^{(1)}v_n^\alpha,
L_{-1}^{(2)}v_n^\alpha,
L_{-A}^{(1)}L_{-B}^{(2)}v_{n-1}^\alpha:
|A|+|B|=4n-1
\right\}.
\]

However, the displayed second equation in the main notes does not pass. In
that equation the coefficients of \(L_{-1}^{(1)}v_n^\alpha\) and
\(L_{-1}^{(2)}v_n^\alpha\) are both fixed to one. The residual

\[
L_{-1}v_n^\alpha
-L_{-1}^{(1)}v_n^\alpha
-L_{-1}^{(2)}v_n^\alpha
\]

does not lie in the stated descendant space of \(v_{n-1}^\alpha\). Thus the
two level-one terms must have nontrivial coefficients. The symbol multiplying
the last sum in the displayed equation is also written as
\(\mathbb V_{1,n}^{AB}\), although the sentence following it refers to
\(\mathbb V_{-1,n}^{AB}\).

The prose before the display says \(L_1v_n^\alpha\) twice. The calculation
interprets the second occurrence as \(L_{-1}v_n^\alpha\), in agreement with
the displayed equation. A structurally correct version of that displayed
equation must leave the coefficients of both level-one descendants
undetermined and use an independent coefficient set for the descendants of
\(v_{n-1}^\alpha\).

At the sample point used below, the fitted coefficients of
\(L_{-1}^{(1)}v_n^\alpha\) and \(L_{-1}^{(2)}v_n^\alpha\), respectively, are

| \(n\) | first copy | second copy |
|---:|---:|---:|
| \(3/4\) | \(-2.659648053149\) | \(-7.610359891824\) |
| \(5/4\) | \(-3.188656011067\) | \(-5.803822128370\) |
| \(7/4\) | \(-3.558482770347\) | \(-5.951760476958\) |
| \(9/4\) | \(-3.819382110849\) | \(-6.223694041158\) |

The two parity copies give the same coefficients within numerical precision.
In particular, none of these coefficients is compatible with the value one.

## Numerical data

The generic exact rational sample is

\[
b=\frac32,\qquad P=\frac25,\qquad Q=b+b^{-1}=\frac{13}{6}.
\]

Every descendant matrix has full column rank. The table gives the largest
relative residual over \(\alpha=0,1\).

| \(n\) | \(L_1\) support | general \(L_{-1}\) support | displayed unit-coefficient formula |
|---:|---:|---:|---:|
| \(3/4\) | \(2.22\times10^{-16}\) | \(1.44\times10^{-14}\) | \(7.29\times10^{-1}\) |
| \(5/4\) | \(2.73\times10^{-15}\) | \(3.05\times10^{-14}\) | \(6.34\times10^{-1}\) |
| \(7/4\) | \(3.34\times10^{-14}\) | \(1.59\times10^{-13}\) | \(5.72\times10^{-1}\) |
| \(9/4\) | \(1.17\times10^{-13}\) | \(2.60\times10^{-12}\) | \(4.77\times10^{-1}\) |

The first two columns are at numerical zero, including the largest matrices
at \(n=9/4\). The last column is of order one and therefore represents a
genuine failure rather than accumulated floating-point error.

## What the code computes

The script starts from the ordered Ramond strings in the main-note convention,

\[
\chi_0(P)\chi_{-1}(P)\cdots
\chi_{-(2n-1/2)}(P)(u^0\otimes w_\beta^+),
\qquad
\chi_r(P)=\psi_r-i\eta_r(P),
\]

with the additional opposite zero mode for the other parity copy. It then:

1. applies the physical \(L_{\pm1}\) directly in the free-field Fock basis;
2. constructs \(L_{-m}^{(1)}\) and \(L_{-m}^{(2)}\) from \(L_{-m}\),
   \(L_{-m}^{\mathrm F}\), and \(U_{-m}\);
3. evaluates
   \(U_m=\sum_r\psi_{m-r}G_r\) directly, including the graded tensor-product
   sign \((-1)^{|u|}\) when the physical odd operator \(G_r\) acts through an
   auxiliary state \(u\);
4. constructs every two-Virasoro descendant at the required level; and
5. tests membership by a column-normalized numerical rank and least-squares
   calculation.

No form of the proposition is used in constructing the states or the
operators.

Three internal checks guard the implementation:

- the generalized negative-mode oscillator action agrees exactly with the
  existing exact free-field implementation through physical level three;
- every branch state used is annihilated by the positive modes of both
  Virasoro copies; and
- the independently constructed operators satisfy
  \(L_{-1}=L_{-1}^{(1)}+L_{-1}^{(2)}-L_{-1}^{\mathrm F}\) on every tested
  state.

All three checks have zero residual at the working precision.

## Reproduction

From the repository root, run

```bash
python3 -u python/ramond_lpm1_proposition_check/check_proposition_7_13.py \
  --json python/ramond_lpm1_proposition_check/results.json
```

The run takes about eight seconds on the machine used for this check. The
generated `results.json` contains the separate result for each parity, all
matrix dimensions and ranks, absolute and relative residuals, and the fitted
level-one coefficients. Passing `--strict` makes the program exit nonzero
because the displayed unit-coefficient equation fails.

This is strong numerical evidence at one generic parameter point, not a
symbolic proof in \(b\) and \(P\).

## Direct check for negative Ramond labels

The reflected proposition was also tested directly, without assuming the
reflection relation while projecting the states.  For
\(n\leq-\frac34\), the tested statement is

\[
L_1v_n^\alpha\in
\operatorname{span}\left\{
L_{-A}^{(1)}L_{-B}^{(2)}v_{n+1}^\alpha:
|A|+|B|=-4n-3
\right\},
\]

and

\[
L_{-1}v_n^\alpha\in
\operatorname{span}\left\{
L_{-1}^{(1)}v_n^\alpha,
L_{-1}^{(2)}v_n^\alpha,
L_{-A}^{(1)}L_{-B}^{(2)}v_{n+1}^\alpha:
|A|+|B|=-4n-1
\right\}.
\]

The calculation was performed at the generic rational point

\[
b=\frac75,\qquad P=\frac{11}{23},\qquad Q=\frac{74}{35}.
\]

The following table gives the largest relative residual over
\(\alpha=0,1\).  Every displayed matrix had full column rank.

| \(n\) | \(L_1\) level | columns | \(L_1\) residual | \(L_{-1}\) level | columns | \(L_{-1}\) residual | time for both parities |
|---:|---:|---:|---:|---:|---:|---:|---:|
| \(-3/4\) | 0 | 1 | \(2.220\times10^{-16}\) | 2 | 7 | \(1.747\times10^{-13}\) | 0.038 s |
| \(-5/4\) | 2 | 5 | \(4.402\times10^{-15}\) | 4 | 22 | \(2.875\times10^{-13}\) | 0.031 s |
| \(-7/4\) | 4 | 20 | \(4.445\times10^{-14}\) | 6 | 67 | \(7.257\times10^{-13}\) | 0.319 s |
| \(-9/4\) | 6 | 65 | \(4.489\times10^{-13}\) | 8 | 187 | \(1.116\times10^{-11}\) | 6.939 s |

The branch states were constructed in their native \(+1\) free-field
realization, and the physical and embedded Virasoro modes were evaluated in
that same realization.  The boundary state \(v_{1/4}^\alpha\), which enters
the \(n=-3/4\) test, was transported through the common abstract Ramond PBW
basis.  Thus no operator was applied across incompatible free-field charts.
Every tested \(v_n^\alpha\) was annihilated by the positive modes of both
embedded Virasoro algebras, and the independent identity

\[
L_{-1}=L_{-1}^{(1)}+L_{-1}^{(2)}-L_{-1}^{\mathrm F}
\]

had zero oscillator residual.

As a separate check, all fitted coefficients were compared with those of the
positive-label calculation at the reflected momentum.  The largest
difference was \(8.689\times10^{-12}\), confirming numerically that, in the
same ordered descendant bases,

\[
\mathbb V_{\pm1,n,\alpha}^{AB}(P,c)
=\mathbb V_{\pm1,-n,\alpha}^{AB}(-P,c),\qquad
\mathbb V_{n,\alpha}^{(i)}(P,c)
=\mathbb V_{-n,\alpha}^{(i)}(-P,c).
\]

Run the check with

```bash
python3 -u python/ramond_lpm1_proposition_check/check_negative_proposition.py \
  --json python/ramond_lpm1_proposition_check/negative_results.json
```

The final direct negative-family run took 7.328 s.  Including the independent
positive-family reflection comparison, the script took 14.735 s internally
and 15.17 s wall time.
