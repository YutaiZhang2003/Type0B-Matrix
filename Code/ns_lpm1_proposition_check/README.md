# Test of the NS proposition in the main notes

## Statement tested

The script tests exactly the NS-sector support statement appearing after the
Ramond proposition in the main notes:

\[
L_1v_n\in
\operatorname{span}\left\{
L_{-A}^{(1)}L_{-B}^{(2)}v_{n-1}:|A|+|B|=4n-3
\right\},
\]

and

\[
L_{-1}v_n\in
\operatorname{span}\left\{
L_{-1}^{(1)}v_n,
L_{-1}^{(2)}v_n,
L_{-A}^{(1)}L_{-B}^{(2)}v_{n-1}:|A|+|B|=4n-1
\right\}.
\]

The \(\alpha\) labels currently present in the two displayed NS equations in
the main notes were not used: unlike the Ramond branch, the NS branch has no
second parity-copy label.

## Outcome

Both support statements pass for

\[
n=1,\frac32,2,\frac52.
\]

The calculation uses

\[
b=\frac32,\qquad P=\frac25,\qquad Q=\frac{13}{6}.
\]

Every descendant matrix has full column rank:

| \(n\) | \(L_1\) level | rank | relative residual | \(L_{-1}\) lower-branch level | rank including the two level-one states | relative residual |
|---:|---:|---:|---:|---:|---:|---:|
| \(1\) | \(1\) | \(2/2\) | \(8.01\times10^{-16}\) | \(3\) | \(12/12\) | \(1.12\times10^{-14}\) |
| \(3/2\) | \(3\) | \(10/10\) | \(5.81\times10^{-15}\) | \(5\) | \(38/38\) | \(8.69\times10^{-14}\) |
| \(2\) | \(5\) | \(36/36\) | \(3.98\times10^{-14}\) | \(7\) | \(112/112\) | \(3.39\times10^{-13}\) |
| \(5/2\) | \(7\) | \(110/110\) | \(2.72\times10^{-13}\) | \(9\) | \(302/302\) | \(5.99\times10^{-12}\) |

At this sample point, the fitted coefficients of
\(L_{-1}^{(1)}v_n\) and \(L_{-1}^{(2)}v_n\), in that order, are

| \(n\) | first copy | second copy |
|---:|---:|---:|
| \(1\) | \(-2.945069033478\) | \(-6.039710609714\) |
| \(3/2\) | \(-3.390569309921\) | \(-5.838638443749\) |
| \(2\) | \(-3.699542251212\) | \(-6.086796200119\) |
| \(5/2\) | \(-3.922296163751\) | \(-6.354843184914\) |

## Construction and internal checks

The state \(v_n\) is constructed directly from the ordered NS string

\[
\chi_{-(4n-1)/2}\chi_{-(4n-3)/2}\cdots\chi_{-1/2}\phi,
\qquad \chi_r=\psi_r-i\eta_r.
\]

The calculation then constructs \(L_m^{(1)}\) and \(L_m^{(2)}\) from the
definitions in the main notes. In particular,

\[
U_m=\sum_{r\in\mathbb Z+1/2}\psi_{m-r}G_r
\]

is applied directly with the graded tensor-product sign. Every partition pair
at the required double-Virasoro level is included in the span calculation.

The following independent checks give zero residual at the working precision:

- the generalized physical \(L_m\) and \(G_r\) actions agree with the older
  exact NS oscillator implementation through twice-level six;
- all branch states used are annihilated by the positive modes of both
  Virasoro copies; and
- on every tested state,
  \(L_{-1}=L_{-1}^{(1)}+L_{-1}^{(2)}-L_{-1}^{\mathrm F}\).

## Reproduction

From the repository root, run

```bash
python3 -u python/ns_lpm1_proposition_check/check_ns_proposition.py \
  --json python/ns_lpm1_proposition_check/results.json
```

The generated `results.json` contains all ranks, residuals, singular values,
and fitted level-one coefficients. The calculation is numerical at one generic
parameter point; it is not a symbolic proof in \(b\) and \(P\).
