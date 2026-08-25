# Ramond theta-block torus-limit audit

This folder implements the requested (q_1\to0) cross-check through total
level six at the generic point

\[
b=\frac75,\qquad
P_1=\frac{11}{23},\qquad
P_2=\frac{13}{29},\qquad
P_3=\frac{17}{31}.
\]

Run

```sh
python3 python/ramond_torus_limit_check/check_torus_limit.py
```

The code keeps the two spin parities on the surviving Ramond edges separate,
so each reported comparison simultaneously checks all four numerical choices
of \((\eta_2,\eta_3)\). It computes the canonical diagonal blocks
\(\mathbb F_0^{(+,+)}\) and \(\mathbb F_1^{(-,-)}\).

## The two calculations

The direct calculation sets the NS edge to its primary, constructs the two
Ramond PBW bases

\[
L_{-M}G_{-K}w^\alpha,
\]

forms and inverts their BPZ Gram matrices, evaluates both NS--R--R
three-point functions by Ward reduction, and contracts the indices. The same
calculation is performed for the auxiliary Ramond fermion. The two series are
then combined with the Ramond convolution in `SCblock.tex`.

The branching calculation sets \(n_1=0\), solves the finite first-Ward system
for the normalized Ramond branching coefficients, constructs each ordinary
Virasoro torus two-point block from Virasoro PBW Gram matrices and Ward
identities, and multiplies the two Virasoro copies. At total level six the
Ramond branch labels are

\[
n=\pm\frac14,\ \pm\frac34,\ \pm\frac54,\ \pm\frac74.
\]

`results.json` contains the direct SCA series, the auxiliary-fermion series,
the convolved enlarged series, the double-Virasoro series, every coefficient
difference, and timings.

## Outcome

The two constructions agree through total level six. For example, in the
even block,

\[
[q_3\eta_2^0\eta_3^0]_{\rm PBW}=0.874201752560,
\qquad
[q_3\eta_2^0\eta_3^0]_{\rm branch}=0.874201751811.
\]

The maximum absolute errors by total level are:

| total level | \(f=0,\eta=+\) | \(f=1,\eta=-\) |
|---:|---:|---:|
| 0 | 1.09e-10 | 1.72e-10 |
| 1 | 1.25e-9 | 7.11e-10 |
| 2 | 7.21e-8 | 2.87e-8 |
| 3 | 1.97e-7 | 7.31e-8 |
| 4 | 1.06e-6 | 3.56e-7 |
| 5 | 2.55e-6 | 3.15e-6 |
| 6 | 1.59e-5 | 2.25e-5 |

The largest relative discrepancies are (4.21\times10^{-7}) and
(4.55\times10^{-7}), respectively. They are consistent with the numerical
conditioning of the finite Ward and Gram systems. The full run took 3.57
seconds internally on the recorded machine.

The phase that resolves the earlier apparent failure is

\[
\rho^{\mathsf F}(\mathbf 1,u^0,u^0)=1,
\qquad
\rho^{\mathsf F}(\mathbf 1,u^1,u^1)=i.
\]

The second number is not the BPZ norm: the norm remains
\(\langle u^1|u^1\rangle=-1\). Confusing the NS--R--R chiral three-point
phase with the R--NS--R BPZ pairing produces a false mismatch already at
level one. With the phase above, the free-fermion mode Ward identity and its
Virasoro \(L_{-1}\) Ward identity agree, and the mixed-reflection
embedded-Virasoro descendant test agrees to (3.4\times10^{-16}).

There is also a source/geometry correction: arXiv:1207.5740 computes torus
one-point blocks, not torus two-point blocks. Removing one edge of the theta
graph leaves a connected genus-one graph with two punctures. The code uses the
paper's PBW/inverse-Gram sewing prescription and extends it to the two-edge
contraction dictated directly by the theta-block definition.
