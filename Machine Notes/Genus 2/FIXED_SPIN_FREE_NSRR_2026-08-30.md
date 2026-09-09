# Fixed-spin free boson–Majorana factor in NSRR theta plumbing

The physical fixed-spin free denominator can now be computed. The new
implementation uses charged Heisenberg pants sewing and bosonization, and
has an independent finite-descendant Ramond check. No checked PBW,
double-Virasoro, branching, or c-recursion kernel was changed. The existing
NS free evaluator and archived runs were also left unchanged.

This solves the free-factor subproblem. It does **not** supply the remaining
nonchiral super-Liouville Ramond contraction, and does not certify an old
Liouville numerator's identification with a particular geometric spin.

## Formula and normalization

Let geometric puncture order be `(zero, one, infinity)`. Define the charged
Heisenberg block by the same local coordinates and `q^L0` propagators:

\[
F_X(a_0,a_1;\mathbf q)=P(\mathbf q)
\exp\left\{\frac12[a_0^2\log q_0+a_1^2\log q_1
+(a_0+a_1)^2\log q_\infty]+E(a_0,a_1;\mathbf q)\right\}.
\]

Here `P` is the boson vacuum Fredholm determinant and `E` is its charged
Schur complement. Both are evaluated by the existing charged-boson code.
The full quadratic exponent, including the separately retained principal
`log(q)` branches, defines

\[
F_X(\mathbf a;\mathbf q)=P(\mathbf q)
 e^{i\pi\mathbf a^T\Omega_{\rm charge}\mathbf a}.
\]

The two-Majorana/Dirac theory is the odd charge lattice. For binary
characteristics `[alpha|beta]`, sum `a=n+alpha/2` with phase
`exp(i*pi*a.beta)`. Up to an irrelevant unit phase this gives

\[
Z_{\rm Dirac,L}^{\rm pl}
=P(\mathbf q)\vartheta[\alpha|\beta](0|\Omega_{\rm charge}).
\]

The sewing bosonization identity is described in
[Tuite–Zuevsky, equation (78)](https://arxiv.org/html/1007.5203).
The implementation here constructs its Heisenberg factor in the *theta*
plumbing coordinates; it does not import a torus-sewing frame factor.

One **nonchiral real Majorana**, rather than a Dirac fermion, contributes
`abs(Z_Dirac,L)`. With scalar weight `h(a)=a^2/2`, completeness
`da_zero da_one`, and unit connected target-space zero-mode volume,

\[
Z_X^{\rm pl}=\frac{|P|^2}{\sqrt{\det(2\operatorname{Im}\Omega_{\rm charge})}},
\qquad
Z_{X+\psi,\delta}^{\rm pl}
=\frac{|P|^3\,|\vartheta[\delta](0|\Omega_{\rm charge})|}
{\sqrt{\det(2\operatorname{Im}\Omega_{\rm charge})}}.
\]

The factor `2` inside the genus-two determinant is explicit: the nonchiral
Gaussian is `exp(-2*pi*a^T Im(Omega) a)`. Changing to a momentum measure
rescaled by `sqrt(2)` on both independent loops multiplies the answer by
two. No such rescaling or fitted constant is used here. Common edge Casimir
powers are stripped consistently; no Ramond primary power is stripped.

For `alpha=(1,1)` the charges at zero and one are half-integer and their sum
is integer. These are sectors `(R,R,NS)` geometrically, or `(NS,R,R)` in the
Human Note's slot order. The two leading lattice charges
`(+1/2,-1/2)` and `(-1/2,+1/2)` fix the nonchiral Majorana normalization:

\[
Z_\psi^{\rm pl}\sim 2|q_0q_1|^{1/8}
\]

in the joint small-plumbing limit. Squaring this expression would incorrectly
double the Majorana theory and overcount its zero modes.

## Marked periods and affine spin transport

For the saved five surfaces the independently extracted charged periods obey

\[
\Omega_{\rm charge}=\Omega_{\rm marked}+B,
\quad
B_{\rm source}=\begin{pmatrix}0&0\\0&1\end{pmatrix},
\quad
B_{\rm target}=\begin{pmatrix}-1&-1\\-1&0\end{pmatrix}.
\]

These matrices are fixed across the family, not rounded independently to
force agreement at each point. The new evaluator requires them explicitly
and rejects a period mismatch. Binary spin transport is

\[
\alpha_{\rm charge}=\alpha_{\rm marked},\qquad
\beta_{\rm charge}=\beta_{\rm marked}-B\alpha_{\rm marked}
+\operatorname{diag}B\pmod2.
\]

The source marked spin `[11|00]` stays `[11|00]`; the target marked spin
`[00|00]` becomes `[00|10]` in the charged period basis. Omitting the affine
term selects the wrong all-NS spin. The explicit marked characteristic is
the input: no dictionary for package Ramond parity lifts is guessed.

Maximum charged-versus-marked period residuals are `5.53e-13` on the source
and `1.303e-9` on the target. The latter reflects the saved target plumbing
accuracy; the geometry is not silently changed under its saved numerator.

## Why the former NS-to-R conversion failed

The legacy `theta_physical_fermion_fredholm` returns

\[
F_{\rm legacy}=\tfrac12(-D_{+++}+D_{-++}+D_{+-+}+D_{++-}).
\]

Its constituent *unfiltered* determinants satisfy

\[
D_\eta^2=P\,\vartheta[00|\beta_{\rm charge}],\qquad
\beta_{\rm charge}=
\bigl(\eta_0\eta_\infty<0,\eta_1\eta_\infty<0\bigr),
\]

as a complex identity, to `4.0e-15` in the ten saved charts. The filtered
quantity is a linear combination of four spin blocks. In general its
absolute square is **not** a single fixed-spin Majorana determinant.
Consequently `abs(F_legacy)^2/abs(theta_delta)` need not be spin independent.
This is the source of the failed conversion test, not evidence against the
checked PBW/double-Virasoro algebra.

The quadratic-parity block can remain a valid Human-Note *block convention*.
The mistake was identifying one such block directly with the complete
fixed-spin free path integral. Do not remove quadratic signs inside checked
block code. Spin projection/nonchiral assembly must be done at its boundary.
For the four even NS parity sectors the above four-component transform is
an involution, so converting between these block bases is explicit.

The earlier all-NS free denominator must also be replaced when a genuine
fixed-spin comparison is assembled. Merely replacing that denominator and
calling the old quotient a corrected physical Q is not justified until the
Liouville numerator's spin projection is checked in the same dictionary.

## Five-point numerical result

The original family is `Omega_11=Omega_22=i`,
`Omega_12=t+i/2`; the source is re-marked to place NS at infinity.
The two columns below use different plumbing frames and are **not** expected
to coincide directly.

| t | NSRR free factor, source frame | NSNSNS free factor, target frame |
|---:|---:|---:|
| 0.52 | 0.553067029082 | 0.539105450464 |
| 0.56 | 0.564655845239 | 0.551426940800 |
| 0.60 | 0.575409592372 | 0.563892457040 |
| 0.64 | 0.585617043586 | 0.576604161947 |
| 0.68 | 0.595598097971 | 0.589629896408 |

The scalar determines the frame change. Since its central charge is one,
whereas the superfield's is `3/2`, the independent consistency quantity is

\[
\frac{Z_{X+\psi,S}^{\rm pl}/Z_{X+\psi,T}^{\rm pl}}
{(Z_{X,S}^{\rm pl}/Z_{X,T}^{\rm pl})^{3/2}}-1.
\]

Its maximum absolute value is `4.90e-10`, compatible with the saved target
period accuracy. Oscillator cutoffs 24 and 32 give identical displayed
double-precision values. Charge-lattice cutoffs 4 and 5 are stable at double
precision on this family.

A genuinely separate check enumerates charged current Fock states, their
Gram norms and Wick contractions, with no period matrix, theta function,
Fredholm determinant, or charged Gaussian resummation. Half-integer charge
sewing through oscillator levels 6, 10 and 14 converges to the NSRR answer;
the maximum relative discrepancy at level 14 is `2.71e-12`.
Thirteen new unit tests plus the protected-kernel hash test pass. Including
the existing physical-free and legacy-conversion regressions, all 28 tests
pass. Saved implementation, geometry and protected-kernel hashes were also
independently verified against the files on disk.

The free partition itself is independent of Liouville b. For the requested
generic-b comparison its power is

\[
\kappa=\frac{c_{\rm SL}}{3/2}=1+2(b+b^{-1})^2,
\quad \kappa(1.4)=9.940408163265307.
\]

Hence this denominator has precisely the frame-anomaly power required to
cancel a correctly assembled super-Liouville numerator. No cosmological
constant is present in the free factor. A common Liouville normalization
must still be applied consistently on both numerator sides.

## Files and reproduction

- `Code/genus_2/fixed_spin_free_plumbing.py`: new standalone evaluator and
  independent finite-Fock oracle.
- `Code/genus_2/test_fixed_spin_free_plumbing.py`: spin, branch, Gaussian,
  fractional-charge, zero-mode and determinant regressions.
- `Code/genus_2/run_fixed_spin_free_check.py`: deterministic five-point audit.
- `Data Set/fixed_spin_free_NSrr_20260830/summary.json`: full factors,
  convergence sweeps, branch/spin ledgers, source hashes and checks.
- `Data Set/fixed_spin_free_NSrr_20260830/free_factors.csv`: compact table.

From the repository root:

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime \
  python3 Code/genus_2/run_fixed_spin_free_check.py

env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime \
  python3 -m unittest Code/genus_2/test_fixed_spin_free_plumbing.py \
  Code/genus_2/test_nsrr_checked_kernel_boundary.py
```

This result is ready as the fixed-spin free denominator, but is not a result
for `Q_NSrr`. The remaining comparison work is the physical super-Liouville
spin projection and Ramond nonchiral assembly, using this denominator on
both sides and leaving the checked block kernels intact.
