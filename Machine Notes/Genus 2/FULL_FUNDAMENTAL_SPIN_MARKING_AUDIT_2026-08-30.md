# Full q-to-period-to-fundamental-domain spin audit

## Conclusion and correction of the previous interpretation

The saved **geometric period/spin marking chains pass** when all coordinate
changes are composed, including integer B-period branches and reduction to
a common Gottschling fundamental-domain representative. The user's point
about transporting the characteristic during this reduction is essential.

The previous free-spin diagnosis should not be read as a demonstrated
error in this geometric marking. Its numerical comparison concerns the
value of the legacy filtered free sewing expression versus the independently
evaluated fixed-spin free factor. That difference survives the complete
marking chain, but is a separate amplitude/assembly question. It is not
proof of an incorrectly marked Liouville numerator or an error in the
Human Note or checked chiral blocks.

No production code, Human Note, block algorithm, coefficient, free
denominator, or saved Liouville result was changed in this audit.

## 1. Conventions and checks

For every symplectic matrix, the action is

\[
\Omega'=(A\Omega+B)(C\Omega+D)^{-1},
\]

and for binary characteristics,

\[
\alpha'=D\alpha-C\beta+\operatorname{diag}(CD^T),\qquad
\beta'=-B\alpha+A\beta+\operatorname{diag}(AB^T)\pmod2.
\]

Each matrix is checked to be integral and exactly symplectic. The spin
rule is not merely compared against another call to the labeling helper:
independent theta sums verify

\[
|\theta[\delta'](\Omega')|
=|\det(C\Omega+D)|^{1/2}|\theta[\delta](\Omega)|.
\]

This identity was tested for all ten even characteristics on every native
historical/current chart-to-fundamental-domain map. The largest relative
error is `2.9e-15`.

The invariant used to compare the same physical free theory between
frames is

\[
\mathcal I_\delta(\Omega)
=\det(2\operatorname{Im}\Omega)^{1/4}|\theta[\delta](\Omega)|
=\frac{Z_{X+\psi,\delta}^{\rm plumbing}}
       {(Z_X^{\rm plumbing})^{3/2}}.
\]

The last equality uses the existing `da_1 da_2`, `h(a)=a^2/2` convention.
It removes the relative plumbing Weyl factor before the fundamental-domain
comparison. No auxiliary double-Virasoro fermion or change of graded metric
is involved.

## 2. Historical all-NS geometry: the branch was included

The original record was recovered from
`/Users/yutaizhang/Desktop/Project/StringMC/plumbing/results/genus2_plumbing_moduli_samples/q06_search_N256/overlap_samples.csv`.
The five production points retain the atlas word

`W = T12^-1 T22 gl-shear-12 T11 full-s I`.

The rightmost operation acts first. It reconstructs

\[
W=\begin{pmatrix}
2&-1&-1&-1\\-2&1&0&-1\\1&0&0&0\\-1&1&0&0
\end{pmatrix}.
\]

The atlas theta period differs from the saved native theta period by the
recorded shift

\[
B_{\rm hist}=\begin{pmatrix}-1&1\\1&-1\end{pmatrix},\qquad
\Omega_\Theta= W\cdot\Omega_{\rm Gl}+B_{\rm hist}.
\]

The exact product `T_B W` equals the saved branch-composed matrix

\[
M_{\rm Gl\to\Theta}=
\begin{pmatrix}
0&0&-1&-1\\0&0&0&-1\\1&0&0&0\\-1&1&0&0
\end{pmatrix}.
\]

The characteristic chain, including the intermediate change, is

\[
\underbrace{[00|00]}_{\text{native glasses}}
\xrightarrow{W}
\underbrace{[00|11]}_{\text{atlas theta}}
\xrightarrow{T_{B_{\rm hist}}}
\underbrace{[00|00]}_{\text{saved native theta}}
\longrightarrow
\underbrace{[00|00]}_{\text{common FD}}.
\]

The native glasses route also reaches `[00|00]` in that same fundamental
marking. This holds at all five historical points. Omitting the branch
would be wrong, but the production matrix does include it.

For the reference point `o0243`, one certified pair of native-to-FD maps is

\[
N_{\rm Gl}=
\begin{pmatrix}
0&0&0&-1\\0&0&1&0\\0&1&0&0\\-1&0&0&0
\end{pmatrix},\qquad
N_\Theta=
\begin{pmatrix}
0&1&0&0\\-1&1&0&0\\0&0&1&1\\0&0&-1&0
\end{pmatrix}.
\]

They satisfy `N_theta M_Gl_to_Theta = N_Gl` exactly, not just numerically
at this period. The common representative is

\[
\Omega_{\rm FD}\simeq
\begin{pmatrix}
-0.0323849821+2.3267489266i & -0.0268190090+0.2222910398i\\
-0.0268190090+0.2222910398i & 0.0242581756+2.3455142295i
\end{pmatrix}.
\]

Its minimum Gottschling-domain margin is `0.0187653`, strictly positive.

### Independent q-to-Omega reconstruction and B-path bookkeeping

The historical saved theta matrices are reproduced directly by the
Schottky word-length-9 map used in their provenance. An independent
holomorphic-one-form computation at basis order 32 gives the same
markings only after its own B-path convention is accounted for:

| Point | `Omega_collocation - Omega_saved` integral branch | Spin in raw collocation basis |
|---|---|---|
| o0243 | zero | `[00|00]` |
| o0127 | zero | `[00|00]` |
| o0015 | `diag(0,-1)` | `[00|01]` |
| o0167 | `diag(0,-1)` | `[00|01]` |
| o0239 | `diag(0,-1)` | `[00|01]` |

Transporting the latter three back by `T22` gives `[00|00]` in the saved
native basis. These shifts are determined solely by the integrated periods,
not from spin amplitudes. The executable audit supplies them explicitly
and fails if the corrected periods or characteristics disagree.

Glasses periods were independently recomputed from normalized one-forms
at basis order 40 with 240 samples per seam. The largest historical
one-form period residual, after the explicit branches, is `3.50e-10`.

## 3. Current NSRR/all-NS geometry

The original family is

\[
\Omega_0(t)=\begin{pmatrix}i&t+i/2\\t+i/2&i\end{pmatrix},
\qquad \delta_0=[01|10].
\]

The saved source re-marking sends this to `[11|00]` in the NS-at-infinity
source chart, while the target transformation sends it to `[00|00]` in
the all-NS target chart. Those are **native chart labels**, not unchanged
fundamental-domain labels.

At `t=0.60`, choose the original-to-FD word

`T12 full-s bridge-sign T12^-1 I`.

The resulting native-to-FD matrices are

\[
N_{\rm source}=
\begin{pmatrix}
1&-1&1&0\\-1&0&0&1\\-1&0&0&0\\1&-1&0&-1
\end{pmatrix},\qquad
N_{\rm target}=
\begin{pmatrix}
0&0&-1&-1\\1&-1&0&-1\\1&0&0&0\\0&0&0&-1
\end{pmatrix}.
\]

They satisfy `N_target M_source_to_target = N_source` exactly. Both reach

\[
\Omega_{\rm FD}\simeq
\begin{pmatrix}
-0.4048173262+0.9209594171i & 0.4292075701+0.2985527781i\\
0.4292075701+0.2985527781i & -0.4048173262+0.9209594171i
\end{pmatrix},
\]

with characteristic

\[
[11|00]_{\rm source}\xrightarrow{N_{\rm source}}[00|01]_{\rm FD},
\qquad
[00|00]_{\rm target}\xrightarrow{N_{\rm target}}[00|01]_{\rm FD}.
\]

The holomorphic-one-form reconstruction is repeated from the saved q's,
not just from the nominal target matrices. Across all five current points,
the maximum q-to-marked-period residual is `1.303e-9`.

The charge-log periods used by the fixed-spin free evaluator have their
own already explicit branches

\[
B_s=\begin{pmatrix}0&0\\0&1\end{pmatrix},\qquad
B_t=\begin{pmatrix}-1&-1\\-1&0\end{pmatrix}.
\]

The audit uses `N_source T_(-B_s)` and `N_target T_(-B_t)` from those charge
frames, transporting the characteristics at both stages. These are not
additional branches to omit or apply twice.

The chosen common-FD spin labels along the five-point family are:

| t | FD characteristic |
|---:|---|
| 0.52 | `[01|00]` |
| 0.56 | `[01|00]` |
| 0.60 | `[00|01]` |
| 0.64 | `[00|01]` |
| 0.68 | `[00|01]` |

The reduction word changes between the sampled points 0.56 and 0.60.
This is a change of reduced marking, not a physical discontinuity in spin.
These points lie on a fundamental-domain boundary, so a representative
and its stabilizer must be handled consistently. The audit compares exact
composed matrices into one common marking; it does not infer spin equality
from equality of unmarked period matrices alone.

As a negative control, incorrectly retaining the target's native
`[00|00]` at the `t=0.60` FD matrix would change its free factor by
`+12.3775251%`. The actual current fixed-spin denominator does not make
that mistake: it agrees with the correctly transported FD evaluation to
`1.71e-11` at this point, and `4.90e-10` over all five points.

## 4. What remains of the earlier free-amplitude observation

The fixed-spin factors, after dividing by `(Z_X)^(3/2)`, agree with the
correctly marked `I_delta(Omega_FD)` above. The legacy theta filtered
expression does not:

- old reference `o0243`: relative difference `-5.4677797e-6` in the common FD;
- current target `t=0.60`: relative difference `-0.0472446707332` in the common FD.

Thus these differences were not caused by forgetting the FD change of
characteristic. However, they should be described as a comparison of two
free-sewing/amplitude prescriptions, **not as an established failure of the
saved geometric marking**. The physical relation of the legacy filtered
expression to the Human-Note block assembly still needs to be derived;
no change to that assembly follows merely from this audit.

## Artifacts and verification

- `Code/genus_2/audit_full_fundamental_spin_marking.py` reconstructs the
  complete chains, verifies theta covariance, and repeats the one-form
  forward period maps.
- `Data Set/full_fundamental_spin_marking_audit_20260830.json` records every
  matrix, before/after period, characteristic, residual, and source hash.
- `Code/genus_2/test_audit_full_fundamental_spin_marking.py` adds ten tests,
  including controls that deliberately omit the required branch/spin change.

The new and previous spin/free suites pass **42 tests**. All eight
protected kernel hashes remain unchanged.
