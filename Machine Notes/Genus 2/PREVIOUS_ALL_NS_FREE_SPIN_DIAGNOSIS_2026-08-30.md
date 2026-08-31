# What the previous all-NS check establishes about the free spin

Clarification after the user's marking objection: the complete
`q -> native/atlas Omega -> fundamental domain` chain has now been checked
in `FULL_FUNDAMENTAL_SPIN_MARKING_AUDIT_2026-08-30.md`. The saved geometric
markings **pass**, including their characteristic changes. The amplitude
discrepancies below persist after that transport, but must not be described
as an omitted or incorrect fundamental-domain marking, or as proof that
the Human-Note/Liouville numerator is incorrectly marked.

This is a diagnostic audit. No Human Note, protected PBW/double-Virasoro
kernel, c-recursion kernel, three-point coefficient, or Liouville numerator
has been changed. In particular, no denominator substitution below is
promoted to a new physical NSRR partition function.

## Result

The final all-NS theta/glasses check really did show near agreement after
the odd-coefficient phase correction. But it did **not** independently
establish that its filtered theta free factor was the fixed-spin Majorana
partition function it was labelled as. The geometry suppressed the one
parity sector which distinguishes that filtered expression from the
claimed spin. The glasses free factor passes the independent spin test.

This supports investigating the physical-spin identification of the sewn
numerator. It does **not** support choosing a different free characteristic
to fit the current four-percent discrepancy: changing the free spin on
both sides with the correct modular transport leaves their frame ratio
unchanged, as explicitly verified below.

## 1. Which earlier calculation was audited

The final source is
`Data Set/ns_genus2_human_note_fivepoint_certificate_2026-08-25.json`,
using `Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json`.
The audit verifies the config's SHA256 against the certificate.

The setup was:

- all-NS theta and glasses c-recursion at `R=24`, momentum quadrature `N=10`;
- `c=27/2`, hence the anomaly-cancelling quotient `Q_L=Z_L/(Z_free)^9`;
- geometric theta edge order `(0,1,infinity)`, lifts `(+,-,+)`;
- glasses edge order `(left,right,bridge)`, lifts `(+,+,+)`;
- declared characteristic `[00|00]` in both marked period bases;
- the full affine characteristic action of the saved glasses-to-theta
  symplectic matrix, not just its linear part;
- `C_HN^(0)=C_BRY`, `C_HN^(1)=i*C_BRY_tilde`, and the Human-Note
  descendant/nonchiral signs retained. The odd coefficient product supplies
  `i^2=-1`; this is not a change to the physical free fermion.

Both free denominators were evaluated directly in their respective
plumbing frames. The final certificate used mode 28, checked against
modes 16, 20 and 24. It did not use the auxiliary double-Virasoro fermion.

The older `ns_genus2_cross_sewing_glasses_parity_audit.md` reports a
pre-coefficient-phase 4--5% residual. That is not the final all-NS result.

## 2. Independent fixed-spin test, with the scalar frame held fixed

For binary characteristic notation `[alpha|beta]`, define the charge sum
with charges `n+alpha/2` and insertion `exp(i*pi*(n+alpha/2).beta)`.
The independent control is the bosonization identity

\[
D_\delta(q)^2=P(q)\,\theta_\delta(\Omega_{\rm charge}),
\qquad
Z_{X+\psi,\delta}
=\frac{|P(q)|^3|\theta_\delta(\Omega_{\rm charge})|}
       {\sqrt{\det(2\operatorname{Im}\Omega_{\rm charge})}}.
\]

Here `P` and the **complex** charge period are obtained from charged
Heisenberg pants sewing in the same local frame. The charge period is
read from the quadratic exponent, retaining the specified logarithms of
the plumbing parameters; it is not fitted to a fermion answer. The factor
2 is from `h(a)=a^2/2` and the measure `da_1 da_2`.

For the general bosonization principle see Tuite--Zuevsky,
[equation (78), section 5.4](https://arxiv.org/html/1007.5203#S5.SS4).
Their explicit sewing construction uses sewn tori. In this audit the
theta-pants implementation is checked independently against its own charge
sum; no local-coordinate prefactor is borrowed from another chart.

The historical period dictionary is, for every one of the five points,

\[
\Omega_{\rm charge}^{\Theta}=\Omega_{\rm marked}^{\Theta},
\qquad
\Omega_{\rm charge}^{\rm Gl}
=R\Omega_{\rm marked}^{\rm Gl}R^T,
\quad R=\operatorname{diag}(1,-1).
\]

The second relation reverses the orientation of one charge cycle. Its
off-diagonal sign must not be mistaken for an integer branch shift. No
integer translation is needed in either historical chart. The old
Schottky inversion's recorded integer branch is already incorporated into
the marked period; it is not an extra shift to apply to the charge period.

The charge-derived period residual is at most `3.50e-10` in theta and
`1.4e-16` in glasses. All four unfiltered NS determinants in each channel
obey the **complex** bosonization identity to `3.8e-15` at mode 32.

As a second, spin-sensitive check, the scalar has central charge one and
the scalar-plus-Majorana has central charge `3/2`. For correctly transported
spins in two plumbing frames,

\[
\frac{Z_{\rm free}^{\Theta}/Z_{\rm free}^{\rm Gl}}
     {(Z_X^{\Theta}/Z_X^{\rm Gl})^{3/2}}=1.
\]

The fixed-spin factors pass this check to `4.3e-11`, limited by the saved
period matching. The legacy filtered theta factors fail at the ppm levels
listed below. Scalar normalization and local frame are identical in this
comparison, so this is not a change of metric convention or target-volume
normalization.

## 3. Why the theta denominator was nearly right on the old points

Separate the unfiltered free chiral sewing into its four allowed fermion
parity sectors, in geometric edge order:

\[
S_{000},\quad S_{110},\quad S_{101},\quad S_{011}.
\]

The independent fixed-spin `[00|00]` amplitude in the historical charge
marking is the unfiltered all-plus amplitude,

\[
D_{[00|00]}=D_{+++}=S_{000}+S_{110}+S_{101}+S_{011}.
\]

The historical implementation instead evaluates the quadratic-parity
filtered expression, at lifts `(+,-,+)`:

\[
\boxed{F_{\rm legacy,+-+}
=S_{000}+S_{110}-S_{101}+S_{011}
=D_{[00|00]}-2S_{101}.}
\]

This is an exact parity decomposition, verified at the complex-amplitude
level. It is not a finite-order approximation. Equivalently, in lift order
`+++`, `+-+`, `-++`, `--+`, the four legacy amplitudes satisfy
`F=(J/2-I)D`, where every entry of `J` is one.

At the reference point `o0243`,

\[
\begin{aligned}
q_0&=8.37794\times10^{-7}+3.84815\times10^{-7}i,\\
q_1&=0.24390749-0.04149429i,\\
q_\infty&=1.80796\times10^{-6}-6.32533\times10^{-8}i.
\end{aligned}
\]

The distinguished sector is odd on both exceptionally thin endpoint tubes:
`S101` starts at `sqrt(q0)*sqrt(q_infinity)`. Thus its contribution was tiny.
The other two leading two-fermion sectors have the same sign in the
filtered amplitude and the claimed fixed spin.

| Historical point | `Z_free,legacy / Z_free,[00|00] - 1`, theta | Original `Q_theta/Q_glasses - 1` | Fixed denominators only, same saved numerators |
|---|---:|---:|---:|
| o0243 | -5.46775e-6 | -2.983919e-4 | -3.475859e-4 |
| o0127 | -3.58583e-6 | -1.258953e-4 | -1.581632e-4 |
| o0015 | -6.35816e-6 | -3.331207e-4 | -3.903236e-4 |
| o0167 | -1.27735e-6 | -4.428686e-5 | -5.578244e-5 |
| o0239 | -9.27065e-7 | -4.895938e-5 | -5.730252e-5 |

Glasses legacy/fixed discrepancies are below `2.1e-15`. The old free values
are reproduced to `2.6e-15`; mode 24 to 32 changes the free factors by at
most `1.9e-14`. Increasing mode cutoff cannot remove the theta difference.

The last column is a **counterfactual denominator-only diagnostic**, not
a recomputed or newly certified physical partition function. It shows
that the previous near agreement survives, but was not sensitive enough
to validate this subtle spin identification. For `o0243`, the effect on
the quotient ratio is only about `4.92e-5`, smaller than its numerical
Liouville residual.

A free-only stress test makes the hidden sector visible. Multiply `q0`
and `q_infinity` by a common factor, keep `q1` fixed, and use the resulting
charge marking throughout:

| Endpoint scale | Theta legacy/fixed free residual |
|---:|---:|
| 1 | -5.46775e-6 |
| 10 | -5.45237e-5 |
| 100 | -5.40369e-4 |
| 1000 | -5.24928e-3 |
| 10000 | -4.76127e-2 |

All three plumbing moduli remain below `0.248`. This is a family of new
free-theory geometries, not a Liouville cross-channel run. The fixed-spin
bosonization identity continues to pass throughout.

## 4. Where the earlier spin certification was incomplete

`ns_genus2_partition._spin_characteristic_from_lifts` assigns the theta
beta bits with a hardcoded second-bit offset. The production spin audit
and the rerun driver call this same helper. Checking agreement with its
declared `[00|00]` label, followed by a correct affine transformation of
that label, does not independently determine the spin of the actual
filtered function being evaluated.

Specifically, in the historical charge/marked frame, the *unfiltered*
`+-+` determinant is `[00|01]`, while the *filtered* `+-+` expression is
the combination displayed above, not a single fixed-spin determinant.
The helper's docstring and the rerun metadata describing the latter as
a fixed-spin determinant therefore overstate what was established.
The old direct-Fock-versus-Fredholm tests verify the resummation of the
same filtered prescription; they are not independent physical-spin tests.

This does not identify an error in the checked PBW/double-Virasoro block
algorithms. Nor does it authorize deleting the Human-Note quadratic sign.
The missing identification is between the package's ordered local
sewing prescription and the global physical spin structure, including
the spin-frame phases and physical nonchiral pairing.

## 5. Consequence for the current NSRR/all-NS comparison

At the current original period
`Omega(t)=[[i,t+i/2],[t+i/2,i]]`, `t=0.60`, `b=1.4`, the target chart has

\[
\Omega_{\rm charge}=\Omega_{\rm marked}
+\begin{pmatrix}-1&-1\\-1&0\end{pmatrix}.
\]

Thus the desired marked `[00|00]` is charge characteristic `[00|10]`,
whose unfiltered representative is `-++`, not the historical `+++`.
The same selected filtered `+-+` expression now obeys

\[
F_{\rm legacy,+-+}=D_{[00|00],\rm marked}+2S_{110}.
\]

Here `S110=-0.0100939641118-0.0315912533653i` is not suppressed by two
ultrathin endpoint tubes. The full free values are

\[
Z_{\rm free,legacy}=0.5372515435695151,\qquad
Z_{\rm free,[00|00]}=0.5638924570404817,
\]

giving a `-4.724467%` discrepancy. This is a *free-factor control*, not an
explanation that quantitatively solves the remaining `-3.948799%`
Liouville quotient discrepancy. In particular, the current trial already
uses the fixed-spin free denominator, not the legacy filtered one.

There are two compatible nonzero even-spin pairs with the required
source NSRR and target all-NS cuts. Re-evaluating both gives:

| Source marked spin | Target marked spin | Source free factor | Target free factor | Source/target |
|---|---|---:|---:|---:|
| `[11|00]` | `[00|00]` | 0.575409592371721 | 0.563892457040482 | 1.02042434720919 |
| `[11|11]` | `[00|10]` | 0.512619621002060 | 0.502359261026763 | 1.02042434721782 |

The other two source beta choices are odd characteristics and have
fermion zero modes; they cannot provide a nonzero free denominator.

Changing the paired even spin, with both Liouville numerators held fixed
only for this diagnostic, multiplies the quotient ratio by
`0.999999999915968` at `kappa=9.940408163265307`. This is the expected
spin independence of the **relative local-frame anomaly**. The individual
physical partition functions are of course spin dependent.

## 6. What should be derived next

The useful next check is the local-to-global sewing dictionary, not a
search over denominator spins and not a higher block cutoff:

1. Start from physical Majorana half-differentials on each pant. Track
   boundary orientation, the square root of the sewing Jacobian, and the
   physical graded pairing through the two pants contractions.
2. Express the resulting parity-resolved amplitude in the Human-Note
   ordered block basis. Require it to reproduce the unfiltered fixed-spin
   free determinant and its charged period, for generic nonsymmetric
   plumbing values, before assigning a global characteristic.
3. Carry that dictionary to the NSRR zero-mode/ground-state contraction
   and to the all-NS numerator. Keep the existing three-point `i` convention
   and the independently checked block algorithms unchanged.

This audit localizes the failure of the earlier *spin identification*;
it does not yet supply that full boundary dictionary or solve the physical
NSRR nonchiral assembly.

## Reproduction

New diagnostic driver:
`Code/genus_2/audit_previous_all_ns_free_spin.py`.
Full numerical output:
`Data Set/previous_all_ns_free_spin_audit_20260830.json`.
The driver refuses to overwrite an existing report; importing and calling
`run()` recomputes without writing files.

The ten new tests and the existing fixed-spin and spin/quadrature suites
pass: **32 tests** in total. All eight protected kernel SHA256 values match
the saved manifest before and after the audit. No production fix or new
cluster job was made.
