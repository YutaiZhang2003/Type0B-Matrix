# NSRR normalization: current dictionary and sphere checks

This is the current machine-note normalization rule for the native sewing
calculation. It supersedes the identification `c_+=E/2,c_-=O/2` in the older
NSRR notes. The correction is fixed by the actual Liouville identity limit
and tested by mixed sphere crossing, independently of genus-two matching.
It does not establish the full interacting genus-two Ramond pairing.

**Scattering-metric follow-up:** The Type 0B scattering code's half-sum
couplings use an effective `D_R/D_NS=1/2`, verified from its actual
R-exchange primary sewing and identity residue. Thus the equal-metric
dictionary here is a field-and-metric conversion, not a rule to double
every NSRR vertex in another convention while leaving its other factors
fixed. The external axion coefficient must transform with that field.
See [the sewing-metric audit](../../Code/type0b_rr_sphere/normalization_audit_20260924/README.md).

## 1. Dictionary to use

Let `E,O` be the unchanged numerical Upsilon functions returned by
`rr_ns_structure_constants` or `GenericSuperLiouvilleConstants.rr_ns_constants`.
At `b=1` these are the functions printed in BRY (3.9), including their
common prefactor. With equal NS and physical R-family metrics,

\[
D_{\rm NS}(P,P')=D_{{\rm R},\pm}(P,P')=\pi\delta(P-P'),
\qquad P,P'>0,
\]

and completeness measure `dP/pi` in either sector, use

\[
\boxed{c_+=E,\qquad c_-=O,\qquad d_+=E+O,\qquad d_-=E-O.}
\]

Here `c` labels coefficients of the normalized ordered three-forms;
`d` labels physical equal-family ground three-point amplitudes. The linear
relation `c_+=(d_++d_-)/2,c_-=(d_+-d_-)/2` remains unchanged. It is the
identification of `d` with the raw numerical inputs that has been corrected.
The odd NS conversion also remains `C_0=C_BRY,C_1=i*Ctilde_BRY` at the
Human-Note graded sewing boundary, applied once as specified in
[the master convention ledger](../conventions.md#41-global-scope-rule-bry-coefficients-versus-the-human-note-graded-basis).

Replacing the old `E/2,O/2` adapter by `E,O` doubles each NSRR vertex and
multiplies a two-vertex contribution by four, with all other data fixed.
**Apply the correction once at the coefficient boundary. Do not also
multiply the corrected result by four.** The R two-point metric, continuum
measure, chiral blocks, Gram matrices, primary powers and BPZ phases do not
change in this normalization correction.

The native convention is
`unit_identity_upsilon_coefficients_2026-09-24`, with
`normalization_factor: 1`; see
[the native normalization note](../../C++/PARTITION_NORMALIZATION.md).
The explicit `1/2` in the current native NSRR partition contraction is a
separate ground-pairing factor for the paper's unprojected `w^+,w^-` block
basis. It is retained; see [the partition prescription](../../C++/PARTITION.md).

For comparison with historical data only, holding the old block basis and
pairing prescription fixed would change the old matrix coefficient
`B_L B_R/8` to `B_L B_R/2`. This is coefficient bookkeeping, not a proof that
the historical pairing and the current native geometric BPZ prescription
are equivalent. Saved historical results retain their original convention.

## 2. Actual identity limit

Insert an NS field with `alpha_NS=epsilon -> 0+` and compare the actual
NSRR functions with `C_NS` at the same momenta and identity regulator.
The shared external identity normalization cancels. The Upsilon reflection
and zero identities give

\[
\frac{E}{C_{\rm NS}}\longrightarrow1,\qquad
\frac{O}{C_{\rm NS}}\longrightarrow0,
\qquad
\frac{d_\pm}{C_{\rm NS}}\longrightarrow1.
\]

This also holds across the delta-function peak with
`P2-P1=epsilon*t` at fixed finite `t` and positive limiting momentum.
Thus `d_+=(E+O)/2,d_-=(E-O)/2` would reproduce half the required R metric.
The analytical ratios and generic-`b` leg conventions are written out in
[the native normalization derivation](../../C++/PARTITION_NORMALIZATION.md#how-the-normalization-is-fixed-without-the-genus-two-comparison).

The earlier module/torus checks assigned `E=2,O=0` by hand. They verified
a Clifford contraction but never took the identity limit of the imported
Liouville functions. Their claim to exclude this factor-four correction
is withdrawn. After dividing out the common NS identity residue, the
actual inputs approach `E=1,O=0`.

The new checks comprise 30 generic-`b` cases at `b=0.8,1,1.4`, including
offsets across the peak, and 18 direct `b=1` Barnes-G cases independent of
the generic-`b` leg machinery. At `epsilon=1e-8` the direct test gives a
maximum deviation of `(E+/-O)/C_NS` from one of `2.153e-8`; the half-sums
approach one half. Reports and reproduction commands are in
[the sphere audit](../../C++/experiments/sphere_ramond_normalization_2026-09-24/README.md#identity-residue-and-the-correction-to-the-earlier-reasoning).

## 3. Mixed sphere four-point test

For `G(z)=<V4(infinity) V3(1) R2^+(z) R1^+(0)>`, the direct NS channel
contains one NSRR vertex and one NSNSNS vertex. The crossed R channel
contains two NSRR vertices and one inverse R metric. If `a` scales the
NSRR vertex and `g_R` scales the R metric relative to the convention above,

\[
G_{\rm NS}\mapsto aG_{\rm NS},\qquad
G_{\rm R}\mapsto\frac{a^2}{g_R}G_{\rm R},\qquad
\frac{G_{\rm R}}{G_{\rm NS}}\mapsto
\frac{a}{g_R}\frac{G_{\rm R}}{G_{\rm NS}}.
\]

With `c=(E,O)` and equal metrics, the fresh `b=1` calculation at
`z=0.37+0.11i`, block order 12 and 48 Gauss nodes on `0<P<5` gives:

| External momenta `(P1_R,P2_R,P3_NS,P4_NS)` | NS channel | R channel | Relative difference |
|---|---:|---:|---:|
| `(0.20,0.40,0.30,0.30)` | 0.164677179072911 | 0.164677179070223 | 1.6321e-11 |
| `(0.17,0.38,0.26,0.49)` | 0.197755394262737 | 0.197755394257739 | 2.5273e-11 |

An extra vertex half with the metric fixed makes `R/NS` approximately
`1/2`; doubling only the R metric does the same. A consistent field
rescaling `R -> R/sqrt(2)` has `a=g_R=1/2`, preserves crossing, and cancels
on a closed graph with two NSRR vertices and two R edges. Such a consistent
rescaling cannot repair a closed-graph normalization discrepancy.

These multiplier controls are algebraic reweightings of the fresh
integrals. Block-order, quadrature and central-charge-regulator refinements
were checked separately. Halving the symmetric regulator about `c=13.5`
changes individual integrals by up to `1.03e-10` relative, so the smaller
channel differences are not certified absolute errors. Independent
Ward/Gram versus recursion tests also passed. See the
[full sphere audit and its saved reports](../../C++/experiments/sphere_ramond_normalization_2026-09-24/README.md).

## 4. Literature and remaining scope

BRY explicitly specifies equal NS/R metrics in (3.1),(3.6), prints the
half-sum relation in (3.8), and gives the raw functions in (3.9).
The literal identification of all three with our unchanged raw inputs
and normalized ordered forms fails the identity test above. The current
native dictionary is therefore stated explicitly. This is not a claim
of an author-issued erratum or a complete reconciliation of every printed
BRY normalization statement. [BRY, section 3.1](https://arxiv.org/html/2201.05621#S3.SS1)

Four-R sphere crossing alone cannot fix the common NSRR scale: both
channels contain two NSRR vertices. The legacy `RRRRSphereCorrelator`
passes the checked crossing comparisons, but retains a separate overall
`1/2`; relative to ground amplitudes `d=E+/-O` and unit NS primary metric,
its primary degeneration gives half their product. That absolute-scale
dictionary remains to be reconciled. No legacy sphere production formula
or frozen Type-0B scattering assembler was changed by this audit.

The coefficient correction does not prove the full interacting Ramond
projector, fixed-spin genus-two sewing or spin transport, nor a string
amplitude after moduli/PCO integration. Those tasks require their own
checks; they must not be declared solved from the normalization tests.

## 5. Field normalization must accompany the sewing metric

The subsequent [Type 0B metric audit](../../Code/type0b_rr_sphere/normalization_audit_20260924/README.md)
checks the reduced `R+` vector, its BPZ dual and the actual leading
coefficients in all three frozen sphere channels. The vector includes its
`1/sqrt(2)` and has reduced norm one. In terms of the half-sum *field*
couplings, however, the R channel has inverse metric `2/pi`, while the
NS channel has `1/pi`; this agrees with the identity residue.

Writing the scattering field as `R_sc`, the consistent conversion is
`R_unit=sqrt(2)*R_sc`, `D_R_unit=2*D_R_sc`, and `d_unit=2*d_sc`.
Consequently the same physical axion combination is
`(sigma R_sc^+ +/- mu R_sc^-)/sqrt(2)` or
`(sigma R_unit^+ +/- mu R_unit^-)/2`.
Its four-point amplitude is unchanged by this complete conversion.
Only changing the Liouville correlator would multiply it by four.

The matrix-model comparison therefore does not independently establish
the absolute asymptotic-state normalization. Reconciliation with BRY's
printed equal metrics and its physical string-vertex coefficients remains
open; a full string BPZ/BRST picture pairing is needed for that claim.
