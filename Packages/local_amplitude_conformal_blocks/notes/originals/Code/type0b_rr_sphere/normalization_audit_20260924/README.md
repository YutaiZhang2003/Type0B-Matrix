# Ramond primary metric and asymptotic-state normalization audit

Checked 2026-09-24 after the user required an independent sewing-metric
check before interpreting the Type 0B four-R scattering comparison.

**Result:** the finite `R+` small-representation vector is normalized, but
this does not fix the continuum normalization of the field multiplying it.
With the half-sum ground couplings used by the scattering code, both the
actual identity residue and its R-exchange primary factorization give
`D_R/D_NS=1/2`. If `D_NS=pi delta(P-P')`, the effective field metric is
`D_R=(pi/2) delta(P-P')`. It must not also be treated as `pi delta` without
converting the field, couplings and external-vertex coefficients together.

This is a statement about the implemented normalization dictionary. BRY
prints equal metrics; we have not reconciled that statement, its half-sum
couplings and the full asymptotic string-state normalization simultaneously.
The retained matrix-model match is not an independent proof of that map.
No production formulas or frozen data are changed by this audit.

## Finite ground state and BPZ dual

In the ordered tensor basis `(++,+-,-+,--)`, the actual
`literature_component_blocks.ramond_state` implements

\[
u_+=\frac1{\sqrt2}(1,0,0,-i)^T,\qquad
u_-=\frac1{\sqrt2}(0,1,1,0)^T.
\]

These are the small-representation states in
[Suchanek (7)](https://arxiv.org/pdf/1012.2974).
Writing `U=(u_+,u_-)`, the graded bilinear BPZ frame used in the native
notes has

\[
D=\operatorname{diag}(1,-1,-1,-1),\qquad
g=U^TDU=\operatorname{diag}(1,-1).
\]

The dual is `g^{-1} U^T D`, and the physical projector is

\[
\Pi=U g^{-1} U^T D,
\qquad \Pi^2=\Pi.
\]

In this finite frame `Pi=U U^dagger`; in particular the `R+` BPZ norm is
one. This equality is explicitly checked, not assumed as a replacement
of bilinear sewing by a Hermitian metric. Omitting the `1/sqrt(2)` would
give norm two and require the corresponding inverse Gram factor.
The production embedding includes that normalization correctly.

This calculation strips the continuum primary metric. Its result cannot
establish the normalization of `R_P(0)|0>` relative to the unit reduced
vector `u_+`. The momentum-conjugated analytic block table likewise
implements phases and analytic continuation; it does not independently
fix that continuum scalar.

## Identity residue fixes the relative field metric

Let `E,O` be the unchanged raw BRY (3.9) functions. The scattering code
uses `d_+=(E+O)/2` and `d_-=(E-O)/2` in its NS-channel factorization check.
For the same NS identity insertion `alpha=epsilon -> 0+`, the literal
Barnes-G functions give

\[
\frac{E}{C_{\rm NS}}\to1,\qquad
\frac{O}{C_{\rm NS}}\to0,
\qquad
\frac{d_+}{C_{\rm NS}}\to\frac12.
\]

With the common identity normalization, the last ratio measures
`D_R/D_NS`. The fresh checks use `P=0.23,0.67` and
`epsilon=1e-4,1e-6,1e-8`; see [sewing_metric.json](sewing_metric.json).
The earlier audit additionally checked offsets across the delta-function
peak. BRY's printed metric and coefficient statements are in
[§3.1, (3.1),(3.6),(3.8),(3.9)](https://arxiv.org/html/2201.05621#S3.SS1).

## Check the inverse metric in the actual production banks

For each channel, the audit extracts the leading primary coefficient
using the saved holomorphic and analytic-dual tables and the actual
`ALGEBRA.sewing_terms`. It compares it with the product of the ground
three-point amplitudes, before the common `dP/pi` integration measure.
For the R channel the right vertex retains its reflected internal momentum
`-P`; the block-sign reflection is retained independently.

| External fields and exchanged sector | Three-point product in the scattering convention | Actual leading coefficient / product |
|---|---|---:|
| Four `R+`, NS exchange | `d_L d_R` | 1 |
| Two `R+`, two NS, NS exchange | `C_NS d_R` | 1 |
| Two `R+`, two NS, R exchange | `d_L d_R` | 2 |

These results hold in all 27 probes: three momentum nodes in each channel
at each of `t=0.10,0.25,0.70`. The maximum deviation from the displayed
integers is `5.14e-15`. Loading these probes verifies all 288 relevant
native bank records and the frozen scientific-source manifest.

Consequently, in terms of the half-sum field couplings, the R-channel
sewing contains the effective inverse primary metric `2/pi`, despite
the common integration code writing `dP/pi`. A measure label alone does
not specify the metric after the other normalization factors have been
absorbed into the spectral density.

The existing `test_primary_factorization_fixes_external_normalization`
only tests the first two rows. Its success did not check the R-exchange
metric. All seven existing worldsheet tests passed in the current session
(13.674 seconds), but that does not add the missing absolute state-to-field
identification.

## Consistent conversion and the scattering comparison

The two field conventions are related by

\[
\widehat R=\sqrt2\,R_{\rm sc},\qquad
\widehat D_R=2D_{R,\rm sc}=D_{\rm NS},\qquad
\widehat d_\pm=2d_{\pm,\rm sc}=E\pm O.
\]

This is the interpretation of `c_+=E,c_-=O` in the current equal-metric
native convention. With unchanged external NS fields, a mixed correlator
scales by two and a four-R correlator by four. An internal R inverse
metric scales by one half. A closed graph with two NSRR vertices and two
R edges is invariant under the *consistent* change: `2^2 / 2^2 = 1`.
Using the old half-sum couplings with an equal NS/R metric instead mixes
the two conventions and suppresses that closed graph by four.

The matter part of the scattering vertex is written as

\[
A^\eta_{\rm sc}=\frac1{\sqrt2}
 (\sigma^0 R^+_{\rm sc}+\eta\mu^0 R^-_{\rm sc})
=\frac12(\sigma^0\widehat R^++\eta\mu^0\widehat R^-),
\]

with the common `g_s omega`, ghosts and time exponential omitted.
Converting the Liouville fields while retaining this *same* physical
vertex therefore supplies `1/sqrt(2)` per external leg from its coefficient.
The factor four in the Liouville correlator cancels against
`(1/sqrt(2))^4=1/4`.

The retained 64-node `t=0.25` raw integral, freshly converted without any
fit, gives:

| Operation | Amplitude / matrix-model target |
|---|---:|
| Original complete scattering prescription | 0.999999259277 |
| Multiply the Liouville four-R correlator by four, external coefficients fixed | 3.999997037108 |
| Convert both the Liouville fields and the coefficients of the same external vertex | 0.999999259277 |

These are exact normalization reweightings of a retained moduli integral,
not fresh integrations. They show why matching the complete amplitude
cannot determine the field metric separately from the external-state
coefficient. They do not prove that the old axion vertex is canonically
normalized as a string asymptotic state. That identification still needs
the full BRST/BPZ picture pairing and the matrix-model state dictionary.
The Liouville `R+` field should also not be confused with the right-sea mode
`mathcal R^+=T^++A^+` in BRY (2.9)–(2.10).

## Reproduction

From the repository root, with Python 3.11, numpy, mpmath, scipy and sympy:

```sh
python -B Code/type0b_rr_sphere/normalization_audit_20260924/check_sewing_metric.py
```

The report records all probes, source hashes, the finite inverse metric,
the identity regulators and the retained scattering input hash. This is a
primary-normalization audit, not a proof of every descendant pairing or
of the full physical genus-two projector.
