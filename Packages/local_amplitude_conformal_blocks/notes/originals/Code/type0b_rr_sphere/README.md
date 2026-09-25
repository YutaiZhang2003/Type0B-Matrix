# Type 0B mixed and four-RR sphere amplitudes

**To our knowledge, the first direct numerical sphere-moduli integration
of the mixed NSNS–RR and four-RR amplitudes in two-dimensional Type 0B.**
The exact scope and prior-work qualification are in [NOVELTY.md](NOVELTY.md).

The self-contained, frozen production is in
[type0b_rr_sphere_production_20260921](../type0b_rr_sphere_production_20260921/README.md).
Use that release and its separate-workspace launcher for reproduction.
This directory remains the working source.

This is a direct worldsheet computation at `b=1`, `alpha'=2`, zero RR flux.
It reuses the **super-Liouville coefficient banks** in the frozen HetSO(23)
bundle and assembles Type 0B external states, free time fermions, ghosts and
picture changes. The heterotic amplitudes and Spin(7) tensor are not input
amplitudes for this computation.

The numerical results and qualifications are in
[the worldsheet report](../../Machine%20Notes/TYPE0B_RR_WORLD_SHEET_2026-09-21.md).
`T` denotes the NSNS tachyon and `A` the RR axion. The cached nine-energy
scan covers the independent mixed `AATT` and four-RR `AAAA` amplitudes,
evaluated as `A -> ATT` and `A -> AAA`. `T -> TAA` is related to the mixed
amplitude by crossing and is not computed separately. These are genus-zero,
equal-outgoing **imaginary-energy**
calculations, not a real-energy continuation or a higher-genus computation.

## Input provenance and analytic continuation

The prepared input directory is
`Data Set/type0b_rr_sphere_20260921/het_inputs`, relative to the project root.
It was created with the original frozen bundle's `run.py prepare --mode banks`.
The 515-file release verification passed before copying. The release is
`706bef6ef7b6306210b29c7c05a2ef5d9817e7bc55da06127918a9bb116d867a`.

`frozen.py` verifies the manifest, scientific-source hashes and every used
bank's task and payload checksum. It imports the frozen coefficient-table,
graded sewing and local-series machinery. It does not run a heterotic
amplitude driver. Native tables include both analytic chiralities and
physical levels 4 and 5, assembled at 60 decimal digits and extrapolated
from `b=1.01,1.02` at fixed physical momenta. The exact `b=1` prefactors and
structure constants are then used.

The cached mixed banks have ordered sectors `(R,R,NS,NS)` and momenta
`(it/3,it,it/3,it/3)`. Thus the incoming particle is an axion. The ordered
momentum labels are preserved when reusing these banks. Crossing supplies
the other external-state placements of this mixed amplitude.
All new momentum-refinement banks are outside `het_inputs`.

For complex momenta, the antiholomorphic block is the conjugate of the
stored **dual** table, whose momentum parameters are conjugated first.
This conjugates convention phases and the coordinate while preserving
the physical analytic momenta. A real-momentum `abs(block)**2` API is
not used as an analytic-continuation prescription.

## External states and picture changes

We use [BRY, arXiv:2201.05621](https://arxiv.org/html/2201.05621), equations
(2.6)-(2.7):

\[
T^\eta_\omega=g_s c\bar c\,e^{-\phi-\bar\phi}e^{i\eta\omega X^0}V_\omega,
\qquad
A^\eta_\omega={g_s\omega\over\sqrt2}c\bar c\,
e^{-\phi/2-\bar\phi/2}e^{i\eta\omega X^0}
(\sigma^0 R^+_\omega+\eta\mu^0 R^-_\omega).
\]

Here `eta=+1` is incoming and `eta=-1` outgoing. Every RR leg supplies
its explicit factor of `omega`.

Four RR vertices already have picture `(-2,-2)` in total. For two RR and
two NS vertices, one NS vertex is raised in **both** chiralities. The code
computes both choices of NS vertex. All four terms of BRY (4.2) are kept:
`W`, `psi psibar V`, `psi Lambdabar`, and `psibar Lambda`.

The mixed frame is `(R_0,R_z,NS_1,NS_infinity)`. In the code `k_i` is
positive for an outgoing energy and negative for an incoming energy;
the exponential in the vertex is `exp(-i k_i X^0)`. For the raised NS leg
`j`, put `epsilon=-k_j`, `u=1-a`, `v=1-b`, where `(a,b)` is its pair of
Liouville descendant bits. Define

\[
f_r={e^{-i\pi(1-2r)/4}\over\sqrt2},\qquad
B_1={\sqrt z\over\sqrt{1-z}},\qquad B_\infty=\sqrt z.
\]

After transport to the stored HJS graded frame, the term coefficient is
`eta_0^r eta_z^s / 2` times the following table, with `r+s=u+v mod 2`:

| SL component `(a,b)` | Time insertions | Coefficient, excluding `B_j^u Bbar_j^v` |
| --- | --- | --- |
| `(1,1)` | none | `1` |
| `(0,0)` | `psi psibar` | `i (-1)^r epsilon^2 / 2` |
| `(0,1)` | `psi` | `+epsilon f_r` |
| `(1,0)` | `psibar` | `-epsilon conjugate(f_r)` |

The infinity frame additionally supplies `(-1)^(a*b+u+v)`: reversal of
the two odd SL modes in the bra and transport of the time fermions.
The common free-time and superghost factor has chiral exponents

\[
z^{-k_0k_z-3/8}(1-z)^{-k_zk_1-\delta_{j,\infty}/2},
\]

and the analogous antiholomorphic factor, with the momenta unchanged.
In this stored full-field convention the common raised-vertex factor is
`-i/4`. The explicit mixed kernels satisfy the pointwise picture-changing
Ward identity; this test is performed before moduli integration and uses
no matrix-model target.

## Four-RR chiral GSO projection

Let `I_0,I_1` be the identity and fermion free-Majorana chiral blocks,
both with leading coefficient one. In particular,

\[
I_0=[z(1-z)]^{-1/8}\sqrt{{1+\sqrt{1-z}\over2}},\qquad
I_1=[z(1-z)]^{-1/8}{\sqrt z\over\sqrt{(1+\sqrt{1-z})/2}}.
\]

The physical fermion-channel coefficient is `1/2`. For three outgoing
axions and one incoming axion the two chiral combinations are

\[
K_0=F_{\rm e}^{-,+}I_0-\tfrac12F_{\rm o}^{-,+}I_1,
\qquad
K_1=F_{\rm o}^{+,-}I_0-\tfrac12F_{\rm e}^{+,-}I_1.
\]

Sew `rho_{-,+} K_0 Kbar_0 + rho_{+,-} K_1 Kbar_1`. Bars here mean analytic
antiholomorphic continuation, not conjugation of physical momenta. The
structure-constant signs follow the outgoing/outgoing and incoming/outgoing
RRNS trinion projections. The relative spinor sign follows the Ramond
zero-mode constraint, and the factor `1/2` is the free-Ising OPE coefficient.

There is an equivalent sixteen-component construction. In the canonical
chiral block frame it requires both the defect-line pairing sign
`(-1)^(r_0*r_infinity+r_z*r_1)` and the odd-form conversion
`(s_L*s_R)^(q_h+q_anti)`. `four_r_finite_terms` retains these separately.
Tests compare it to the two chiral combinations above. Omitting the
transport signs can give a superficially close amplitude while failing
one crossing transformation. The production uses the chiral expression.

## Normalization and integration

**Sewing-metric audit, 2026-09-24:** The factors below reproduce the saved
scattering results, but their half-sum field couplings have effective
`D_R/D_NS=1/2`, as shown by the identity residue and R-exchange primary
factorization. The finite `R+` ground vector itself has unit reduced BPZ
norm. See [the metric audit](normalization_audit_20260924/README.md) before
identifying the continuum field with an equal-metric asymptotic state or
transferring a normalization correction from another assembler. A consistent
field rescaling must also transform its inverse sewing metric and the
coefficients of the physical external vertices.

Write `N_NS,N_R` for the BRY factors in section 3.1. The stripped stored
Liouville densities are restored by

\[
\mathcal N_{RRNN}=-\frac18\prod_i N_{s_i}(p_i),\qquad
\mathcal N_{RRRR}=\frac1{16}\prod_i N_R(p_i).
\]

These factors are checked against products of **three-point** structure
constants in the NS primary channel. In the saved scattering dictionary the RRNS
constant is `(C_even +/- C_odd)/2`; dropping these halves changes the
four-point normalization if other factors are held fixed. The integration
measure is `dP/pi`, applied once; the spectral density also contains relative
primary-metric factors, as the audit above makes explicit.

The three charts partition the complex plane into images of
`D={|z|<1, Re z<1/2}`. A full disk around zero in each native chart is
integrated by analytic local series; the remaining region uses Gaussian
quadrature on both cut lips. The Jacobian cancels the complete-vertex
conformal factor in the infinity chart. Four identical outgoing-momentum
RR configurations use the three equivalent chart contributions. Mixed
charts are assembled separately, including the exchanged NS picture.

Using `g_s=4/(pi mu_F)`, `C_sphere=pi/g_s^2`, the reported amplitudes are
`mu_F^2 M`, with the energy delta function omitted. The known comparison is

\[
\mu_F^2 M={i t^4(1-2t)\over27},\qquad
\omega=it,\quad\omega_1=\omega_2=\omega_3=it/3.
\]

`matrix_prediction` is used only in reports, after the worldsheet integral.
It supplies no integrand coefficients, phases or fitted scale.

## Reproduction

Python 3.11 or newer is needed. Dependencies are listed in `requirements.txt`.
From the project root:

```sh
python3.11 -B -m unittest discover -s Code/type0b_rr_sphere -p 'test_*.py' -v
python3.11 -B Code/type0b_rr_sphere/compute.py overlap --output /tmp/type0b-overlap.json
python3.11 -B Code/type0b_rr_sphere/compare_integrands.py --output /tmp/type0b-pointwise.json
python3.11 -B Code/type0b_rr_sphere/compute.py integrate --energy t0250 --angular 48 --output /tmp/type0b-t0250.json
python3.11 -B Code/type0b_rr_sphere/refine_momentum.py --nodes 64 --workers 3 --output /tmp/type0b-rr-N64
```

The fresh-bank commands checkpoint and resume; they do more work than
reusing the cached integrator. No cluster jobs are submitted.

`report.py` regenerates the normalized scan and figure from the retained
production outputs. See the [dataset inventory](../../Data%20Set/type0b_rr_sphere_20260921/README.md)
for the selected fields and superseded development outputs.

The present scan tests an analytic, convergent imaginary-energy region.
The reused quadrature and self-dual extrapolation are numerical
approximations. Block-order stability alone is not a bound on momentum,
moduli or regulator error. Real-energy amplitudes require a separate
continuation and collision/residue prescription. General unequal outgoing
energies require matching ordered coefficient banks.
