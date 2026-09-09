# Type-0B NS five-tachyon: direct physical \(+i\epsilon\) subtraction ledger

Date: 2026-08-28

Status: **stopped under the 3.3-hour limit; worldsheet not frozen;
matrix-model blind**.

This is the replacement for the retracted remote-complex-ray freeze path.
The external energies remain infinitesimally close to the physical domain,
the internal super-Liouville momenta remain on \(P_1,P_2\in\mathbb R_+\),
and the moduli divergences are defined by local BRY-style polynomial
subtraction.  No matrix-model expression or target value is loaded.

## 1. Physical boundary value

Let \(E_a>0\) and \(\nu_a>0\), and set

\[
 \omega_a=E_a+i\epsilon\nu_a,\qquad
 \omega_0=\sum_{a=1}^4\omega_a,\qquad \epsilon\to0^+.
\]

The signed timelike momenta are

\[
 k=(\omega_0,-\omega_1,-\omega_2,-\omega_3,-\omega_4).
\]

For an outgoing--outgoing divisor, both the real and imaginary parts of
\(K_{ij}=k_i+k_j\) are negative.  For an incoming--outgoing divisor they are
both positive.  Consequently

\[
 \operatorname{Im}K_{ij}^2>0
\]

for every divisor.  Each radial denominator therefore approaches the
physical boundary value from below,

\[
 \lambda_{ij,n}(P)
 =P^2-K_{ij}^2-\tau_{ij}-\frac{\delta c}{12}+2n
 \longrightarrow
 P^2-K_{ij,\mathrm{phys}}^2-\tau_{ij}+2n-i0.
\]

For the default equal tilt \(\nu_a=1\), no external super-Liouville pole
crosses the positive-real internal contour provided \(5\epsilon<1\).  The
production scan stays far inside this chamber.  Hence this prescription has
no Liouville residue forest.

## 2. Complete divergent regions

The fixed gauge and picture choice are

\[
 (z_0,z_1,z_2,z_3,z_4)=(\infty,1,0,z,w),\qquad
 R_{\rm PCO}=\{0,1,2\}.
\]

For a pair \(\{i,j\}\), let \(r_{ij}\) be its number of picture-zero
vertices.  The picture threshold is

\[
 \tau_{ij}=r_{ij}-1,
\]

except for \(\{3,4\}\), where the nonchiral superghost factor
\(|z_3-z_4|^{-2}\) gives \(\tau_{34}=1\).  If the angularly diagonal normal
series is indexed by \(n\ge0\), its degree-\(n\) term is divergent on

\[
 \boxed{
 0\le P\le P_{ij,n}^*,\qquad
 (P_{ij,n}^*)^2=
 \tau_{ij}+\frac{\delta c}{12}
 +\operatorname{Re}K_{ij}^2-2n>0.
 }
\]

Only finitely many \(n\) obey this inequality.  The analytic radial finite
part used for each diagonal coefficient is

\[
 \boxed{
 \operatorname{FP}\!\int_{|q|<\rho}d^2q\,|q|^{\lambda-2}
 =\frac{2\pi\rho^\lambda}{\lambda},
 \qquad
 \lambda=P^2-K_{ij}^2-\tau_{ij}-\frac{\delta c}{12}+2n.
 }
\]

This formula is evaluated at finite \(\epsilon>0\), and only after the
subtracted moduli integral is numerically controlled is \(\epsilon\to0^+\)
taken.

## 3. First physical benchmark

Use

\[
 E_1=E_2=E_3=E_4=\frac14,\qquad E_0=1,
 \qquad \nu_1=\nu_2=\nu_3=\nu_4=1.
\]

At \(\epsilon=0\) and \(\delta c=0\), the ten face records are:

| divisor | \(|K_{ij}|\) | \(\tau_{ij}\) | divergent interval for \(n=0\) |
|---|---:|---:|---:|
| \(D_{01}\), \(D_{02}\) | \(3/4\) | 1 | \(0\le P\le5/4\) |
| \(D_{12}\) | \(1/2\) | 1 | \(0\le P\le\sqrt5/2\) |
| \(D_{03}\), \(D_{04}\) | \(3/4\) | 0 | \(0\le P\le3/4\) |
| \(D_{13}\), \(D_{14}\), \(D_{23}\), \(D_{24}\) | \(1/2\) | 0 | \(0\le P\le1/2\) |
| \(D_{34}\) | \(1/2\) | 1 | \(0\le P\le\sqrt5/2\) |

For every row the degree-one threshold is already negative.  Thus the full
face subtraction polynomial has degree zero on all ten divisors.  Every pair
of disjoint divisors is compatible, giving the fifteen stable corners of
\(\overline{\mathcal M}_{0,5}\); each corner requires the single overlap
degree \((n_L,n_R)=(0,0)\).

## 4. Recursion partition and freeze conditions

At every bulk, face, and corner evaluation, first choose the geometrically
proper convergent comb.  Use multipoint \(h\)-recursion if and only if every
active CCY plumbing parameter satisfies

\[
 \max_e|q_e|<0.3.
\]

Use \(c\)-recursion on the complement, including equality.  Both recursions
use the same plumbing-series cutoff so the artificial interface does not
introduce a finite-depth mismatch.

This terminology matters.  The implemented five-point fixed-difference
\(h\)-recursion is the sphere-linear multipoint recursion in ordinary
plumbing/cross-ratio variables
\((0,q_1q_2,q_2,1,\infty)\); it is not the special four-point elliptic
normalization \(q_{\rm ell}(z)=\exp[-\pi K(1-z)/K(z)]\).  There is presently
no canonical five-point analogue of that elliptic normalization in this
calculation.  Applying the four-point map independently to both edges after
proper-channel selection would make the \(0.3\) complement numerically empty
in the atlas audit and would therefore not implement the requested nontrivial
hybrid partition.

The worldsheet number will not be frozen until it is stable under:

1. \(\epsilon\to0^+\), with the correct \(-i0\) denominator sign;
2. finite-part collar and normal-projection radii;
3. internal-momentum quadrature, including the near-on-shell thresholds;
4. block level, momentum cutoff, and central-charge shift;
5. independent randomized-QMC replicates;
6. pure-\(h\)/pure-\(c\) overlap checks around \(|q|=0.3\).

Only after these tests pass and the worldsheet result is marked frozen may a
separate module load and compare the matrix-model prediction.

## 5. Implementation audit and current numerical status

The first physical preflight exposed and fixed an important projector bug:
the generic linear-\(q\) face/corner density initially omitted the
nonchiral superghost factor.  Its valuation changes by two powers between
boundary charts.  After restoring it, direct log-slope measurements on all
ten faces agree with the predicted \(\beta_{ij}(P)\) to better than
\(2\times10^{-3}\) already between radii \(10^{-3}\) and \(10^{-4}\), and
the most sensitive corner coefficient has a stable \(10^{-4}\to10^{-5}\)
projection limit.

The near-on-shell internal-momentum factor is treated by a second, purely
numerical subtraction which does not change the worldsheet scheme.  For a
smooth coefficient \(f(P)\), write

\[
 f(P)=f(P_*)+[f(P)-f(P_*)].
\]

The universal constant-coefficient integral of
\(2\pi\rho^{\lambda(P)}/\lambda(P)\) is evaluated at high precision with
threshold-centered panels.  The bracketed remainder is uniformly regular
as \(\epsilon\to0^+\) and is integrated by staggered composite Gauss rules.
At a corner the same identity is applied successively in \(P_1\) and
\(P_2\).  A scalar regression recovers the positive Feynman delta term
\(\operatorname{Im}J\to\pi^2/P_*\).

Two numerical pathologies in this step have now been removed.  First, an
equal pair of threshold momenta hit confluent denominators in the correlated
fixed-difference representation even though the double-primary block is
regular.  The double-primary coefficient is now evaluated directly as the
common \(\mathfrak{osp}(1|2)\) seed of h- and c-recursion.  Second, the old
momentum ``shell'' sequence changed its panel partition whenever the nominal
order was changed, so its oscillation was not a polynomial-convergence test.
The production rule now fixes factor-four panels from the physical width
\(\Gamma/(2P_*)\) out to both endpoints and increases only the Gauss order.

At \(\epsilon=0.02\), \(P_{\max}=1.5\), collar radius \(0.08\), and
\(\delta c=10^{-5}\), the multiplicity-weighted sum of all fifteen corner
finite parts is

\[
 C_{(4,5)}=7.7742436535+1.0180488863i,
 \qquad
 C_{(5,6)}=7.7746854919+1.0182910932i.
\]

The order increment is
\(4.42\times10^{-4}+2.42\times10^{-4}i\).  This is the first converged
corner diagnostic; it is not a full worldsheet number because the face and
bulk integrals, regulator limit, collar limit, and momentum-cutoff limit
remain to be certified.  The machine-readable orbit ledger is
`type0b_ns_five_tachyon_physical_iepsilon_e002_corner_momentum_convergence.json`.

The next order gives
\(C_{(6,7)}=7.7746565037+1.0182778442i\) at \(P_{\max}=1.5\), a change of
only \(-2.90\times10^{-5}-1.32\times10^{-5}i\).  The cutoff-stable corner
scan at order \((4,5)\) gives

\[
 C_{P_{\max}=2}=7.7723662976+1.0168720264i,
 \quad
 C_{P_{\max}=3}=7.7724048532+1.0169038805i,
 \quad
 C_{P_{\max}=4}=7.7724132083+1.0169057749i.
\]

For the face continuum, the un-subtracted surviving momentum now uses
unit-width fixed panels: raising \(P_{\max}\) appends panels without moving
any lower-momentum node.  At the representative lower modulus
\(z=0.185+0.1274814104i\), \(P_{\max}=2\), the fixed-panel sequence locks
the normal order at five and gives

\[
 F_{(5,6)}=-3.0208589350-2.3167865453i,
 \qquad
 F_{(5,7)}=-3.0208737355-2.3167949616i.
\]

At smooth order seven, raising the normal order from five to eight changes
the value by only \(-2.26\times10^{-6}+0.64\times10^{-6}i\).  At lower
per-panel order the same face point changes by less than \(5\times10^{-6}\)
when \(P_{\max}\) is raised from two to three or four.  The provisional
production momentum grid is therefore normal order five, smooth order seven,
automatic threshold panels, and \(P_{\max}=2\); other face orbits and the
integrated face QMC still require certification.

The stored file
`type0b_ns_five_tachyon_physical_iepsilon_e002_level0_preflight.json` is a
variance diagnostic only.  It uses block level zero, momentum orders
\((2,3)\), and \(P_{\max}=1.5\); its reported QMC uncertainty is larger than
the mean.  It is explicitly not a freeze candidate.  The next numerical
stage is the converged face-modulus integral followed by bulk QMC, collar,
projection, block-level, momentum-cutoff, and \(\epsilon\) scans.

A final implementation audit tested the algebraic single-primary face
factorization separately in both recursion backends.  It agrees with the
unfactorized five-point direct c-series to relative error below
\(2\times10^{-12}\).  The naively reduced four-point h-recursion, however,
does not have the same finite-recursion normalization as the selected-edge
five-point h block.  That optimization is therefore enabled only in the c
region.  Every strict \(\max_e|q_e|<0.3\) point continues to use the verified
unfactorized five-point h-recursion.  Dedicated regressions enforce both
statements.  A matched-cutoff pure-h/pure-c regression also agrees to relative
error below \(2\times10^{-12}\) immediately on both sides of the 0.3 gate.
The full five-point/domain suite passes all 49 tests; the 25 generic
multipoint h/c recursion tests pass as well.

The production face-only scrambled-Sobol pilot was deliberately stopped
before completion: its measured cost could not finish the requested orbit
sum and replicate count within the user-imposed 3.3-hour total goal limit.
It emitted no result file and supplies no amplitude estimate.  Consequently
the worldsheet amplitude remains **not frozen**.  In particular, there is no
certified face-modulus integral, bulk integral, collar extrapolation,
block-depth scan, \(\epsilon\to0^+\) extrapolation, or independent-QMC error
bar.  The matrix-model comparison remains unopened.

A cheaper level-two face-only preflight did finish with two scrambled Sobol
replicates and two moduli samples per replicate.  At momentum orders
\((4,5)\), \(P_{\max}=1.5\), and \(\epsilon=0.02\), its replicate estimates
are

\[
 -25.1528775-11.4742723i,
 \qquad
 -2.8720901-1.4212376i.
\]

Their mean is \(-14.0124838-6.4477549i\), with independent-replicate standard
errors \(11.1403937\) and \(5.0265174\) for the real and imaginary parts.
This order-one relative uncertainty proves that the face contribution is not
numerically resolved.  The machine-readable record is
`type0b_ns_five_tachyon_physical_iepsilon_e002_level2_face_qmc_power1_preflight.json`;
its status explicitly says it is not a worldsheet freeze.

Extending the same nested scrambles to four samples per replicate gives
replicate estimates

\[
 -17.9046805-8.3687047i,
 \qquad
 -18.8164989-8.8388730i,
\]

and hence the mean \(-18.3605897-8.6037889i\), with formal two-replicate
standard errors \(0.4559092\) and \(0.2350841\).  Despite the closer pair,
the mean moved by \(-4.3481059-2.1560339i\) relative to the nested power-one
estimate.  Four samples and two scrambles therefore do not establish QMC
convergence, especially in the presence of the observed heavy individual
samples.  Each replicate exercised 341760 c-backend and 341760 h-backend
calls, confirming that the strict gate is active on both sides.  The record
is
`type0b_ns_five_tachyon_physical_iepsilon_e002_level2_face_qmc_power2_preflight.json`.
