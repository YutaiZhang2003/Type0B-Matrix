# Type-0B NS five-tachyon worldsheet freeze ledger: strict \(|q|=0.3\) hybrid

Date: 2026-08-28

Status: **RETRACTED as a freeze path; retained only as an audit record**.

> This ledger used a remote complex one-divisor continuation ray.  It is not
> the requested direct physical-domain prescription and none of its numerical
> values may enter the worldsheet freeze.  The replacement calculation keeps
> positive real physical energies, takes the boundary value through a small
> common \(+i\epsilon\), leaves the internal Liouville contours on
> \(P\in\mathbb R_+\), and performs BRY-style local polynomial subtraction on
> all divergent Deligne--Mumford strata.

This ledger supersedes only the recursion-backend policy in
`TYPE0B_NS_SPHERE_FIVE_TACHYON_1TO4_2026-08-25.md`.  The PCO integrand,
Möbius/spin lifts, path-ordered Liouville residue forest, and corrected
one-divisor kinematic certificate continue to be taken from that audited
derivation.  No matrix-model formula, target value, or fit coefficient is
used below.

## 1. Kinematics and the only divergent stratum

Use the fixed gauge and pictures

\[
 (z_0,z_1,z_2,z_3,z_4)=(\infty,1,0,z,w),\qquad
 R_{\rm PCO}=\{0,1,2\},
\]

and the certified ray

\[
 \omega_a(t)=t c_a,\qquad 0.96\le t\le1.00,
\]

\[
 (c_1,c_2,c_3,c_4)=
 (0.0815+0.1284i,-0.3063+0.1280i,
 -0.2409+0.6785i,-0.5234+0.8075i).
\]

The complete reflected, path-ordered depth-two pole ledger contains 40
records throughout this interval.  Exactly one is non-integrable: the
positive-real continuum on the raised-pair divisor

\[
 D_{12}:\quad z_1\longrightarrow z_2.
\]

There is no moving-line subtraction and no corner subtraction.  The smallest
margin among all other records is (0.04418598399999962), the minimum
pole-wall clearance is (0.02134400000000003), and the minimum external
frequency separation is (0.29813719258086535).  After removing the
continuum primary on (D_{12}), the first allowed NS descendant has margin

\[
 M_{12}^{(1)}\ge0.014013775871999923>0.
\]

Thus the required normal subtraction polynomial has degree zero: it is the
endpoint primary coefficient.  Every positive-degree normal term is
integrable.

## 2. Local polynomial subtraction and analytic finite part

Let (q) be a plumbing coordinate normal to (D_{12}), let (u) be the
remaining four-point modulus, and let (P\ge0) be the normal NS Liouville
momentum.  The continued density has the local form

\[
 d^2q\,d^2u\int_0^\infty dP\;
 A_0(P;u,\bar u)\,|q|^{\beta(P)}
 \left[1+O(|q|)\right],
\]

where (A_0) includes the (D_{12}) bubble, the lower four-point continuum,
all crossed lower-point quotient-contour residues, and the BRY spectral
measures.  At (c_{\rm L}=27/2+\delta c),

\[
 \beta(P)=-2-\frac{Q^2}{4}+P^2-(\omega_1+\omega_2)^2,
 \qquad
 \frac{Q^2}{4}=1+\frac{\delta c}{12}.
\]

Write

\[
 \lambda(P)=\beta(P)+2
 =P^2-\frac{Q^2}{4}-(\omega_1+\omega_2)^2.
\]

The raw radial integral diverges when

\[
 \operatorname{Re}\lambda(P)\le0.
\]

At the central production point (t=0.98),

\[
 \omega_1+\omega_2=-0.220304+0.251272i,
\]

so at the physical central charge the divergent normal-momentum interval is

\[
 0\le P\le P_*,\qquad
 P_*^2=1+\operatorname{Re}(\omega_1+\omega_2)^2
      =0.985396234432,
\]

\[
 P_*=0.9926712620157793.
\]

With the numerical regulator (delta c=10^{-5}), this endpoint becomes
(P_*=0.9926716817585426).  Over the full certified ray the physical
endpoint ranges only from (0.9929683902964888) at (t=0.96) to
(0.9923679156441929) at (t=1).

For a collar (|q|<\rho), analytic continuation of the normal integral is

\[
 \operatorname{FP}\int_{|q|<\rho}d^2q\,|q|^{\beta(P)}
 =\frac{2\pi\rho^{\lambda(P)}}{\lambda(P)},
\]

with the continuous value (2\pi\log\rho) at (lambda=0).  Equivalently,
for a cutoff (epsilon), the divergent term

\[
 -\frac{2\pi\epsilon^{\lambda(P)}}{\lambda(P)}
\]

is cancelled analytically before (epsilon\to0).  The implemented finite
part is therefore

\[
 \begin{aligned}
 \operatorname{FP}\int_{\mathcal M_{0,5}}\!\mathcal I
 ={}&\int_{\mathcal M_{0,5}\setminus\{|q|<\rho\}}\!\mathcal I
 +\int_{|q|<\rho}(\mathcal I-\mathcal I_0)\\
 &+\int d^2u\int_0^\infty dP\,
 A_0(P;u,\bar u)\frac{2\pi\rho^{\lambda(P)}}{\lambda(P)}.
 \end{aligned}
\]

The code omits the endpoint primary coefficient-by-coefficient rather than
subtracting two large floating-point values.  A symmetric angular quadrature
adds the integrable difference between the exact leading fixture and its
strict plumbing asymptotic.  Collar-radius independence remains a required
freeze test.

## 3. Proper-channel strict hybrid recursion

For each moduli point, construct all (5!=120) oriented linear combs and
select a convergent representative minimizing

\[
 \rho_\alpha=\max(|q_{1,\alpha}|,|q_{2,\alpha}|).
\]

Only after this geometric choice is made is a right-incoming comb reversed
to the standard residue orientation.  Since reversal preserves
(\rho_\alpha), it does not change the cell.

The production backend is

\[
 \boxed{
 \begin{array}{ll}
 h\text{-recursion},&\max(|q_1|,|q_2|)<0.3,\\[2mm]
 c\text{-recursion},&\max(|q_1|,|q_2|)\ge0.3.
 \end{array}}
\]

Equality belongs to the (c)-recursive complement.  On a divisor face the
same rule is applied to the one surviving nome.  This includes the
(D_{12}) primary, descendant remainder, analytic face density, and fixture
correction; there is no separate forced-(c) degeneration layer.

At finite order, the two representations must use the same plumbing-series
truncation.  A common finite recursive *depth* is not a common truncation:
at a representative point a depth-two comparison differed by about 35%.
With matched per-edge and total twice-level cutoffs, the same representative
five-point component agreed to relative (1.9\times10^{-43}).  The hybrid
production driver therefore requires `recursion_max_twice_level=None` and
varies only the matched global plumbing-series cutoffs.

## 4. Worldsheet normalization to freeze

The primary numerical observable is the matrix-blind moduli integral

\[
 I_{T^5}=\operatorname{FP}\int d^2z\,d^2w\,\mathcal I_{\rm NS}(z,w).
\]

The literal all-NS diagram is

\[
 \mathcal A_{T^5}=
 \frac{i}{64}g_s^5 C_{S^2}\,\delta(E)\,I_{T^5}.
\]

The additional factor of sixteen sometimes used to infer a full right-mode
amplitude requires equality of all even-axion diagrams and is not part of
this worldsheet freeze.

## 5. Current numerical status and freeze criteria

The strict-gate implementation passes the multipoint recursion, PCO,
channel-lift, residue-forest, domain, subtraction reconstruction, and
matrix-import-isolation tests.  A complete 120-chart end-to-end preflight at
(t=0.98) used both backends (108000 h evaluations and 135920 c evaluations)
without overflow.  Its moduli statistics were intentionally negligible and
its value is not an amplitude datum.

The saved level-4 preflight is
`Data Set/type0b_ns_five_tachyon_hybrid_q03_level4_preflight.json`.  Its
statistical uncertainty is of the same order as its central value, so it is
also not freeze-ready.

The worldsheet result may be frozen only after independent stability in:

1. per-edge and total block twice-level;
2. both Liouville momentum quadrature orders and the momentum cutoff;
3. scrambled-Sobol power and replicate count;
4. collar radius and fixture-correction quadrature;
5. structure-constant and block working precision;
6. the regulator (delta c\to0^+); and
7. pure-h/pure-c overlap samples on both sides of the (0.3) gate.

Only after a hashed dataset satisfies these tests may a separate module load
a matrix-model prediction.
