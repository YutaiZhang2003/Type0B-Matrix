# Sphere five-point finite-part recipe in the c=1 string

## Scope and conventions

This note generalizes the BRY sphere four-point prescription to the tree-level
five-tachyon amplitude.  Xi's normalization is used:

\[
V^\pm_\omega=g_s^{\rm Xi}\,\widetilde c c\,
e^{\pm i\omega X^0/\sqrt{\alpha'}}V_{P=\omega/2},\qquad
\widetilde K_{S^2}=\frac{2}{\sqrt{\alpha'}},\qquad
\mu^{-1}=4\pi g_s^{\rm Xi}.
\]

The amplitude is defined by analytic continuation from complex energies with
positive imaginary parts.  Energy conservation is imposed in the complex
domain before the physical limit.  The worldsheet result must be completed,
including its extrapolation and error budget, before the matrix-model curve is
evaluated.

With labelled external states, the BRY factorized sphere-five integrand is

\[
\mathcal A^{\rm tree}_{1\to4}
=i\,(g_s^{\rm BRY})^5 C_{S^2}\,I_5,
\qquad C_{S^2}=\frac{2\pi}{(g_s^{\rm BRY})^2}.
\]

Using \(2\pi g_s^{\rm BRY}=4\pi g_s^{\rm Xi}=\mu^{-1}\) gives

\[
\boxed{\mu^3\mathcal A^{\rm tree}_{1\to4}=\frac{i}{4\pi^2}I_5}.
\]

There is no extra \(1/4!\): the four outgoing vertex operators are labelled
before the equal-energy specialization.  This coefficient is fixed by sphere
factorization, not by comparison with a target-space answer.  It is useful to
strip the universal soft factor by defining

\[
Q(\omega)=\frac{I_5}{16\pi^2\omega^5},\qquad
\mu^3\mathcal A^{\rm tree}_{1\to4}=4i\omega^5Q(\omega).
\]

Choose the moving insertions to be the incoming state at \(z_1\) and one
outgoing state at \(z_2\); put the other outgoing states at \(0,1,\infty\).
The signed target-time energies are

\[
k=(+\omega_{\rm in},-\omega_2,-\omega_3,-\omega_4,-\omega_5),
\qquad \sum_i k_i=0.
\]

For the equal-energy slice requested here,
\(\omega_2=\cdots=\omega_5=\omega\) and
\(\omega_{\rm in}=4\omega\).

## The complete boundary set

The Deligne-Mumford compactification \(\overline{\mathcal M}_{0,5}\) has ten
boundary divisors.  Label a divisor by the two external legs on the
three-punctured component of the limiting surface.  In the \((z_1,z_2)\)
chart, seven divisors are visible directly:

\[
z_1\to0,1,\infty,\qquad z_2\to0,1,\infty,\qquad z_1\to z_2.
\]

Blowing up the three coincident corners adds the exceptional divisors

\[
(z_1,z_2)\to(0,0),(1,1),(\infty,\infty).
\]

These ten loci are generated and checked by
`sphere_five_point_subtraction.py`.  Two divisors intersect precisely when
their two-label sides are disjoint.  There are fifteen such pairs, exactly the
fifteen trivalent trees with five labeled leaves.  Since
\(\dim_{\mathbb C}\mathcal M_{0,5}=2\), no higher forest is possible.

For equal outgoing energies, four divisors carry signed tube energy
\(\kappa=3\omega\) (incoming plus one outgoing leg) and six carry
\(\kappa=-2\omega\) (two outgoing legs).  The fifteen corners split into
twelve of type \((3\omega,2\omega)\) and three of type
\((2\omega,2\omega)\).

## Five-point conformal block

For external weights \(d_1,\ldots,d_5\) at
\((0,z_1,z_2,1,\infty)\) and internal weights \(h_1,h_2\), define

\[
q_1=\frac{z_1}{z_2},\qquad q_2=z_2.
\]

The chiral correlator in the linear channel is

\[
z_1^{h_1-d_1-d_2}z_2^{h_2-d_3-h_1}
\mathcal F(q_1,q_2).
\]

`ccy_sphere_five_point.py` implements CCY equation (3.26), specialized to
\(N=5\).  In the variables

\[
a=h_2-h_1,\qquad e_1=d_1-h_1,\qquad e_5=d_5-h_1,
\]

the two residue families are

\[
\frac{q_1^{rs}A_{rs}
P_{rs}\!\begin{bmatrix}d_{rs}+e_1\\ d_2\end{bmatrix}
P_{rs}\!\begin{bmatrix}d_{rs}+a\\ d_3\end{bmatrix}}
{h_1-d_{rs}}
\mathcal F(d_{rs}+rs,a-rs,e_1-rs,e_5-rs)
\]

and

\[
\frac{q_2^{rs}A_{rs}
P_{rs}\!\begin{bmatrix}d_{rs}-a\\ d_3\end{bmatrix}
P_{rs}\!\begin{bmatrix}d_{rs}-a+e_5\\ d_4\end{bmatrix}}
{h_1+a-d_{rs}}
\mathcal F(d_{rs}-a,a+rs,e_1,e_5).
\]

At \(c=25\), coincident Kac weights make the individual \(h\)-recursion
terms singular at \(b=1\), although their sum is finite.  The implemented
definition uses \(b=e^\eta\) and extrapolates the coefficients polynomially
in \(\eta^2\) to zero.  An independent fixed-weight CCY \(c\)-recursion is
implemented in the same file.  Both recursions, and a direct Verma-module
descendant contraction, agree through total level four in the automated
checks.  The regulated \(c=25\) \(h\)-recursion is also compared directly
with the exact \(c=25\) \(c\)-recursion.

The Liouville five-point function in this channel is

\[
\int_0^\infty\frac{dP_1}{\pi}
\int_0^\infty\frac{dP_2}{\pi}
C(P_1,P_2^{\rm ext},P_1^{\rm ext})
C(P_2,P_3^{\rm ext},P_1)
C(P_5^{\rm ext},P_4^{\rm ext},P_2)
\,|\text{primary powers}\times\mathcal F|^2,
\]

with the arguments permuted according to the selected trivalent tree.  All
fifteen trees use the same linear-channel block after relabeling.

### Fast logarithmic DOZZ evaluation

The three DOZZ factors are never multiplied as separately exponentiated
numbers.  For one momentum-grid point the implementation forms

\[
L=\log\!\frac{w_1w_2}{\pi^2}
 +\log C(P_a,P_b,P_1)
 +\log C(P_1,P_c,P_2)
 +\log C(P_2,P_d,P_e)
\]

and inserts \(e^L\) only once, together with the primary powers.  Each
\(\log\Upsilon_b\) is evaluated from a cached seed in the central fundamental
cell and transported by

\[
\log\Upsilon_b(x+b)-\log\Upsilon_b(x)
=\log\gamma(bx)+(1-2bx)\log b.
\]

At \(b=1\), the seed uses
\(\Upsilon_1(x)=G(x)G(2-x)\); on the physical central line this becomes the
real quantity \(2\operatorname{Re}\log G(1+iy)\).  Thus neighboring shifted
arguments require gamma functions rather than new Upsilon integrals, repeated
arguments are served from the cache, and exponentially large intermediate
DOZZ factors never appear.  The shift identity, multi-cell recurrence, and a
large-momentum combined-log stress test are included in
liouville_torus_checks.py.

## Divisor subtraction

Let \(q_D\) be a plumbing coordinate normal to a boundary divisor and let
\(P_D\) be the Liouville momentum on its tube.  The local OPE expansion of
the complete matter integrand contains terms

\[
d^2q_D\,
|q_D|^{-2-\kappa_D^2/2+2P_D^2+n+\bar n}
q_D^{n-\bar n}/|q_D|^{n-\bar n}.
\]

Angular integration eliminates nonzero spin.  The spin-zero term at level
\(n=\bar n\) is power divergent precisely when

\[
P_D^2+n<\frac14\operatorname{Re}(\kappa_D^2).
\]

Thus the divisor projector \(S_D\) retains

\[
n=0,1,\ldots,
\qquad 0\le P_D<
\sqrt{\frac14\operatorname{Re}(\kappa_D^2)-n}.
\]

The retained coefficient is not a number: it is the factorized sphere
four-point integrand on the other component, including the complete
dependence on its remaining modulus.  It must be evaluated with the already
validated four-point finite-part prescription.

For

\[
\alpha_D=P_D^2+n-\frac{\kappa_D^2}{4},
\]

the radial integral in a collar \(|q_D|<\rho_D\) is restored by analytic
continuation,

\[
\operatorname{FP}\int_{|q_D|<\rho_D}d^2q_D
|q_D|^{-2+2\alpha_D}
=\pi\frac{\rho_D^{2\alpha_D}}{\alpha_D}.
\]

The \(i\epsilon\) prescription keeps \(\alpha_D\) away from its logarithmic
pole.  Numerically, the momentum grid must be split at every endpoint above,
because the physical-limit imaginary part is concentrated there.

## Corner subtraction and the forest formula

In a chart for a trivalent tree, let \(q_1,q_2\) be its two plumbing
coordinates and \(S_1,S_2\) the corresponding spin-zero divergent
projectors.  The locally integrable bulk remainder is

\[
(1-S_1)(1-S_2)I=I-S_1I-S_2I+S_1S_2I.
\]

The last term is essential: subtracting the two faces without it subtracts a
simultaneous degeneration twice.  The complete finite part is

\[
\begin{aligned}
\operatorname{FP}_{1}\operatorname{FP}_{2}\int I
={}&\int(1-S_1)(1-S_2)I\\
&+A_1\int S_1(1-S_2)I
+A_2\int(1-S_1)S_2I
+A_1A_2S_1S_2I,
\end{aligned}
\]

where \(A_i\) means applying the analytic radial primitive term by term.
The corner coefficient factorizes into three sphere three-point functions,
with two internal Liouville momenta and the one external state at the middle
vertex.  This formula makes collar-radius independence an explicit numerical
check rather than an assumption.

## Numerical atlas and integration order

1. Cover \(\overline{\mathcal M}_{0,5}\) by a thick interior and plumbing
   collars attached to the fifteen trivalent trees.  Use a partition of unity
   whose simultaneous collar supports occur only for compatible divisors.
2. In every collar use the matching linear-channel five-point block.  Select
   the chart that minimizes \(\max(|q_1|,|q_2|)\) in the overlap and verify
   agreement with at least one alternate channel before production.
3. Precompute the logarithm of the product of the three DOZZ factors on a
   tensor-product \((P_1,P_2)\) grid, using the Upsilon shift recurrence above.
   Take one exponential only after adding the quadrature weights and primary
   powers.  Split each momentum direction at every OPE endpoint.  Use mapped
   Gauss-Legendre panels on the finite intervals and a Gaussian-tail map on
   the remainder.
4. Integrate the forest-subtracted bulk in the four real moduli directions.
   Integrate face remainders with the four-point finite-part routine and add
   the analytic radial primitives.  Add the double primitive at each corner.
5. Repeat at block orders \(N,N+2\), at two momentum grids, at two moduli
   grids, and at at least three collar radii.  Keep block, momentum, moduli,
   collar, Kac-regulator, and \(i\epsilon\) errors separate.
6. For each complex energy, first extrapolate the Kac regulator
   \(\eta\to0\), then the numerical grids, then verify collar independence,
   and only then extrapolate \(\epsilon\to0^+\).
7. Fix the overall coefficient from worldsheet factorization onto the
   Xi-normalized \(1\to2\) and four-point amplitudes.  No normalization may be
   fitted to the matrix-model prediction.
8. Freeze the worldsheet table and its error bars.  Only after that step may
   the matrix-model prediction be evaluated and overlaid on the amplitude
   plot.

## Equal-energy numerical result

The previous convergent-ray fit is quarantined: it is not a physical-domain
result and is not used by the current driver.  The production path now calls
`integrate_physical_equal_energy_finite_part_qmc` at each requested
\(\omega+i\epsilon/4\) independently.  It reports the three separately large
pieces

\[
I_5=I_{\rm bulk}+\sum_{D=1}^{10}I_D
    +\sum_{D\cap E=1}^{15}I_{DE},
\]

so the cancellation can be audited rather than hidden in a fitted function.
The bulk is integrated only where both plumbing radii in the best tree exceed
\(\rho\).  Each face is split into the six BRY four-point crossing cells; its
normal radius is integrated analytically.  Each compatible corner receives
the product of the two analytic radial primitives.

A worldsheet point is promotable only if the totals at three collar radii
agree within the combined moduli, momentum, and block errors.  The current
low-order run is a smoke test of this cancellation, not a frozen amplitude
table.  Consequently no physical five-point curve and no matrix-model overlay
are claimed here yet.

## Current executable checks

Run

```text
python3 plumbing/ccy_sphere_five_point_checks.py
python3 plumbing/sphere_five_point_subtraction_checks.py
python3 plumbing/sphere_five_point_liouville_checks.py
python3 plumbing/liouville_torus_checks.py
python3 plumbing/sphere_four_point_subtraction_checks.py
python3 plumbing/sphere_five_point_physical_checks.py
python3 plumbing/sphere_five_point_continuation_checks.py
python3 plumbing/sphere_five_point_physical_scan.py --help
python3 plumbing/sphere_five_point_imaginary_ray_fit.py
python3 plumbing/sphere_five_point_imaginary_ray_fit_checks.py
python3 plumbing/sphere_five_point_matrix_comparison.py
```

The first command checks the \(h\)-recursion, \(c\)-recursion, direct
descendant definition, and the regulated \(c=25\) limit.  The second checks
the ten divisors, fifteen corners, the blown-up chart, equal-energy channel
multiplicities, OPE thresholds, analytic radial primitive, and the two-face
forest signs.  The remaining commands check the conformal atlas and Jacobians,
the logarithmic DOZZ/Upsilon implementation, the verified four-point collar
baby step, the vectorized five-point face/corner primitives, and the direct
physical scan interface.  The final three commands reproduce and check the
separate target-free imaginary-ray fit before running its hash-verified
comparison.  Matrix comparison remains intentionally
disabled for the direct physical branch until a future physical worldsheet
table is frozen.  The separate imaginary-ray comparison described below does
not make a direct physical-\(i\epsilon\) claim.

## Parallel analytic-continuation program

The convergent-ray approach can be extended beyond its first Liouville
momentum wall without introducing the moduli-space subtraction forest.  On
\(\omega=it\), the incoming--outgoing cherry coefficient contains

\[
P_\pm=\pm\left(\frac52\omega-i\right),
\]

and this pair pinches the quotient-contour endpoint at \(t=2/5\).  For
\(2/5<t<1/2\), the continued channel is the real-momentum integral plus
\(-2i\) times the residue at \(P=(5/2)\omega-i\).  This term is inserted only
when the chosen OPE channel pairs the incoming operator with an outgoing
operator; it lowers the momentum-integral dimension by one.

The continued scan uses \(P=P_{\max}u^{5/4}\) to resolve the Liouville
threshold.  A quadratic map places nodes too close to the degenerate weight
\(h=1\) and can destabilize the fixed-weight \(c\)-recursion.  A production
higher-point integrator should instead combine a Gauss--Jacobi threshold
panel, ordinary bulk panels, and a separate tail map.  The full proposed
momentum-residue forest and tensor-network contraction are described in
`sphere_n_point_momentum_integration.md`.

This continuation program remains worldsheet-only and independent of the
direct physical collar calculation.  For the analytic-continuation branch,
the points-only table is frozen and fitted under the explicit ansatz

\[
Q_4(it)=a+bt+ct^2.
\]

The primary unweighted fit uses all seventeen validated points through
\(t=0.48\), including the three residue-corrected points above \(t=2/5\).
The \(t=0.49\) point is excluded because it is a diagnostic near the next
wall; an eighteen-point sensitivity fit is recorded separately.  The fit
program contains no matrix-model coefficients, and the comparison program
first verifies the frozen points hash.  This establishes an imaginary-energy
analytic-continuation result under the stated degree-two ansatz.  It does not
promote the independent direct physical collar smoke tests to a physical
amplitude table.
