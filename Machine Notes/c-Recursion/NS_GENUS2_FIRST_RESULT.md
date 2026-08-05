# First genus-two NS super-Liouville computation

This is the first theta/glasses computation here that is not already covered
by a known genus-zero or genus-one (h)-recursion.  I evaluated the genus-two
(b=1), (c=27/2) ((\hat c=9)) NS partition at the certified common-modulus
sample `o0026` and formed

\[
Q_L=\frac{Z_{\mathrm{SL}}}{Z_{X+\psi}^{,9}},
\]

where one (X+\psi) is one noncompact real scalar and one NS Majorana
fermion.  Both numerator and denominator use the same raw plumbing-cylinder
convention, with the common Casimir factors omitted edge by edge and unit
scalar zero-mode volume.

## What was evaluated

The super-Virasoro blocks are evaluated with the functional (c)-recursion.
The reported recursion order bounds only the accumulated twice-level of Kac
residues.  It is not a truncation of the plumbing-(q) expansion.  Each leaf
is a direct, tolerance-controlled large-(c) regular block: a lifted NS
Schottky vacuum product times a directly summed global
(\mathfrak{osp}(1|2)) graph.

The two graphs are independent implementations:

- theta: two trinions joined by the three (0,1,\infty) tubes;
- glasses: two self-sewn handles joined by the separating bridge.

For glasses, direct graded sewing gives

\[
Q_{\rm orient}=e_B(e_L+e_R)\pmod 2.
\]

The right-handle frame transition is fixed by the exact separating limit, not
by theta/glasses agreement.  With it, the global glasses block at (q_B=0)
equals the product of two independently known torus one-point global blocks
to (2.4\times10^{-16}) relative error.  Its ((3,1)) handle residue agrees
with the independent torus (c)-recursion residue to displayed precision.

The same spin structure is transported through the certified symplectic word.
Starting from the glasses characteristic ([00|00]), the word gives the theta
characteristic ([00|11]).  The corresponding plumbing lifts are
((+,+,+)) in glasses and ((+,-,-)) on the theta
((0,1,\infty)) edges.

## Numerical result

At common recursion and quadrature order (R=4,N=6),

| channel | (Z_{\rm SL}) | (Z_{X+\psi}) | (Q_L) |
|---|---:|---:|---:|
| theta | (7.51991819\times10^{-24}) | (2.96040084\times10^{-1}) | (4.30584553\times10^{-19}) |
| glasses | (5.04481318\times10^{-5}) | (3.44251386\times10^{1}) | (7.42957221\times10^{-19}) |

Thus

\[
\frac{Q_L^{\theta}}{Q_L^{\rm glasses}}
=0.57955497,
\qquad
\frac{Q_L^{\theta}}{Q_L^{\rm glasses}}-1
=-0.42044503.
\]

No matching was imposed.  A higher theta quadrature (N=8) gives
(Q_L^\theta=4.29596476\times10^{-19}), a further (0.23\%\) movement.  If
combined only as a channel-adaptive convergence indicator with glasses
(N=6), the ratio is (0.57822505).

At (N=4), increasing the recursion order gives

| recursion order | (Q_L^\theta/Q_L^{\rm glasses}) |
|---:|---:|
| 0 | 0.57601660 |
| 3 | 0.59486693 |
| 4 | 0.59775804 |

The (N=4\to6) changes are (3.65\%\) in theta and (0.63\%\) in glasses.
The free-superfield primitive product was carried to word length 13; its
largest last-step change is (2.34\times10^{-6}) per superfield.  Increasing
the glasses global occupation ceiling from 16 to 20 at (N=4,R=4) changes
the integrated (Z_{\rm SL}) by (2.1\times10^{-10}) relative.

## Interpretation

The current order-four calculation does **not** show theta/glasses matching;
the difference is about (42\%\), far larger than the audited direct-sum,
primitive-product, and quadrature errors.  This should not yet be called a
failure of crossing or of the quotient prescription.  At (b=1), recursion
orders above four require the confluent-pole finite-part machinery, which is
deliberately not hidden behind a detuning in this first computation.  The
honest conclusion is therefore:

> the first independently evaluated order-four genus-two blocks do not match;
> higher recursion order or an as-yet-unidentified sewing normalization must
> be resolved before drawing a physical conclusion.

The complete numerical ledger is in
`Data Set/ns_genus2_theta_glasses_hatc9.json`.
