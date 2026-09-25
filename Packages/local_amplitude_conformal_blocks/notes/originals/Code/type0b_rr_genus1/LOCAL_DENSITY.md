# Local Type 0B RR reflection density

**Later interacting audit:** channel overlap and S transport at omega=i/4
expose a relative factor-two NS/R normalization inconsistency. The finite
local algebraic checks below do not certify the continuum normalization.
The saved local bank is unchanged and must not be used as a production
integrand. See [the new radial trace and audit](RADIAL_TRACE.md).

`local_density.py` assembles the interacting Liouville RR pair, the BRY
incoming/outgoing states, all four torus spin structures, picture changing,
and the time/ghost determinants in the **collision OPE chart**. It returns
both pointwise densities and the coupled Taylor coefficients for the
collision-disk machinery. `prepare_local_bank.py` saves a fresh spectral
pilot and its individual spin contributions.

This completes a local layer of the RR calculation. The implemented domain
is a standard fundamental-domain modulus and `0 < |z| < 1/4`, with a
contractible line joining the two Ramond punctures. The finite OPE cutoff
must still be checked at any intended evaluation point. The code rejects
puncture translations outside this patch; it is not a global callback for
`matched_integration.py`.

The result is in the explicitly ordered, canonical `G_h G_a` convention
used by the RR Ward tensors. No overall phase or normalization has been
chosen by fitting the matrix prediction. Conversion of the completed
worldsheet integral to the BRY S-matrix convention remains a separate audit.

## Fixing the local spin/disorder frame

For the free time Majorana field, the unnormalized spin two-point function
in characteristic delta is

\[
S_\delta(z,\bar z)=|E(z)|^{-1/4}
 \left|\frac{\theta_\delta(z/2|\tau)}{\eta(\tau)}\right|.
\]

The disorder correlator with a shrinking line is

\[
\langle\mu(z)\mu(0)\rangle_\delta
 =\epsilon_\delta S_\delta(z,\bar z),\qquad
(\epsilon_1,\epsilon_2,\epsilon_3,\epsilon_4)=(-1,1,1,1).
\]

The sign follows from
[Di Francesco–Saleur–Zuber, section 5.6, equations (5.40)–(5.42)](https://www.lpthe.jussieu.fr/~zuber/MesPapiers/dfsz_NP87b.pdf).
Their discussion after (5.42) also explains why winding the disorder line
changes the sign assignment. In the even characteristics the leading OPE
is the identity. In the odd characteristic it is the energy field, whose
coefficient has opposite signs in the spin and disorder OPEs. This checks
the local sign without selecting it to match an amplitude.

The mixed spin/disorder two-point functions vanish by total fermion parity.
In the [BRY axion vertices, equations (2.6)–(2.8)](https://arxiv.org/html/2201.05621v2),
the outgoing disorder term has a minus sign. Regrouping the two products
`mu^0 V_R^-` into time and Liouville correlators supplies another minus.
Thus, for each of the four raised components,

\[
 C_{pq,\delta}=\frac12\left(L^{++}_{pq,\delta}
                  +\epsilon_\delta L^{--}_{pq,\delta}\right),
 \qquad p,q\in\{0,1\}.
\]

Here `0` means `G0`, `1` means `G_-1`, and both operators act on the
Liouville field at zero, with the holomorphic operator ordered first.
The time spin magnitude is already included in `free_outer_factor`.
The BRY state normalization `1/2` in this equation is distinct from the
Type 0B GSO `1/2` already included in that free factor.

## Sewing the NS bridge

Set `h=(1+p_bridge^2)/2` and `h_R=9/16+omega^2/2`. Each chiral block
contains the sphere RR Ward vector, the inverse NS Gram matrix, the
one-point handle trace, and the flat-coordinate transport. Its family
`f=0,1` includes all retained NS grades `j+f/2` and has primary power

\[
(2\pi i)^{h+f/2}z^{h+f/2-2h_R-p}.
\]

The `G_-1` polynomial includes its `3 K z G0/4` coordinate correction.
The odd family's inverse norm `1/(2h)` is already in each chiral block.
It is not inserted a second time during the nonchiral contraction.

Use the physical Ramond ket embedding

\[
E_R=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&1\\0&1\\-i&0\end{pmatrix}.
\]

For holomorphic ground labels `(a,c)` at `(z,0)` and antiholomorphic
labels `(b,d)`, the raised-pair contraction multiplies the two chiral
blocks by

\[
\mathcal K_{f;e}(a,c;b,d)=
(E_R)_{2a+b,e}(E_R)_{2c+d,e}
(-1)^{c+f(f+a+c+1)+b(c+1)+f}.
\]

The exponent contains the action of the anti operator on the ground
tensor, the trinion regrouping, and the inverse full NS Gram sign
`(-1)^f`. This is a bilinear contraction. Neither a sphere coupling nor
an external momentum is replaced by its complex conjugate.

For the antiholomorphic RR sphere Ward tensor, the chiral constructor must
receive **`-conjugate(omega)` before its result is conjugated**. This is
because its parameter is `beta=i*omega/sqrt(2)`. Passing `conjugate(omega)`
gives the correct conformal weights but the wrong zero-mode signs. The
handle uses the separately normalized reflected traces described next.

## Handle normalization and descendant phase

The theta labels refer directly to the closed OPE handle:

| Characteristic | Trace |
| --- | --- |
| 3 | NS ordinary trace |
| 4 | NS parity supertrace |
| 2 | R ordinary trace |
| 1 | R parity supertrace |

Write `c_eta=(C_even/2,C_odd/2)`, with eta ordered `(+,-)`.
The sphere factor is `c_eta(omega,omega,p_bridge)`. The coefficients
multiplying a product of the stored **full chiral handle traces** are

\[
H^{NS}_f=i^f(C,\widetilde C)_f,
\qquad
H^{R}_{f,t}=\frac{i^f}{2}c_t(P_{loop},P_{loop},p_{bridge}).
\]

The NS odd phase expresses the canonical three-form convention
`C_HN^(1)=i Ctilde_BRY`. The full NS bridge metric on a pair of odd words
has the opposite sign to the product of its chiral metrics. Both signs
are retained, rather than folding either into an absolute square.

The R coefficient follows by restricting the full graded RNR operator to
the two-family Ramond representation. In the normalized numerical RNR
basis its matrix is

\[
\mathcal V_f=\frac{i^f}{2}
 \left(D_h\otimes D_a+i\,O_h(-1)^{F_h}\otimes O_a\right),
\]

where `D` preserves chiral state parity and `O` reverses it. The trace of
this operator on the physical small space is
`i^f Tr_h Tr_a/2`, also with parity insertions. The star and trace structure
can be compared with
[Hadasz–Jaskólski–Suchanek, section 2.2](https://arxiv.org/pdf/1207.5740);
the factor here is fixed using this repository's normalized physical kets
and full traces.

In particular, the primary ground trace is `C^+ + C^- = C_even`, and
the supertrace is `C^+ - C^- = C_odd`. For the canonical odd NS bridge,
the unit eta-minus ground matrix elements are both `2 i P_loop^2`.
These anchors fix the multiplicity and phase without a genus-two trial
factor. The validation also constructs the physical projector explicitly
at handle levels zero, one and two, rather than checking only the ground
degeneracy.

## Density and collision data

With the existing free factor `B_delta`, the local contribution is

\[
\rho_\delta=\mathcal B^{RR}_\delta\left[
C_{11,\delta}+D_\delta C_{01,\delta}
 +\overline D_\delta C_{10,\delta}
 +\left(D_\delta\overline D_\delta-
 \frac{\pi}{\omega^2\tau_2}\right)C_{00,\delta}\right].
\]

`evaluate` returns the four entries and their sum; it does **not** average
them again. Their measure is
`d²tau d²z dP_bridge dP_loop/pi²`, with the spectral measure supplied
separately by the quadrature weights. The single torus Casimir is included
in the blocks. The finite time-boson Gaussian contact term is retained.

`component_polynomials` exports the actual coupled `C_pq` Taylor arrays,
with the individual powers of `z` and `zbar` removed. Passing these to
`RRCollisionKernel` gives `collision_polynomials`, whose families multiply

\[
 |z|^{p_{bridge}^2+f-3-\mathbf1_{\delta=1}}.
\]

This is a factored power, not a claim that its leading coefficient is
nonzero. These polynomials can be integrated using the existing annulus
or disk finite-part primitives. A global subtraction prescription has
not been inferred from local Taylor data.

## Checks and saved pilot

`LOCAL_DENSITY_CHECKS.json` records:

- 256 comparisons with successive full nonchiral sphere Ward reduction,
  including independent holomorphic and antiholomorphic bridge grades.
- 468 explicit small-space Ramond traces through handle level two,
  including the ordinary trace, supertrace, and unequal left/right levels.
- Eight ground trace anchors and eight checks of
  `L_00^{ee}=(-i,+i)_e omega² L_bare^{ee}/2`.
- Short-distance Majorana checks in all four spin structures, and complex
  energy analyticity checks with freshly evaluated structure constants.
- Direct pointwise density versus its coupled collision polynomial, and
  independent angular/radial integration of the actual finite annulus
  coefficients.
- Separate bridge-order and handle-level diagnostics at two energies.

`RR_LOCAL_DENSITY_PILOT.npz` uses omega=i/4, a 4 by 4 Gauss rule on
`[0,2.5]^2`, bridge relative order three, and NS/R handle level two. It
saves four local geometry points, all four spin entries, all four PCO
components, both bridge families, the coupled collision polynomials, and
source/array hashes. `RR_LOCAL_DENSITY_POINTS.json` is its readable point
summary. The reported spectral sums are **finite pilot densities**, with
no spectral-tail or quadrature-convergence claim. They are not scattering
amplitudes; all amplitude fields remain null.

Reproduce with the NumPy/SciPy/SymPy/mpmath Python runtime:

```sh
python3.11 -B Code/type0b_rr_genus1/check_local_density.py \
  --output /tmp/type0b-rr/LOCAL_DENSITY_CHECKS.json
python3.11 -B Code/type0b_rr_genus1/prepare_local_bank.py \
  --output /tmp/type0b-rr/RR_LOCAL_DENSITY_PILOT.npz \
  --summary /tmp/type0b-rr/RR_LOCAL_DENSITY_POINTS.json
```

The remaining amplitude work is to transport this local convention into
the mixed bulk and cusp charts, check their overlap after spectral
integration, establish spectral and truncation convergence, implement the
global boundary prescription, and audit the S-matrix normalization before
comparison with the independent RR matrix prediction.
