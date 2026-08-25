# Liouville Torus One-Point Layer

This note accompanies `liouville_torus.py`. It supplies the Liouville CFT data
that multiply the universal torus one-point Virasoro blocks in
`virasoro_blocks.py`.

## 1. Xi-Yin Momentum Convention

The public integral follows the Balthazar-Rodriguez-Yin convention used in the
`c=1` string paper:

\[
Q=b+b^{-1},\qquad c=1+6Q^2,
\]

and Liouville primaries are labelled by a real momentum \(P\),

\[
V_P,\qquad h_P=\bar h_P={Q^2\over4}+P^2.
\]

Their two-point normalization is

\[
\langle V_{P_1}V_{P_2}\rangle\propto \pi\,\delta(P_1-P_2),
\]

so the diagonal torus one-point function uses the completeness measure
\(dP/\pi\).

The relation to the `virasoro_blocks.py` momentum is

\[
\lambda = 2iP,\qquad h={Q^2-\lambda^2\over4}.
\]

## 2. Upsilon And DOZZ Data

`UpsilonB` evaluates \(\Upsilon_b(x)\) for positive real \(b\). It uses the strip
integral

\[
\log\Upsilon_b(x)=\int_0^\infty {dt\over t}
\left[
\left({Q\over2}-x\right)^2e^{-t}
-
{\sinh^2\left(({Q\over2}-x)t/2\right)
\over
\sinh(bt/2)\sinh(t/(2b))}
\right],
\qquad 0<\operatorname{Re}x<Q,
\]

and continues with the standard shift identities. The normalization is
\(\Upsilon_b(Q/2)=1\), so \(\Upsilon_b'(0)=\Upsilon_b(b)\).

The Xi-normalized three-point coefficient is implemented as
`yin_structure_constant_momentum`. It is the coefficient \(C(P_1,P_2,P_3)\) in
their equation (2.6), with the momentum-independent cosmological prefactor off
by default. At \(b=1\), the implementation evaluates their equation (2.9)
directly. This fixes the square-root branch in equation (2.6), including the
sign change under analytic continuation \(P_i\to-P_i\). It satisfies the useful
check

\[
C(P_1+P_2,P_1,P_2)= (2P_1+2P_2)(2P_1)(2P_2).
\]

For cross-checking, the file also exposes the standard alpha-space DOZZ constant
and the Hadasz-Jaskolski-Suchanek lambda convention.

## 3. Torus One-Point Integral

For external momentum \(P_{\rm ext}\), the implemented quantity is

\[
\langle V_{P_{\rm ext}}\rangle_\tau
=
\int_0^\infty {dP\over\pi}
C(P,P_{\rm ext},P)
\left|F^{h_{P_{\rm ext}}}_{c,h_P}(q)\right|^2,
\qquad q=e^{2\pi i\tau}.
\]

Following the Hadasz-Jaskolski-Suchanek modular-bootstrap normalization, the
Liouville integral evaluates the known eta factor separately:

\[
F^{h_{\rm ext}}_{c,h_P}(q)
=q^{h_P-(c-1)/24}\eta(q)^{-1}H^{h_{\rm ext}}_{c,h_P}(q),
\]

and truncates only the recursively computed \(H(q)\). This gives much better
modular behavior than truncating the full \(\eta^{-1}H\) product as a single
q-series.

In code:

```python
quadrature = LiouvilleTorusOnePointQuadrature.for_q_values(
    b=0.8,
    external_momentum=0.2,
    q_values=[cmath.exp(2j * cmath.pi * 0.9j)],
    block_order=4,
)
value = quadrature.full_one_point(cmath.exp(2j * cmath.pi * 0.9j))
```

or from the command line:

```bash
python3 plumbing/liouville_torus.py --b 0.8 --external-momentum 0.2 --tau 0.9i --block-order 4
```

For scans in \(q\), build one `LiouvilleTorusOnePointQuadrature` and reuse it.
The object precomputes the Gauss-Legendre \(P\) nodes, the DOZZ coefficient
\(C(P,P_{\rm ext},P)\), and the Virasoro block object for each node. The
structure constant is independent of \(q\), so this avoids recomputing the
Upsilon/DOZZ data during q-expansion or modular-covariance scans.

## 4. Exact b=1 Caveat

The Xi paper's `c=1` string application uses the \(c=25\) Liouville theory,
which has \(b=1\). The DOZZ side is handled here in that convention, including
the option to omit the singular cosmological prefactor.

The current block recursion, however, is the generic Zamolodchikov recursion.
At exactly \(b=1\), Kac labels collide in the residue products. The code
therefore rejects resonant \(b\) values before the integral is attempted. For a
numerical `c=25` experiment with the current block code, use a nearby
non-resonant regulator such as \(b=1+\epsilon\) and extrapolate. A literal
`b=1` computation would require implementing the analytic collision limit of
the recursion residues.

## 5. Genus-One Free-Energy Normalization

Three Liouville momentum descriptions can occur in comparisons. The BRY/Xi
variable used by the higher-genus CFT code has character
\(q^{P_{\rm BRY}^2}/\eta\). The physical momentum in the string note is defined
instead by

\[
e^{-\pi\alpha'\tau_2P_{\rm note}^2},\qquad
P_{\rm note}={2P_{\rm BRY}\over\sqrt{\alpha'}}.
\]

A historical coordinate called an `alpha'_L=2` Liouville field instead has

\[
 \kappa=\sqrt2P_{\rm BRY},\qquad h=1+{\kappa^2\over2}.
\]

It must not be substituted directly for \(P_{\rm BRY}\) in a DOZZ
coefficient. In the delta-normalized \(\kappa\) basis,
\(V_\kappa^{[2]}=2^{-1/4}V_{\kappa/\sqrt2}\) and the normalized cubic carries
\(2^{-3/4}\); these factors cancel the momentum Jacobians in a closed sewing.
The production integral therefore continues to use \(P_{\rm BRY}\).

Both half-line measures are written \(dP/\pi\), but their integration
variables differ. Per ordinary Liouville coordinate volume,

\[
{Z_L^{\rm note}\over V_\phi}
={2\over\sqrt{\alpha'}}Z_L^{\rm BRY}.
\]

The complete string-note vacuum density per \(V_\phi\) is

\[
 {1\over2}{|\eta(\tau)|^4 Z_{\rm compact}(R,\tau)
 (Z_{\rm Liouville}(\tau)/V_\phi)\over\tau_2}.
\]

The factor \(1/\tau_2\) divides the translation CKV volume. The factor
\(1/2\) divides the residual reflection automorphism of the unmarked torus.
After the eta functions cancel, the density is

\[
 {R\over4\pi\alpha'\tau_2^2}
 \sum_{m,n\in\mathbb Z}
 \exp\left[-\pi R^2 {|m\tau-n|^2\over\alpha'\tau_2}\right].
\]

Direct integration over the standard fundamental domain gives equation (4.95)
of the string note,

\[
 \boxed{
 {A_1(R)\over V_\phi}=\int_{\mathcal F}d^2\tau\,
 {R\over4\pi\alpha'\tau_2^2}\Theta_R(\tau)
 ={1\over12}\left({R\over\alpha'}+{1\over R}\right).}
\]

At \(\alpha'=1\), this is \((R+R^{-1})/12\), or \(1/6\) at the self-dual
radius. This is still the answer **per \(V_\phi\)**, not a coefficient of
\(\log\mu\).

For the canonically normalized interaction \(\mu e^{2b\phi}\), shifting the
zero mode gives

\[
V_\phi^{\rm ren}=-{1\over2b}\log\mu+C_{\rm cutoff}.
\]

Only after this separate substitution, at \(b=\alpha'=1\), does one obtain

\[
\mathcal F_1\big|_{\log\mu}
=-{R+R^{-1}\over24}\log\mu,
\]

which is the matrix-model coefficient. The additive constant in
\(V_\phi^{\rm ren}\) is regulator dependent; its slope with respect to
\(\log\mu\) is fixed by the exponent \(2b\phi\).

On the ribbon side, the ordered genus-one theta-graph lengths satisfy

\[
\ell_1+\ell_2+\ell_3={L\over2}.
\]

Cyclic permutations of \((\ell_1,\ell_2,\ell_3)\) give the same oriented
complex structure up to \(SL(2,\mathbb Z)\).  Odd permutations reverse the
orientation, sending the reduced modulus to the conjugate orientation
\(\tau\mapsto-\bar\tau\), and should not be included in the oriented moduli
quotient.  Therefore the effective graph automorphism factor for the oriented
ribbon integral is

\[
{1\over 3}.
\]

With the current `bghost_density` normalization, the implemented
ribbon-coordinate density is

\[
{1\over 3}
{1\over \tau_2(\ell)}
{(2\pi)^{18}\over 3}
b_{\rm ghost}^{\rm ribbon}(\ell)
Z_{\rm compact}(1,\tau(\ell))
Z_{\rm Liouville}(e^{2\pi i\tau(\ell)}).
\]

Equivalently,

\[
{(2\pi)^{18}\over 9\,\tau_2(\ell)}
b_{\rm ghost}^{\rm ribbon}(\ell)
Z_{\rm compact}(1,\tau(\ell))
Z_{\rm Liouville}(e^{2\pi i\tau(\ell)}).
\]

This formula reaches the unquotiented \(1/6\) result at \(R=1\).  The cyclic
factor \(1/3\) does not by itself demonstrate that the residual torus
reflection has been divided out.  Until the vertex-exchange/ribbon
automorphism is audited independently, the ribbon result must not be used to
set the absolute free-energy normalization.  The direct tau-coordinate result
\(1/12\) is the benchmark.  For higher genus, the corresponding generic
automorphism weight must likewise be included exactly once.

## 6. Checks

Run

```bash
python3 plumbing/liouville_torus_checks.py
.venv/bin/python plumbing/genus1_tau_normalization_checks.py
```

The checks verify:

- \(\Upsilon_b(Q-x)=\Upsilon_b(x)\) and the \(b\)-shift identity;
- agreement between the HJS lambda convention and alpha-space DOZZ;
- the Xi \(b=1\) resonance normalization;
- early rejection of resonant \(b=1\) in the generic block recursion;
- stability of a small generic-\(b\) torus integral under quadrature refinement;
- equality of the unsimplified CFT product and Poisson-resummed vacuum density;
- direct fundamental-domain integration at seven radii;
- T-duality, cusp-cutoff convergence, and the factor-of-two failure without
  the torus reflection weight.

## 7. Modular Covariance Check

The script `liouville_modular_check.py` follows the numerical test in section 5
of Hadasz-Jaskolski-Suchanek. For the full Xi-normalized scalar one-point
function it compares

\[
G(-1/\tau)\quad\hbox{with}\quad |\tau|^{2h_{\rm ext}}G(\tau).
\]

It also has an `--form hjs-stripped` mode, which removes the eta factor and
checks the equivalent HJS equation (25):

\[
I_H(-1/\tau)=|\tau|^{2h_{\rm ext}+1}I_H(\tau).
\]

Example:

```bash
python3 plumbing/liouville_modular_check.py --b 1.05 --external-momentum 0.2 --tau-imag-min 0.2 --tau-imag-max 5 --form hjs-stripped
```

The script `liouville_torus_plumbing_modular_check.py` is the plumbing-level
version of this diagnostic.  It first solves the torus plumbing seam problem for
the multiplier

\[
q=\exp(2\pi i\tau),
\]

and independently solves it again for

\[
q_S=\exp(2\pi i(-1/\tau)),\qquad
q_T=\exp(2\pi i(\tau+1)).
\]

It then evaluates the Liouville one-point function from the reconstructed
plumbing periods.  The \(S\)-check is

\[
G(-1/\tau)=|\tau|^{2h_{\rm ext}}G(\tau)
\]

for the full one-point function.  For \(T\), the plumbing multiplier is
unchanged, \(q_T=q\), so the seam solver reconstructs the principal period
rather than the chosen lift \(\tau+1\).  The diagnostic therefore reports both
the integer lift difference and the invariant one-point value.

Example:

```bash
.venv/bin/python plumbing/liouville_torus_plumbing_modular_check.py \
  --tau 0.2+0.9j \
  --b 0.8 \
  --external-momentum 0.2 \
  --block-order 3
```
