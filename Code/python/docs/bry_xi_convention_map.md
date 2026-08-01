# BRY-to-Xi Convention Map

> **c=1 sphere-topology correction (2026-07-21; newest verdict).** The local
> BRY/Xi Liouville map and the correlated free-field replacement remain
> exactly as stated below. They do not, however, replace the critical sphere
> metric by the c=1 sphere metric. With
> `K_S2^crit=8*pi/alpha'` and
> `Khat_S2^c1=2*pi*sqrt(alpha')*(2/sqrt(alpha'))=4*pi`, the genus-two
> topology density acquires `Lambda_top=2/alpha'`. More generally its genus-g
> vacuum factor is `(2/alpha')^(g-1)`: the genus-one vacuum is therefore
> unchanged. At `alpha'=1`, the final c=1 kernel coefficient is `2/pi`. This
> is analytically pinned down but not yet applied to the production kernel or
> saved integrations.

> **Xi free-field correction (2026-07-21).** The genus-two scalar bridge is
> no longer undetermined. The sewing code's dimensionless loop Gaussian and
> Xi's physical-momentum convention obey
> `Z_X^Xi=Z_X,p/(4 pi^2 alpha')`; the compact connected zero mode is
> `2 pi R_phys`. When the 26-boson critical seed is converted at the same
> time, the complete c=1 density acquires exactly
> `1/(2 pi sqrt(alpha'))`. Hence at `alpha'=1` old stored absolute values are
> divided by `2 pi`, with no change to radius ratios. A normalized
> genus-one-anchored separating sewing then fixes the correlated local
> critical/scalar/ghost bridge to one; it is not an adjustable residual. The
> sphere-topology factor recorded above is a separate Euler-characteristic
> normalization.

## Verdict

The BRY-to-Xi conversion is nontrivial for a **full string amplitude**, but it
is trivial for the intrinsic Liouville CFT used by the plumbing code.

```text
Liouville primary/metric/DOZZ/dP/pi/q:  BRY = Xi exactly
genus-one physical-volume conversion:   J_phi^(1)=2/sqrt(alpha')
intrinsic closed genus-two sewing:       A_L^sewn=1
string coupling:                        g_s^BRY = 2 g_s^Xi
one complex modulus:                    i dz wedge dbar(z) = 2 d^2z
free-scalar replacement at genus two:   fixed exactly
correlated local CFT bridge:             Lambda_local = 1, certified
c=1 sphere-topology bridge:              Lambda_top = 2/alpha', certified
topology factor applied to production:   no
```

The current Monte Carlo kernel is not a BRY-normalized full amplitude. It uses
BRY/Xi-common Liouville data inside Xi's positive real period-coordinate
kernel and Xi's coupling dictionary. Therefore neither the factor two in the
coupling nor the factor eight in the genus-two real measure should be applied
again.

## 1. Intrinsic Liouville CFT

BRY and Xi use the same normalized primary

```text
V_P = S(P)^(-1/2) V_in,
```

the same two-point metric and inverse metric

```text
<V_P V_P'> = pi delta(P-P'),
1 = integral_0^infinity dP/pi |P><P|,
```

and the same `b=1` DOZZ coefficient, with `Upsilon_1(1)=1`. Xi equation
(4.119) is the BRY formula used by the code. Both sewing constructions also
use the literal local relation `u*v=q`, with a primary propagator `q^h`.

Hence a genus-two Liouville partition assembled from two pants vertices and
three inverse two-point metrics is identical in the two conventions:

```text
Z_L^Xi,sewn = Z_L^BRY,sewn.
```

There is no independent factor `A_L` in this intrinsic closed-surface sewing
statement.  This must be distinguished from the volume-normalized
unpunctured torus trace, whose conversion is nontrivial.

### Genus-one audit of the Liouville volume-density convention

Let `P_CFT` denote the dimensionless Liouville momentum used in Appendix I,
the DOZZ coefficient, and the plumbing code.  At `b=1`,

```text
h(P_CFT)=1+P_CFT^2,
<P_CFT|P_CFT'>=pi delta(P_CFT-P_CFT').
```

After the nonchiral `c=25` vacuum shift, the BRY/CCY continuum character
trace is

```text
Z_L^BRY,(1)
 = integral_0^infinity dP_CFT/pi
     exp(-4*pi*tau2*P_CFT^2)/|eta(tau)|^2
 = 1/[4*pi*sqrt(tau2)*|eta(tau)|^2].
```

Equivalently, if `B_L^raw,(1)` is the unshifted plumbing block,

```text
Z_L^BRY,(1)=|q|^(-25/12) B_L^raw,(1).
```

Xi equation (4.92) instead writes the torus trace per ordinary Liouville
length using the dimensionful asymptotic momentum `P_note`:

```text
Z_L^Xi,(1)/V_phi
 = integral_0^infinity dP_note/pi
     exp(-pi*alpha'*tau2*P_note^2)/|eta(tau)|^2
 = 1/[2*pi*sqrt(alpha'*tau2)*|eta(tau)|^2].
```

Matching the characters fixes

```text
P_note=2*P_CFT/sqrt(alpha'),
dP_note/pi=(2/sqrt(alpha')) dP_CFT/pi.
```

Therefore the genus-one volume-density conversion is the pointwise identity

```text
Z_L^Xi,(1)/V_phi
 = [2/sqrt(alpha')] Z_L^BRY,(1)
 = [2/sqrt(alpha')] |q|^(-25/12) B_L^raw,(1),
```

It is useful to name this conversion separately:

```text
J_phi^(1),per-V_phi = 2/sqrt(alpha').
```

At `alpha'=1` this is exactly `2`.  It also gives Xi's genus-one equality

```text
Z_L^Xi,(1)/V_phi = Z_X^Xi,(1)/V_X
 = 1/[2*pi*sqrt(alpha'*tau2)*|eta(tau)|^2].
```

This factor must not be copied directly into the closed genus-two partition.
The symbol `P` is overloaded in the string note.  The `P_note` in the
dimensionful Gaussian of (4.92) is the physical asymptotic wave number used
to define the ordinary length `V_phi`.  The momentum in Appendix I, in the
weight `h=1+P_CFT^2`, and in the DOZZ coefficient (4.119) is the dimensionless
intrinsic CFT label `P_CFT`.  The genus-two sewing integral keeps this latter
momentum unchanged:

```text
Z_L^(2)=integral product_e dP_CFT,e/pi
          C(P_CFT,1,P_CFT,2,P_CFT,3)^2 |F(h(P_CFT,e);q_e)|^2.
```

In particular, Xi does **not** replace the DOZZ arguments by
`2*P_CFT/sqrt(alpha')`.  If one formally reparameterized the entire Hilbert
space by `P_CFT=a k`, the correctly normalized cubic coefficient would be

```text
C_k(k1,k2,k3)=a^(3/2) C_CFT(a*k1,a*k2,a*k3),
h_k(k)=1+a^2*k^2.
```

The arguments and weights change as well as the overall state normalization;
the resulting closed sewing integral is only a change of variables.  Writing
merely `C_k=a^(3/2) C_CFT` would be wrong because the DOZZ function is not
homogeneous.  No such reparameterization is made in the production formula.

Thus the convention ledger contains two different, compatible statements:

```text
J_phi^(1),per-V_phi = 2/sqrt(alpha') (unpunctured torus volume density),
A_L^sewn             = 1             (intrinsic normalized DOZZ sewing).
```

The genus-one computation therefore checks the `V_phi` density convention,
but it does not determine an additional intrinsic `A_L` for genus two.  A
genuinely new genus-two constant would have to survive normalized
once-punctured-torus factorization; it cannot be inferred by inserting the
physical `P_note` into the DOZZ coefficient.

## 2. Coupling Dictionary

BRY determine

```text
mu^-1 = 2*pi*g_s^BRY,
```

whereas Xi equation (4.122) gives

```text
mu^-1 = 4*pi*g_s^Xi.
```

Therefore

```text
g_s^BRY = 2*g_s^Xi.
```

For a genus-`g`, `n`-point amplitude the coupling power is `2g-2+n`. At genus
two with no punctures, changing the name of the coupling changes the displayed
coefficient by four. This does not imply a factor four in the current result,
because the current target and integration kernel already use `g_s^Xi`.

## 3. Moduli Differential

BRY write ordinary area integrals `d^2z`. Xi works with differential forms.
For each complex modulus,

```text
i dz wedge dbar(z) = 2 d(Re z) d(Im z) = 2 d^2z.
```

Thus the bare Xi/BRY real-measure factor is `2^d`, where
`d=3g-3+n`. Two useful cases are

```text
(g,n)=(1,2): d=2, measure factor=4, coupling weight=1/4;
(g,n)=(2,0): d=3, measure factor=8, coupling weight=1/4.
```

The first product equals one and reproduces Xi's explanation of the
lower-genus coupling convention. The genus-two bare product equals two, but
it excludes state metrics, the critical-to-`c=1` replacement, and the stack
quotient. It is not a correction factor for the code.

## 4. Genus-Two Coefficient Ledger

Xi equation (4.105) displays

```text
g_s_Xi^2 * alpha'/(8*pi)
```

on the complex six-form. With `N_(2,0)=-i`, the sign in equation (4.106), and

```text
d^3Omega wedge d^3bar(Omega) = -8i d^3X d^3Y,
```

the positive real coefficient is

```text
g_s_Xi^2 * alpha'/pi.
```

Separately, extrapolating BRY's lower-genus constants with

```text
C_Sigma2*C_S2=C_T2^2,
C_S2=2*pi/g_s_BRY^2,
C_T2=1,
```

would give

```text
C_Sigma2^BRY = g_s_BRY^2/(2*pi)
              = 2*g_s_Xi^2/pi.
```

The residual factor two is real bookkeeping, but it is not yet a contradiction:
BRY do not write a genus-two vacuum formula, and the extrapolated coefficient
does not specify the same scalar/ghost state metric and automorphism convention
as Xi's critical-boson construction. In particular, one must not use this
extrapolation and then independently reapply the coupling and differential-form
changes; that double-counts factors already bundled in the two coefficients.

## 5. Fixed Scalar and Full-Partition Bridge

For the scalar replacement, let

```text
A_crit = (4*pi^2*alpha')^-26,
Z_X^Xi = A_X Z_X,p,                  A_X=(4*pi^2*alpha')^-1,
Z_XR^Xi = A_XR [R Z_X,p Theta_R],    A_XR=1/(2*pi*sqrt(alpha')).
```

Since the Liouville factor is one, the exact scalar relation is

```text
I_c=1^Xi / I_c=1^code,old
 = A_crit*A_XR/A_X^26
 = 1/(2*pi*sqrt(alpha')).
```

These constants must be applied together because the critical Mumford density
already contains 26 scalar normalizations. Treating `A_X` in the quotient
while leaving `A_crit` fixed would be inconsistent. The production code now
performs the correlated conversion.

An independent cross-check is a normalized maximal genus-two sewing of

```text
26 free bosons times bc ghosts
```

in the same pants coordinates, followed by comparison with Xi equation
(4.105), now using the fixed scalar dictionary above. The genus-one-anchored
separating audit already performs the normalization-sensitive check with the
correct once-punctured torus states and gives `Lambda_local=1`. Since genus
one has zero Euler characteristic, that check cannot determine the sphere
constant; the separate sphere audit gives `Lambda_top=2/alpha'`.

## 6. The Near-`2 pi 2^12` Numerical Discrepancy

The bare BRY/Xi measure and coupling changes give

```text
genus-two measure factor = 8,
genus-two coupling weight = 1/4,
bare product = 2.
```

The bare product `2` agrees with the independently derived c=1
sphere-topology correction at `alpha'=1`; it must not be applied a second
time. The raw-theta-product coefficient `2^12` is already incorporated once
through the `2^24` nonchiral Mumford conversion, so an additional
`2*pi*2^12` factor is not permitted by the BRY-to-Xi map.

## Executable Audit

```text
python plumbing/audit_bry_xi_convention_map.py
python plumbing/audit_bry_xi_convention_map_checks.py
python plumbing/genus2_integrand_factorization_audit.py
```

The machine-readable convention ledger is written to
`plumbing/results/genus2_c1_moduli_mc/bry_xi_convention_map.json`.

Primary sources: [BRY](https://arxiv.org/abs/1705.07151) and Xi's local
`string notes-3.pdf`, especially equations (4.58)-(4.72), (4.97)-(4.109), and
(4.111)-(4.122).
