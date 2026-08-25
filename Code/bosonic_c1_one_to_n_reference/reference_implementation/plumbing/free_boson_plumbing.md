# Genus-Two And Genus-Three Free-Boson Plumbing

This note documents the free-boson calculations used in the genus-two
theta/glasses comparison and their five-channel genus-three extension:

```text
free_boson_pair_of_pants.py   direct Heisenberg Fock-space sewing
free_boson_plumbing.py        plumbing oscillator product and Bergman F
```

The direct state sum gives a finite-level construction in a specified pants
frame.  The primitive-word product is the all-level resummation of the same
Heisenberg module sum, for both the two established genus-two charts and the
five genus-three markings.

## 1. Direct Heisenberg pair-of-pants sewing

The rank-one Heisenberg vacuum module has basis

```text
a_-lambda |0> = a_-lambda_1 ... a_-lambda_k |0>
```

labelled by integer partitions.  Its Gram matrix is diagonal,

```text
<lambda|mu> = delta_lambda,mu z_lambda,
z_lambda = product_n n^m_n m_n!.
```

The three-punctured-sphere coefficient is computed directly as

```text
rho(lambda, mu, nu) = <lambda|Y(a_-mu|0>, 1)|nu>.
```

`free_boson_pair_of_pants.py` evaluates this matrix element by Wick
contractions of the Heisenberg current and its derivatives.  It then contracts
the pants vertices with inverse Gram matrices and the same unshifted `q^L0`
propagators used in the CCY plumbing convention.

For the theta graph, whose public plumbing labels are `(q_zero, q_one,
q_infty)`, the chiral sum is

```text
Z_theta^pl = sum_lambda,mu,nu
  q_infty^|lambda| q_one^|mu| q_zero^|nu|
  rho(lambda,mu,nu)^2 / (z_lambda z_mu z_nu).
```

The reversed first/third ordering is required because `rho` is ordered as
`(infinity bra, insertion at one, zero ket)`.

For the glasses graph,

```text
Z_glasses^pl = sum_lambda,mu,nu
  q_left^|lambda| q_right^|mu| q_bridge^|nu|
  rho(lambda,nu,lambda) rho(mu,nu,mu)
  / (z_lambda z_mu z_nu).
```

The nonchiral oscillator partition is `|Z_chiral^pl|^2`.  No `-c/24`
Casimir factor is inserted.

Run the direct checks with

```text
python3 plumbing/free_boson_pair_of_pants_checks.py
```

They verify the Heisenberg Gram factors, elementary pants coefficients, the
theta level-two identity, glasses separating factorization, and agreement
with the independent plumbing-frame oscillator product.

## 2. Plumbing-frame primitive-word oscillator product

The all-level oscillator answer is resummed by

```text
Z_X,osc^Sch(q) = product_primitive_gamma product_n>=1 |1-k_gamma^n|^-2,
```

where `k_gamma` is the attracting multiplier of a primitive Schottky word.
The generators are constructed from the same theta or glasses plumbing maps.

At a small generic point, direct total-level 7 sewing agrees with the
word-length 10 product at relative differences

```text
theta   1.41e-12
glasses 2.43e-12.
```

At the period-matched overlap point, convergence is slower because all three
glasses parameters are `0.15`:

```text
direct level   theta/glasses
12             0.4661738200075466
16             0.4661616323875529
18             0.4661610259958539
20             0.4661608730773036

Schottky       0.4661608209183812   (word length 12)
```

Thus the primitive-word oscillator product is a high-accuracy all-level
evaluation of the direct plumbing state sum, not a substitute for having
defined that state sum.

It must not be confused with the CCY large-`c` Virasoro vacuum seed:

```text
free Heisenberg oscillator:  product_gamma product_(m>=1) |1-k_gamma^m|^-2
CCY seed, oriented classes:  product_gamma product_(m>=2) (1-k_gamma^m)^-1/2
CCY seed, inverse-paired:     product_{gamma,gamma^-1} product_(m>=2) (1-k_gamma^m)^-1
```

The CCY expression is only part of the regular seed for the Liouville
central-charge recursion.  The free-boson expression is exact for the Gaussian
Heisenberg oscillators: Wick contractions exponentiate into this product.  The
code's primitive-word enumerator identifies `gamma` with `gamma^-1`, so the
last, exponent-`-1` CCY convention is the one implemented.

The primitive-word oscillator product is also **not** the Bergman spectral
determinant.  The two comparisons are logically separate:

```text
direct Fock plumbing = primitive-word oscillator product   (same frame)
W_a = Z_a^plumbing / Z^Bergman(Omega)                      (Weyl change)
```

Only the second line contains the Weyl anomaly.

## 3. Five genus-three plumbing channels

`free_boson_pair_of_pants.py` directly contracts four Heisenberg
three-punctured-sphere vertices and six inverse Gram propagators for any of
the five marked genus-three channels:

```text
one-tadpole-double-triangle
opposite-double-edge-cycle
tetrahedron
three-tadpole-star
two-tadpoles-double-bridge
```

The edge order, local \(0,1,\infty\) slots, spanning tree, and chord marking
come from `genus3_plumbing_channels.py` and agree with the genus-three
period-matrix note.  The three Schottky generators are all expressed in one
root-sphere coordinate, including channels with parallel edges and
self-plumbing loops.

The finite-level pants-frame call is

```python
from plumbing.free_boson_pair_of_pants import genus3_heisenberg_plumbing_partition

result = genus3_heisenberg_plumbing_partition(
    "three-tadpole-star",
    {
        "q01": 0.02,
        "q02": 0.02,
        "q03": 0.02,
        "q11": 0.02,
        "q22": 0.02,
        "q33": 0.02,
    },
    max_total_level=8,
)
```

It returns the truncated chiral sum, its nonchiral absolute square, and every
nonzero six-edge level contribution.

For the genus-three Liouville quotient, the production oscillator factor is
the primitive-word resummation below.  It is multiplied by
`noncompact_scalar_loop_momentum_factor(native_omega)` using the native period
matrix returned by the same channel's plumbing solve.  A period matrix
transformed to a common `Sp(6,Z)` marking is used only to certify an overlap;
it is not inserted into one channel's scalar.

`free_boson_plumbing.py` separately constructs the three exact Schottky
generators and evaluates their all-level primitive-word product:

```python
from plumbing.free_boson_plumbing import genus3_free_boson_product

result = genus3_free_boson_product(
    "three-tadpole-star",
    {
        "q01": 0.02,
        "q02": 0.02,
        "q03": 0.02,
        "q11": 0.02,
        "q22": 0.02,
        "q33": 0.02,
    },
    max_word_length=5,
    max_mode=40,
)
```

This canonical Schottky product resums the raw \(0,1,\infty\) Heisenberg pants
sum in the same plumbing frame.  The current generic-graph direct contraction
does not reproduce this identity in several genus-three slot patterns even
after its total-level series has stabilized.  That discrepancy is therefore
a bug diagnostic for the direct CFT sewing (most likely in the endpoint
transport/BPZ convention), not evidence for a second scalar frame.  Until the
direct contraction is repaired, it must not replace the product when the
scalar is subsequently raised to the 25th power.

At the saved opposite-double-edge-cycle/tetrahedron genus-three locality
point, the direct level-12 oscillators gave a misleading scalar ratio.  The
resummed values are

```text
Z_osc,A = 0.9771481658069647
Z_osc,B = 1.0004817793809655
```

After multiplying by the native-period Gaussians and forming
`Q_L = Z_L / Z_X^25`, the block-14, momentum-quadrature-7 comparison changes
from `Q_A/Q_B = 0.2372807389` to `0.9997593958`.  The residual locality
difference is `-2.406e-4`; the Schottky word-tail contribution is negligible
at that scale, so the remaining error follows the Liouville momentum cutoff.

The dimensionless handle Gaussian has also been generalized from
\(\det(\operatorname{Im}\Omega)^{-1/2}\) at genus two to any square period
matrix.  Xi's physical-momentum convention is

```text
det(Im Omega)^(-1/2) / ((2 pi)^g alpha'^(g/2)).
```

Run the five-channel checks with

```text
python3 plumbing/free_boson_genus3_checks.py
```

They test direct total-level convergence in every pants channel, direct and
Schottky conjugation, Schottky word-length stability, rank reduction when a
chord is pinched, and the genus-three loop-momentum normalization.  A
genus-three Bergman determinant and both local-coordinate and Weyl
conversions are separate work and are not inferred from the genus-two
Igusa-\(\chi_{10}\) formula below.

## 4. Canonical Bergman determinant

The canonical comparison object is

```text
F(Omega) = det(Im Omega)^(5/2) product_even |theta[delta](0|Omega)|.
```

Klein-Kokotov-Korotkin give

```text
det' Delta_B = C_B F(Omega)^(1/3),
```

where `C_B` is moduli-independent.  For one real scalar with the constant zero
mode omitted,

```text
Z_X^B = (det' Delta_B)^(-1/2)
      = C_B^(-1/2) F(Omega)^(-1/6).
```

Therefore the plumbing-to-Bergman frame factor in channel `a` is

```text
W_a = Z_X,a^pl / Z_X^B
    = C_B^(1/2) Z_X,a^pl F(Omega_a)^(1/6).
```

At a period-matched point one common `Omega` must be used.  Both `F(Omega)` and
the unknown constant cancel:

```text
W_theta / W_glasses = Z_X,theta^pl / Z_X,glasses^pl.
```

The direct Heisenberg sum is only the oscillator factor.  For a noncompact
scalar, sewing the continuous handle momenta adds

```text
Z_X,zero(Omega) = det(Im Omega)^(-1/2).
```

This is distinct from the common constant target-space zero mode.  Because the
saved theta and glasses period matrices use different symplectic markings,
their determinants of imaginary parts differ.  The factor in those markings is

```text
W_theta/W_glasses
  = (Z_X,osc^theta/Z_X,osc^glasses)
    * sqrt(det Im Omega_glasses / det Im Omega_theta).
```

At direct sewing level 20 at the central overlap point,

```text
oscillator ratio, c=1             0.4661608730773036
continuous-momentum ratio, c=1    0.08928555539163692
full scalar frame ratio, c=1      0.04162143245455742
full scalar frame ratio, c=25     3.0403639419278054e-35
```

The direct oscillator sum is not a hidden compact momentum sum.  For a
rank-one lattice theory Mason--Tuite's sewing theorem gives

```text
Z_lattice^pl,a = Z_Heisenberg^pl,a * Theta_L^(2)(Omega).
```

After transporting both formulas to one marking, the same
`Theta_L^(2)(Omega)` cancels from `theta/glasses`.  If the two recorded markings
are retained, its modular automorphy must be retained as well.  The noncompact
Gaussian above is the decompactified form used by the current comparison.

## 5. Absolute normalization from the separating tube

The absolute normalization can be tested without assigning a value to the
unknown Bergman determinant constant.  Use the normalized Arakelov scalar as
an intermediate canonical frame.  With the connected target zero mode divided
by

```text
V_X/(2*pi),
```

its genus-one partition function is

```text
Z_1(tau) = 1/(sqrt(Im tau) |eta(tau)|^2).
```

This convention is also the plumbing convention in which momentum states obey
`<p|p'>=delta(p-p')` and completeness uses `dp`.  The genus-two factor

```text
det(Im Omega)^(-1/2)
```

is therefore the Gaussian integral over the two handle momenta.  It is not the
connected constant target-space zero mode.

The normalized Arakelov determinant is

```text
Z_X^Ar = 2 |prod_even theta[delta](0|Omega)^2|^(-1/12)
           Phi(Omega)^(-1/6) / sqrt(det Im Omega).
```

In a separating glasses degeneration,

```text
Omega_12 = 2*pi*i*t + O(t^3),
||t|| = 4*pi^2 |t| |eta(tau_L) eta(tau_R)|^2,
Phi(Omega) = |eta(tau_L) eta(tau_R)|^2 + O(t^2),
Z_X^Ar -> ||t||^(-1/6) Z_1(tau_L) Z_1(tau_R).
```

The raw torus sewing convention has

```text
Z_1^pl(q) = |q|^(1/12) Z_1(tau).
```

Consequently the directly predicted Weyl quotient is

```text
W_pl/Ar^asym = |q_L q_R|^(1/12) ||t||^(1/6).
```

No constant is fitted.  The number extracted from the two calculations is

```text
kappa_X(q_B)
  = (Z_X^pl/Z_X^Ar) / (|q_L q_R|^(1/12) ||t||^(1/6)).
```

`audit_free_boson_long_tube_normalization.py` evaluates `Omega` with normalized
holomorphic one-forms, evaluates the plumbing scalar with the all-level
primitive-word product and the handle Gaussian, and evaluates the canonical
side with exact `chi10` and the leading separating expression for `Phi`.  For
`q_L=0.08` and `q_R=0.11` it gives

```text
|q_B|     4*pi^2|t|/|q_B|    kappa_X-1
1e-2      1.005039885830      +7.013e-7
1e-3      1.000500395736      +6.952e-9
1e-4      1.000050003954      +6.792e-11
1e-5      1.000005000043      +7.233e-12
```

A linear boundary extrapolation in `|q_B|^2` gives
`kappa_X(0)=0.9999999999809734`.  Thus the plumbing and normalized canonical
free-boson calculations use the same scalar normalization at the tested
precision.  A phase-deformed check at `|q_B|=1e-3` gives
`|kappa_X-1|=5.66e-9`.  Omitting `det(Im Omega)^(-1/2)` fails, while mixing a
partition per ordinary target volume `V_X` with one per `V_X/(2*pi)` shifts
the result by exactly `2*pi`.  There is no hidden factor `2^12` in this scalar
conversion.

Both graph implementations use the same unit sphere three-point coefficient,
inverse Heisenberg state metric, and three unshifted propagators; their
level-zero theta and glasses sums are separately one.  The separating audit
therefore fixes their common local scalar normalization.  The same-period
theta/glasses calculation then determines their relative frame transition.

This check uses the fully normalized Arakelov determinant of
[Vandermeulen](https://arxiv.org/abs/1902.02420).  Converting its absolute
normalization to the unit-area Bergman determinant still requires the direct
Arakelov-to-Bergman Polyakov action; setting the KKK constant `C_B=1` is not a
substitute for that metric conversion.

Run the audit and its checks with

```text
python3 plumbing/audit_free_boson_long_tube_normalization.py
python3 plumbing/audit_free_boson_long_tube_normalization_checks.py
```

## 6. What the coefficient 64 checks

The Bergman function has the boundary asymptotics

```text
F_glasses^asym
  = 64 |q_bridge| (Im tau_1 Im tau_2)^(5/2)
      |eta(tau_1) eta(tau_2)|^12,

F_theta^asym
  = 64 |q_zero q_one q_infty|^(1/2)
      det(Im Omega_trop)^(5/2).
```

Both plumbing boundaries reproduce the coefficient `64`.  This verifies the
theta-constant and degeneration conventions.  It does not replace the direct
free-boson sewing calculation and does not determine the unknown determinant
constant `C_B`.

Dividing each channel by its full, q-dependent boundary asymptotic produces a
finite-part diagnostic.  With determinant exponent `-1/2`, its current ratio is

```text
c=1   0.9873822740248464
c=25  0.7280029800809033.
```

This is not the relative same-Omega Weyl factor.  The earlier use of this
finite-part ratio as the Liouville correction was incorrect.

## 7. Applying the resummed frame factor

The legacy postprocessor now defaults to the Schottky-resummed oscillator
times the native-period Gaussian.  Its direct sewing value is printed only as
a finite-level diagnostic:

```text
python3 plumbing/apply_free_boson_weyl_to_overlap.py \
  --direct-level 20 \
  --max-word-length 12 \
  --max-mode 100 \
  --out-json plumbing/results/theta_glasses_weyl_overlap_c25_q015_w7_seed_10_12.json \
  --out-csv plumbing/results/theta_glasses_weyl_overlap_c25_q015_w7_seed_10_12.csv
```

For the old under-resolved Liouville momentum quadrature, the order-12 row was

```text
raw Liouville theta/glasses              3.166919401457097e-36
```

That run used only three global momentum nodes and missed the narrow theta
threshold peak.  With primary-Gaussian quadrature, the full scalar factor, and
the inverse-paired CCY vacuum seed, the block-order-10, quadrature-order-10
central result is

```text
raw Liouville theta/glasses              3.043438944355056e-35
full-free-boson-frame corrected          1.001011392874664
```

Four independently period-matched real deformations give corrected ratios
between `1.0000748` and `1.0016330` at the same numerical order.  These are
controlled channel checks, not threshold-shifted diagnostics.

## 8. Main commands

```text
python3 plumbing/free_boson_pair_of_pants_checks.py
python3 plumbing/free_boson_plumbing_checks.py
python3 plumbing/normalization_selfcheck.py
python3 plumbing/audit_free_boson_long_tube_normalization_checks.py
```
