# Type-0B NS sphere multipoint crossing check

Date: 2026-08-25

## Scope

This check assembles the bottom-component NS Liouville sphere correlator from
the multipoint fixed-weight central-charge recursion and compares two pants
decompositions. It is the matter-CFT crossing test relevant to Type 0B. It is
not yet a complete Type-0B string amplitude: timelike matter, ghosts, picture
changing, and the supermoduli measure are not included.

The computation is at the self-dual point

\[
 c=\frac{27}{2},\qquad \widehat c=9,\qquad
 h(P)=\frac{1+P^2}{2}.
\]

No displacement in the central charge is used.

## Five-point sewing formula

For a standard comb frame

\[
 (z_1,z_2,z_3,z_4,z_5)=(0,z_2,z_3,1,\infty),
 \qquad q_1=\frac{z_2}{z_3},\quad q_2=z_3,
\]

the chiral block reconstructed from the coefficient recursion is

\[
 \mathcal F_{\boldsymbol\alpha}(q_1,q_2)
 =q_1^{h_a-h_1-h_2}
  q_2^{h_b-h_1-h_2-h_3}
  \sum_{\ell_1,\ell_2\geq0}
  B^{\boldsymbol\alpha}_{\ell_1,\ell_2}
  q_1^{\ell_1}q_2^{\ell_2}.
\]

The functional implementation evaluates the same `c`-recursion with finite
global `osp(1|2)` series at its leaves. If
\(C^{(0)}=C\) and \(C^{(1)}=\widetilde C\), the nonchiral standard-frame
correlator is

\[
 G_5=\int_0^\infty\frac{dP_a}{\pi}
         \int_0^\infty\frac{dP_b}{\pi}
 \sum_{\alpha_0+\alpha_1+\alpha_2\in2\mathbb Z}
 C^{(\alpha_0)}(P_1,P_2,P_a)
 C^{(\alpha_1)}(P_a,P_3,P_b)
 C^{(\alpha_2)}(P_b,P_4,P_5)
 \left|\mathcal F_{\boldsymbol\alpha}\right|^2.
\]

Thus all four allowed vertex-sector assignments
`000`, `011`, `101`, and `110` are included. The block supplies the
odd-null parity transport; the structure-constant layer supplies `C` or
`tilde C` at each trinion.

### Convention boundary

This formula is BRY-native.  Its \(\widetilde C\) is the real coefficient of
BRY's locally normalized top component, so no extra factor of \(i\) is
inserted in this sphere correlator.  This differs from the separate Human
Note's graded sewing basis.  There the raw ordered state
\((G_{-1/2}V)\otimes(\widetilde G_{-1/2}\widetilde V)\) has the opposite BPZ
norm, and the one-time interface map is

\[
C_{\rm HN}^{(1)}=\sigma i\,\widetilde C_{\rm BRY},
\qquad \sigma=\pm1,
\qquad
(C_{\rm HN}^{(1)})^2=-\widetilde C_{\rm BRY}^{2}.
\]

Only the squared relation is fixed without an explicitly ordered component
matrix element.  This map must not be applied to the BRY-native product
written above; doing so would double-count the convention change.

For finite physical punctures, each ordered channel is mapped to
\((0,\ldots,1,\infty)\). The primary-field Jacobian, including the finite
limit of the puncture sent to infinity, is restored before channels are
compared. Raw standard-frame correlators are therefore not compared.

## Numerical point and channels

The external momenta and finite punctures are

\[
 (P_1,\ldots,P_5)=\left(\frac12,\frac13,\frac14,\frac35,\frac25\right),
 \qquad
 (z_1,\ldots,z_5)=\left(0,\frac1{20},\frac1{10},1,2\right).
\]

The two comb orders are

```text
left  = (0,1,2,3,4)   # ((0,1),2)
right = (2,1,0,3,4)   # ((2,1),0)
```

Their plumbing parameters and covariance factors are

| channel | `(q1,q2)` | `max |q|` | covariance factor |
|---|---:|---:|---:|
| left | `(19/39, 1/19)` | `0.48717949` | `0.12628968712068464` |
| right | `(20/39, -1/18)` | `0.51282051` | `0.15362229438702713` |

The cluster was chosen so that both local series lie well inside their
convergence domains. Different Gauss-Legendre orders are used on the two
internal edges. This avoids sampling the measure-zero diagonal
\(P_a=P_b\), where equal internal weights create an artificial coincident
fixed-`c` pole in a tensor-product quadrature.

## Block-cutoff convergence

The continuum is truncated at \(P_a,P_b\leq3\), with Gauss-Legendre orders
8 and 9 on the two edges. `R` is the maximum accumulated recursion
twice-level, while `(L1,L2)` are the global-leaf series cutoffs.

| `R` | `(L1,L2)` | left physical | right physical | relative residual |
|---:|---:|---:|---:|---:|
| 4 | `(10,4)` | 1.783017174996658 | 1.775694535990144 | 4.1069e-3 |
| 6 | `(14,6)` | 1.787601563919021 | 1.784341847841007 | 1.8235e-3 |
| 8 | `(18,8)` | 1.789745105687501 | 1.788323773123422 | 7.9415e-4 |

The channel difference decreases by a factor of roughly 2.3 at each step.
No equality is imposed or used to average the two values.

## Momentum-quadrature check

At fixed `(R;L1,L2)=(6;14,6)` and `Pmax=3`, increasing the base quadrature
order gives

| edge orders | left physical | right physical | relative residual |
|---:|---:|---:|---:|
| `(8,9)` | 1.787601563919021 | 1.784341847841007 | 1.8235e-3 |
| `(10,11)` | 1.779439443915569 | 1.776900405055665 | 1.4269e-3 |
| `(12,13)` | 1.778741506884299 | 1.776547975112673 | 1.2332e-3 |

The individual channel values still move at the few-per-mille level, so the
best block-cutoff residual above is an observed finite-cutoff residual, not
a certified error bound on the infinite continuum integral.

## Four-point control

The same general assembly was reduced to four points at

```text
momenta = (1/2,1/3,1/4,3/5)
points  = (0,2/3,1,2)
channels = (0,1,2,3) and (2,1,0,3)
```

Both standard frames have `q=1/2`. With recursion twice-level 8, global
twice-level 20, `Pmax=5`, and quadrature order 20, the physical-frame values
are

```text
left  = 0.07762053515262507
right = 0.07761760337712030
relative residual = 3.7771e-5
```

This independently checks the Mobius covariance and spectral sewing layer
against the already established sphere four-point crossing setup.

## Reproduction

From the repository root:

```bash
PYTHONPATH=Code/c_Recursion:Code python3 \
  Code/c_Recursion/stress_ns_multipoint_crossing.py
```

The machine-readable ledger is

```text
Data Set/ns_sphere_fivepoint_crossing_c_recursion.json
```

Targeted regression tests are

```bash
PYTHONPATH=Code/c_Recursion:Code python3 -m unittest \
  Code/c_Recursion/test_ns_multipoint_c_recursion.py \
  Code/c_Recursion/test_sphere_multipoint.py
```

## Limitations

1. The global leaves and recursion depth are finite.
2. Both internal momentum integrals use a finite cutoff and fixed quadrature.
3. The computation covers bottom-component NS external primaries. Other
   component choices require their corresponding trinion tensors.
4. This is a Liouville matter correlator. A full Type-0B string `N`-point
   amplitude still needs the timelike, ghost, picture-changing, and
   supermoduli factors.
