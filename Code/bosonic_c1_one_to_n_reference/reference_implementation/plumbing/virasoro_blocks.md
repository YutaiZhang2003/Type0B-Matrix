# Virasoro Blocks For Plumbing

This note accompanies `virasoro_blocks.py`. It implements universal chiral
Virasoro factors for the genus-one one-point channel and the torus multipoint
necklace channel. The dynamical CFT input, such as two-point normalizations and
three-point coefficients, is not included here.

## 1. Torus One-Point Decomposition

For a primary insertion with holomorphic weight \(h_i\), the chiral part of the
genus-one one-point function is assembled as

```text
sum_O C_{O i O} F_O^{(1)}(h_i; q)
```

and the full non-chiral answer multiplies this by the analogous anti-holomorphic
block.

The implemented chiral block has the convention

\[
F_{c,h}^{h_i}(q)
=
q^{h-c/24}
\sum_{n\geq 0} q^n F_{c,h}^{h_i,n},
\qquad F_{c,h}^{h_i,0}=1.
\]

Equivalently, it is computed through the elliptic block \(H\):

\[
F_{c,h}^{h_i}(q)
=
q^{h-c/24}
\left[\prod_{m\geq 1}(1-q^m)^{-1}H_{c,h}^{h_i}(q)\right].
\]

In code this is returned by `TorusOnePointVirasoroBlock.chiral_block`.

## 2. Momentum Convention

The recursion is written with Liouville-style momenta:

\[
c=1+6Q^2,\qquad Q=b+b^{-1},
\]

and

\[
h_\lambda={Q^2-\lambda^2\over 4}.
\]

Given `c` and `external_weight`, the code chooses a branch for \(b\) and then
solves for \(\lambda_i\). A user can override this by passing `b=` or
`external_lambda=`.

## 3. Recursion

The elliptic block is

\[
H_{c,h}^{h_i}(q)=\sum_{n\geq0}q^n H_{c,h}^{h_i,n},
\qquad H_{c,h}^{h_i,0}=1.
\]

The coefficients obey

\[
H_{c,h}^{h_i,n}
=
\sum_{1\leq rs\leq n}
{R_{r,s}^{\rm torus}(c,h_i)\over h-\Delta_{r,s}}
H_{c,\Delta_{r,s}+rs}^{h_i,n-rs}.
\]

The degenerate weights are

\[
\Delta_{r,s}
=
{Q^2-(rb+s/b)^2\over 4}.
\]

The residue is

\[
R_{r,s}^{\rm torus}
=
A_{r,s}
P_{c}^{r,s}\!\left[{}^{h_i}_{\Delta_{r,s}+rs}\right]
P_{c}^{r,s}\!\left[{}^{h_i}_{\Delta_{r,s}}\right].
\]

This is the Hadasz-Jaskolski-Suchanek recursive representation of the
Poghossian/Zamolodchikov torus one-point block.

## 4. Torus Two- and Three-Point Necklace Blocks

For `N >= 2`, internal edge `i` carries weight \(h_i\) and cylinder nome
\(q_i\). The external primary \(d_i\) joins edge \(i\) to edge \(i+1\), with
indices understood cyclically. The block convention is

\[
\left[\prod_i q_i^{h_i-c/24}\right]
\sum_{n_1,\ldots,n_N\geq0}
F_{n_1\ldots n_N}\prod_iq_i^{n_i}.
\]

`TorusNecklaceVirasoroBlock` implements the simultaneous internal-weight
recursion of Cho--Collier--Yin, arXiv:1703.09805, equation (3.20). The reduced
block has unit large-weight seed. `descendant_coefficients` restores the
non-degenerate torus character

\[
\prod_{m\geq1}\frac{1}{1-(q_1\cdots q_N)^m}.
\]

The two requested specializations are available directly:

```python
two_point = TorusTwoPointVirasoroBlock(c, h1, h2, d1, d2)
coefficients_2 = two_point.descendant_coefficients((order1, order2))
value_2 = two_point.chiral_block((q1, q2), (order1, order2))

three_point = TorusThreePointVirasoroBlock(
    c, h1, h2, h3, d1, d2, d3
)
coefficients_3 = three_point.descendant_coefficients((order1, order2, order3))
value_3 = three_point.chiral_block(
    (q1, q2, q3), (order1, order2, order3)
)
```

Passing one integer in place of the order tuple applies the same rectangular
cutoff on every edge. `reduced_coefficients` and `reduced_block` omit the torus
character, while `chiral_block(..., include_prefactor=False)` retains the full
descendant sum but omits primary propagation. As with the one-point recursion,
the implementation expects generic data away from degenerate poles; coincident
internal-weight poles should be evaluated by taking a generic limiting family.

## 5. Plumbing Use

For a self-plumbing tube with parameter \(q\), each exchanged primary
\(\mathcal O\) contributes the universal factor

```python
TorusOnePointVirasoroBlock(c, h_O, h_i).chiral_block(q, order)
```

and the full contribution is schematically

```text
C_{O i O}
* holomorphic_block(c, h_O, h_i, q)
* antiholomorphic_block(cbar, hbar_O, hbar_i, qbar)
```

The code intentionally does not assume diagonal normalization. If your later CFT
data uses a non-orthonormal two-point metric, the contraction with the inverse
two-point form should happen outside this universal block.

## 6. Checks

Run

```bash
python3 plumbing/virasoro_blocks_checks.py
python3 plumbing/torus_necklace_blocks_checks.py
```

The current checks verify:

- identity insertion reduces the elliptic block to \(H=1\);
- the full identity-insertion coefficients become partition numbers;
- the first recursion coefficient agrees with the level-one Ward-identity
  result;
- the value is invariant under \(b\mapsto b^{-1}\);
- the small-\(q\) truncated series is stable under increasing order.

The necklace checks compare every two-point coefficient through level `(3,3)`
and every three-point coefficient through `(2,2,2)` against independent direct
Virasoro descendant contractions. They also verify rectangular truncations and
the primary-propagation prefactor.
