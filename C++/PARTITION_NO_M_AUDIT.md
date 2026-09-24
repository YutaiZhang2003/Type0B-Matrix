# SCblock factorization and the open fixed-spin dictionary

The all-NS theta decomposition in `Human Notes/SCblock.tex` is an exact
factorization of the **graded state sum stated there**. For even
primaries, the note's full reordering sign and the three mixed
inverse-Gram signs obey, in its edge notation,

`K(A+tilde A) + sum_i A_i tilde A_i = K(A) + K(tilde A)
 + (sum_i A_i)(sum_i tilde A_i) (mod 2)`.

The three-point selection rule makes the two parity sums equal. Their
product is then the common parity `a`. The two `K` factors belong inside
the chiral blocks, leaving the overall `(-1)^a`. With general primary
parities, replace each `A_i` by `A_i+p_i` and likewise on the
antiholomorphic side; the overall factor becomes
`(-1)^(a+p_1+p_2+p_3)`, exactly as written in SCblock.
The identity was checked exhaustively for all 2,048 allowed assignments
of edge and primary parities. No extra matrix is needed for this formal
factorization.

What has **not** been established by that algebra is the identification
of its formal `q_e,eta_e` with a fixed geometric spin structure. The
source and target plumbing parameters have the same marked periods to
`6.3524e-11` after the saved symplectic map, and the map transports
`[11|00]` to `[00|00]` and `[11|11]` to `[00|10]`. This checks the
surfaces and the marked characteristics, but it does not by itself say
which local spin lift the paper's `eta_e` denotes in the bosonization
calculation.

Under the **currently assumed linear** map from `eta_e` to charge-lattice
theta signs at fixed plumbing `q_e`, the first three two-fermion terms of
the literal all-NS block have signs `-eta_1 eta_2`, `-eta_1 eta_3`, and
`-eta_2 eta_3`. Their product is `-1`, while that particular linear
theta-sign assignment has product `+1`. This rules out repairing the
assumed map by flipping only `eta_e`; it does not refute SCblock's
decomposition. A direct Majorana Wick calculation through total level 8
has 228 nonzero monomials and differs from bosonization by 5.53–31.86%
under the current map. These numbers test the map, including its
plumbing and BPZ spin lifts.

For an odd field, geometric inversion `w=q/z` carries a branch of
`sqrt(dw/dz)`. The Human Note defines the chiral Gram by an algebraic
adjoint but does not give a full local-coordinate and spin-lift dictionary
for this genus-two comparison. [Tuite–Zuevsky](https://arxiv.org/pdf/1007.5203)
make that branch explicit in their fermionic sewing construction.
The earlier coefficientwise BPZ adjustment produced near-one numerical
ratios, but it changed the literal chiral block. It is therefore not a
derivation that the Human Note's unchanged block has those ratios. It
also does not prove that the paper's block definition needs alteration:
the geometric dictionary may instead account for the difference.

The current C++ driver retains exactly the Human Note's `F` and labels
its diagonal output `Z_diagonal_trial` until that dictionary is fixed.
The saved 40-digit source/target comparison at total levels 5 and 8 gives
diagnostic ratios `0.987817` and `0.931222` under the assumed map.
The independent full Ramond source contraction in the paper's `w^+,w^-`
basis is also outstanding. Neither diagnostic ratio should be presented
as a physical crossing test. The draft has not been edited.
