# NS descendants in the displayed literature basis

Date: 2026-09-15.

## Result and precise scope

The numerical and exact NS modules now share the descendant-vector ordering
displayed in the literature. In particular,

\[
\begin{aligned}
\mathcal B_{3/2}&=(G_{-1/2}L_{-1},G_{-3/2})|h\rangle,\\
\mathcal B_2&=(L_{-1}^2,L_{-2},G_{-1/2}G_{-3/2})|h\rangle.
\end{aligned}
\]

The old NS code already used the same ordering of generators **inside each
word**. My earlier suggestion that its two-supercurrent word was reversed
was incorrect. This code change orders the **vectors in the basis list**,
and therefore the rows and columns of matrices and the indices of tensors.
It does not rephase chiral vertices or mix the four spin-lift blocks.

The generator convention is Hadasz--Jaskolski--Suchanek's. The displayed
matrix order is Belavin--Geiko's, with `c_BG=2*c/3` relative to the ordinary
central charge in our code. Beyond the displayed examples, our explicit
extension orders by the number of G modes and then by their positive mode
indices. [HJS, sections 3–4 and Appendix A](https://arxiv.org/pdf/hep-th/0611266),
[Belavin–Geiko, (2.5)–(2.8) and Appendix B](https://arxiv.org/pdf/1806.09563).

## Code changes

- [ns_pbw_basis.py](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/c_Recursion/ns_pbw_basis.py>)
  supplies the common basis at every finite level.
- [mixed_ns_ramond_descendant_blocks.py](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/c_Recursion/mixed_ns_ramond_descendant_blocks.py>)
  and
  [ns_genus2_symbolic_low_order.py](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/c_Recursion/ns_genus2_symbolic_low_order.py>)
  consume it. Their Gram matrices, mode-action coordinates and three-point
  tensors are rebuilt in that same order. This also updates the NS edge in
  the mixed NSNSRR direct oracle.
- [ns_genus_c_recursion_checks.py](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/c_Recursion/ns_genus_c_recursion_checks.py>)
  uses the displayed level-3/2 Gram order and permutes both test vectors with it.
- [test_ns_pbw_basis.py](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/c_Recursion/test_ns_pbw_basis.py>)
  checks the explicit literature Gram entries in both algebra implementations.

No opaque saved descendant-coordinate arrays are loaded by these modules.
Their basis/Gram/action caches are generated in memory. Previously saved
scalar block coefficients therefore require no conversion. Historical
source hashes in previous audits remain historical; this audit has its own
before/after source hashes.

## Why the partition function is unchanged

At each level write the new vectors as `e'=S e`, where S is the real
permutation recorded in `summary.json`. Then

\[
G'=S G S^T,\qquad r_L'=S r_L,\qquad r_R'=S r_R,
\]

and hence

\[
(r_L')^T(G')^{-1}r_R'=r_L^T G^{-1}r_R.
\]

For a three-legged vertex apply `S_1 tensor S_2 tensor S_3` to the vertex
tensor and the matching inverse Gram matrix on each edge. Every S cancels
in the sewn contraction. The same argument applies to the Hermitian
reflection pairing because S is real. No change of norm or phase occurs.

S acts within a fixed level and parity, so it commutes with both descendant
propagation and the spin-lift character. In the user's convention,

\[
Z=\sum_{A,B}M_{AB}\,P_A\widetilde P_B\,
F_A\widetilde F_B,\qquad
P_A=\exp\!\left(\sum_i h_{i,A}\operatorname{Log}q_i\right),
\]

the descendant block F itself is invariant. Thus the supplied three-point
products in M, all channel-dependent primary weights, and the chosen
logarithm branches retain their roles. This permutation introduces no
factor of four, no factor of i and no change to a single-Majorana square-root
phase. The argument holds before the momentum integral and at every level;
the numerical checks below exercise finite truncations.

## Independent descendant checks

The accompanying
[audit](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/audit_ns_descendant_literature.py>)
constructs a component-ordered vertex directly from HJS (4.11), with
reflection (4.14)–(4.15). It does not call the Human-Note vertex recurrence.
It shares the NS algebra module, which is separately checked against the
printed Gram matrices. For even NS highest states it verifies

\[
\rho_{\rm HN}^{a}(x_\infty,x_1,x_0)
=(-1)^{a\,p(x_0)}
\rho_{\rm component}(x_\infty,x_1,x_0).
\]

All **141 exact vertex comparisons** pass, including non-global and
multi-supercurrent descendants: each edge level at most 2, total level at
most 3. This fixed-parity/component-vertex dictionary is distinct from
the vector permutation above and is kept explicit.

There is also a generic torus one-point check through level 2. With
external weight d, it reproduces the literature's Gram and inserted
matrices (15 entries each). Its first descendant coefficients are

\[
F_{1/2}=\frac{2h-d}{2h},\qquad
F_1=\frac{2h+d(d-1)}{2h}.
\]

Setting `d=0` gives the identity trace
`1 + q^(1/2) + q + 2q^(3/2) + 3q^2 + ...`. The separate generic genus-one
character regression passes through level 6, including coefficient 28 at
that level. These statements concern descendant-only traces; primary
propagation is outside them.

## Before/after sewing results

`before.json` was captured from the actual old modules **before any basis
edit**, and is never overwritten by the audit. Two generic fixtures are
used: `(c,h)=(81/5,(7/10,11/10,13/10))` and
`(149/4,(73/100,91/100,117/100))`. Theta and glasses coefficients are checked
through total physical descendant level 3. Complex plumbing parameters
test block phase as well as norm; both three-form sectors and all four
canonical lifts are included.

| Check | Result |
|---|---:|
| Exact theta coefficients before/after | 84 identical |
| Numerical theta/glasses coefficients | 336; maximum scaled error `2.634e-15` |
| Truncated genus-two evaluations | 16 |
| Maximum relative complex block error | `4.454e-17` |
| Maximum relative block-norm error | `0` in double precision |
| Maximum block-phase error | `5.180e-17` radians |
| Maximum relative partition error | `0` in double precision |
| External primary factors | Identical |
| Direct theta versus c-recursion | 84 coefficients; maximum error `1.777e-15` |
| Targeted unit tests | 12 passed |

The partition evaluations use two independent nonzero even/odd
three-point products as algebraic fixtures. They test equality under the
basis change, including the existing reflected theta pairing; they are
not a new physical momentum integration or a comparison of different
decomposition channels at a common moduli point. The global interacting
NSRR/all-NS spin-transport question is not settled by a basis permutation.

One old diagnostic initially failed: its positive literature odd global
seed was compared directly with the signed Human-Note seed. The same
failure was reproduced from the pre-change source. The checker now inserts
the explicit `(-1)^a` dictionary at that comparison boundary. The production
seed is unchanged; the global/vacuum diagnostic now passes.

## Reproduce and inspect

Results and commands are in
[the data README](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/ns_literature_basis_switch_20260915/README.md>).
The before/after snapshots include source hashes, basis permutations,
individual complex coefficients, separate primary factors and both
sector contributions. CSV files expose coefficient, norm and phase errors.
