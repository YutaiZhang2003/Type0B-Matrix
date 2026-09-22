# Ramond descendants: literature tests and two Ward corrections

Date: 2026-09-15.

## Result

Further testing found **two genuine descendant errors** in the older
R-R-NS direct Ward implementation. They are corrected. They concern
operator order and a prematurely truncated Ward sum; they cannot be
repaired by changing an overall sign or normalization.

The separate generalized NS-R-R engine used in the maintained physical
sewing passes the extended tests below. Those tests support the existing
**local radial-reflection** coefficient matrix. They do not determine the
global spin/local-coordinate transport between the interacting genus-two
decompositions.

## 1. Bra operator order

For a ket `A_1 ... A_k |w>`, the bra is
`<w| A_k^dagger ... A_1^dagger`. It is therefore the **first** lowering
operator's adjoint that meets the inserted vertex.

The two affected routines removed the last lowering operator. Mixed
descendants reveal the error already at level 2:

\[
[L_{-1},G_{-1}]=\tfrac12G_{-2}.
\]

It was fixed in
[RamondThreePointWardMatrix](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/h_recursion/ramond_descendant_blocks.py>)
and
[RRNSDescendantThreeForm](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/c_Recursion/ramond_genus2_direct.py>).
New tests check this mixed-mode identity and, independently, require the
entire identity-insertion matrix to equal the Gram matrix. Checking only
`Tr(G^-1 G)` would miss some erroneous entries.

An independent reference is the explicit torus coefficients in
[Hadasz–Jaskolski–Suchanek, Appendix C](https://arxiv.org/pdf/1207.5740).
For `b=1.1`, `h=c/24+0.73^2`, external weight `d=0.83`, their level-2
descendant coefficients give:

| HJS vertex sign | Old direct result | Corrected result | Published formula |
|---|---:|---:|---:|
| `+` | 3.426453151608114 | 3.362808099292724 | 3.362808099292725 |
| `-` | 0.163099548792197 | 0.166629026258388 | 0.166629026258388 |

The formula transcription was checked visually against both complete
Appendix C pages. In particular, the outer factor 4 in part of the minus
coefficient multiplies all terms inside its parentheses.

## 2. The Ramond Ward sum does not terminate at the mode number

In the general RRNS three-descendant routine, moving an integer Ramond
mode `G_n` through an NS insertion gives coefficients

\[
\binom{n+1/2}{j}\,V(G_{j-1/2}\xi).
\]

These binomial coefficients do not vanish when `j>n+1`. The old code cut
off there. The correct finite bound comes from the inserted NS state:
`j <= floor(level(xi)+1/2)`. For example, `binomial(3/2,3)=-1/16`, and
`G_(5/2)` can act nontrivially on a level-5/2 NS descendant.

The corrected bound is tested against the independently implemented
primary/superpartner Ward matrix through total R level 5, and against
translation covariance with descendants on all three legs. The formulas
use the R-R Ward framework of
[Suchanek, section 2](https://arxiv.org/pdf/1012.2974).

## 3. Norms and phases beyond the Ramond ground states

The two algebra implementations use different ground conventions:

\[
u=G_0w^+=a\,w^-,\qquad
a=i\beta e^{-i\pi/4},\qquad
h_R=c/24-\beta^2.
\]

Let T include the basis permutation and the factor `a` on every descendant
based on `w^-`. With basis vectors as columns,

\[
B_u=T^T B_wT.
\]

On the physical line with imaginary beta and real weights, the Hermitian
matrix in the `w^+/w^-` basis is

\[
H_w=\operatorname{diag}((-i)^{\text{ground}})\,B_w,
\qquad H_u=T^\dagger H_wT=B_u.
\]

The transpose belongs to the algebraic bilinear form; the conjugate
transpose belongs to radial Hermitian reflection. They must not be
interchanged for a complex T. These identities were tested in both total
R parities through level 5, in two generic fixtures. The Hermitian matrices
are positive, and `G_0^2=L_0-c/24` is verified exactly on all 212 tested
states. The zero-mode convention agrees with
[HJS, section 2.2](https://arxiv.org/pdf/1207.5740).

The generalized NSRR three-forms also satisfy the exact phase relation

\[
i^{\alpha+\gamma}\overline{\rho_f^\eta}
=(-i)^f(-1)^{p(x_0)}\rho_f^\eta,
\]

where alpha and gamma are the two ground labels and `p(x_0)` is the full
ket parity, including its G descendants. This was checked for 4,480
combinations, with both form parities, both eta signs, all ground labels,
NS descendant level at most 2, each R level at most 3 and total chiral
level at most 4. Another 8,960 exact checks verify the two Clifford
intertwiner identities. The NS highest state is even in this audit.

## 4. Torus and physical genus-two sewing

| Check | Result |
|---|---:|
| Independent R Gram entries through level 5 | 3,428; maximum scaled error `9.566e-15` |
| Exact zero-mode-square checks | 212 |
| Exact bilinear/Hermitian conversion matrices | 24 |
| Exact Clifford and bra-phase identities | 13,440 |
| Published torus coefficients, levels 0–2 | 18; maximum scaled error `6.263e-16` |
| Odd-parity torus relation at the same points | 18 further comparisons pass |
| Maximum phase error in the published-coefficient comparison | `2.432e-15` radians |
| Identity-insertion R character | Passes through level 6, both parities |
| New physical NSRR genus-two coefficients | 128; maximum scaled error `5.106e-15` |
| Targeted unit tests, including affected consumers | 19 pass |

The torus comparisons include asymmetric real parameters and a complex
internal/external-weight fixture. They check the coefficients themselves,
not just their absolute values. For a bottom NS insertion, HJS section 2.2
gives `F_o^(eta)=eta F_e^(eta)`, which the new tests also verify.

For the identity, one parity of the generic long chiral R module has

\[
\prod_{n\ge1}\frac{1+q^n}{1-q^n}
=1+2q+4q^2+8q^3+14q^4+24q^5+40q^6+\cdots.
\]

The complete long-module trace is twice this; its graded trace vanishes.
This is a representation trace with its Ramond doublet. It is not by itself
the single-Majorana Pfaffian or the physically restricted nonchiral trace.

The factor 4 in the paper's torus formula written using only `F_e` follows
from doing the two chiral parity sums. It is not a new two-point
normalization to insert into an already projected genus-two block.

The 128 new genus-two comparisons include R levels 2 and 3, mixed
descendants on both R edges, and separate holomorphic/antiholomorphic
levels. The four coefficient products `c_L,eta*c_R,eta-prime` are kept
independent. They continue to give the local reflected reduction

\[
Z_s^{\rm refl}=\int\frac{d^3p}{\pi^3}|P_s|^2
\left[\frac{E_LE_R}{4}|\widehat F_0^{++}|^2
+\frac{O_LO_R}{4}|\widehat F_0^{--}|^2\right],
\]

with `c_+=E/2`, `c_-=O/2` and the saved two-lift block convention. No
coefficient is fitted to the target partition function.

Every test uses descendant coefficients. The channel-dependent `q^h`
powers stay in P, outside the blocks and M. The paper's cylinder
`q^(h-c/24)` prefactor is distinguished from the saved plumbing convention.

## 5. Consequences for saved results

The changed RRNS routines feed the older direct genus-two diagnostic and
the direct R torus two-point implementation. They are separate from the
generalized NSRR Ward engine tested in the physical-state comparison.

The old level-5 direct RRNS table is superseded, and its corresponding
regression anchors have been updated transparently. For example, its
`(twice_NS,R1,R2)=(0,5,0)` coefficient changes from `4.9524423357081` to
`0.14104101264833407`. The updated numeric anchors are regression values,
not independent evidence; the literature, Gram, mode-commutator and
translation tests provide that evidence.

All 91 coefficients of that diagnostic through total level 5 were
regenerated. These are chiral diagnostic coefficients, not a new physical
partition-function integral. The maintained NSRR physical decomposition
still needs its separate global spin-transport check.

## Files and reproduction

- [Audit script](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/audit_ramond_descendant_literature.py>)
- [New regression tests](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/test_ramond_descendant_literature.py>)
- [Results, source hashes and commands](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/ramond_descendant_literature_20260915/README.md>)
