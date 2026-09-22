# A single convention for physical, auxiliary, and enlarged vertices

These are the conventions now used by the production C++ pipeline, its direct
PBW executable, and the updated graph drivers. The machine notes give the same
self-contained definitions. The manuscript and human notes were not edited.

## Modules and pairings

Use the puncture order $(\infty,1,0)$, even NS highest-weight vectors, and
bilinear BPZ pairings. Take

\[
 L_n^\dagger=L_{-n},\qquad G_r^\dagger=G_{-r},\qquad
 \psi_r^\dagger=-\psi_{-r},\qquad \{\psi_r,\psi_s\}=\delta_{r+s,0}.
\]

Keep $G_0w^\pm=i\beta e^{\mp i\pi/4}w^\mp$ and
$\psi_0u^{\mathsf a}=u^{1-\mathsf a}/\sqrt2$.
With the even ground norms set to one, these adjoints imply the Ramond
ground Gram matrices $\operatorname{diag}(1,i)$ and
$\operatorname{diag}(1,-1)$, respectively. All mixed ground pairings vanish.
The enlarged pairing is the product of these two pairings, without an
additional sign. The operators $G$ and $\psi$ anticommute; consequently
$(\psi G)^\dagger=\psi G$ with reversed mode indices, as required by the
two Virasoro embeddings.

## Ground values

The only nonzero physical NS-R-R ground components are

\[
\begin{aligned}
 \rho_0^{(\eta)}(\phi_1,w_2^+,w_3^+)&=1,&
 \rho_0^{(\eta)}(\phi_1,w_2^-,w_3^-)&=\eta,\\
 \rho_1^{(\eta)}(\phi_1,w_2^+,w_3^-)&=1,&
 \rho_1^{(\eta)}(\phi_1,w_2^-,w_3^+)&=-i\eta.
\end{aligned}
\]

The minus sign in the last entry is part of this convention. Auxiliary
ground values are

\[
 \rho_{\mathsf F}(\mathbf1,u^0,u^0)=1,\qquad
 \rho_{\mathsf F}(\mathbf1,u^1,u^1)=i.
\]

For all-NS vertices use
$\rho_0(\phi_1,\phi_2,\phi_3)=1$,
$\rho_1(\phi_1,G_{-1/2}\phi_2,\phi_3)=1$, and
$\rho_{\mathsf F}(\mathbf1,\mathbf1,\mathbf1)=1$.

## Ward identities

Choose the square root with expansions

\[
 \sqrt{z(z-1)}=z\sqrt{1-z^{-1}}\quad(z\sim\infty),\qquad
 =\sqrt{z-1}\sqrt{1+(z-1)}\quad(z\sim1),\qquad
 =i\sqrt z\sqrt{1-z}\quad(z\sim0).
\]

The last expression is the upper-lip continuation of the branch cut
between the Ramond punctures. For integers $m,n$, apply the residue
theorem to the current correlator multiplied by
$z^m(z-1)^n\sqrt{z(z-1)}$ for $G$, and by
$z^m(z-1)^n/\sqrt{z(z-1)}$ for $\psi$.
At infinity use the stated BPZ adjoints. Thus the supercurrent identity is

\[
\begin{aligned}
 &\sum_{j\ge0}(-1)^j\binom{n+\frac12}{j}
 \rho_f^{(\eta)}(G_{j-m-n-\frac12}x_1,x_2,x_3)\\
 &\quad=\sum_{j\ge0}\binom{m+\frac12}{j}
 \rho_f^{(\eta)}(x_1,G_{n+j}x_2,x_3)
 +i(-1)^{\epsilon_2+n}\sum_{j\ge0}(-1)^j\binom{n+\frac12}{j}
 \rho_f^{(\eta)}(x_1,x_2,G_{m+j}x_3),
\end{aligned}
\]

where $\epsilon_2$ is the parity of the middle physical state $x_2$.
The auxiliary identity is

\[
\begin{aligned}
 &\sum_{j\ge0}(-1)^j\binom{n-\frac12}{j}
 \rho_{\mathsf F}(\psi_{j-m-n+\frac12}a_1,a_2,a_3)\\
 &\quad=-\sum_{j\ge0}\binom{m-\frac12}{j}
 \rho_{\mathsf F}(a_1,\psi_{n+j}a_2,a_3)
 +i(-1)^{\epsilon'_2+n}\sum_{j\ge0}(-1)^j\binom{n-\frac12}{j}
 \rho_{\mathsf F}(a_1,a_2,\psi_{m+j}a_3),
\end{aligned}
\]

with $\epsilon'_2$ the middle auxiliary parity. Every sum terminates on
finite-level states. No three-point parity label occurs in the contour
sign: the sign counts the state that the current passes at $z=1$.

For example, on Ramond ground states the first identity gives

\[
 \rho_f^{(\eta)}(G_{-1/2}\phi_1,w_2^\alpha,w_3^\gamma)
 =\rho_f^{(\eta)}(\phi_1,G_0w_2^\alpha,w_3^\gamma)
 +i(-1)^{|\alpha|}\rho_f^{(\eta)}(\phi_1,w_2^\alpha,G_0w_3^\gamma).
\]

The auxiliary identity with $m=n=0$ and grounds
$(\mathbf1,u^0,u^1)$ forces
$\rho_{\mathsf F}(\mathbf1,u^1,u^1)=i\rho_{\mathsf F}(\mathbf1,u^0,u^0)$.
Thus the remaining auxiliary $i$ has a direct zero-mode Ward explanation.

The Virasoro Ward identities use the ordinary residue theorem with
$z^m(z-1)^nT(z)$, or explicitly, for either three-point form,

\[
\begin{aligned}
 &\sum_{j\ge0}(-1)^j\binom nj\rho(L_{j-m-n+1}x_1,x_2,x_3)\\
 &\quad=\sum_{j\ge0}\binom mj\rho(x_1,L_{n+j-1}x_2,x_3)
 +(-1)^n\sum_{j\ge0}(-1)^j\binom nj\rho(x_1,x_2,L_{m+j-1}x_3).
\end{aligned}
\]

For all-NS vertices the currents are single-valued: omit the square-root
multiplier and use the same BPZ and operator-ordering prescription.

## Enlarged three-point form

For the tensor basis with auxiliary factor first, define

\[
\begin{aligned}
 &\widehat\rho_f^{(\eta)}(
 \Psi_{-\mathsf A}\mathbf1\otimes\mathbf L_{-A}\phi_1,
 \Psi_{-\mathsf B}u^{\mathsf b}\otimes\mathbb L_{-B}w_2^\alpha,
 \Psi_{-\mathsf C}u^{\mathsf c}\otimes\mathbb L_{-C}w_3^\gamma)\\
 &\quad=(-1)^{(B+|\alpha|)(\mathsf C+\mathsf c)}
 \rho_{\mathsf F}(\Psi_{-\mathsf A}\mathbf1,
                  \Psi_{-\mathsf B}u^{\mathsf b},
                  \Psi_{-\mathsf C}u^{\mathsf c})
 \rho_f^{(\eta)}(\mathbf L_{-A}\phi_1,
                 \mathbb L_{-B}w_2^\alpha,
                 \mathbb L_{-C}w_3^\gamma).
\end{aligned}
\]

$A,B,C$ in signs count physical supercurrent modes; $\mathsf A,
\mathsf B,\mathsf C$ count auxiliary modes. The sign moves the middle
physical operators past the ket auxiliary operators. The bra uses the
product BPZ pairing already specified. For all-NS, the same prescription
gives $(-1)^{B\mathsf C}$. No extra phase multiplies either definition.

## Enlarged block

At every internal edge sum over the two physical PBW words and ground
labels, and over one common auxiliary word and ground label (the
auxiliary Gram matrix is diagonal). Denote the resulting tensor PBW state
at a vertex slot by $U_{v,i}$. Write $\epsilon_e,\epsilon'_e$ for its
physical and auxiliary parities, and use $\mathbf G_e$ or $\mathbb G_e$
for the physical Gram matrix according to the edge sector. Then

\[
\begin{aligned}
 \widehat{\mathbf F}_{\hat\Gamma}
 ={}&\sum_{\mathrm{PBW}}(-1)^{K(\epsilon+\epsilon')}
 \prod_e q_e^{|A_{e,1}|+|\mathsf A_e|}
          \eta_e^{\epsilon_e+\epsilon'_e}(-1)^{\epsilon'_e}\\
 &\times\prod_{e:\mathrm{NS}}\mathbf G_e^{A_{e,1}A_{e,2}}
 \prod_{e:\mathrm R}\mathbb G_e^{A_{e,1}\alpha_{e,1},A_{e,2}\alpha_{e,2}}
 \prod_v\widehat\rho_v(U_{v,1},U_{v,2},U_{v,3}).
\end{aligned}
\]

Upper Gram indices mean inverse matrices. External states are fixed,
not summed. $K$ is the Koszul sign of the fixed permutation from the
ordered edge-end pairs (followed by external states) to the ordered
vertex triples. For theta it is
$K(\epsilon)=\epsilon_1\epsilon_2+\epsilon_1\epsilon_3+
\epsilon_2\epsilon_3$. The factor $(-1)^{\epsilon'_e}$ is the inverse
auxiliary Gram entry. Ground powers of $q$ are excluded. These definitions
give ordinary inverse-Gram sewing, with no additional NS-R-R vertex weight.

The algebraic factorization into physical and auxiliary sums follows
by expanding this definition. The convolution sign for general graphs is
the polarization of $K$, together with the middle-to-ket signs just
displayed. At theta each local middle-to-ket sign occurs twice and
cancels, leaving the usual polarization of the theta sign.

## What was checked

`unified_ward.hpp` constructs the physical and auxiliary forms directly
from the two displayed residue identities and the ground values. It does
not call either production physical Ward evaluator or the production auxiliary
vertex evaluator. It reuses the SCA module-action algebra, not its Ward
solutions. `check.cpp` tests additional residue identities and the two
embedded Virasoro Ward identities using oscillator/PBW expansions.
`blocks.cpp` builds full Gram matrices and sews the new physical and
enlarged forms, then compares against fresh production CCY calculations
with Schottky vacuum restoration.

All calculations use 40 decimal digits at $b=7/5$ and
$(P_1,P_2,P_3)=(11/23,13/29,17/31)$.

| Directed check | Cases | Maximum scaled error |
|---|---:|---:|
| Physical NS-R-R residue identities | 4608 | $3.25\times10^{-41}$ |
| Auxiliary NS-R-R residue identities | 396 | $1.72\times10^{-41}$ |
| Ramond odd-ground identity | 512 | $5.74\times10^{-41}$ |
| Both embedded Virasoro Ward identities, all three slots | 1536 | $3.01\times10^{-31}$ |
| All-NS physical residue identities | 450 | $3.45\times10^{-41}$ |
| All-NS auxiliary residue identities | 180 | $0$ |
| Physical PBW versus CCY, total level 3 | 960 | $7.32\times10^{-28}$ |
| Enlarged PBW versus CCY, total level 3 | 960 | $2.20\times10^{-27}$ |
| Enlarged PBW versus convolution, total level 3 | 960 | $4.60\times10^{-41}$ |
| Odd-loop enlarged block without insertion | 480 | $2.87\times10^{-41}$ |

Counts include parity-forbidden zero components. The local residue tests
use input triples of total level at most 2 and $(m,n)\in\{-1,0,1\}^2;
the identities also evaluate descendants created by their modes.
The Virasoro tests use $n_1=0,1/2$, $n_2,n_3=1/4,3/4$, both ground
parities, both $f$, both $\eta$, both Virasoro copies, and $L_{-1},L_{-2}$
inserted in each of the three slots.
The zero mode on a branching primary is evaluated by its exact conformal
weight; the existing nonzero-mode oscillator helper is not used for $L_0$.

The block tests include both $f=0,1$ and both $\eta\eta'=\pm1$.
In the negative sector, full physical and enlarged coefficients are
compared after the established diagonal $Q\Theta\psi_0$ insertion on
edge 2. Its auxiliary action is $Q(-1)^{\mathsf b}/\sqrt2$ on the
Ramond ground label, and it commutes with the nonzero auxiliary modes.
The ordinary negative-sector enlarged block is separately checked to
vanish. This is a total-level-3 theta test, not a new arbitrary-genus
numerical verification. The general sewing rule above is defined
algebraically for any graph.

`local_results.json` and `block_results.json` retain the exact test
metadata and results; the latter also stores all new block coefficients.
The final local suite took 5.94 s and the block suite 1.42 s, excluding build.
These are validation process times, not comparative performance benchmarks.

Reproduce in this directory:

```sh
make check blocks
./check
./blocks
```

The Ward identities, ground normalizations, product pairing, and vertex
factorization must be adopted together. In particular the odd physical
ground value is $-i\eta$, and the auxiliary descendants are defined by
the Ward identity above. Keeping different descendant conventions while
using this shortened enlarged vertex would not implement these conventions.

## Graph driver migration checks

The current graph drivers use the same local forms and product sewing; there is
no physical-coefficient phase conversion before convolution or after recovery.
Fresh 40-digit total-level-1 runs include all 64 Mercedes choices and all eight
Ramond/Ramond glasses choices (zero, one, or two insertions). All comparisons pass.
Mercedes has 53,248 recovered parity components with maximum scaled error
3.29e-31; glasses has 320 with maximum 1.32e-30. The unequal split powers are
also checked. These cutoffs do not activate CCY residues. Source and saved
coefficients of the previous high-level tests are not represented as fresh tests
of the migrated conventions.

```sh
make -C ../total_level6_2026-09-16 graph_validate
../total_level6_2026-09-16/graph_validate mercedes 1
../total_level6_2026-09-16/graph_validate glasses_rr 1
```

`check_cli.py` additionally compares `bin/ramond` and `bin/pbw` in machine
precision at individual edge level 1, for both f and all four vertex-sign pairs.
It saves each executable's coefficients under `cli_results/` and its comparison
in `cli_results.json`. Both executables identify this convention in output
metadata. All production branching is computed fresh; no old branching table
is loaded.

The machine-precision CLI comparison passes all 768 components, maximum scaled
error 2.72e-13, with a total process runtime of 0.13 s.
