# Direct low-state check of the Ramond branching recursion

## Scope

The check uses only the definitions and recursion in `SCblock.tex`.  The
generic numerical point is

\[
b=\frac75,\qquad P_1=\frac{11}{23},\qquad
P_2=\frac{13}{29},\qquad P_3=\frac{17}{31}.
\]

The states are constructed for every label in

\[
n_{\rm NS}=0,\pm\frac12,\pm1,\pm\frac32,
\qquad
n_{\rm R}=\pm\frac14,\pm\frac34,\pm\frac54,
\]

and for both Ramond parities \(\alpha=0,1\) on both Ramond legs.  This is 31
state checks in total.

## What is computed directly

1.  The code constructs \(v_n\) and \(v_n^\alpha\) from the corresponding
    \(\chi\)-strings.  For a negative label it constructs the reflected
    free-field realization and transports it to the fixed SCA module.

2.  At each physical level, the code independently generates the SCA PBW
    basis

    \[
    L_{-A}G_{-C}\phi,qquad L_{-A}G_{-C}w^\pm,
    \]

    expands every PBW vector in free-field oscillators, and inverts that
    finite change-of-basis matrix.  Thus the three-point calculation uses
    explicit SCA descendant states rather than treating a branching
    coefficient as an unknown in a Ward lattice.

3.  BPZ norms are contracted directly.  The auxiliary rule
    \(\psi_r^\dagger=-\psi_{-r}\), the physical rules
    \(L_n^\dagger=L_{-n}\), \(G_r^\dagger=G_{-r}\), and the Ramond ground
    metrics are applied mode by mode.

4.  The auxiliary free-fermion three-point function and the physical
    NS--R--R SCA three-point function are reduced separately by their mode
    Ward identities.  Their product is multiplied by the tensor sign written
    in the main notes,

    \[
    (-1)^{A\mathsf A+(B+|\alpha|)(\mathsf C+\mathsf c)}.
    \]

    The numerical three-point checks use the coupled canonical forms
    \(\eta=(-1)^f\), so both \(f=0\) and \(f=1\) occur.  The branching
    recursion itself is independent of \(\eta\).  The second coupled pair is
    not manufactured by reusing the same scalar supercurrent reducer; doing
    that gives an invalid check on reflected components.

5.  For each test triple, the code computes

    \[
    \mathbb B_f^{(\eta)}(n_1,n_2,n_3,\alpha_2,\alpha_3)
    \]

    directly from the three explicit states.  It separately computes the
    three child coefficients, substitutes those direct child values into one
    step of the recursion, and compares the two answers.  No branching value
    is obtained by solving a global Ward system.

## Negative labels

For \(|n_1|\geq1\), the NS action is solved with the neighbor

\[
n_1-\operatorname{sgn}(n_1).
\]

For \(|n|\geq\frac34\), the Ramond \(L_{-1}\) action contains the two
same-branch level-one descendants and descendants of the neighbor

\[
n-\operatorname{sgn}(n).
\]

Consequently, the same first Ward identity recurses toward the boundary in
both the positive and negative chambers.  The direct BPZ norm used by this
recursion is fixed by reflection:

\[
\|v_n(P)\|^2=\|v_{-n}(-P)\|^2,qquad
\|v_n^\alpha(P)\|^2=\|v_{-n}^\alpha(-P)\|^2.
\]

Directly substituting \(n<0\) into the positive-label NS formula would differ
from the reflected state normalization by \(2^{4|n|}\), so the code does not
make that substitution.

As a separate code-path smoke test, the full negative recursion expands
\((-3/2,-5/4,-5/4)\) into the three boundary triples
\((-1/2,-5/4,-5/4)\), \((-3/2,-1/4,-5/4)\), and
\((-3/2,-5/4,-1/4)\), for all four choices of
\((\alpha_2,\alpha_3)\).

## Numerical outcome

| \((n_1,n_2,n_3)\) | four parity checks | runtime (s) | largest relative difference |
|---|---:|---:|---:|
| \((1,3/4,3/4)\) | 4 | 0.0124 | \(5.05\times10^{-14}\) |
| \((3/2,5/4,3/4)\) | 4 | 0.1215 | \(1.21\times10^{-13}\) |
| \((3/2,3/4,5/4)\) | 4 | 0.0992 | \(1.77\times10^{-13}\) |
| \((-1,-3/4,-3/4)\) | 4 | 0.0027 | \(3.45\times10^{-14}\) |
| \((-3/2,-5/4,-3/4)\) | 4 | 0.0353 | \(1.00\times10^{-11}\) |
| \((-3/2,-3/4,-5/4)\) | 4 | 0.0337 | \(2.66\times10^{-11}\) |
| \((-1,3/4,3/4)\) | 4 | 0.0028 | \(8.58\times10^{-15}\) |

The summary diagnostics are

\[
\begin{aligned}
\max\frac{|\|v\|^2_{\rm direct}-\|v\|^2_{\rm formula}|}
{\max(1,|\|v\|^2_{\rm formula}|)}&=6.18\times10^{-14},\\
\max\|L_{m>0}^{(i)}v\|_{\rm relative}&=4.75\times10^{-15},\\
\max|h^{(i)}_{\rm direct}-h^{(i)}_{\rm formula}|&=1.56\times10^{-14},\\
\max(\text{action-fit residual})&=9.22\times10^{-13},\\
\max(\text{direct/recursion difference})&=2.67\times10^{-11}.
\end{aligned}
\]

The action decompositions take 0.097 seconds, the direct three-point
comparisons take 0.308 seconds, and the complete run takes 0.468 seconds.
Every state expansion, direct norm, child coefficient, branching coefficient,
and per-case timing is stored in `direct_state_results.json`.
