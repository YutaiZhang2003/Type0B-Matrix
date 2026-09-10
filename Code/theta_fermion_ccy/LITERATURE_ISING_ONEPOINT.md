# Explicit Ising one-point blocks at higher genus

Read September 10, 2026. This is a source audit, not a new implementation
or numerical validation. The original APS PDFs were downloaded from the
official APS harvest endpoint and the relevant equations were inspected
visually.

## Direct genus-two source

N. Behera, R. P. Malik and R. K. Kaul, **Genus-two correlators for
critical Ising model**, *Physical Review D* **40** (1989) 1993–2003,
[DOI](https://doi.org/10.1103/PhysRevD.40.1993),
[original PDF](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevD.40.1993/fulltext).

Figure 4, p. 1996, enumerates six one-point energy blocks in a separating
pants decomposition. Equation (14), p. 1997, writes their nonchiral
contributions as squared linear combinations of square roots of odd
theta derivatives. Thus the relevant chiral weight-1/2 blocks are
explicitly present. Their Eqs. (4)–(6) retain a common scalar determinant
factor; the displayed anomaly-free ratios do not themselves fix our
plumbing-coordinate normalization. The printed Eq. (14) has defects:
the third contribution repeats a characteristic used in the first two,
and a sum in the second contribution has upper limit 1. Do not directly
transcribe that equation into production.

## Cleaner all-genus chiral formula

Jae Hoon Choi and Jae Kwan Kim, **Higher-genus characters of the Ising
model**, *Physical Review D* **41** (1990) 484–491,
[DOI](https://doi.org/10.1103/PhysRevD.41.484),
[original PDF](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevD.41.484/fulltext).

Equation (26), p. 488, gives the unnormalized **chiral** one-point block:

\[
 \langle\epsilon(z)\rangle_{\boldsymbol n}^{(\boldsymbol m)}(\Omega)
 =\frac{N(\boldsymbol n)}{2^g Z_1^{1/4}}
 \sum_{\boldsymbol b'}(-1)^{c(\boldsymbol n,\boldsymbol m)}
 \left[\sum_{i=1}^g v_i(z)\partial_i
       \theta[\boldsymbol\alpha](0\mid\Omega)\right]^{1/2}.
\]

Here their chiral energy field has weight 1/2. The characteristic is odd;
\(v_i\) are the normalized holomorphic one-forms; and
\(Z_1^{-1/4}\) is the common scalar factor in their convention. The
labels \(n_k=0,1,2\) denote \(1,\sigma,\epsilon\) on the loop and
\(m_k=0,2\) the intermediate representation. Equations (14)–(17)
determine the characteristic, sign and normalization. After Eq. (26),
the last characteristic component is modified for the insertion. Their
Eqs. (25) and (27) check the two separating factorization possibilities.
The formula uses the full period matrix, not a special one-parameter
family of genus-two surfaces.

### Genus-two specialization

For clarity define, only in this source audit,

\[
 H_{a_1a_2;b_1b_2}(z)
 =\left[\sum_{i=1}^2v_i(z)\partial_i
 \theta\!\begin{bmatrix}a_1&a_2\\b_1&b_2\end{bmatrix}
 (0\mid\Omega)\right]^{1/2},\qquad C=Z_1^{-1/4}.
\]

Their insertion is on the final loop of the chain. The six allowed
blocks specialize to:

| Loop labels \((n_1,n_2)\), bridge \(m_1\) | Chiral block |
| --- | --- |
| \((0,1),0\) | \(C[H_{0,1/2;0,1/2}+H_{0,1/2;1/2,1/2}]/(2\sqrt2)\) |
| \((2,1),0\) | \(C[H_{0,1/2;0,1/2}-H_{0,1/2;1/2,1/2}]/(2\sqrt2)\) |
| \((1,1),0\) | \(C H_{1/2,1/2;0,1/2}/2\) |
| \((1,1),2\) | \(C H_{1/2,1/2;1/2,0}/2\) |
| \((1,0),2\) | \(C[H_{1/2,0;1/2,0}+H_{1/2,0;1/2,1/2}]/(2\sqrt2)\) |
| \((1,2),2\) | \(C[H_{1/2,0;1/2,0}-H_{1/2,0;1/2,1/2}]/(2\sqrt2)\) |

This table is an algebraic specialization of Choi–Kim's characteristic
rules, with each unfixed \(b'_k\) summed once. Branches and phases must
be chosen by their factorization convention. It is not a claim that
these labels equal our theta-channel labels without a change of basis.
It also shows why the single-characteristic terms cannot simply use
the same normalization as the paired terms in the earlier printed
genus-two equation.

## Relevance to the present auxiliary block

These are irreducible Ising Virasoro conformal blocks, so the literature
does cover the mathematical space containing our auxiliary chiral
fermion one-point function. The odd theta derivative is the square of
the holomorphic zero-mode half-differential. The papers do not include
our operator \(\Theta\), spin-lift convention, or ordered Ramond ground
frames: those determine which linear combination and phases we need.

To turn this into the current four-edge plumbing series one still needs:

1. the full map from our plumbing parameters to \(\Omega\) and the
   insertion's Abel/local coordinate;
2. the common scalar determinant in precisely the sphere-sewing
   trivialization, including the external half-differential factor;
3. the pants-basis and spin-frame transport to the split theta channel;
4. removal of the prescribed primary powers and ground normalizations.

These are geometric conversion tasks. They do not require a minimal
model Verma quotient, but neither paper evaluates the blocks by CCY
central-charge recursion. A determinant/theta implementation would
therefore be a different computational route and must be identified
as such. The existence of these formulas is not an established extension
of our current CCY implementation beyond total level 5.

## Hyperelliptic sphere shortcut: central-charge distinction

A two-sheeted cover of the Ising theory is represented on the sphere by
twists of the permutation orbifold of two copies. Its diagonal Virasoro
central charge is 1, and the identity twist has chiral weight 1/32.
It is not the single-copy Ising spin field of chiral weight 1/16.
See T. Dupic, B. Estienne and Y. Ikhlef, **Entanglement entropies of
minimal models from null-vectors**, *SciPost Physics* **4** (2018) 031,
Sec. 5.2,
[DOI](https://doi.org/10.21468/SciPostPhys.4.6.031),
[primary PDF](https://inspirehep.net/files/01775d6541ea5b242491f6faa7c915cc).
Consequently a naive replacement by six ordinary Ising spin fields is
not a derived Ising \(c=1/2\) recursion for our genus-two auxiliary.
