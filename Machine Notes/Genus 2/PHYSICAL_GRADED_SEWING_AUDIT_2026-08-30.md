# Physical graded sewing versus the auxiliary SCA+F pairing

This audit implements the user's clarification: the physical free-fermion
partition function must not depend on the auxiliary metric convention used
to facilitate the double-Virasoro decomposition. Here “metric convention”
means the state-space pairing, not the surface Weyl frame. The physical
free factor and the Liouville numerator must still use the same local frame.

## Two distinct pairings in the Human Note

`Human Notes/SCblock.tex`, equation labelled `eq: Gram_Factorialization`,
uses the physical graded holomorphic/antiholomorphic BPZ pairing. On
normalized product states it is

\[
\mathcal G_{a\tilde a;b\tilde b}
=(-1)^{p_b\tilde p_{\tilde a}}
B_{ab}\widetilde B_{\tilde a\tilde b}.
\]

The section “Ramond blocks from double Virasoro” explicitly defines a
different auxiliary SCA+F tensor pairing without the exchange sign, and
sets the auxiliary odd Ramond ground norm to minus one. Its branching
coefficients and star-product formula must retain those conventions.
That auxiliary definition is not permission to omit the exchange sign
from the physical nonchiral theory. It is also not justified to identify
the two complete prescriptions by a rephasing of one ground state alone.

## Grading reproduces the note's decomposition factor

For the three holomorphic parities \(p\) and antiholomorphic parities
\(\tilde p\), let \(K(p)=\sum_{i<j}p_i p_j\) modulo two. The identity

\[
K(p+\tilde p)+p\cdot\tilde p
=K(p)+K(\tilde p)
+\Bigl(\sum_i p_i\Bigr)\Bigl(\sum_i\tilde p_i\Bigr)\pmod2
\]

combines the full pants-orientation sign with the three inverse graded
pairings. For equal total parity \(f\), the residual factor is
\((-1)^f\). For all-NS even primaries this is precisely the explicit
decomposition sign in the note; it must coexist with the quadratic signs
inside each chiral block. The odd three-point coefficient
\(C^{(1)}_{\rm note}=i\widetilde C\) remains separate:
\((-1)(i\widetilde C)^2=\widetilde C^2\).

A one-term example detects the error immediately. Put the only nonzero
chiral and antichiral pants component at parity `(1,0,0)`, with unit
chiral pairings. The first full edge has state `(odd,odd)`, whose graded
norm is minus one. The full sewing contribution is therefore minus one;
an ungraded Kronecker metric gives plus one.

## What is and is not established

- `Code/genus_2/graded_sewing_audit.py` is a finite-dimensional bilinear
  contraction audit, not a physical NSRR partition evaluator. It requires
  caller-supplied physical pants tensors, pairings, and state restrictions.
- Eight tests check the tensor metric, all 64 parity identities, explicit
  full-versus-factorized contractions, the odd-sector counterexample,
  parity-preserving complex basis changes, odd rephasing, and coefficient
  phases. Changing the Gram matrix alone fails the covariance check.
- The independent free evaluator remains
  `Code/genus_2/fixed_spin_free_plumbing.py`. Its bosonized charge sewing
  does not take the SCA+F auxiliary metric or its auxiliary fermion block
  as input. None of its numerical results was changed by this audit.
- The earlier exploratory Hermitian chiral contraction, including the
  apparent disappearance of mixed-sign components under that contraction,
  is **not** a derivation of the physical Ramond sewing projection. Do not
  use it to discard mixed-sign Human-Note blocks, undo quadratic signs in
  the physical all-NS blocks, or assign physical spin lifts.
- No formula such as half the diagonal diagnostic norm, or an unproved
  sum of absolute squares of the Hermitian diagnostic, is promoted to
  physical \(Z_{\rm NSRR}\) or \(\mathcal Q_{\rm NSRR}\).
- The physical Ramond ground-state restriction and its nonchiral
  three-point assembly still need to be explicitly implemented and
  checked with this grading retained. This is an implementation task,
  not a request to choose a new convention in the Human Note.

Validation: 8 new audit tests + 13 independent fixed-spin free tests +
1 protected-kernel manifest test pass (22 tests). All eight protected
source hashes match the existing manifest. No Human Note, checked kernel,
partition evaluator, or archived numerical result was changed.
