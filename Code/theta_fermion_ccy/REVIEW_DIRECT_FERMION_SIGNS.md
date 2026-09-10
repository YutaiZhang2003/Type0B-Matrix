# Direct fermion sewing: algebraic sign review

This review uses the mode algebra and the existing human/machine-note
conventions only. No numerical comparison or additional component test
was run.

Write an incoming Ramond Fock state as
\(\psi_{-B}u^g\), with the positive integers in \(B\) strictly
descending. Let \(b=\#B+g\pmod2\). The notes fix
\(\psi_0u^g=u^{1-g}/\sqrt2\),
\(\Theta\psi_{-B}u^g=(-1)^{\#B+g+1}\psi_{-B}u^{1-g}\),
and the state norm \((-1)^b\).

For the action coefficients of \(K=Q\Theta\psi(1)\):

- The zero mode acts diagonally, with coefficient
  \(Q(-1)^g/\sqrt2\). The two factors \((-1)^{\#B}\) cancel.
- For \(k>0\), let \(t=\#\{m\in B:m>k\}\). If \(k\notin B\),
  \(\psi_{-k}\) inserts it; if \(k\in B\), \(\psi_k\) removes it.
  In either case the fermion coefficient is \((-1)^t\), and the
  subsequent \(\Theta\) coefficient is \((-1)^b\). Thus the full
  coefficient is \(Q(-1)^{t+b}\), with the occupied set toggled and
  \(g\mapsto1-g\).

The full action preserves \(b\). Opposite occupancy conditions give zero.
The field is evaluated at one, so its nonzero modes carry no additional
coordinate factor. The mode connecting incoming level \(\ell_L\) to
outgoing level \(\ell_R\) has index \(\ell_L-\ell_R\).

The incoming state is the ket at the middle sphere's zero puncture; the
outgoing state is the bra at infinity. Therefore use \(K_{R,L}\), the
coefficient of the outgoing state in \(K|L\rangle\). This is an
**action coefficient**, not a raw BPZ matrix element. The latter is
\((-1)^bK_{R,L}\).

Let the auxiliary NS and third-edge parities be \(a,c\). Both outer
vertices vanish unless \(a+b+c=0\pmod2\). The four inverse Fock norms
multiply to \((-1)^{a+2b+c}\); multiplying the middle BPZ matrix element
leaves \((-1)^{a+b+c}=1\). Hence the sewn coefficient is

\[
 (-1)^a(-1)^{ab+ac+bc}
 \rho(\mathrm{NS},L,3)\rho(\mathrm{NS},R,3)K_{R,L}.
\]

The linear \((-1)^a\) is the NS contour sign in the human definition;
the quadratic sign is the existing `theta_sign(index)`, with
`index = a + 2*b + 4*c`. There is no second independent split-edge
parity, and no additional norm sign should be inserted.

The series key is
`(2*NS_level, left_level, right_level, 2*third_level)`.
With \(\hat q_{2,1}=u\sqrt{q_2}\) and
\(\hat q_{2,2}=u^{-1}\sqrt{q_2}\), its physical degree is half the
sum of this key. The cutoff `sum(key) <= 2*N` is therefore correct.
Off-diagonal split levels must remain present.

As an algebraic normalization consequence, the ground terms are
\(Q/\sqrt2\) in parity slot zero and \(-Q/\sqrt2\) in parity slot six:
\(\rho(1,u^1,u^1)^2=i^2=-1\), while the theta sign at slot six is
also negative. Thus the constant is
\(Q(1-\eta_2\eta_3)/\sqrt2\), as required.

The same conventions also give \(\Theta^\dagger=\Theta\) and
\((\Theta\psi_m)^\dagger=\Theta\psi_{-m}\). Since the two states
connected by the insertion have the same norm, the real mode action
coefficients are symmetric under exchanging the incoming and outgoing
states. This is an algebraic orientation check; it does not license
collapsing the two independent propagation powers.

Implementation inspection will be appended once `direct_fermion.py` is
available.

## Static implementation audit completed

Inspected `direct_fermion.py` after the level-ten benchmark. No
implementation edits or numerical runs were made. No mathematical or
cutoff discrepancy was found in the inspected code.

The rational basis is \(e_0=u^0,e_1=\sqrt2u^1\). Its zero-mode
coefficients are \(1/2\) and \(1\), exactly as `_act` implements. The
stored form satisfies

\[
 \rho(\mathrm{NS},B u^g,C u^h)
 =i^b\,r(\mathrm{NS},B e_g,C e_h)/2^{(g+h)/2}.
\]

The two outer phases therefore multiply to \((-1)^b\), explaining
`common = (-1)**(a+b) * theta`. For the diagonal term the conversion
denominator is \(2^{g+h}\). For a nonzero-mode pair, the two second-edge
ground labels sum to one, so its \(\sqrt2\) denominator cancels the
\(\sqrt2\) in \(K/(Q/\sqrt2)\), leaving \(2^h\). The implemented
`zero_value` and `value` match these expressions.

The contour Ward coefficients follow from the three test sections

\[
 z^{k-1/2}(z-1)^{-1/2},\qquad
 z^{-1/2}(z-1)^{-n-1/2},\qquad
 z^{-n-1/2}(z-1)^{-1/2},
\]

respectively. Expanding at infinity, one and zero gives the code's
phases \((i,1,-i)\), mode indices and half-binomial coefficients.
Stripping the second-slot phase produces precisely the exponent in
`add`. Each recursion removes the largest mode in its first nonempty
slot; every non-target contribution lowers the total nonzero-mode
level, so the recurrence is triangular.

Each dynamic upper bound contains all possible creation/zero-mode
terms and every annihilation mode no larger than the largest occupied
mode on its leg. There is no inherited fixed expansion cutoff.

The sewing loop enumerates each pair by adding its unique toggled mode
to the lower-level state. Reverse removal has the same coefficient and
outer product, so accumulating both split exponent orders is correct.
A diagonal state needs `2*lower_level <= budget`; an off-diagonal pair
needs `2*lower_level + mode <= budget`. These are exactly the two
implemented bounds. Levels omitted beyond them cannot contribute.
Skipping a pair whose lower-state three-point form vanishes is valid
for both orientations because both products contain that same form.
The third ground label determined by `c = a ^ b` exhausts all
parity-allowed states, rather than imposing an additional restriction.
