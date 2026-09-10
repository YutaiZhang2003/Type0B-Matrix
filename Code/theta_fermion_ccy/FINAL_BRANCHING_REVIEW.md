# Final branching and convolution review

Reviewed the selected direct-auxiliary algorithm in
`Machine Notes/theta_fermion_ramond_recovery.tex`, its two included
auxiliary subsections, `outer_branching.py`, `middle_branching.py`, and
the sector division in `series_algebra.py`. This is an adversarial algebra
and source review. No numerical calculation or production-source change
was performed. The level-ten production run is separate from this review.

## Conclusion

No mathematical defect was found in the selected theta-channel construction
at generic parameters with \(Q\ne0\). In particular, the reflected outer
action optimization preserves the original primary normalization; the raw
middle recurrence has the correct BPZ orientations and anchors; and the
raw four-edge contraction agrees with the displayed normalized branching
formula when its explicitly required compatible norm roots are used.
The direct Fock denominator removes the previously unresolved need to
compute an irreducible higher-genus Ising block by a Verma-module regulator.
It does not resolve that separate regulator problem, and the selected
algorithm does not require it to do so.

## Normalization and recurrence checks

The negative-label optimization uses a common module intertwiner, not an
independent normalization of each primary. In the free-field code, comparing
`(P, realization=+1)` with `(-P, realization=-1)` leaves the product
`realization * P` invariant. Multiplication by minus one on the physical
odd ground component changes precisely the physical fermion zero-mode sign.
This intertwines the displayed physical \(L,G\) actions and the two embedded
Virasoro actions. It also sends the entire defining chi string, including
the optional opposite-chart zero mode, to the reflected string without a
label-dependent scalar. Composing with the old native-to-target conversion
gives the Human-Note reflection with ground action
\(\operatorname{diag}(1,-1)\). Since the same intertwiner acts on all
branches and descendants, the action coefficients are unchanged and all
output labels are simply reversed. The current outer override implements
exactly this rule, using independent providers for the two momenta.

The physical Ward identity is
\([L_m,\Theta Q\psi(1)]=0\). Therefore the first recurrence uses
\(L_1\) on the outgoing endpoint and \(L_{-1}\) on the incoming endpoint;
the second exchanges these roles. It is essential that these are the
physical generators. The insertion does not commute with either embedded
Virasoro algebra. The code and notes maintain this distinction.

The other-primary term in the smaller endpoint's \(L_{-1}\) action is
separated from the larger endpoint by \(3/2\), including the
\(1/4\to-3/4\) boundary. It vanishes by the external degenerate fusion
rule. It is not omitted on the basis of a numerical estimate. The surviving
scalar pivot is the appropriate sum of the two same-primary coefficients
times \(h+h_\psi-h'\), or its outgoing counterpart. Negative labels and
the \(3/4\to-1/4\) crossing reduce to the two oriented ground pairs.

The displayed ground primaries have norms \(2\) and \(-1\). Their ordered
zero-mode actions produce the positive raw anchors \(\sqrt2Q\) and
\(Q/\sqrt2\). The minus sign in the parity-one action is canceled by
its negative BPZ norm. These agree with `MiddleBranching.raw` by direct
algebra from the displayed definitions.

For normalized coefficients, if
\(\Theta(v_n^\alpha/\|v_n^\alpha\|)
=-i v_n^{1-\alpha}/\|v_n^{1-\alpha}\|\), anticommuting \(\Theta\)
through the fermion gives the displayed factor
\(i\|v_{1/2}\|\mathbb B\). The norm identity
\(\|v_{1/2}(Q/2)\|^2=-Q^2\) is consistent with the auxiliary BPZ
convention. Arbitrary independent principal roots need not satisfy the
compatibility relation; the production contraction correctly uses raw
matrix elements and squared norms instead. Retain this distinction.

## Convolution and direct denominator

The identity SCA field on the middle sphere contracts the two physical
propagators by \(G^{-1}GG^{-1}=G^{-1}\), enforcing equal physical levels
on the split halves. Their propagation factors multiply to
\(q_2^{|B|}\). Auxiliary levels remain independent because the complete
field \(\psi(1)\), rather than only its zero mode, is inserted.

The contour phases are needed before comparing quadratic parity signs.
Their product cancels the physical part of the enlarged NS contour sign;
the remaining auxiliary sign is exactly the one in the direct Fock sum.
Polarizing the theta quadratic sign then gives the displayed symmetric
convolution cocycle. Subdividing the edge introduces no fourth parity bit,
because the ordered operator \(\Theta\psi\) is even.

The direct denominator's action coefficients, its outgoing BPZ norm, and
the four inverse Fock norms are consistently distinguished. The inverse
norm product times the outgoing norm is one on allowed outer vertices.
The rationalized diagonal and toggle summands then follow from the stated
ground rescaling and phase stripping. The auxiliary Ising decomposition
shows that the entire denominator lies in the minus ideal, not merely its
constant term. Its constant acts there by \(\sqrt2Q\), which makes
triangular division unique. The implementation checks the ideal residual
without replacing the answer by its projection.

Restoring two copies of the universal large-central-charge vacuum factor
after division is correct for a reduced double-Virasoro numerator divided
by this full direct denominator. The factor is independent of parity and
depends on the split variables only through their product. It therefore
commutes with the convolution. The notes now distinguish this case from
the older reduced-Ising denominator, which required one restored copy.

## Remaining presentation and scope refinements

1. State \(Q\ne0\) alongside \(b\ne0\) and \(b^2\ne1\) at the first
   parameter assumptions. The inverse subsection already states this
   restriction correctly, but at \(b=\pm i\) the chosen external state
   \(Q\psi\) is zero even though the embedding restrictions hold.
   Moving the restriction earlier avoids an apparent unrestricted claim.
   Continuing a differently normalized insertion would be a separate
   convention, not something to perform silently in the displayed formula.

2. Write the spin-lift composition as the adopted transported-frame
   convention. Eliminating the sphere proves the product of plumbing
   parameters, but does not itself choose a square root of the derivative
   of the inversion. A suitable sentence is: “We choose the transported
   Ramond spin frame so that the composed lift is
   \(\eta_2=\hat\eta_{2,1}\hat\eta_{2,2}\).”

3. The direct subsection accurately states an arbitrary-genus construction
   of the auxiliary factor. It does not by itself establish the complete
   inverse algorithm for every graph. Such a claim additionally requires
   the chosen graph's quadratic sewing sign, its polarized convolution
   cocycle, the physical sector identities, and compatible insertion and
   projector choices for every independent Ramond component. The particular
   theta projector \((1-\eta_2\eta_3)/2\) and the single insertion must
   not be asserted to suffice unchanged for all channels. This distinction
   was discussed with the independent punctured-CCY reviewer.

4. The explanation of the middle fusion rule could be made easier to
   reconstruct by identifying the two external representations as the
   respective level-two degenerate representations before imposing their
   compatible half-step changes in the common label. This is a clarity
   improvement, not a detected error in the rule or implementation.

5. The direct subsection's phrase about a nonzero mode connecting one pair
   should explicitly be conditional on an incoming state: each mode maps
   that state to at most one outgoing state. This wording correction was
   sent to the owner of that subsection; it does not affect the formula.

6. Keep the benchmark scope explicit: the requested independent comparisons
   concern the stated \(f=p_1=0\), \((\eta,\eta')=(+,-)\) sample through
   total level five. The other parity/form transports have an algebraic
   derivation but no additional numerical coverage is being claimed. The
   full level-ten run must be reported from its actual saved result and
   timing, not inferred from the auxiliary-only benchmark.

The first two items improve the order of assumptions and make an existing
coordinate convention explicit. The third is a substantive scope boundary
if the final prose claims arbitrary-genus recovery, rather than only the
correctly described arbitrary-genus auxiliary sewing construction.
