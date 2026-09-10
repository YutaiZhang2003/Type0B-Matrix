# Adversarial mathematical review of the machine notes

Reviewed `Machine Notes/theta_fermion_ramond_recovery.tex`, its Ising input,
and the corresponding CCY, middle, outer, and parity implementations. This
review used algebra and source inspection, not an additional numerical
comparison suite. The parent reports that the two authorized level-five
end-to-end checks now pass; that evidence does not settle level ten.

## Finding requiring correction: scope of the action-level bound

The final paragraph of “Central-charge recursion and truncation” states
that the relevant reusable action coefficients require descendant levels
at most six at physical level ten. This is correct for the **new middle
recurrence**: its largest endpoint is `9/4`, so the lowering `L_1` action
has level `4*(9/4)-3=6`; the opposite endpoint is `7/4`, so its `L_-1`
action has other-primary level `4*(7/4)-1=6`.

It is not a bound on the present entire implementation. The outer
`BranchingGrid.build_actions` calls `solve_ramond_lminus` for every Ramond
label in the grid, including `+/-9/4`. Those action coefficients have
other-primary descendant level `4*(9/4)-1=8`. The notes should explicitly
restrict the six-level claim to the middle recurrence and distinguish the
current outer implementation's level-eight action solve.

## Finding requiring correction: odd-form eta transport

The archived branching oracle and the independent human-convention PBW
oracle use opposite eta labels for odd form parity. This is visible from
their displayed Ward formulas, without a new numerical test. Set intrinsic
NS parity to zero and form parity to one, and consider
`rho(G_-1/2 phi,w2+,w3+)`. The human generalized Ward identity gives

\[
i e^{i\pi/4}(\eta_{\rm physical}\beta_2-\beta_3).
\]

The native branching `PhysicalThreePoint` has Ramond ground basis
`f1=exp(3 pi i/4) w-`, `G0 f0=-i beta f1`. Its first contour equation
is `-i rho(G phi,...)+rho(phi,G0..., ...)+i rho(phi,...,G0...)=0`.
Its ground table therefore gives

\[
e^{i\pi/4}(\eta_{\rm native}\beta_2+\beta_3).
\]

Taking `eta_native=-eta_physical` turns this into `i` times the human
expression, the known descendant coordinate phase compensated in the
enlarged sewing convention. Without the label transport the relative
coefficient of the two independent momenta is wrong; an overall phase
cannot repair it. This reproduces the archived prototype's explicit
`eta_native=(-1)^f eta_physical` map from source-level algebra.

`OuterBranching.prepare` should retain physical eta keys but call the
native `grid.solve` with `(-1)^self.f*eta`. The completed f=0 validation
does not probe this distinction and must not be presented as an f=1 test.

## Proof gaps that should be filled in the writeup

1. **The spin and BPZ sewing sign is asserted too quickly.** Evenness of
   the combined insertion is necessary, but is not by itself the complete
   derivation of the convolution. State that the middle physical identity
   has matrix `G` between the two inverse metrics, so
   `G^{-1} G G^{-1}=G^{-1}` within each parity subspace. The auxiliary
   combined insertion preserves parity, identifying the two split-cut
   parity bits. Write the original quadratic theta sign at total parity
   `epsilon+delta`, subtract its separate physical and auxiliary quadratic
   signs, and display the remaining bilinear cocycle. This directly proves
   why the fourth cut does not introduce another independent sign.

2. **The universal vacuum seed is not derived sufficiently explicitly.**
   Its pullback by `q2=qL*qR` is sound: CCY's vacuum factor is independent
   of all light weights, so the external puncture can be forgotten; all
   the sewing and collapsing maps are Möbius, with no Schwarzian term.
   The notes should say this rather than merely state the dependence.
   Include the exact oscillator norm with multiplicities and the three
   pair-contraction formulas from `CCY_COMPONENT.md`. At present the
   description “Gaussian contractions” does not specify the implemented
   coefficients sufficiently for reproduction.

3. **The middle recurrence imports rather than proves the sparse action
   coefficients.** This is legitimate when stated as an input from the
   human notes. Say explicitly which `L_1` and `L_-1` decompositions are
   used and how their total descendant levels follow from the difference
   of the branch levels. The boundary `n=1/4` action needs its explicit
   special case; the general human-notes proposition is stated only for
   `n>=3/4`. The new code uses the appropriate boundary action, but the
   prose should not suggest that the proposition alone covers it.

4. **The numerical precision claim needs scope.** The completed
   level-five comparison has a scaled residual around `7.05e-16`, despite
   using higher-precision production arithmetic. Report this measured
   agreement, not “agreement to 60 digits.” Separate arithmetic precision,
   regulator error, and reference accuracy. No level-ten performance or
   quotient result follows from this comparison.

5. **Explicit residue data would improve self-containment.** The notes
   give the correct endpoint table but leave `c_rs`, `A_rs`, and `P_rs`
   implicit. Reference exact human-note equations or supply their finite
   products. Also record the noncancelling root continuation used for
   negative double-Virasoro weights and the generic-input restriction.

## Arguments challenged and found consistent

### Geometry and the local insertion

Eliminating `z3` from the two displayed equations gives exactly the
original edge-2 plumbing relation. The relation of spin lifts requires
the transported Ramond spin frame stipulated by the notes; it is not an
unframed statement about arbitrary independent square-root choices.

The physical irreducible vacuum condition removes `G_-1/2|0>`, giving the
human-notes normalization `v_1/2(Q/2)=Q psi_-1/2|0>`. The distinction
between an outgoing Ramond intertwiner and an NS zero mode is essential
and correctly stated.

From the explicit auxiliary action, `Theta` anticommutes with every
fermion mode. In the graded tensor product the physical odd generators
contain auxiliary parity, so they also anticommute with `Theta`.
Consequently both the auxiliary stress tensor and `psi G` commute with
`Theta`; the displayed commutation with the two Virasoro algebras follows.
This justifies the use of ordinary descendant three-point forms for the
combined insertion. It does not justify replacing the full field by its
zero mode, and the notes correctly avoid that replacement.

### Branch selection and normalized middle coefficient

The two degenerate fusion rules shift the double-Virasoro momenta by
`+/-b` and `+/-1/b`. Within the fixed physical-momentum module they give
precisely `n'=n+/-1/2` at generic momentum. The odd field flips parity;
`Theta` flips it back.

The four inverse primary norms in the raw formula are correct. The two
outer normalized coefficients contribute the two NS and edge-3 norm
roots and one root on each split edge; the normalized middle contributes
the second root on each split edge. Its external norm root remains
unsewn and cancels the factor introduced when defining its normalized
branching coefficient.

The raw `Theta` coefficients agree with the two raw Ramond norms. Choosing
both parity norm roots from one common square root of their shared norm
product gives `Theta e_alpha=-i e_(1-alpha)`. Anticommuting `Theta` through
the fermion field then yields the `+i ||v_1/2||` factor in the normalized
formula. Independent principal square roots need not implement this
choice; the raw formula is appropriately used in production.

### Ward recurrence

The physical `L_m` commutes with the insertion. Because the BPZ form is
bilinear, the coefficients from the `L_1` action are not complex
conjugated. The two displayed middle Ward identities have the correct
orientation. The branch rejected from the `L_-1` action is separated by
`3/2`, so the double-degenerate fusion rule excludes it. The remaining
same-branch level-one matrix elements have exactly the two displayed
weight factors. Reflection of the physical ground components is an
isometry and commutes with this auxiliary even operator, so it gives the
negative-label recurrence.

### CCY residues

The endpoint table agrees with the code. A useful sign check is the
formal level-one null: at null slot infinity the spectator order
`(zero,one)` gives `h_one-h_zero`; at slot one the order `(zero,infinity)`
gives `h_infinity-h_zero`; at slot zero the order `(infinity,one)` gives
`h_one-h_infinity`. These are the corresponding Ward factors. Thus the
table has not silently interchanged odd-level fusion factors.

The external weight must remain fixed at recursive pole evaluations.
Both the prose and implementation do this. The generic Verma recursion
alone is not an irreducible Ising quotient; the notes now distinguish
these two functions.

### Auxiliary primary signs

The four nonzero parity coefficients can be obtained without appealing
to numerical agreement. In the NS-fermion branch, for auxiliary Ramond
grounds `(b,c)=(0,1)`, the outer product is `+1/2`, the middle raw matrix
element is `Q/sqrt(2)`, and the product of four squared norms is `+1`.
The extra NS-fermion sign and the theta quadratic sign are both minus,
so the coefficient is `+Q/(2 sqrt(2)) eta1 eta3`. For `(b,c)=(1,0)`, the
outer product and norm product are both minus, giving the same positive
coefficient multiplying `eta1 eta2`. In the NS vacuum branch the grounds
`(0,0)` give `+Q/sqrt(2)` and `(1,1)` give
`-Q/sqrt(2) eta2 eta3`. This establishes the auxiliary polynomial in the
notes and shows that it belongs to the minus ideal of the star algebra.

### Truncation and the pole from interacting null states

The balanced degree is additive and the retained index set is downward
closed. Computing all off-diagonal split powers before division is
therefore appropriate. The physical product constraint is not built
into the quotient algorithm.

The Ising agent's exact symbolic triple-null matrix element contradicts
the tempting assumption that three null legs force a second-order zero.
Its first-order zero leaves an uncancelled pole when all four level-two
null sectors are sewn. At that fixed edge-level tuple each edge has only
one level-two null direction; terms with fewer null edges do not cancel
the leading all-four pole. Thus the stated obstruction at physical level
`13/2` is substantive, not a loss of arithmetic precision. The bounded
level-five cycle correction cannot be promoted to a level-ten algorithm.

## Review conclusion

The new middle Ward recurrence, raw sewing formula, and generic CCY
reduction are mutually consistent under the stated conventions. The
presentation should expose the omitted norm/sign and seed steps and fix
the action-bound scope. The irreducible Ising quotient through level ten
remains an actual missing mathematical component, so the overall goal
must remain incomplete.

## All-order proof of the physical sector identity

This proof uses only the physical Ramond ground conventions and the
superconformal Ward identities; it does not infer the sector relation from
the numerical comparison. On either Ramond module make the replacement

\[
\mathbb L_{-B}w^+\longmapsto
 (-1)^B e^{-i\pi/4}\mathbb L_{-B}w^-,\qquad
\mathbb L_{-B}w^-\longmapsto
 -(-1)^B e^{i\pi/4}\mathbb L_{-B}w^+.
\]

Here \((-1)^B\) counts physical supercurrent modes. This odd map preserves
level and anticommutes with every \(G_r\), while commuting with every
\(L_m\). Its anticommutation with \(G_0\) follows immediately from the
two displayed Ramond ground actions in the human notes. The ground BPZ
metric is \(\operatorname{diag}(1,i)\), which this map preserves; BPZ
contravariance extends that preservation to all descendants.

For the following three formulas only, denote the map by \(J\). The
graded action on the second and third factors together is
\((-1)^{\epsilon_2}\rho(\xi_1,J\xi_2,J\xi_3)\), where
\(\epsilon_2\) is the total parity of \(\xi_2\). This even tensor-product
map commutes with the combined Ward action. Evaluating the four nonzero
ground components gives

\[
(-1)^{\epsilon_2}
\rho_f^{(\eta)}(\xi_1,J\xi_2,J\xi_3)
=-i\eta\,\rho_f^{(\eta)}(\xi_1,\xi_2,\xi_3).
\]

Since the two sides obey the same Ward identities and agree on every
ground component, the identity holds for arbitrary descendants.

Change basis by this BPZ isometry on both Ramond edges in the theta
contraction. Both Ramond parity bits flip, while all levels and inverse
metrics are unchanged. The two vertex factors multiply the contraction
before its theta sign by \((-i\eta)(-i\eta')=-\eta\eta'\); the two
\((-1)^{\epsilon_2}\) factors cancel. The quadratic theta sign changes by

\[
(-1)^{\epsilon_1(\epsilon_2+1)+
       \epsilon_1(\epsilon_3+1)+
       (\epsilon_2+1)(\epsilon_3+1)
       -\epsilon_1\epsilon_2-\epsilon_1\epsilon_3-
       \epsilon_2\epsilon_3}
=(-1)^{\epsilon_2+\epsilon_3+1}.
\]

Therefore, coefficient by coefficient at every level,

\[
\mathbb F_{\epsilon_1,\epsilon_2+1,\epsilon_3+1}
=\eta\eta'(-1)^{\epsilon_2+\epsilon_3}
 \mathbb F_{\epsilon_1,\epsilon_2,\epsilon_3}.
\]

Finally, the given star cocycle acts by
\((\eta_2\eta_3)\star\eta^\epsilon
=(-1)^{\epsilon_2+\epsilon_3}\eta^{\epsilon+(0,1,1)}\).
Combining the last two equations proves
\((\eta_2\eta_3)\star\mathbb F_f^{(\eta,\eta')}
=\eta\eta'\mathbb F_f^{(\eta,\eta')}\).
The temporary letter \(J\) can be omitted from the final notes by referring
to the displayed replacement as “this map.”

The argument includes both intrinsic NS primary parities \(p_1=0,1\)
and both relative form parities \(f=0,1\). Moving the two odd maps through
the NS factor contributes its parity twice, hence no additional
\((-1)^{p_1}\). This is why the factor remains
\(-i\eta(-1)^{\epsilon_2}\) also for an intrinsically odd NS primary.

## Final audit: the human notes mix two vertex conventions

This supersedes any blanket assertion above that all printed human-note
sewing formulas are mutually consistent. No human-note file was changed.
The inconsistency is visible directly, without a numerical calculation:

- `Human Notes/SCblock.tex:1751` explicitly defines the enlarged vertex
  as the physical human three-point form times the auxiliary form and
  the tensor-product Koszul sign.
- Its enlarged tensor-basis contraction at line 1742 includes the
  additional sign \((-1)^{A+\mathsf A}\).
- The physical contraction at line 1210 has no linear \((-1)^A\), while
  the auxiliary contraction at line 1755 has \((-1)^{\mathsf A}\).
- Squaring the explicitly defined tensor-product vertex cancels its
  Koszul sign but cannot supply the missing \((-1)^A\). Thus the claimed
  convolution does not follow from these literal definitions.
- The descendant branching contraction at line 1774 omits
  \((-1)^{2n_1}\), whereas the primary branching sum at line 1785 includes
  it. Factoring ordinary Virasoro descendant forms cannot generate that
  extra sign.

The actual branching oracle uses a different physical contour frame.
Express its Ramond ground coordinates in the physical human basis first:
\(f^0=w^+\), \(f^1=e^{3\pi i/4}w^-\). Write
\(a=A\bmod2\) and \(\epsilon_2=B+|\alpha|\bmod2\) only in this
audit. Its exact three-point transport is

\[
 \rho_{\mathrm{native},f}^{((-1)^f\eta)}
   (\xi_1,\xi_2,\xi_3)
 =(-i)^a(-1)^{f(a+\epsilon_2)}
   \rho_{\mathrm{human},f}^{(\eta)}(\xi_1,\xi_2,\xi_3).
\]

The exponent of \(-i\) is the parity bit, not the integer number of
NS supercurrent modes. The formula holds for both values of \(p_1\)
and \(f\). The ground tables give the factor
\((-1)^{f\epsilon_2}\). In the NS contour recursion the ratio between
the factors with \(a+1\) and \(a\), reduced modulo two, is
\(-i(-1)^{a+f}\); a flip of the second Ramond parity gives
\((-1)^f\). These reproduce the infinity phase in the native equation
and its second-slot recursion. For completeness, in the third-slot
recursion the native coefficients of the NS and second-slot
supercurrents are respectively
\((-1)^{p_1+a+\epsilon_2}\) and
\(i(-1)^{p_1+\epsilon_2}\). After multiplying by the two transport
ratios they become
\(+i(-1)^{p_1+a+\epsilon_{3,\mathrm{rest}}}\) and
\(-i(-1)^{p_1+a+\epsilon_{3,\mathrm{rest}}}\), exactly the human Ward
coefficients, using
\(a+\epsilon_2+\epsilon_{3,\mathrm{rest}}+1=f\bmod2\).
Virasoro Ward reduction preserves these parity bits. This establishes
the transport throughout the Ward recursion, not only for ground forms.

The code-consistent separate note can retain the existing branching
sum and its \((-1)^{2n_1}\) if it **explicitly corrects** its enlarged
vertex convention to

\[
\begin{aligned}
 &\hat\rho_f^{(\eta)}
 (\Psi_{-\mathsf A}\mathbf L_{-A}\phi_1,
  \boldsymbol\Psi_{-\mathsf B}u^{\mathsf b}\otimes
       \mathbb L_{-B}w_2^\alpha,
  \boldsymbol\Psi_{-\mathsf C}u^{\mathsf c}\otimes
       \mathbb L_{-C}w_3^\gamma)\\
 &\quad=(-i)^{A\bmod2}(-1)^{f(A+B+|\alpha|)}
 (-1)^{A\mathsf A+(B+|\alpha|+p_1)(\mathsf C+\mathsf c)}
 \rho_{\mathsf F}
 (\Psi_{-\mathsf A}\mathbf1,
  \boldsymbol\Psi_{-\mathsf B}u^{\mathsf b},
  \boldsymbol\Psi_{-\mathsf C}u^{\mathsf c})\\
 &\qquad\qquad\times
 \rho_f^{(\eta)}(\mathbf L_{-A}\phi_1,
       \mathbb L_{-B}w_2^\alpha,\mathbb L_{-C}w_3^\gamma).
\end{aligned}
\]

Here the unhatted physical form on the right is exactly the human one;
its label is the physical \(\eta\). No new permanent three-point symbol
is needed, but the note must state that the displayed contour transport
corrects the missing phase in the old enlarged-vertex factorization.
At the two outer vertices the physical transport factors multiply to
\((-1)^A\), independently of \(f\). They cancel the physical part of
\((-1)^{A+\mathsf A}\), leaving the auxiliary linear sign
\((-1)^{\mathsf A}\). The remaining comparison is then precisely the
quadratic spin cocycle already displayed in the main note.

Simply deleting \((-1)^{2n_1}\) from the branching formula while using
the literal old enlarged vertex does not repair the desired identity:
in a tensor basis it deletes \((-1)^{A+\mathsf A}\), whereas the
required auxiliary linear sign is still \((-1)^{\mathsf A}\).
Moreover auxiliary parity alone is not preserved by each embedded
Virasoro algebra, so that residual sign cannot be pulled out as one
universal scalar on each branching summand. The explicit vertex
convention is therefore essential to a formula independent of the code.
