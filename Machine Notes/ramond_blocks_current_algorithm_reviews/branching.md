# Adversarial branching, normalization, and sector review

Reviewed the initial `ramond_blocks_current_algorithm.tex` against the current C++ headers and `Human Notes/SCblock.tex`. No production code, main notes, or manuscript was edited. No PBW calculation or numerical run was performed. Source line numbers below refer to the initial reviewed versions; equation labels remain the more durable locators.

## Corrections required

1. **The displayed NS raw-primary order is not the code order.** In `free_field.hpp:415–429`, `primary` loops from the largest mode down and left-applies each new operator. Thus the C++ primary is

   \[
   2^{-2n}\ell(Q+2P,4n)\chi_{-1/2}\chi_{-3/2}\cdots\chi_{-(4n-1)/2}\phi,
   \]

   whereas the initial note and Human Notes use the reversed word. There are `2n` anticommuting factors, so the two words differ by \((-1)^{n(2n-1)}\). This is a **normalization convention discrepancy**, not a failed block computation: norms are unchanged, and the NS primary phase occurs twice in the sewn block. However, the individual raw vertices and action coefficients must follow one convention consistently. For notes describing only the current implementation, display the C++ order and state the conversion to the Human-Note order. Root has independently confirmed this correction. Do not silently present the Human-Note action normalization as the raw C++ one.

2. **The componentwise and graded two-intertwiner actions are conflated in the sector proof.** For the map in `eq:physical-intertwiner`, the componentwise identity is

   \[
   \rho_f^{(\eta)}(\xi_1,T\xi_2,T\xi_3)
   =-i\eta(-1)^{\epsilon_2}\rho_f^{(\eta)}(\xi_1,\xi_2,\xi_3).
   \]

   The graded tensor operator \(T\otimes T\) supplies the additional \((-1)^{\epsilon_2}\), giving the constant intertwining factor \(-i\eta\). The initial sentence says the parity-dependent factor already includes that graded sign; it does not. For example, the ground input \((w^-,w^+)\) gives componentwise ratio \(+i\eta\), but graded ratio \(-i\eta\). State these as two successive assertions. The final sewn factor \(-\eta\eta'\), theta-sign change, and sector identity are correct: the paired vertex parity factors cancel. This is a proof/wording correction, not a change of the sector theorem.

3. **The middle-recurrence domain includes an excluded ground pair.** “For \(n'=n+1/2>0\)” includes \((n',n)=(1/4,-1/4)\). That pair is an anchor, and `MiddleBranching::raw` returns it before entering the recurrence (`branching.hpp:298–310`). State instead \(n'=n+1/2\) with \(n\ge1/4\). Also append \(n\ge3/4\) to the Ramond \(L_1\) line of `eq:actions`. The remaining orientation-dependent formulas match the code, including the special \(1/4\to-3/4\) other-primary contribution and its exclusion by fusion.

4. **The explanation after `eq:fermion-sum` refers to a nonexistent middle metric in ordinary sewing.** The formula agrees with `fermion.hpp:172–184`, but its explanation should distinguish the cases. The three Fock norms are \((-1)^{\delta_i}\), so their inverse product is \((-1)^{\delta_1+\delta_2+\delta_3}=1\) on an allowed triple. In inserted diagonal sewing, the additional edge-2 inverse norm cancels the BPZ norm in the middle matrix element, leaving \(Q(-1)^{\mathsf b}/\sqrt2\). The CCY reviewer independently challenged this sentence and accepted this resolution.

## Missing data needed for a self-contained algorithm

5. **Specify the physical ground forms and both zero-mode metrics.** The initial note refers to low Ward anchors but does not supply their initial physical values. Add

   \[
   G_0w^\pm=i\beta e^{\mp i\pi/4}w^\mp,\qquad
   \langle w^\alpha,w^\gamma\rangle=\operatorname{diag}(1,i)_{\alpha\gamma},
   \]

   and the physical ground table: for \(f=0\), \((++,--)=(1,\eta)\); for \(f=1\), \((+-, -+)=(1,i\eta)\); the other entries vanish. These are the Human-Note conventions (`SCblock.tex:1189–1204`). For the auxiliary factor state \(\langle u^{\mathsf b},u^{\mathsf c}\rangle=\operatorname{diag}(1,-1)_{\mathsf b\mathsf c}\), \(\psi_0u^{\mathsf b}=u^{1-\mathsf b}/\sqrt2\), and the auxiliary-first graded tensor convention. These data make the sector derivation and the insertion's zero-mode normalization independently checkable.

6. **Explain the C++ physical ground coordinates before invoking contour transport.** `anchors.hpp:209–220` uses the coordinates \((w^+,e^{3\pi i/4}w^-)\), not the raw Human-Note \((w^+,w^-)\). Its `eta` argument is \((-1)^f\eta\) (`branching.hpp:prepare`). Once these ground coordinates are transported, the native physical form is precisely

   \[
   (-i)^{a\bmod2}(-1)^{f(a+b+\alpha)}\rho_{f,\mathrm{human}}^{(\eta)}.
   \]

   Consequently the displayed `eq:vertex-frame` is consistent with the implementation. It is **not** the literal untransported enlarged-vertex definition in the old Human Notes. Say which unhatted form the equation uses. The transport can be justified without new numerical tests: the ground table supplies \((-1)^{f\epsilon_2}\), the NS contour step supplies \(-i(-1)^{a+f}\), the second-slot parity change supplies \((-1)^f\), and Virasoro reduction preserves those parity bits. This settles the factor throughout the Ward recursion.

7. **Give the free-field action used to construct the finite columns.** The algorithm currently says to expand into free-field oscillators but omits the actual representation. Both reviewers independently identified this as a reconstruction gap. A compact display or appendix should provide

   \[
   [c_m,c_n]=m\delta_{m+n,0},\quad
   \{\eta_r,\eta_s\}=\{\psi_r,\psi_s\}=\delta_{r+s,0},\quad
   \{\eta_r,\psi_s\}=0,
   \]

   \[
   L_m=\tfrac12\sum_{k\ne0,m}c_kc_{m-k}
       +\tfrac12\sum_r r\eta_{m-r}\eta_r
       +\tfrac i2(Qm-2P)c_m\quad(m\ne0),
   \]

   \[
   G_r=\sum_{k\ne0}c_k\eta_{r-k}+i(Qr-P)\eta_r,
   \]

   together with oscillator `L0` and the ground shift \((1-2\nu)/16\). This is `free_field.hpp:185–256` on the positive chart and `SCblock.tex:1567–1570`; the reflected chart uses \(P\to-P\). State that physical odd operators acquire the auxiliary spectator parity when acting on the auxiliary-first tensor product.

8. **Make the inserted field's fusion restriction auditable.** The two displayed external weights are the level-two degenerate weights \(h_{2,1}^{(1)}\) and \(h_{1,2}^{(2)}\) in the embedding convention. Their momentum shifts, together with a common physical momentum on both middle halves, require \(n'=n\pm1/2\) at generic parameters. This one sentence identifies the actual null relations behind “the two degenerate fusion rules,” rather than leaving the selection rule unexplained.

## Checks that survived the challenge

- The two embeddings, branch weights, shifts \(2n_1^2\) and \(2n_R^2-1/8\), signed BPZ norms, parity transport ratio, and ordinary/insertion assembly factors agree with their C++ formulas.
- The physical stress tensor commutes with \(Q\Theta\psi\); this statement is about the physical \(L_m\), not either embedded \(L_m^{(i)}\). The note already distinguishes them. The middle recurrence has the correct incoming/outgoing descendant factor and no BPZ complex conjugation.
- The optional normalized middle coefficient \(i\|v_{1/2}\|\mathbb B\) times the two internal norm roots is consistent with the stated coherent root convention \(\Theta(v_n^\alpha/\|v_n^\alpha\|)=-i v_n^{1-\alpha}/\|v_n^{1-\alpha}\|\). It is not needed numerically, and raw matrix elements remain preferable.
- From the raw ground primaries and auxiliary metric, the two middle anchors are \(\sqrt2Q\) and \(Q/\sqrt2\) in either orientation. They are the values hardcoded in `branching.hpp:300–301`.
- The sector ideals are orthogonal under the displayed commutative twisted product. Diagonal projection and multiplication/division by the scalar vacuum series commute with the parity operator, so they preserve these sector identities. The CCY reviewer explicitly challenged and checked this point.
- The note correctly treats the limited action supports as structural input tested by all-row residuals, rather than claiming a new all-level proof. Numerical residuals do not establish the symbolic support identity at arbitrary parameters; retain that qualification.
- Reuse claims match the present C++ code: reflected charts, parity transport, interned states, spectator-independent images, shared descendant suffixes, paired actions, closure of required outer supports, cached single-leg factors, and factorization reuse across the two vertex signs. The code does not identify the different numerical Virasoro copies; “commuting-copy scheduling” should retain that distinction.

## Presentation recommendations

- Put the small ground table and oscillator representation before the first calculation that uses them. The present sector proof asks the reader to trust data only mentioned much later.
- Preserve the opening route map, but add a short numbered operational sequence: construct reused actions/outer data; add middle data if required; evaluate the two reduced CCY factors at branch-specific cutoffs; assemble; divide by the direct auxiliary factor in its sector; restore the common vacuum square. This connects the substantial formula sections without introducing a new algorithm.
- In `eq:vertex-frame`, avoid using plain `b` both for the Liouville coupling and for a supercurrent count even within a declared local scope. The Human-Note convention (PBW words `A,B`, auxiliary words in mathsf) is clearer and does not need new permanent notation.
- Keep the full mathematical inserted sewing relation to establish factorization, but call the numerical object the **diagonal projection on a downward-closed coefficient domain**. It should never read as if production first computes the full four-variable block.
- Retain the two precise limitations: generic-momentum action/recurrence pivots, and residual-based rather than certified numerical accuracy. They are directly relevant to the current algorithm; no discussion of abandoned alternatives is needed.

## Review dialogue

The CCY reviewer challenged auxiliary sector closure after diagonal restriction and the ordinary Fock-metric explanation. We resolved the first by commutation of the parity operator with `Diag` and scalar vacuum multiplication, and corrected the second using the even auxiliary parity constraint and the additional middle-cut metric cancellation. I challenged the use of a parity-dependent componentwise intertwiner factor while saying its graded tensor sign was included; the distinction above preserves the sector result while repairing the proof. We independently identified the missing free-field representation and agreed it is required for the intended self-contained notes. Root confirmed the NS string-order discrepancy and requested explicit ground-basis transport rather than a convention gloss.

## Second round: verification of revisions

The revised main LaTeX was read again against the C++ implementation. No numerical tests were run.

- **NS normalization resolved.** The display now has the ascending left-to-right mode order actually constructed by `FreeField::primary`. The conversion factor to the Human-Note descending order is correct for every allowed half-integer label: `2n` is an integer and the reversal parity is `n(2n−1)`.
- **Ground/frame data resolved.** The physical table, metrics, and `G0` action match the Human Notes. In the stated native physical coordinates the metric becomes `(1,1)` and `G0=-iP η0` with `η0` exchanging the grounds by `1/sqrt(2)`, as in the current C++ positive chart. The native `eta=(-1)^f eta_physical` transport and revised `eq:vertex-frame` agree with `PhysicalForm` and `LowAnchors`.
- **Sector proof resolved.** The prose now correctly separates the componentwise parity-dependent factor and the graded constant factor. The resulting sewn sign and sector identity are unchanged and valid.
- **Free-field construction resolved.** The inserted oscillator algebra and `L_m`, `G_r`, `L0` formulas agree with the positive-chart implementation; the Ramond `L0` shift is correctly `1/16`. The auxiliary-first tensor ordering and reflected-chart qualification supply the needed graded-action convention.
- **Action/recurrence domains resolved.** The Ramond `L1` expansion now says `n>=3/4`. The middle formula now says `n'=n+1/2` with `n>=1/4`, so it excludes the independently specified ground anchor. The special other-primary boundary remains explicit.
- **Fock metric explanation resolved.** Ordinary inverse norms cancel by even total auxiliary parity; the inserted additional cut cancels its middle BPZ norm. Both cases now explain the implemented state sum accurately.

No new mathematical obstruction was found in the revised branching and recovery account. The qualifications concerning generic pivots, assumed structural action supports, and residual checks remain necessary and are still present.

## Second round: presentation debate and smallest agreed edits

I challenged further restructuring after the new five-step overview and early total-level definition were added. The CCY reviewer agreed that these already resolve the navigation problem, so moving entire sections would add churn without improving the mathematics. We agreed on only three small local refinements:

1. Replace the newly added symbols `h_{2,1}^{(1)}` and `h_{1,2}^{(2)}` by the prose **“Kac labels (2,1) in the first copy and (1,2) in the second.”** The labels themselves are correct, but `h_{2,1}^{(1)}` otherwise collides exactly with the already established edge/branch notation `h_{j,n}^{(i)}`. This avoids introducing an ambiguous new weight notation.
2. Immediately after the formal inserted branch sum, add one sentence that production evaluates only its shifted diagonal, referring to the target-domain section. The full mathematical block remains useful for proving factorization; the computational scope should be clear at its first display.
3. Call `G` in `G^{-1}GG^{-1}=G^{-1}` the **physical Gram matrix**, to distinguish that temporary matrix notation from the physical supercurrent modes `G_r`.

The CCY reviewer also challenged the PBW runtime table's presentation of one model estimate while another follows nearby. I agreed that the first table should either display the range across models or explicitly identify its chosen model. Timing validity is outside this branching review; the agreed writing principle is to label estimates and their model dependence where the numbers first appear.
