# Adversarial review: CCY, Schottky, target truncation, and recovery

Reviewed the initial `ramond_blocks_current_algorithm.tex` against the current C++ implementation (`ccy.hpp`, `schottky.hpp`, `pipeline.hpp`, `fermion.hpp`) and the primary CCY paper, [arXiv:1703.09805](https://arxiv.org/html/1703.09805). No physical PBW calculation or new expensive numerical check was performed. Findings below refer to equation labels, since the author is revising the shared notes concurrently.

## Findings that should be corrected

1. **Ambiguous recursion summation bound.** In `eq:ccy`, `\sum_{e,r\ge2,s\ge1}` visually imposes `e>=2` and omits edge 1. Write `\sum_e\sum_{r\ge2,s\ge1}`. Root has already acknowledged this correction.

2. **The explanation of the auxiliary inverse norms conflates the two graphs.** The paragraph following `eq:fermion-sum` mentions cancellation against a middle BPZ metric even in ordinary sewing, where no middle vertex exists. The branching reviewer independently derived the correct explanation: each auxiliary Fock norm is `(-1)^delta_i`; the product of the three inverse norms is one on the allowed even triples. In inserted sewing the extra edge-2 inverse norm cancels the BPZ norm in the inserted matrix element, leaving `Q(-1)^b/sqrt(2)`. The displayed state-sum formula itself matches the code.

3. **“The full inserted Virasoro blocks ... are nevertheless required” overstates the workload and contradicts the main truncation optimization.** The code requires unequal middle levels *inside each factor on the exact downward-closed domain*. It does not construct unrestricted four-variable blocks before diagonal extraction. Replace that sentence accordingly.

4. **A sign-convention sentence in the physical-sector proof is inaccurate.** In discussion with the branching reviewer, the parity-dependent factor `-i eta (-1)^epsilon_2` belongs to the componentwise action on two arguments. Extending to the graded tensor operator contributes another `(-1)^epsilon_2`, giving the constant `-i eta`. The notes currently say the parity-dependent expression already includes the graded sign. The final sector identity is unchanged; the intermediate explanation should distinguish the two operations.

## Challenges resolved mathematically

### The larger-magnitude Kac root does not lose a branch

This needed proof, rather than a numerical-stability assertion. Put `A=1-r^2`, `B=1-s^2`, and `H=rs-1+2h`. For `r,s>=2`, the two roots are `x_±=(H±sqrt(Delta))/A` and obey `x_+ x_-=B/A`. Swapping `r,s` gives

`x'_±=(A/B)x_±=1/x_∓`.

Both orientations select the same sign by magnitude because their candidate pairs differ by the common factor `A/B`. Thus the ordered pairs `(r,s)` and `(s,r)` supply both central-charge values `c(x_+)` and `c(x_-)`. When `r=s`, the roots are reciprocal, so they give the same central charge and the same residue after exchanging the identical Kac-grid axes; `(r,r)` occurs only once. For `s=1`, one root is zero and the other is the unique finite-pole root. This explains both the magnitude choice and omission of `r=1`. Collisions and exceptional limits remain subject to the stated genericity restriction.

The combined derivative-times-normalization factor also checks algebraically. From

`h(x) = [(1-r^2)x + 2(1-rs) + (1-s^2)/x]/4`,

one obtains `-dc/dh * A_rs = -12 x^(2rs-1)(x^2-1) / ([(1-r^2)x^2-(1-s^2)] prod(ux+v))`. This is precisely the expression whose `x±1` factors are cancelled in the code. No residue is dropped merely because an uncancelled representation would be `0/0`.

### Fusion signs and global coefficients

The notes' ordered spectator pairs agree with the current four-edge graph. The middle vertex is ordered `(h_R,h_psi,h_L)`. Changing pair order can change an odd-level fusion sign, so keeping the explicit pair table is warranted. The paired-root polynomial in the notes reproduces the code, including the unpaired origin factor `h_b-h_a`.

The global three-point expression agrees with CCY Eqs. (4.20)–(4.22), and the punctured seed uses one inverse global norm per internal edge. In particular there is no extra external-field norm and no duplicated norm on edges 1 or 3.

### Forward propagation retains the information needed by later residues

The forward state must contain both the accumulated internal shifts and its current central-charge label. Different incoming central charges have different outgoing denominators. The code correctly propagates them separately, including when their amplitudes sum to zero. Only after propagation does it use their sum for the terminal global coefficient. Every transition increases a shift by at least two, so the finite downward-closed index set gives a directed acyclic computation. The notes correctly limit completed-block reuse: different outer branching tuples instantiate separate CCY engines.

### The target domain is exactly the required downward closure

For a fixed branch, an individual-factor index `(a,b,c,d)` extends to a permissible shifted diagonal target if and only if

`a+d+max(2n^2-1/8+b, 2n'^2-1/8+c) <= K`.

Necessity follows because the final common middle power must be at least both displayed middle powers. Sufficiency follows by choosing that common power to be their maximum and raising only the smaller middle exponent. This is exactly the pair of inequalities `a+b+d<=L` and `a+c+d<=R`. It proves that the domain is sufficient without a doubled total cutoff, and that imposing `b=c` inside the individual Virasoro factors would be wrong. I recommend adding this one-line maximum criterion to the notes.

The product targets `(a,L-t,R-t,d)` with `a+d<=t` exhaust the required shifted diagonal. BPZ transposition swaps the two middle weights and exponents; the implementation reuses only the scalar product, retaining the distinct oriented branching factors.

### Diagonal projection is used only where it commutes with the convolution

Diagonal extraction is not multiplicative on arbitrary four-variable series. Here it works because the physical factor depends on the split parameters only through their product. A physical contribution `(j,j)` forces a diagonal auxiliary contribution at any diagonal output target. The notes give precisely this restricted argument, which is correct. Projection and multiplication by the parity-scalar Schottky vacuum commute with the parity-sector operator.

### Schottky normalization and word cutoff

CCY's product uses exponent `-1/2` on primitive conjugacy classes; the implementation identifies inverse pairs and consequently uses exponent `-1`. Directed reduced closed walks on the theta graph, modulo cyclic rotation and inversion, implement these pairs. Keeping the starting vertex in an arc label prevents an odd graph path from being incorrectly treated as a closed root.

For a projective matrix, `det(M)/tr(M)^2=q/(1+q)^2`; inversion yields the Catalan series shown in the notes. Every nonbacktracking closed walk has a unit cusp trace and determinant degree equal to its length. Since oscillator powers begin at two, a word longer than `floor(N/2)` cannot contribute. The vacuum has leading coefficient one and is independent of internal/external weights. It may therefore be removed once from each Virasoro factor and restored as its square after the restricted fermion division. The extra sphere's identity insertion collapses to the same plumbing product `q_2=q_L q_R`.

### Sector division has a genuine uniqueness statement

Writing `x=eta_2 eta_3`, the given cocycle gives `x star x=1`. The two eigenspaces are ideals, with units `(1±x)/2`. The ordinary auxiliary series belongs to the plus ideal; its constant acts there as 2. The inserted diagonal auxiliary series belongs to the minus ideal; its constant acts there as `sqrt(2) Q`. Since real allowed `b` gives nonzero `Q`, triangular division has a unique solution in the appropriate ideal. It is not an inverse in the entire eight-dimensional parity algebra. The code's numerical projection/record policies should remain explicitly separated from this exact statement.

## Presentation and self-containment

- Define the physical total-level cutoff `N` in the opening conventions, before the Schottky discussion first uses it. Clarify that the output retains half-integer NS powers and integer Ramond powers whose sum is at most `N`.
- Add a short execution-order paragraph near the beginning: find needed branches and cached actions; solve outer and, when needed, middle Ward systems; evaluate reduced Virasoro factors on the exact target domains; assemble and divide by the direct fermion factor in its sector; restore the vacuum square.
- Move or preview the diagonal identity and target-domain logic before the implementation detail. It is the reason the new insertion is computationally practical and should not first become clear after the lengthy CCY subsection.
- The oscillator-column construction currently relies on `L_m`, `G_r`, and the free-field fermion without giving the physical free-field realization. Both reviewers independently consider a compact oscillator-algebra/representation paragraph necessary for literal self-containment. The branching reviewer is supplying the minimal formulas.
- Keep the fusion-pair table, the distinct incoming-central-charge warning, and the explanation of the inverse-pair Schottky exponent. These prevent plausible but incorrect simplifications.
- Do not describe Schottky as computing the complete global block. It computes the universal large-central-charge vacuum; the global `SL(2)` coefficient is the separate closed finite sum. The present mathematical formulas distinguish these correctly.

## Review outcome

No blocking mathematical inconsistency was found in the reviewed CCY, Schottky, exact target truncation, or sector-recovery algorithms. The finite action supports and high-level numerical accuracy are separate proof/validation issues; this review does not certify them or treat completion of a timed run as an accuracy bound. The four concrete text corrections and the self-containment improvements above should be incorporated before final publication of the notes.

## Second-round review of the revised notes

The revised manuscript resolves the four concrete first-round findings. Its execution-order overview and upfront physical-level cutoff now explain the organization. The free-field realization and ground data fill the prior reconstruction gap. The Kac-root coverage argument and maximum criterion for exact diagonal closure are now explicit and correct. The auxiliary norm paragraph now correctly distinguishes ordinary sewing from cancellation against the added middle metric. No numerical calculations were run in this round.

Only small material presentation changes remain advisable:

1. The newly written Kac weights `h_{2,1}^{(1)}` and `h_{1,2}^{(2)}` collide with the existing notation `h_{j,n}^{(i)}` for edge and branch labels. The Kac labels are correct in the human notes' embedding convention, but should be given in prose as `(2,1)` and `(1,2)`, without reusing the already defined weight symbols.
2. In `G^{-1}GG^{-1}=G^{-1}`, explicitly identify `G` as the physical Gram matrix. This prevents confusion with the physical supercurrent modes `G_r`.
3. An optional single sentence after the inserted branch sum can say that production evaluates its shifted diagonal and that the exact domain is proved later. The branching reviewer and I agree that the new overview makes further section rearrangement unnecessary.

### PBW comparison and measurement honesty

I read the PBW include and checked its interpretation against the stated loop-count model. It correctly distinguishes integer operation proxies from timed stages; accounts for halving distinct three-point requests, rather than the entire positive-sector runtime; identifies saved level-10 measurements and extrapolated level-15 estimates; and separates 53-bit native, 384-bit FLINT, and fresh 136-bit MPC arithmetic. Its explicit refusal to interpolate across arithmetic backends is justified. It also correctly notes that saved PBW internal timings contain repeated serialization, whereas the C++ internal timer excludes final serialization, making parent-process timings the proper end-to-end comparison.

One material presentation issue remains: the headline level-15 table gives only the full-feature estimates (40 and 252 minutes in the negative sector), while the next paragraph reports materially smaller lean-model estimates (18 and 76 minutes). Because the fitted features are correlated and internal Ward recursion is not counted directly, the single headline numbers could appear more determinate than the evidence supports. Both reviewers recommend either showing the ranges across both models at first occurrence or explicitly labeling the table “full-feature estimates.” The model-spanning ranges are native positive 12–40 minutes and negative 18–40 minutes; 384-bit positive 61–252 minutes and negative 76–252 minutes. These are sensitivity ranges, not confidence intervals, and should remain labeled accordingly.

The fresh-run timing include is still a progress placeholder at this review snapshot. This review therefore does not certify final recorded timing values or diagnostics. The author must replace that placeholder with completed-run evidence before delivery, as the notes already promise.

**Second-round outcome:** the mathematics reviewed here is ready, subject to the small notation/presentation changes above and completion of the actual timing table. No alternative or historical recovery algorithm needs to be added.

## Final audit of completed measurements

The four fresh runs are complete. This final audit read the timing manifest and each saved result, inspected their metadata and array structure, and checked current source hashes. It did not evaluate or compare block coefficients numerically.

| Level | Sector | Internal seconds | Full-process seconds | Physical vectors |
|---|---|---:|---:|---:|
| 10 | positive | 38.209002 | 38.352457 | 506 |
| 10 | negative | 110.324932 | 110.414582 | 506 |
| 15 | positive | 471.291275 | 471.398110 | 1,496 |
| 15 | negative | 693.747822 | 693.835638 | 1,496 |

All displayed stage times, branch counts, engine counts, reuse counts, and residuals agree with `timings.json` and the individual result files. Actions/support and outer Ward systems are correctly marked as subcomponents of branching; the separately measured total need not equal the sum of rounded rows. For every run, the engine count equals twice the branch count minus twice the transpose-reuse count, as required by two Virasoro factors per newly evaluated product. All saved physical vectors have eight slots and obey their stated diagonal/cutoff convention.

Every result records 40 decimal digits, 136 arithmetic bits, the documented rational input parameters, `p=f=0`, the appropriate vertex signs, and the `record` sector policy. Saved start and end times confirm that the four processes ran sequentially. All eleven source hashes in the manifest match the current production sources. These facts substantiate the measurement provenance; they do not establish 40 accurate output digits.

One final wording correction was sent to the author: `maximum_sector_residual` is evaluated in `recover()` on the quotient before restoring the Schottky vacuum square. The table should therefore say **recovery parity sector**, and its definition should state **before vacuum restoration**. The vacuum factor preserves the exact parity ideal, but multiplication need not preserve the normalized numerical residual. This diagnostic must not be represented as a fresh residual computation on every final physical coefficient.

The mean-middle-level candidate condition added to the branching section accurately describes `OuterBranching`: the initial support uses the sum of the two middle primary levels against twice the physical cutoff, then closes under action changes. The stricter maximum-middle-level condition belongs to diagonal assembly and the individual Virasoro target domains. Separating these stages prevents an incorrect claim that all finite action preparation is pruned to the final diagonal domain.

The PBW table is now explicitly labeled as the full-feature model and warns at first occurrence that the lean model gives substantially smaller estimates. Its stated precision/backend limits and serialization distinctions remain accurate. No new PBW run was made. The revised Kac-label prose and explicit physical-Gram definition also resolve the remaining second-round notation issues.

**Final signoff:** the reviewed algorithm, optimization descriptions, saved timing values, and qualified PBW comparison are consistent with the current implementation and evidence. Apply the small pre-restoration residual-label correction above. No further numerical block test or historical-algorithm discussion is warranted for this documentation task.

**Author resolution:** the author reports that the precision section now identifies the residual's pre-restoration stage and that the regenerated table uses “Recovery parity sector.” This resolves the last outstanding correction; final audit has no remaining requested changes.
