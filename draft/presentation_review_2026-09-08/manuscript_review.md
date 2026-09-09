# Presentation and correctness review of the superconformal-block manuscript

**Recovery follow-up:** The global inverse obstruction below is valid, but
equal vertex-form signs impose an additional physical parity identity. A
restricted inverse on the corresponding ideal now recovers the physical
block without PBW input. The proof, implementation, validation, and remaining
opposite-sign limitation are documented in
[PHYSICAL_RECOVERY.md](../../Code/full_ramond_block_runtime/PHYSICAL_RECOVERY.md).
This supersedes the audit's statement that such a restricted prescription had
not been supplied. The manuscript's equations have not been changed by this
follow-up.

Reviewed 8 September 2026. This document contains **proposed changes only**. The reviewers did not edit the manuscript, bibliography, compiled PDF, or code. The working manuscript changed externally during the audit; the added integration discussion and acknowledgments were reviewed before delivery.

The source reviewed is [`../draft_scblock.tex`](../draft_scblock.tex), with the six-page compiled [`../draft_scblock.pdf`](../draft_scblock.pdf) and [`../references.bib`](../references.bib). Line numbers below refer to that source snapshot. Existing author changes were preserved. Commented-out material is not treated as part of the manuscript or as an available explanation. In particular, a definition present only in a source comment does not make the published argument self-contained.

“Edit and verify” in the request is interpreted consistently with “Do not change anything in the draft”: formulate and check proposed repairs here, leaving their adoption to the authors. A complete audit is not a certification that every scientific claim is established. Unresolved mathematical claims and missing numerical evidence are explicitly identified.

The review covers the title/front matter, all 28 source prose paragraphs in order, their displayed equations and continuation text, all four active content footnotes, the rendered references, and PRL presentation requirements. Long paragraphs are subdivided within their entries so that no equation-introducing or equation-explaining passage is omitted. The latest source snapshot has SHA-256 `259f80d25bde8d1b64ac60d18556e9771a5412f88ed005b6a738dd412a6058f8`; its modification time is 8 September 2026, 14:48:23 JST. The subsequently compiled PDF is dated approximately 14:50 JST.

## 1. Findings that affect the main claim

1. **The formal inverse in Eq. (23) does not exist in the coefficient algebra described by the draft.** The displayed auxiliary block has constant term `1 + eta_2 eta_3`, a zero divisor under the displayed star product. This is also explicitly recognized by the current code. A complementary recovery prescription or a justified restriction of the domain is required; changing the wording alone does not repair the algorithm. See P17 and the verification notes below.
2. **The documented numerical validation and runtime concern the enlarged block.** The local record checks `Fhat = F_F star F_PBW`; it does not establish that a unique physical block was recovered by Eq. (23). The 82.9-second level-10 benchmark is for `Fhat`. See P23.
3. **The current exposition does not establish the advertised generality.** It displays one closed genus-two channel, omits the general graph construction from the active manuscript, leaves Supplemental Material as a placeholder, and does not explain recovery for arbitrary spin structures or external insertions. See P04, P10, P18, and P19.
4. **Several formula-level corrections are definite:** the NS mode shift is `1/2`, the background-charge convention must be `c = 3/2 + 3Q^2`, the free-field Ramond `L_0` needs `+1/16`, and the inverse Gram contraction needs the missing primed indices. See P08, P11, P12, and P14.
5. **The proposed applications use an excluded parameter value.** Two-dimensional type 0 and the stated heterotic background require the super-Liouville value `b=1`, at which the displayed double-Virasoro embedding is singular. A controlled limiting prescription must precede claims of direct applicability at that value. See P12 and P26.
6. **The recurrence needs an explicit closure argument.** The branch-label shift does not lower total propagation level by one, the boundary labels must be specified, and the second displayed Ward identity introduces an unresolved physical `L_0` action. See P21–P22.

Use the following labels throughout: **definite correction** means an internal contradiction, copy error, or directly checked failure; **missing specification** means that the result may be correct under an unstated convention; **unverified claim** means that the inspected evidence does not establish the assertion; **style** means an optional improvement rather than an error.

## 2. Grammar and phrasing inventory

These are the identified grammar, usage, and copy errors. The separate style list and paragraph entries distinguish optional improvements from actual errors. Mathematical mistakes are audited in Section 3 rather than disguised as copyedits.

| Source line(s) | Current wording | Proposed correction |
|---|---|---|
| 66 | “a NS superconformal primary” | “an NS superconformal primary” |
| 70 | “a basis of states in a supermultiplets” | “a basis of states in a supermultiplet” |
| 75 | “respectively in the NS and the R sector” | “in the NS and R sectors, respectively” |
| 75 | “for R sector”; “for NS sector” | “in the R sector”; “in the NS sector” |
| 75 | “denote ... to be” | “define ... by” or “write ... for” |
| 77, 81, 288 | “a (NS, NS, NS) sphere”; “a (NS, R, R) sphere” | “an ... sphere,” or preferably “at an NS–NS–NS/NS–R–R vertex” |
| 81, 86 | “three-point function ... are defined” | “three-point forms ... are defined” |
| 86 | “at infinity, 1, 0 respectively” | “at infinity, 1, and 0, respectively” |
| 88 | “one plumb together” | “we sew together” or “one plumbs together” |
| 88 | “two edges that connects” | “two edges that connect” |
| 98 | “eta, eta' and f labels” | “eta, eta', and f label” |
| 98 | “ground state label [four labels]” | “ground-state labels ...” |
| 98 | “NS and the R-sector” | “NS and R sectors” |
| 103 | “h, beta labels” | “h and beta label” (also fix what they label; P11) |
| 103 | “moving the operator order” | “reordering the operators” |
| 125 | “is that, the enlarged algebra” | “is that the enlarged algebra” |
| 139 | “satisfiy” | “satisfy” |
| 167 | “For each ... one can directly solve them” | “At each level, we express their action ...” (representation-dependent; P14) |
| 169 | “constructed from acting ... on” | “generated by acting with ... on” |
| 169, momentum footnote | “as the following” | “as follows” |
| 169, momentum footnote | “For h is greater than ...” | “For h greater than ...” or “When h>...” |
| 192 | “a enlarged” | “an enlarged” |
| 192 | “defined similarly with” | “defined analogously to” |
| 209 | “reduce ... into” | “express ... in terms of” |
| 209 | plural “free fermion blocks. The latter is defined” | “The auxiliary block is defined ...” |
| 220 | “the blocks as a formal series” | “the blocks as formal series” |
| 225 | “define the of the convolutive inverse” | Delete “the of”; use “convolution inverse” only after fixing its existence, P17 |
| 229 | “analogous convolution formula ... exists” | “an analogous convolution formula ... exists” |
| 240 | “F(...) are Virasoro conformal blocks” | “F(...) denotes the Virasoro block ...” |
| 248 | “two types ... which corresponds” | “two types ... corresponding to” |
| 253 | “branching coefficient have” | “branching coefficients have” |
| 253 | “there exists [plural coefficients]” | “there exist ...” |
| 262 | “by acting the free-field realization ... on” | “by applying the free-field realization ... to” |
| 264 | “After solving the coefficients” | “After determining the coefficients” or “After solving for the coefficients” |
| 264 | “V's” alongside another coefficient symbol | use the symbol consistently, without an unnecessary plural apostrophe |
| 272 | “which becomes recursion relations that relates” | “which yield recursion relations that relate” |
| 272 | “relates ... with those with 1 lower total level” | “relates ... to coefficients at lower branch level,” with the precise nonunit decrement; P22 |
| 272 | “coefficients ... with all 8 sign configurations takes” | “Computing all eight configurations takes ...” |
| 272 | “On a laptop ... on a laptop” | delete the repeated phrase and give the actual benchmark conditions |
| 274 | “Section ... and ...” | “Secs. ... and ...” |
| 274 | “results ... passes” | “results ... pass” |
| 274 | “coefficients ... matches with” | “coefficients ... agree with” |
| 274 | “block ... matches with” | “block ... agrees with” |
| 276 | “parameters ... which gives the same ...” | “plumbing data ... that describe the same ...” |
| 280 | “partition functions, computed from our superconformal blocks agrees” | rewrite with the ratio as the explicit subject and the modifier properly attached |
| 288 | “in arbitrary plumbing channel” | “in an arbitrary plumbing channel” or “in arbitrary plumbing channels” |
| 288 | “with arbitrary spin structure” | “with an arbitrary spin structure” or “for arbitrary spin structures” |
| 290 | “Our algorithm provide” | “Our algorithm provides” |
| 290 | “strings that are proposed recently” | “recently proposed strings,” or a precise statement of the proposals |
| 290, new sentences | “similarly to [footnote]”; “an algorithm similar to [footnote]” | “using the method of Ref. ...,” identifying the actual reference and its scope |
| 290, new sentences | “one sample ... and convert” | “one samples ... and converts” or “by sampling ... and converting” |
| 296, new acknowledgment | “YW and XY thanks Yukawa Institute ...” | “Y. Wang and X. Yin thank the Yukawa Institute ...” |
| 296 | mixed list “drafting, editing, literature search, implementing ... and conducting ...” | “drafting and editing, literature searches, algorithm implementation, and numerical testing” |

### Style and professional presentation, also in source order

| Source | Proposed improvement and reason |
|---|---|
| 22, 39 | “All-Genus 2D” and “arbitrary genera” are grammatical. Consider less compressed title phrasing and “at arbitrary genus”; match the scope to the result actually proved. |
| 39 | Replace “utilizes” with “uses”; state the Ramond advance and a concrete result rather than repeating “arbitrary.” |
| 46 | Remove repeated “on a spin Riemann surface” and repeated “computed”; say “spectrum and three-point coefficients”; explain which bootstrap data are constrained. |
| 48 | Shorten “both the case ... and that ...,” “Meanwhile,” and “due to the fact that”; distinguish existing recursions from a general impossibility claim. |
| 50 | Use “In this Letter,” “two commuting Virasoro subalgebras”; avoid repeating “in the context of”; define or omit AGT. |
| 52 | Delete or drastically shorten the roadmap. It repeats headings and overstates what the active sections establish. |
| 58–64 | Avoid repeated “using,” “additionally,” “connected together,” and “data ... is specified”; say directly what each graph label means. |
| 66–86 | Prefer “two,” “four,” and “three-point” in running prose; use a consistent “NS/R sectors” style and present tense for definitions. These are editorial choices, not mathematical corrections. |
| 88 | Delete “for simplicity,” “easily,” and the humorous Supplemental Material placeholder; supply the actual figure and the missing extension. |
| 90–103 | Replace “aforementioned” with “this,” “forced to be the same” with “must agree”; define Gram matrices before their inverses and replace the long operator string by reconstructible sign data. |
| 125–143 | Break the sentence spanning the auxiliary algebra definition; “Their central charges are” is sufficient. |
| 145–155 | Separate ground-state definitions from the branching theorem; distinguish enlarged ground states from embedded highest-weight states. |
| 157–186 | Replace repeated “can be directly constructed” and “It can be proved” with a precise construction/proof reference. Explicit oscillator formulas may move to End Matter after retaining their role and conventions. |
| 192 | Define the enlarged block concretely; “the new descendants” and “replace the signs” do not tell the reader what to compute. |
| 209–227 | Use “denotes an ordered product,” “This gives,” and a concise derivation of the convolution; no fluent replacement should conceal the invalid inverse. |
| 229 | Delete “We should point out that,” “The one presented here,” and “Nevertheless”; give the graph-dependent rule. |
| 231–244 | Replace “On the other hand, one can choose” with “Alternatively, use”; replace the repeated “one can ... which ... one can” by a finite algorithm with a cutoff. |
| 248 | Name the two vertex coefficients directly; avoid “the ... one” and repeated explanations of their role. |
| 253–272 | State the support result, show the computational linear system/recurrence, then report validation and timing separately. Avoid “never” priority claims without support. |
| 274 | Replace “passes many cross-checks” by the tested object, order, parameter point, and measured discrepancy. |
| 276–280 | Explain why the anomaly-canceling ratio is used and show a reproducible comparison; a promised supplement is not itself evidence. |
| 288 | Delete the repeated abstract-level summary if necessary; avoid “coefficients given by branching coefficients.” |
| 290 | Replace “find the potential ... dual” by “constrain a possible ... dual”; distinguish MQM and matrix integrals. Remove the two newly added reference placeholders and scope the period-matrix proposal. |
| 292 | Name Cho, Collier, and Yin; specify the spin-dependent spectra and three-point data. No definite grammar error occurs here. |
| 294–296 | Use American “Acknowledgments”; “during this work” is shorter than “during the course of this work.” “Was used” is natural for completed work. Preserve the truthful AI roles, including the newly added drafting/editing. |

Throughout, use “two-dimensional,” “three-point,” “Neveu–Schwarz (NS),” and “Ramond (R)” consistently; introduce abbreviations once. APS uses American English, so prefer “labeled” to “labelled.” These editorial choices should not be counted as separate grammatical mistakes wherever they recur. P01–P28 below give the substantive and sentence-level context for adopting each change.

## 3. Every paragraph, in manuscript order

### Front matter: title, authors, affiliation, date — lines 22–36

**Meaning.** The title promises a construction covering every genus. The front matter identifies the authors and institution.

**Audit and proposed changes.** “All-Genus” is intelligible, but the title must match the result actually established. Until P17–P18 are resolved, a title such as “Double-Virasoro Decomposition of Two-Dimensional Superconformal Blocks” describes the demonstrated mechanism more accurately. If the full recovery theorem is supplied, “Two-Dimensional Superconformal Blocks at Arbitrary Genus” is a clear alternative; retaining `N=1` is appropriate if it is essential to distinguish the result. Expanding “2D” is a style choice, not a grammatical requirement. No substantive error was found in the names or affiliation; authors should verify their own details. The date matches the PDF inspected.

### P01. Abstract — line 39

**Explanation.** The intended result is a numerical construction of superconformal blocks. Adding an auxiliary Majorana fermion exposes two commuting Virasoro algebras, for which numerical block algorithms already exist.

**Audit.** “At arbitrary genera” is grammatical but less natural than “at arbitrary genus.” “Utilizes” can become “uses.” The second sentence presently jumps from an algebra embedding to the reduction of the physical block. The actual intermediate object is the enlarged block, and its recovery map is precisely the unresolved issue in P17. The abstract does not identify the Ramond obstacle, state a concrete validation, or explain why the construction matters outside the algebraic method.

**Safe proposed wording based on the material inspected:** “We develop a numerical method for superconformal sewing in two-dimensional theories with `N=1` symmetry. An auxiliary Majorana fermion exposes two commuting Virasoro algebras, allowing the enlarged blocks to be evaluated from ordinary Virasoro blocks and recursively determined branching coefficients. We demonstrate the construction in a genus-two channel with Ramond propagation.” Restore the stronger all-genus, all-spin physical-block claim only after the recovery and graph arguments are supplied. Add one brief physical payoff if space permits. Do not present a forward-identity benchmark as a test of a missing inverse.

### P02. Introduction: motivation — line 46

**Explanation.** Blocks separate the kinematic consequences of symmetry from the dynamical data of a particular CFT. Sewing blocks with the spectrum and three-point coefficients reconstructs correlators; comparing different sewings constrains that data.

**Audit.** The paragraph repeats “spin Riemann surface,” “superconformal,” and “computed” without advancing the argument. A single block is insufficient: one needs the relevant collection of chiral and antichiral blocks, their allowed pairings, the spectrum including multiplicities, and three-point coefficients. The central charge is also an input. Clarify whether `N=1` refers to the chiral algebra; an arbitrary full theory can have different left- and right-moving structures. Matter blocks are ingredients of superstring amplitudes, which also require ghosts, spin sums/projections, and supermoduli integration. Do not equate spin-surface moduli with all supermoduli.

**Proposed wording:** “Superconformal blocks encode the contribution of a superconformal representation to a correlation function. Together with the central charges, spectrum, and three-point coefficients, they determine correlators by sewing. Higher-genus blocks therefore provide essential input for superstring perturbation theory and for bootstrap constraints on CFT data.” Specify the chiral scope nearby and explain the physical bottleneck in P03.

### P03. Introduction: existing recursions and obstacle — line 48

**Explanation.** Existing sphere and torus recursions use poles in a representation parameter and a controlled asymptotic contribution. The authors want to explain why higher-genus Ramond propagation requires a different method.

**Definite correction.** The standard weight recursion is in the **internal** weight, not the external conformal weights. “External conformal weights `h`” should become “the internal conformal weight `h`.” The [Cho–Collier–Yin abstract](https://arxiv.org/abs/1703.09805) explicitly distinguishes internal-weight and central-charge recursions.

**Missing support.** Define Neveu–Schwarz (NS) and Ramond (R) at first use. The phrase “the large-`h` or large-`c` limit ... does not exist” needs a normalization, a statement of which parameters are fixed, and either a proof or an explicit growing coefficient. An unnormalized divergent limit does not exclude a rescaled limit or a more general recursion. “Can only be generalized” overstates an obstruction to a particular known method. The blanket “algorithms ... were not known” is also too strong: the descendant/Gram-matrix definition is already an algorithm at finite level. Distinguish an efficient recursive implementation from brute-force sewing. Check that each reference supports the claimed topology; the cited NS paper explicitly illustrates genera zero and one, although it proposes a broader construction.

**Proposed replacement of the key transition:** “Existing recursions exploit poles in the central charge or an internal weight. Extending them to higher-genus channels with Ramond propagation requires asymptotic data that are not supplied by the standard recursion. We address this problem using a double-Virasoro decomposition.” A more specific claim about divergence can replace the middle sentence once its precise statement is proved.

### P04. Introduction: contribution and history — line 50

**Explanation.** The method reorganizes a superconformal module tensored with a free fermion into ordinary Virasoro modules. Sewing in that basis converts the enlarged block into a sum of products of familiar blocks.

**Audit.** Write “In this Letter” and “two commuting Virasoro subalgebras.” “Two copies of commuting Virasoro subalgebras” is awkward. Define AGT or omit that acronym from the core narrative. The AGT paper supplies historical context; a discovery-priority statement needs a reference specifically establishing the embedding. “First numerical algorithm” requires qualification for the reason in P03. The final sentence skips the auxiliary factor and therefore misstates the immediate result.

**Proposed wording:** “We use two commuting Virasoro subalgebras of the superconformal algebra tensored with a free Majorana fermion. Branching into their representations expresses the enlarged block as a sum of products of ordinary Virasoro blocks. The new computational ingredient is a recursive determination of the mixed NS–R–R branching coefficients.” Follow this with the actual recovery statement after P17 is repaired. Keep the history to a short cited clause or move it to End Matter.

### P05. Introduction: roadmap — line 52

**Explanation.** This paragraph previews the order: sewing conventions, embedding, block decomposition, and branching recurrence.

**Audit.** There is no major grammatical defect. It is expensive in a Letter and repeats headings rather than explaining the mechanism. It also promises an arbitrary-channel definition and derivation that are absent from the active text.

**Proposed change.** Delete the five-sentence roadmap, or replace it with “We first describe the sewing construction, then derive the double-Virasoro expansion and the recurrence for its coefficients.” Spend the recovered space on the domain of the recovery map, a graph figure, or quantitative validation.

### P06. Plumbing geometry — lines 58–62; Eq. (1)

**Explanation.** A plumbing fixture glues small annuli around two punctures. Its complex parameter controls the size and twist of the connecting tube. A pants decomposition packages these tubes into a trivalent graph.

**Definite correction.** For a stable genus-`g` surface with `n` punctures, the counts are `2g-2+n` three-punctured spheres and `3g-3+n` internal tubes, not universally `3g-3`. The latter is the closed `g>=2` case. Derivation: if `V` is the number of pants and `E` the number of internal tubes, then `3V=2E+n` and the pants Euler characteristics give `V=2g-2+n`; hence `E=3g-3+n`.

**Missing assumptions.** State `2g-2+n>0` and handle unstable sphere/torus cases separately if the title includes them. Plumbing parameters are local coordinates near a chosen degeneration, not a single global coordinate system on moduli space. For fixed coordinate disks, use `0<|q|<epsilon`; `q=0` is a nodal limit. The assertion `|q|<=1` is not a universal convergence/domain statement. Distinguish external legs from internal edges. Use different names for puncture count and branch label, for example `N_p` versus `n`.

**Proposed wording:** “Near a stable degeneration, a genus-`g` surface with `N_p` punctures is obtained by sewing `2g-2+N_p` three-punctured spheres along `3g-3+N_p` tubes. Each tube identifies local coordinates by `z_1 z_2=q_e`. The parameters `q_e` are local coordinates on moduli space, and the sewing pattern is represented by a trivalent graph.”

### P07. Spin structure and channel labels — line 64

**Explanation.** NS/R labels determine the fermion boundary conditions around tubes; the sewing signs encode their gluing. Each pair of pants has an even number of Ramond boundaries.

**Audit.** Explain the local meaning of `eta_e=+/-1`, the restriction to NS–NS–NS and NS–R–R vertices, and any equivalence under changes of local spin trivialization. Edge-sign assignments can be redundant; they are not automatically distinct physical spin structures. A reader should be able to check the expected `2^(2g)` spin structures in the closed case. A channel with graded three-point functions also needs a leg ordering/local-coordinate convention: an abstract unoriented trivalent graph alone does not specify every sign in later equations. Clarify whether the signs are part of `hat Gamma` or separately listed. Avoid reusing `eta` both for spin signs and for a free-fermion oscillator later.

**Proposed wording:** “Label each internal tube by NS or R boundary conditions and by a spin-gluing sign `eta_e`. Every vertex has either zero or two Ramond legs. Together with ordered local coordinates at the vertices, these labels specify the sewing data; assignments related by changes of local spin trivialization represent the same spin structure.” The precise equivalence belongs in the promised graph prescription, not only in an informal claim.

### P08. Intermediate representations and descendant basis — lines 66–75; Eqs. (2)–(3)

**Explanation.** Sewing inserts a resolution of the identity. In the NS sector one starts from a highest-weight state; a generic Ramond representation has two ground states exchanged by the zero mode. Ordered products of negative `L` and `G` modes generate descendants.

**Definite corrections.** “A NS” becomes “an NS”; “a supermultiplets” becomes “a supermultiplet.” Set `nu=1/2` in NS and `nu=0` in R. Use the same counts `ell,p` in both basis formulas: the current R formula switches to `k,ell` while the definitions still use `ell,p`. Keep `-A` consistently on lowering strings. Distinguish a multi-index from its parity, e.g. `p(A)=number of G modes mod 2`, and use `|A|` only for level. Define `|+|=0`, `|-|=1` before their later use.

**Verified step.** Eq. (2) gives `G_0^2=-beta^2=h-c/24`, consistent with the Ramond algebra. But `beta` itself is a momentum parameter, not the `G_0` eigenvalue of either parity eigenstate `w^+/-`. The two eigenvalues of the mixing matrix are `+/- i beta`.

**Missing assumptions.** State that the discussion uses generic Verma modules with nondegenerate Gram matrices. At `beta=0`, and at null-vector loci more generally, irreducible quotients require separate treatment. Introduce the NS primary parity `p_1` now, or explicitly take it even throughout; its late introduction makes the selection rules ambiguous. Include the super-Virasoro relations or a precise convention reference, particularly the `G,G` anticommutator and BPZ anti-involution.

**Proposed opening:** “Sewing sums over descendants in a fixed superconformal representation. We use generic Verma modules and write their NS highest-weight state as `phi` and their Ramond ground states as `w^+/-`, with parities zero and one, respectively.”

### P09. Three-point forms — lines 77–86; Eqs. (4)–(5)

**Explanation.** Ward identities reduce descendant matrix elements to a finite set of chiral three-point structures. The NS vertex has even and odd structures; the mixed vertex has two structures of each parity in the chosen basis.

**Corrections and missing definitions.** The sewn correlator is a **sum of products** of three-point functions, not a single product. Prefer “normalized chiral three-point forms” over physical “3-point functions,” since arbitrary normalization to one removes dynamical structure constants. At lines 81 and 86, make “function ... are” either “functions ... are” or “form ... is.” State `a,f in {0,1}`, the parity support, and the zero normalizations of the complementary structures. Eq. (4) alone does not say that `rho_0` vanishes on the odd seed and `rho_1` on the even seed. Likewise, Eq. (5) needs its mixed/equal-parity vanishing statements. Use a different sign label, such as `sigma`, for three-point-structure signs to distinguish them from tube signs `eta_e`.

**Conventions to verify.** The phases `eta` and `i eta` may be correct in the authors' BPZ and Ramond-ground conventions; they should not be changed by stylistic intuition. Fix the state-field correspondence at infinity and the ground-state pairings so that these phases can be reproduced. Make clear whether `f` is the parity of the form or the parity relative to the intrinsic primary parity.

**Proposed wording:** “At an NS–NS–NS vertex, the superconformal Ward identities determine all descendant matrix elements from two chiral forms, which we normalize by Eq. (4). At an NS–R–R vertex there are four chiral forms, labeled by their parity and by a sign; Eq. (5) fixes their nonzero ground-state matrix elements.” Supply the missing support conditions immediately after the equations.

### P10. Theta-channel example and Supplemental Material — line 88

**Explanation.** Two three-punctured spheres joined by three tubes give the genus-two theta graph. The example assigns the tube at infinity to NS and the other two to R.

**Corrections.** “One plumb” becomes “we sew” or “one plumbs”; “edges that connects” becomes “edges that connect”; capitalize “Letter.” Delete the Supplemental Material joke entirely. Replace the figure placeholder by an actual labeled graph and a caption. “Can be easily generalized” is both unhelpful and currently unsupported.

**Missing data.** Specify local coordinates at the three punctures, for example `z`, `z-1`, and `1/z`, together with the coordinate/orientation convention actually used in the code. The equations `z_1=z_2=1` name puncture locations; they do not themselves define plumbing near 1. Label the edges consistently with later `q_1,h_1`, `q_2,beta_2`, and `q_3,beta_3`. State which vertex is first in the graded ordering. The figure is useful because this is substantive sign data, not decoration. No submission-ready Supplemental Material file was found under `draft`; working notes elsewhere do not replace it.

**Proposed wording:** “We illustrate the construction in the genus-two theta channel, formed by joining two three-punctured spheres. The edge joining the punctures at infinity is NS; the other two are Ramond. Figure 1 fixes the edge labels, local coordinates, and vertex ordering.” Add the generality claim only with a precise extension and available Supplemental Material.

### P11. Definition of the theta block, Gram matrices, and sewing sign — lines 90–103; Eqs. (6)–(7)

**Explanation.** Eq. (6) propagates a state along each tube, contracts the paired descendants with inverse Gram matrices, and evaluates one three-point form at each vertex. The Grassmann factor is the sign of the permutation from edge-pair order to vertex order.

**Definite corrections.** In Eq. (6), replace the Ramond inverse-metric entries by `mathbb G_beta2^(B alpha,B' alpha')` and `mathbb G_beta3^(C gamma,C' gamma')`. Without the primes, the metric does not contract the displayed second-vertex descendant labels. The last two fields in the prose operator-order string must carry label 3, not 2. “Labels ... labels” becomes “labels ... label”; “ground state label” becomes “ground-state labels.” Describe `h_1,beta_2,beta_3` as “representation parameters,” not three internal weights. Explain `h(beta)=c/24-beta^2`.

**Missing definitions.** Show the suppressed dependence on `q_e,eta_e,p_1` once and explain its subsequent omission. State explicitly that these are blocks with the primary propagation factors stripped off; the displayed powers start at descendant levels, not `q_e^(h_e+level)`. Do not guess a cylinder `c/24` shift: the precise prefactor depends on the chosen sewing frame. Normalize the Ramond minus-ground pairing and mixed pairings, specify the BPZ transformation including phases, and state that inverse Gram matrices are taken separately at each level. A bilinear BPZ form is not automatically a positive Hermitian norm.

**Parity audit.** The appearance of `p_1` in the tube/sign factors but not in the displayed `f` selection rule needs an explicit convention. This may encode a relative parity rather than an error; verify against the chosen definition of `rho_f` before changing it. Equal levels alone do not enforce parity, but an even Gram pairing does; explain why that pairing forces equal vertex parity labels.

**Proposed lead:** “With primary propagation factors removed, the theta block is the graded sewing contraction (6). Its arguments are the representation parameters, the plumbing variables, and the spin-gluing signs.” Replace the long prose operator string by a short permutation diagram or a parity formula with a defined ordering.

### P12. Double-Virasoro embedding — lines 125–143; Eqs. (8)–(11)

**Explanation.** Adding a fermion of central charge `1/2` permits two commuting linear combinations of the physical stress tensor, the auxiliary stress tensor, and a mixed `psi G` current. Their sum is the total stress tensor.

**Definite corrections.** Remove the comma in “the key fact ... is that, the.” Break the sentence that runs across Eq. (8). Correct “satisfiy.” Define `Q=b+b^(-1)` and `c=3/2+3Q^2`; line 143's `c=1+6Q^2` is the incompatible bosonic convention. For a complex unrestricted `b`, the excluded set is `b=0` and `b^2=1`, not just `b=0,1`. Alternatively specify a positive-real domain before excluding 1.

**Verified result.** The two displayed central charges add to `2+3Q^2=c+1/2`. The generator formulas likewise give `L_n^(1)+L_n^(2)=L_n+L_n^F`. These are compact consistency checks worth stating. Explain that the tensor product is graded, as signaled by `{psi,G}=0`, and define the normal-ordering/zero-mode convention in `L_n^F`.

**Application gap.** The physically important `b=1`, `c=27/2` value makes the individual Virasoro charges and generators singular. The final superconformal quantities can conceivably have a regular limit, but that requires a demonstrated cancellation and a stable evaluation procedure. The draft supplies neither. This is a limitation of the present formulas, not a proof that such a limit cannot be constructed.

**Proposed opening:** “The graded tensor product of the superconformal algebra and an auxiliary Majorana fermion contains two commuting Virasoro algebras. With `Q=b+b^(-1)` and `c=3/2+3Q^2`, their generators are (10), and their central charges satisfy `c^(1)+c^(2)=c+1/2`.”

### P13. Branching rules, momenta, and levels — lines 145–155; Eqs. (12)–(13)

**Explanation.** Each enlarged representation splits into products of Virasoro modules. The half-integer or quarter-shifted label `n` determines the primary weight in each factor and the extra propagation level. Ramond branches also carry a parity multiplicity.

**Audit.** Fix the graded tensor-factor order: this paragraph writes physical state first, while the later construction writes auxiliary state first. Such a swap can carry a sign. Define the auxiliary Ramond zero-mode action on `u^0,u^1`, their parities, and their pairings. “Generic momentum” should mean that the relevant modules and basis transformations are nonsingular; state exclusions or a limiting prescription for null-vector loci. Distinguish the Ramond parity index `alpha=0,1` here from the earlier physical ground-state label `alpha=+,-`. Introduce an unambiguous notation for these different roles.

**Verified result.** Eq. (13) satisfies `h_n^(1)+h_n^(2)=Q^2/8-P^2/2+2n^2`. In NS this is `h+2n^2`. In R, using `P^2=2 beta^2` and the corrected central charge, it is `h+1/16+(2n^2-1/8)`, exactly the physical-plus-auxiliary ground weight plus the claimed branch level. Thus the stated levels are consistent after the correction to `Q`. For an intrinsically odd NS primary, the branch parity is shifted by `p_1`; the current unqualified parity statement assumes an even ground state.

**Proposed addition:** “The branch level is measured relative to the ground state of the enlarged module. Consequently only finitely many branch labels contribute at any fixed total plumbing order.” This supplies the computational reason for presenting the branching rule.

### P14. Free-field realization and inversion — lines 157–167; Eqs. (14)–(15)

**Explanation.** Oscillators give explicit finite-level coordinates for the superconformal generators. One can then construct branch primaries and match their components numerically.

**Definite correction.** Add `(1-2 nu)/16` to the displayed `L_0`. In R, `G_0=-i P eta_0` on a ground state and `eta_0^2=1/2`, so `G_0^2=-P^2/2`. The printed `L_0-c/24` equals `-P^2/2-1/16`; adding `1/16` restores the superalgebra. The NS expression is unchanged.

**Misleading claim.** “For each ... mode ... one can directly solve them in terms of ...” sounds like a universal operator identity in the enveloping algebra. The computational statement is a representation-dependent change of basis at a finite level, generically invertible away from singular momenta. The code uses component matching/linear algebra. Say that explicitly. “Free fermion mode eta” should carry its index; distinguish these oscillators from the spin-gluing signs. State the vanishing mixed brackets and the action of the central momentum if not already fixed.

**Proposed wording:** “At generic momentum, expanding superconformal descendants in the oscillator basis gives an invertible change of basis at each level. We use this transformation to express the oscillator-built branch primaries in the superconformal basis.” The oscillator formulas are good candidates for End Matter, provided the main text retains what they enable and the relevant domain.

### P15. Explicit NS and Ramond primary construction, including the momentum footnote — lines 169–186; Eqs. (16)–(19)

**Explanation, in order.** The NS construction fills successive `chi` modes; their levels add to `2n^2`. Reflection provides the negative branch labels. The Ramond construction includes a zero mode and two ground-state choices; reflection must preserve the chosen physical ground-state convention.

**Definite omission.** The enlarged NS Fock space requires one boson and **two** fermions, `c`, `eta`, and `psi`. Line 169 lists only `c` and `psi`. Omitting `eta` gives the wrong character already at level `1/2`: the enlarged module has the two states corresponding to the physical and auxiliary fermion excitations.

**Missing domains and conventions.** State `v_0=phi tensor 1` and restrict the nonempty NS product to positive `n in (1/2)Z`; an ellipsis is not a definition at `n=0`. Explain reflection as an identification of isomorphic representation spaces, not an equality of unrelated oscillator vacua. Define `eta_r(P)` as the finite-level realization in the chosen module. In R specify the interpretation of `chi_0(-P)` acting in the fixed physical module, the auxiliary zero-mode normalization, and why the product order is as printed. For `n=1/4`, show the ground-state expressions explicitly to remove uncertainty in the ellipsis. Define the component projections in Eq. (19). Retain the sign until checked in the same BPZ convention; it is not a cosmetic freedom.

**Footnote repair.** “As the following” becomes “as follows”; “For `h` is greater” becomes “For `h>...`.” The proposed branch rule only orders real numbers; it does not define `P` for general complex `h,beta,Q`. State the real domain and separately choose a branch cut/analytic continuation for complex parameters. Include or explicitly exclude the threshold `P=0`. Also specify the sign relation between `P` and `beta`, because `P^2=2 beta^2` alone leaves a phase choice relevant to Ramond ground states.

**Proposed concise main-text wording:** “We construct the branch primaries explicitly in a free-field basis. Their oscillator expressions, reflection conventions, and finite-level change of basis are given in End Matter. These expressions determine all matrix elements needed by the recurrence.” Keep this shortening conditional on actually supplying those details.

### P16. Definition of the enlarged block — line 192

**Explanation.** The same sewing graph can be evaluated in the tensor-product representation rather than in the physical superconformal representation alone.

**Grammar.** “A enlarged” becomes “an enlarged”; “defined similarly with” becomes “defined analogously to.”

**Missing specification.** Replacing the state space and Gram matrix does not uniquely define the enlarged block: one must also define its chiral forms `hat rho`, the tensor-factor ordering, their graded factorization into physical and auxiliary forms, and the tube phases. `hat rho` is used later without an active definition. The production code includes an additional first-tube sign `(-1)^(A+mathsf A)` in the enlarged block and `(-1)^mathsf A` in the auxiliary block. These are not explained by saying only “replace the Grassmann signs.” The mathematical object must be defined before its identities are asserted.

**Proposed repair.** Give a compact formula for `hat rho` on homogeneous tensor-product states, including its Koszul sign, and state the first-tube convention explicitly. Use that definition to derive Eq. (21). Put the long component expansion in End Matter if necessary. The source-comment version does not count as a published definition.

### P17. Auxiliary block, convolution, and attempted recovery — lines 209–227; Eqs. (20)–(23)

**Explanation, in order.** The tensor-product basis separates physical and auxiliary descendants. Reordering fermions introduces a bilinear parity sign, so the separation is a graded convolution rather than an ordinary product in the current variables. The authors then attempt to undo the auxiliary factor by a formal inverse.

**Definition repairs.** Use “the latter are” if retaining plural “blocks,” or name “the auxiliary block.” State the oscillator basis normalization that lets Eq. (20) omit inverse metrics and primed descendant sums. Explain the otherwise unexplained `(-1)^mathsf A`. Use `rho_F` consistently in its normalization. The ordinary Virasoro representation of the auxiliary Majorana theory has central charge `1/2` and weights `0,1/2,1/16`; these are irreducible Ising representations with null states. Applying a generic Verma-module recursion therefore requires a quotient or an explicit regularization/cancellation prescription.

**Parity algebra.** The formal series are in the plumbing variables, with coefficients in a finite parity algebra. State `epsilon_e,delta_e in Z_2` and `eta_e^2=1`; the signs are not independent unrestricted power-series variables. Define the branch of `q_e^(1/2)` in an NS tube or use `t_e=q_e^(1/2)` as the series variable. Give one short derivation of Eq. (22): polarizing the sewing sign `K(epsilon)=sum_(i<j) epsilon_i epsilon_j` yields `K(epsilon+delta)-K(epsilon)-K(delta)=sum_(i<j)(epsilon_i delta_j+delta_i epsilon_j)` modulo two. This explains the cross term rather than simply announcing it.

**Definite failure of Eq. (23).** At zero descendant level, Eq. (20) has only the `(mathsf b,mathsf c)=(0,0)` and `(1,1)` terms. The former contributes 1. For the latter, the sewing sign is -1 and the squared three-point value is `i^2=-1`, so it contributes `eta_2 eta_3`. Therefore `F_F,0=1+eta_2 eta_3`. Under Eq. (22), `(eta_2 eta_3) star (eta_2 eta_3)=1`, hence `(1-eta_2 eta_3) star F_F,0=0`. A regular formal series inverse requires an invertible constant coefficient. No such inverse exists in the stated algebra.

**What must change.** Correct “define the of the convolutive inverse,” but do not merely substitute fluent prose around the same invalid inverse. Retain the forward identity and supply a mathematically specified recovery problem: for example, division in invertible projected components plus sufficient complementary data, or a proved injectivity statement on the actual admissible physical subspace. Projection alone recovers only that projection, not the full block. Allowing Laurent series or changing auxiliary insertions would be a different prescription and requires justification. Evaluating the existing signs at a single assignment before the star product is not automatically valid division. This is a required new argument, not a verified repair supplied by this review.

**Safe wording:** “The tensor-product basis gives the forward relation (21). Recovering the physical block requires resolving the kernel of multiplication by the auxiliary block.” A successful prescription must then replace Eq. (23) before the abstract can claim recovery of arbitrary physical blocks.

### P18. General-channel convolution — line 229

**Explanation.** Different ordered sewing graphs produce different fermionic permutation signs. The general construction should calculate these signs from the graph.

**Grammar and style.** Delete “We should point out that.” Use “analogous convolution formulas exist,” or “an analogous convolution formula exists.” Replace the empty assurance with a computable prescription.

**Missing derivation.** State the ordering of edge endpoints and vertex slots, the parity associated with every half-edge including external states, the graded local tensor-product signs, and how their product changes under a basis choice. A polarization of the global permutation sign is part of the answer, but vertex conventions can contribute additional cocycles; the theta formula cannot simply be declared universal. Include a second channel with a different ordering as an explicit check and explain how spin-gluing redundancy is respected. The presence of external punctures is a further change, since their auxiliary states and normalization must be specified.

**Proposed main-text statement after deriving it:** “For an ordered sewing graph, the convolution sign is obtained from the fermionic permutation taking edge-pair order to vertex order, together with the local tensor-product signs. This rule applies independently of genus; [specified equation] gives the general construction.” Until this is available, the claim remains an extrapolation from the theta example.

### P19. Double-Virasoro branch sum and mixed branching coefficient — lines 231–244; Eqs. (24)–(25)

**Explanation.** In the alternative basis, each tube propagates a pair of Virasoro descendants. Summing those descendants produces two ordinary blocks; the remaining factors are primary propagation shifts, parity signs, and normalized three-point coefficients at the vertices.

**Notation corrections.** Explicitly sum `alpha_2,alpha_3 in {0,1}` in Eq. (24). Their parity constraint does not itself specify the summation domain. Show their dependence in both branching-coefficient factors, consistently with Eq. (25). Define `h_(e,n)^(a)` using edge momentum `P_e`. Define the meaning of `v_(1,n_1)^(p_1)`; earlier the NS branch primary was simply `v_n`. Distinguish intrinsic primary parity from the branch parity shift. Standardize the notation for `F_f^(eta,eta')`, its arguments, and semicolons. Use “the Virasoro block `F(...)` has ...” instead of “`F(...)` are ...” if discussing a single function.

**Normalization and derivation.** Define `||v||^2` as the chosen BPZ bilinear pairing, including square-root branches if using normalized coefficients. Explain why the product of the two vertex normalizations reproduces the inverse primary Gram factor. Orthogonality or inversion of the Ramond multiplicity pairing cannot be assumed merely from a direct-sum character identity. State the normalization `F=1+...` and its suppressed plumbing variables. Explain the additional `(-1)^(2n_1)` by the enlarged-block convention in P16, rather than presenting it as part of an unexplained sign.

**Finite algorithm.** With `ell_NS(n)=2n^2` and `ell_R(n)=2n^2-1/8`, a total cutoff `N` keeps only branch triples with `sum_e ell_e<=N`; the two Virasoro descendant orders together must not exceed `N-sum_e ell_e`. This makes the infinite branch sum a finite computation and prevents an incorrect independent cutoff in each factor.

**Recovery limitation.** The concluding “One can then use Eq. (23)” must be replaced by the corrected recovery procedure from P17. Eq. (24) by itself computes the enlarged block. It does not settle the kernel of the forward map.

### P20. The two vertex types of branching coefficient — lines 248–251; Eq. (26)

**Explanation.** Every allowed pair of pants is either all NS or mixed NS–R–R, so two families of normalized primary three-point coefficients suffice for sewing.

**Corrections.** “Which corresponds” becomes “which correspond.” Name “the all-NS coefficient” and “the mixed coefficient” instead of repeatedly saying “the ... one.” Define `hat rho_a` just as carefully as the mixed form; it is not defined by Eq. (26). Define `p_1,p_2,p_3` and the parity-selection rule for the all-NS form. The hats, intrinsic primary parities, edge momenta, and branch parities should agree with P13 and P19.

**Proposed wording:** “The two allowed vertex types require all-NS and mixed NS–R–R branching coefficients. Both are normalized three-point forms of the double-Virasoro primaries, as in Eqs. (25) and (26).” Add the corresponding selection rule, which determines which entries actually need to be computed.

### P21. Branch-action coefficients — lines 253–262; Eq. (27)

**Explanation.** The physical generators `L_1` and `L_-1` map a double-Virasoro primary into a small set of neighboring branch modules. Expanding these images in ordinary Virasoro descendants supplies the coefficients needed in a Ward-identity recurrence.

**Language.** Use “The all-NS branching coefficients have been obtained analytically ...” and “there exist coefficients ... .” Replace “both only contain” by “lie in ...” or “receive contributions only from ... .” “Can be solved directly by acting” should become “are determined by expanding ... and matching oscillator components.” The categorical historical claim “has never been computed analytically” should either be supported by a careful literature statement or be replaced by the positive description of the new recurrence.

**Verified level bookkeeping.** For positive `n`, `ell(n)-ell(n-1)=4n-2`; applying `L_1` subtracts one and therefore gives descendant degree `4n-3`, while applying `L_-1` adds one and gives `4n-1`. The degree-one term in the same branch is also consistent. These checks justify the displayed degree labels but do not prove the asserted absence of every other branch module. Cite or supply that support theorem. Explain empty sums and low-label exceptions.

**Computational repair.** State the actual linear problem: expand `L_(+/-1)v_n` and the candidate double-Virasoro descendants in a common finite oscillator basis, assemble a matrix from the latter, and solve for its coordinates. Distinguish exact identities from numerical solves. State rank assumptions, normalization, numerical precision and a residual criterion. Code evidence supports least-squares/QR-style component matching, not a symbolic expression for each individual oscillator operator. Include the `alpha` dependence consistently in the prose notation for `mathbb V`.

**Proposed short addition:** “The coefficients in Eq. (27) are obtained by finite-level component matching in the oscillator basis. At generic parameters the resulting basis matrix has full column rank; we monitor the residual of the solved decomposition.” Only assert the rank statement as a theorem if proved, or qualify it as checked over the reported range.

### P22. Ward recurrence, seeds, and branching benchmark — lines 264–272; Eq. (28)

**Explanation.** Global conformal Ward identities move physical `L` generators between the three insertions. Substituting the decompositions in Eq. (27) relates unknown primary three-point coefficients to coefficients at neighboring branch labels. Seed data close the system.

**Definite correction.** These shifts do **not** lower total branch level by one. For the branch moved toward the origin, the propagation-degree change is `4|n|-2`, with low-label exceptions. Replace “which becomes recursion relations that relates ... with those with 1 lower total level” by a precise recurrence and its decreasing ordering.

**Seeds and closure.** Intersect the stated seed interval with the appropriate lattices: NS labels are `{-1/2,0,1/2}` and R labels are `{-3/4,-1/4,1/4,3/4}`. Include every allowed intrinsic parity, Ramond multiplicity, and three-point-structure label. Show at least one recurrence equation or finite linear system after descendant Ward reduction. State when its denominator/matrix is invertible and how singular parameters are handled. A boundary box plus an identity is not by itself a proof of a closed recursion.

**Second-identity issue.** Eq. (28) contains the physical `L_0 v_(2,n_2)`. A branch primary is an eigenvector of `L_0+L_0^F`, not generally of `L_0` alone. Thus substituting its total double-Virasoro weight would be wrong. Supply its expansion if the second identity is needed. The production README describes closure using the first identity; if that is sufficient, remove the unused second identity and show the actual closure instead.

**Benchmark repair.** “The coefficients ... takes ... on a laptop ... on a laptop” should become “Computing ... takes approximately ... .” Explain what the eight configurations enumerate; `f`, vertex signs, ground parities, and tube signs are different labels. Report hardware, parameter point, precision, inclusion/exclusion of basis construction and caching, and an accuracy measure. The claimed one-second value was not independently reproduced in this review. Do not replace it by an invented timing. Separate the algorithm, validation, and timing into distinct sentences or a compact table.

### P23. Block checks and runtime — line 274

**Explanation.** Independent implementations are intended to verify the new construction, while the runtime indicates its practical advantage over direct descendant sewing.

**Grammar.** “Section ... and ...” becomes “Secs. ... and ...”; “results ... passes” becomes “results ... pass”; “coefficients ... matches with” becomes “coefficients ... agree with.” Avoid reusing `n_e` for descendant exponents after using it for branch labels; use `r_e` or `N_e`.

**Evidence correction.** The authoritative local benchmark record is [`../../Code/full_ramond_block_runtime/README.md`](../../Code/full_ramond_block_runtime/README.md). It computes the enlarged `Fhat`, through total level 10 at `b=7/5`, momenta `(11/23,13/29,17/31)`, `f=0`, and both vertex signs positive. The recorded cold runtime is 82.9134 seconds, of which 75.0996 seconds builds branch-action decompositions; with branching data already available the expansion takes 4.5961 seconds. The document identifies Python 3.14.3 on macOS arm64, but that alone is not a CPU specification.

**What the low-level check establishes.** The recorded check compares the double-Virasoro enlarged series with `F_F star F_PBW` through total level 6, over 1,120 parity-resolved entries. Its maximum absolute/scaled discrepancy is about `9.4e-10` for even intrinsic NS parity and `4.1e-10` for odd parity. This is an independent **forward-identity** check. It does not establish a unique inverse or a separately recovered physical series. A smaller-cutoff versus larger-cutoff comparison in the record reaches about `1.18e-7`; therefore avoid an unqualified implication of exact or uniform precision. These numbers are read from the existing record, not newly timed runs.

**Proposed replacement:** “Through total level six, the double-Virasoro expansion of the enlarged block agrees with an independently sewn product of the physical PBW block and the auxiliary block, with maximum discrepancy [reported measure]. At the benchmark parameter point, constructing the enlarged level-ten series takes [specified runtime and hardware].” Retain the all-NS level-ten agreement only with an identifiable result file, parameters, and error measure; this review did not newly certify that assertion. A small validation table would carry more information than “passes many cross-checks.”

### P24. Cross-channel partition-function check — lines 276–280; Eq. (29)

**Explanation.** A physical partition function should be independent of the pants decomposition used to evaluate it. Dividing by a reference theory with a matching Weyl-anomaly power can remove the dependence on the fiducial metric, allowing comparisons across sewing frames.

**Verified rationale.** A free scalar plus Majorana fermion has central charge `3/2`. With the corrected super-Liouville convention, `c_SL/(3/2)=1+2Q^2`, explaining the exponent in Eq. (29). Say this; otherwise the ratio looks arbitrary.

**Missing conditions.** Explain whether the two sewings are related by a change of homology basis and how their period matrices and spin characteristics are matched. Two different NS/R assignments in a fixed marking are not automatically the same spin surface. State the nonchiral completion, three-point constants, internal-momentum integrals, zero-mode/volume normalization of the free boson, and the choice of spin structure. The denominator can vanish for fermion-zero-mode sectors, so its domain and any regularization matter. Specify the branch if a noninteger power is used. State how the `b=1` limitation, if relevant to the reported check, is handled.

**Language and evidence.** “Parameters ... which gives” becomes “plumbing data ... that describe.” Attach “computed from our blocks” unambiguously to the numerator or to the full construction. Give at least one common moduli point, the two truncation orders, integration accuracy, and the relative difference. “Agrees” with details promised only in unavailable Supplemental Material is not a reproducible result. The latest source already replaces the earlier “two  channels” by “two plumbing channels”; that resolved copy issue is not an outstanding edit.

**Proposed wording:** “We compare two pants decompositions of the same marked spin surface. The ratio (29), whose exponent matches the central charges of the two theories, removes the common Weyl-anomaly factor. At [specified moduli and spin data], the two evaluations agree to [measured accuracy].” Every bracket requires actual author-supplied evidence.

### P25. Discussion: summary — line 288

**Explanation.** The summary returns to the mechanism: double-Virasoro branching and recursively computable vertex coefficients.

**Corrections.** “Develop the algorithm” should be “present an algorithm” or “developed an algorithm.” “At arbitrary genus, in arbitrary plumbing channel, with arbitrary spin structure” needs consistent determiners/plurals. Avoid “coefficients given by branching coefficients.” The currently demonstrated reduction is for the enlarged block; recovery must not disappear from the summary.

**Proposed wording:** “We have constructed a double-Virasoro expansion of the enlarged superconformal block and a numerical recurrence for its mixed NS–R–R branching coefficients.” Add the full all-genus physical-block outcome only once P17–P18 establish it. A separate summary paragraph may be unnecessary if it merely repeats the abstract; use those words for a limitation or a concrete outcome.

### P26. Discussion: string amplitudes and matrix duals — line 290

**Explanation.** Higher-genus matter blocks can be combined with the rest of the worldsheet construction to compute loop amplitudes. Those amplitudes test established type-0 matrix descriptions and may constrain a still-unknown heterotic dual.

**Grammar and style.** “Our algorithm provide” becomes “Our algorithm provides.” Use “two-dimensional type 0A and 0B string theories.” “That are proposed recently” becomes “recently proposed,” or remove the time-relative adjective. “Find the potential matrix quantum mechanical dual” promises too much; use “constrain a possible matrix quantum mechanics dual.”

**Scientific qualifications.** Matter blocks alone do not provide the ghost measure, supermoduli/PCO prescription, spin-structure sum, or integrations. State these necessary additional ingredients in one compact clause. Distinguish the type-0 **matrix quantum mechanics** duals from the super-Virasoro-minimal-string **matrix integrals**. The heterotic dual remains an outlook, not an established consequence. Include the `b=1` limiting requirement in the method before implying that these amplitudes are already numerically accessible. Cite the 2005 heterotic-background paper for the theory and the 2023 Balthazar–Rodriguez–Yin discussion for the amplitude/MQM motivation.

**Proposed wording:** “Combined with the ghost measure and supermoduli integration, and after controlling the `b=1` limit where required, these blocks could extend worldsheet tests of the type 0A/0B matrix-quantum-mechanics dualities and their T-duality. They could also test matrix-integral predictions for super-Virasoro minimal strings and constrain a possible matrix quantum mechanics dual of the two-dimensional `SO(23)` heterotic string.” Retain only the caveats that remain unresolved after revision; do not imply an already completed amplitude calculation.

**Added integration sentences in the latest snapshot, reviewed next.** The intended workflow is to sample surfaces using period data and evaluate blocks after converting to plumbing coordinates. Correct “one sample ... and convert” to “one samples ... and converts.” Identify the actual “c=1 paper” and “ribbon paper”; “similarly to [footnote]” does not identify either a publication or a method.

The proposed workflow needs a genus and a domain. Ordinary period matrices do not encode the positions of external punctures, spin characteristics, or odd supermoduli. At genus zero the period matrix is empty even though four-point moduli are nontrivial. For closed genus `g>=4`, symmetric period matrices have `g(g+1)/2` complex entries while curve moduli have dimension `3g-3`, so samples must satisfy Jacobian-locus constraints; arbitrary positive-imaginary-part matrices are insufficient. See [Grushevsky, The Schottky Problem](https://arxiv.org/abs/1009.0369). The inverse period-to-plumbing map must select a channel, marking, and convergent patch, and use the correct transformed integration measure. The local genus-two ribbon routine is a nonunique numerical section, not an arbitrary-genus inverse theorem.

Ordinary period data also do not by themselves implement superstring integration. See [Witten's account of super Riemann surfaces](https://arxiv.org/abs/1209.2459); [Donagi–Witten](https://arxiv.org/abs/1304.7798) in particular rule out a generally projected supermoduli description for `g>=5`. This does not prohibit numerical integration with a suitable prescription; it means that such a prescription is additional work.

**Proposed replacement for the added workflow:** “Together with the remaining worldsheet contributions and a prescription for odd-supermoduli integration, these blocks could support numerical string-amplitude calculations. At genus two, the bosonic integration could combine period-matrix sampling with numerical inversion of the plumbing map [verified references], while retaining the puncture and spin data.” For a shorter PRL discussion, retain only the first sentence and defer the integration scheme to the actual amplitude work.

### P27. Discussion: modular bootstrap — line 292

**Explanation.** Higher-genus partition functions depend on three-point data as well as the spectrum, so equality under changes of pants decomposition yields constraints beyond genus-one spectral constraints.

**Audit.** The sentence is grammatical. Make “modular covariance” precise for the vector of spin-structure partition functions, since mapping-class transformations can mix spin structures and chiral blocks can carry anomaly factors. Name the quantities constrained, such as NS/R spectra and allowed three-point coefficients. Cite Cho, Collier, and Yin by name rather than saying only “the analysis of [number].” Their work supplies the bosonic genus-two precedent, not a completed supersymmetric bootstrap.

**Proposed wording:** “Mapping-class covariance of higher-genus partition functions, with their spin structures tracked explicitly, could constrain NS and Ramond spectra and three-point coefficients, extending the genus-two modular bootstrap of Cho, Collier, and Yin.” The [cited paper](https://arxiv.org/abs/1705.05865) is the appropriate precedent. If the discussion must remain a single PRL paragraph, merge P25–P27 after removing repeated summary language. A 3D-supergravity outlook is optional; it is not present in this snapshot and is not required to repair it.

### P28. Acknowledgments and AI disclosure — line 296

**Explanation.** This paragraph records discussions, funding, and substantive AI assistance with literature and computation.

**Language.** Correct the new sentence to “Y. Wang and X. Yin thank the Yukawa Institute for Theoretical Physics for its hospitality during this work.” Use “was used” for completed work. Make the expanded AI-use list parallel: “drafting and editing, literature searches, algorithm implementation, and numerical testing.” Verify the exact product/model name and version against the actual usage record rather than assuming “ChatGPT 5.6 Sol” is the correct externally identifiable label.

**Responsibility and policy.** The existing disclosure is valuable; do not remove it for stylistic reasons. APS's current AI policy asks authors to disclose substantive use, identify the tool/version and role, and explain human direction, review, and verification. Research-method assistance should be described in the methods as applicable, with appropriate acknowledgment. Add an author-verification sentence only if it accurately describes what the authors did; never invent a claim that every AI-produced result was independently checked. Funding/name accuracy requires author confirmation and cannot be established from prose alone.

**Proposed wording, conditional on accuracy:** “We used [verified tool and version] to assist with drafting and editing, literature searches, algorithm implementation, and numerical testing under the authors' direction. The authors reviewed the generated material and verified [specific outputs by specified independent checks].” Keep the named discussions, hospitality, and grant acknowledgments, subject to normal author verification.

### References and active footnotes — rendered page 6

1. Replace the joke in reference/note [17] by a real Supplemental Material citation describing its contents. This is a publication-content problem, not a LaTeX-syntax complaint.
2. Repair the momentum-branch note [18] as detailed in P15. Its current real-parameter inequalities do not specify a complex branch.
3. In the latest `references.bib`, lines 320 and 333, the author is stored as `Johnson, V., Clifford`, which renders as “C. Johnson, V.” The intended author is Clifford V. Johnson; correct the author metadata to the same name as the adjacent Johnson reference. This is a bibliographic-identity correction.
4. The rendered list omits article titles and displays several JHEP items without a visibly separate year. Audit the final output against APS's current reference examples, ensuring that journal, volume/year, and article number are unambiguous. Missing titles are a useful completeness/style improvement; do not label every omitted title a mandatory PRL violation without a specific applicable rule. Preprints without journal publication should remain identified as preprints.
5. Replace the newly added “c=1 paper” and “ribbon paper” notes (rendered [30] and [31] in the updated PDF) with the exact verified references. These shorthand notes do not identify publications. The bootstrap citation moves to [32]; the bibliography keys remain the reliable identifiers.
6. Attribute each claim to the right reference: ordinary all-genus Virasoro recursion; NS superconformal recursion; Ramond sphere/torus results; the embedding and branching; established type-0 MQM duals; conjectural super-VMS matrix descriptions; and the open heterotic-MQM outlook are different claims. A citation cluster should not be used to imply they have all been proved in the same scope.

## 4. Verification notes and limits

### Algebraic checks performed

The following identities were checked by symbolic simplification, using the formulas printed in the draft rather than silently importing corrected code:

```text
c^(1)+c^(2) = 2+3Q^2
h_n^(1)+h_n^(2) = Q^2/8-P^2/2+2n^2
printed Ramond L0-c/24 = -P^2/2-1/16
Ramond ground G0^2 = -P^2/2
2n^2-2(n-1)^2 = 4n-2
```

The central-charge sum requires the superconformal convention. The ground-state identity proves the missing Ramond shift. The level identity supports the branch propagation exponents and disproves the asserted unit decrement.

The auxiliary constant term and its annihilator were checked directly from Eqs. (20) and (22), and independently against the local star-algebra implementation. In the eight-component parity basis the constant vector is `[1,0,0,0,0,0,1,0]`; its multiplication map has four zero spectral components. The technical reviewer also checked the locally generated auxiliary series through total level two and found the same four components absent at each generated monomial. The constant-term obstruction alone suffices to disprove the regular formal inverse claimed in the draft; the finite-level experiment is supporting evidence, not an all-order kernel proof.

The code explicitly documents the issue in [`../../Code/double_virasoro/nsrr/nsrr_genus2_block.py`](../../Code/double_virasoro/nsrr/nsrr_genus2_block.py), lines 15–19: it treats the forward identity as unambiguous and says that a full formal quotient needs additional spin-structure data. The runtime README quoted in P23 agrees with that scope.

### Checks not claimed

This review did not rerun the full level-ten timing, every reported level-six PBW comparison, the all-NS level-ten benchmark, or the super-Liouville cross-channel momentum integrations. It did not prove an arbitrary-graph recovery theorem or a controlled `b=1` limit. It inspected the definitions, source-level implementation scope, recorded numerical checks, and elementary algebraic consequences, and identifies where further mathematical or numerical work is needed. No proposed replacement turns an unverified result into an asserted fact.

All six PDF pages were visually inspected. Pages 3 and 5 contain dense formulas near column boundaries; page 4 has conspicuous vertical gaps separating Eq. (21) from its explanation. The missing graph and the joke appear in the rendered PDF. These are readability/content observations, with no proposed LaTeX-syntax changes.

## 5. Presentation debate and recommended order of revision

Three independent reviewers covered language, mathematical/implementation consistency, and APS/PRL requirements. They exchanged challenges rather than merely producing parallel lists. The principal reviewer checked the manuscript, algebraic identities, key code records, and official policy sources and reconciled their findings here.

**Debate 1: shorten technical sections or preserve self-containment?** The language reviewer proposed deleting the roadmap and repeated summary, and moving the oscillator construction out of the core. The PRL reviewer objected to moving the foundations of the claimed all-genus result into Supplemental Material. Resolution: remove duplicated narration first; keep the usable graph/recovery rule, assumptions, recurrence closure, and quantitative evidence in the core. End Matter can hold explicit oscillator formulas and supporting proofs. The central recovery gap must be resolved, not relocated.

**Debate 2: can one simply divide after fixing the tube signs?** The PRL reviewer asked whether restricting to assignments with a nonzero auxiliary constant would fix the inverse. The technical reviewer demonstrated that ordinary evaluation of the original signs is not generally a character of the twisted algebra. Rephasing basis elements by `T(e_epsilon)=(-1)^K(epsilon)e_epsilon`, with `K=epsilon_1 epsilon_2+epsilon_1 epsilon_3+epsilon_2 epsilon_3`, untwists the product. Character evaluation of the transformed auxiliary constant gives `1-s_2 s_3`; four transformed components have constant 2 and four have constant 0. The former permit formal projected division, but that is not a construction of all physical components. No production implementation of this proposed projected recovery was verified. Resolution: the safe demonstrated output remains enlarged blocks and their forward-factorization checks; retain the broader title only after sufficient recovery data and proof are supplied.

**Debate 3: should “first” or “efficient” be added to strengthen the PRL case?** Direct descendant sewing already defines a finite algorithm, and the benchmark does not compare arbitrary physical-block recovery with existing methods. Resolution: identify the new mixed branching construction and the exact computational object rather than adding unsupported priority or performance adjectives. A measured comparison is stronger than “first,” “easily,” or “many cross-checks.”

**Debate 4: can convention-heavy signs be simplified for readability?** Changes to Ramond phases, the intrinsic NS parity, tensor ordering, or BPZ pairings can propagate throughout the formulas. Resolution: simplify notation and explain the conventions, but do not change those phases without checking the entire sewing definition in one convention.

**Concrete revision order:**

1. Resolve the physical-block recovery problem, the generic/singular-parameter scope, and the object actually validated. Decide the title and abstract only after that scope is fixed.
2. Correct the definite formula errors in P08, P11, P12, and P14; fix the graded conventions and missing definitions in P09–P19.
3. Supply a finite recurrence with seeds and closure, then an exact general-graph prescription if the general claim remains.
4. Replace qualitative validation claims by a compact table of objects, parameters, orders, errors, and timing conditions. Supply the actual cross-channel dataset if that claim remains.
5. Remove the roadmap and repeated exposition; move oscillator details and long support derivations into End Matter. Add the graph with a caption that fixes its conventions.
6. Apply the language edits, shorten and qualify the outlook, repair references/placeholders, and prepare accurate data/software and AI disclosures.
7. Recount the completed PRL core after the actual figure, caption, and missing explanations have been inserted.

## 6. APS/PRL compliance audit, excluding LaTeX syntax

Policies were checked on 8 September 2026. The current draft is **not submission-ready**, principally because of its unresolved reconstruction claim, missing supporting material, and incomplete evidence. This is not a finding that its six-page PDF automatically exceeds the word limit.

| Item | Current assessment | Proposed action |
|---|---|---|
| Core length | Current PRL limit: **3,750 equivalent words**. Latest estimate: **3,539**, before the missing figure/caption. | Remove redundancy and reserve space for the actual theorem and evidence. Recount the completed paper. [APS length guide](https://journals.aps.org/authors/length-guide) |
| Abstract | Approximately **403 rendered characters**, one paragraph, below the PRL **600-character** limit. | Formal length is acceptable; improve the scientific framing as in P01. [APS style guide, p. 11](https://res.cloudinary.com/apsphysics/image/upload/APS_Journals_Style_Guide_Authors022326_guzwhx.pdf#page=12) |
| End Matter | PRL permits **up to two pages**, excluded from the core limit. None is currently provided. | Use it for oscillator constructions, reflection details, and support proofs. [PRL author guidance](https://journals.aps.org/prl/authors) |
| Self-containment | PRL requires a paper that is convincing without Supplemental Material. | Keep the actual recovery statement, its hypotheses, and the role of the general graph rule in the paper; do not outsource the decisive gap. [PRL Supplemental Material guidance](https://journals.aps.org/prl/authors#supplemental-material) |
| Significance/accessibility | Editorial acceptance is not certifiable by this review. The promised generality is not yet established in the inspected material. | Lead with the Ramond obstacle, demonstrated advance, and an independent numerical result. [PRL scope and criteria](https://journals.aps.org/prl/about) |
| Section presentation | Six numbered freestanding headings differ from PRL's run-in heading style. | Use brief italic run-in topic headings and American “Acknowledgments.” This concerns visible journal presentation, not source syntax. [APS style guide, p. 15](https://res.cloudinary.com/apsphysics/image/upload/APS_Journals_Style_Guide_Authors022326_guzwhx.pdf#page=16) |
| Figure | Only a placeholder exists. | Supply the theta graph, ordered labels, and a self-contained caption. Meaning should not depend on color alone. [APS style basics](https://journals.aps.org/authors/style-basics) |
| Supplement | The text promises general formulas and numerical checks, but the reviewed draft folder contains no submission-ready supplement. | Provide it with a descriptive main-list citation. References used in the supplement must also appear in the main bibliography. [PRL author guidance](https://journals.aps.org/prl/authors) |
| References | Johnson's name is malformed; several rendered JHEP citations lack the year; three content-reference placeholders remain. | Correct those locators and placeholders. Consistently omitted article titles are allowed; PRL encourages including them. [PRL reference guidance](https://journals.aps.org/prl/authors) |
| Reproducibility | Key parameter points, error measures, benchmark conditions, and the claimed mixed-channel comparison are not available in the paper. | Identify the exact scripts/output and state what was checked; place extensive logs outside the core. [APS reproducibility policy](https://journals.aps.org/authors/editorial-policies#reproducibility) |
| Data/software availability | APS requires a statement in published articles and collects it at submission; absence from this source alone is not proof of a submission violation. | Prepare a truthful statement for the original code, coefficients, and benchmark data. Cite any public archive. Sharing is strongly encouraged, not a universal unconditional mandate. [APS data/software guidance](https://journals.aps.org/authors/data-availability-statements) |
| AI disclosure | Assistance is already disclosed, including the latest drafting/editing addition. | Verify the tool/version and document author direction/verification. Research use belongs with methods as applicable; other substantive assistance belongs in acknowledgments. Do not invent verification work. [Current APS AI policy](https://journals.aps.org/authors/appropriate-use-ai-tools) |
| Submission justification | PRL requests a **100-word justification** at submission. No such submission material was inspected. | Prepare it after the actual scope is settled. It is not an extra paragraph required inside the manuscript. [PRL submission guidance](https://journals.aps.org/prl/authors) |
| Author/funding declarations | Names, affiliation, and funding are present. | Authors verify approvals, identities, grant details, and hospitality themselves. Their correctness cannot be inferred from this audit. |

### How the latest length estimate was obtained

The raw TeXcount result is misleading because the custom display macros are not recognized as equations. Only a **temporary counting copy** expanded those macros; the draft itself was not changed. The count covers Introduction through Discussion, includes content footnotes, and excludes title/front matter, abstract, acknowledgments, and references. Inline formulas are estimated as one word each; final production treatment can differ.

| Component | Latest count |
|---|---:|
| Prose words | 2,218 |
| Heading words | 15 |
| Footnote/caption-category words | 65 |
| Inline mathematical expressions | 217 |
| Single-column displayed rows | 46 |
| Double-column displayed rows | 9 |
| Displayed-math equivalent, `46*16 + 9*32` | 1,024 |
| **Estimated equivalent core total** | **3,539** |

APS counts single-column display rows as 16 words and double-column rows as 32. Figures and tables also consume equivalent words, in addition to their captions. [APS counting rules](https://journals.aps.org/authors/length-guide)

The resulting nominal margin is **211 words**. A square single-column figure alone counts as approximately 170 words under the APS formula, before the caption. Tall formulas and final reflow introduce uncertainty, so report the present length as approximately **3.54 thousand equivalent words**, not an APS-certified total. The required mathematical repairs will also use space. The earlier 3,484-word estimate was superseded by the additions during this review.

### Scope of the policy conclusions

The length cap, abstract cap, and required disclosures are distinguishable from editorial judgments about importance or clarity. A bibliography-title preference is not a mandatory violation; a missing submission-only justification is not a manuscript error; a missing figure that the text explicitly promises is a concrete incompleteness. The newer dedicated APS AI policy governs the AI-use recommendations where an older style-guide passage differs. No journal acceptance, funding compliance, or author approval is claimed.

## 7. Completion record

- All 28 active prose paragraphs were reviewed and explained in order, including equation continuations and the additions at lines 290 and 296.
- The title, front matter, four content footnotes, and updated reference list were checked; commented-out passages were excluded from claims of self-containment.
- Grammar/usage corrections and optional style changes are separated from mathematical corrections and unverified claims.
- Three independent reviewers exchanged concrete challenges; their disagreements and resolutions are recorded above.
- The relevant equations were checked algebraically; the production implementation's documented output was distinguished from the draft's claimed output. Full expensive numerical benchmarks were not rerun.
- Current official APS/PRL guidance was checked, with approximate counting and publication/submission requirements distinguished.
- The reviewers' only persistent project addition is this review file in the new review folder. The original paper, bibliography, PDF, and code were not edited by the reviewers.
