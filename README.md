# Noncritical Type 0B / Matrix-Model Target

This standalone folder develops the duality between two-dimensional
noncritical type 0B string theory and the two-sided double-scaled matrix
quantum mechanics.  It also carries a frozen copy of the relevant genus-two
StringMC machinery, so that the Type 0B adaptation can evolve independently.

## Master convention

All scattering-state and sphere normalizations follow
Balthazar--Rodriguez--Yin (BRY),
[arXiv:2201.05621](https://arxiv.org/abs/2201.05621):

\[
\alpha'=2,\qquad
h=\frac12(p^2-x^2),\qquad
\omega_{\rm MM}=2\omega,\qquad
g_s=\frac4{\pi\mu_{\rm F}},\qquad
C_{S^2}=\frac{\pi}{g_s^2}.
\]

Here \(\alpha'=2\) means that lengths are reported in the BRY unit
\(\ell_{\rm B}=\sqrt{\alpha'/2}\), with \(\ell_{\rm B}=1\). It is not a
dimensionful equality. We reserve \(R_{\rm phys}\) for the physical circle
radius and use

\[
\rho=\frac{R_{\rm phys}}{\ell_{\rm B}},
\qquad
x=\frac{R_{\rm phys}}{\sqrt{2\alpha'}}=\frac{\rho}{2}.
\]

The type 0B fields are the NS--NS tachyon \(T\) and R--R axion \(A\). The
matrix-model left/right scattering modes are normalized as

\[
\mathcal L=T-A,\qquad \mathcal R=T+A,
\]

with no \(1/\sqrt2\) and no additional leg-pole factor in the BRY on-shell
basis. Both sides of the oscillator are filled through \(E=-\mu_{\rm F}\),
with \(\mu_{\rm F}>0\).

BRY is the convention and sphere-data source, not a completed higher-genus
calculation. It computes tree-level \(1\to2\) and \(1\to3\) amplitudes and
provides the \(b=1\) super-Liouville constants
\(C,\widetilde C,C_{\rm even},C_{\rm odd}\). The torus result of Douglas
et al. is used only as a later comparison; the genus-two measure still requires D'Hoker--Phong
gauge-fixing data, an odd-spin derivation, and a degeneration-compatible
PCO/vertical-integration prescription.

## First vacuum benchmark: modular path integral

After Wick rotation, \(X\sim X+2\pi R_{\rm phys}\). In BRY numerical units,
the three even spin structures give, per regulated Liouville coordinate
length,

\[
\widehat{\mathcal F}_{1,\rm even}
=\frac{3\rho}{16\pi}
\int_{\mathcal F}\frac{d^2\tau}{\tau_2^2}\Theta_\rho
=\frac\rho{16}+\frac1{8\rho}.
\]

The modular integral is evaluated by separating the zero orbit and unfolding
the nonzero orbits to the strip. It uses the convergent sum
\(\sum_{k\ge1}k^{-2}=\pi^2/6\), not zeta-function continuation at
\(s=-1\).

For the odd structure, direct integration over the global odd supertorus
modulus inserts \(G\bar G\), soaks the fermion and superghost zero modes, and
retains the \(\langle\partial X\bar\partial X\rangle\) contact term. With the
0B sign,

\[
\widehat{\mathcal F}^{0B}_{1,\rm odd}
=-\frac\rho{48}+\frac1{24\rho}.
\]

Adding even and odd structures gives

\[
\widehat{\mathcal F}_1^{0B}(\rho)
=\frac \rho{24}+\frac1{6\rho}.
\]

The two momentum towers and one winding tower in the BRST spectrum explain
the coefficients as a subsequent check.

BRY use the glyph \(\mu\) for both the Liouville interaction and matrix Fermi
depth. We distinguish \(\mu_{\rm L}\) and \(\mu_{\rm F}\). The BRY action
gives

\[
V_\phi^{\rm reg}\big|_{\rm univ}
=-\log(\mu_{\rm L}/\Lambda_{\rm L}),
\]

and hence

\[
{\mathcal F}^{0B}_1(\rho,\mu_{\rm L})\big|_{\rm univ}
=-\frac{\log(\mu_{\rm L}/\Lambda_{\rm L})}{24}
\left(\rho+\frac4\rho\right).
\]

Here the logarithm is understood as
\(\log(\mu_{\rm L}/\Lambda_{\rm L})\). Only in the comparison stage do we
write \(\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}\); the standard result tests
\(p=1\).

The stationary numerical radius is \(\rho=2\), equivalently
\(R_{\rm phys}=\sqrt{2\alpha'}\), where the coefficient of
\(\log(\mu_{\rm L}/\Lambda_{\rm L})\) is \(-1/6\).

As a T-duality cross-check, fluxless type 0A has one momentum tower and two
winding towers. In the same notation,

\[
\widehat{\mathcal F}^{0A}_1(\rho_A)
=\frac1{12}\left(\rho_A+\frac1{\rho_A}\right),
\qquad
\mathcal F^{0A}_1\big|_{\rm univ}
=-\widehat{\mathcal F}^{0A}_1
\log\frac{\mu_{\rm L}}{\Lambda_{\rm L}}.
\]

The physical map is
\(R_A=\alpha'/R_B\), hence \(\rho_A=2/\rho_B\), and
\(\widehat{\mathcal F}^{0A}_1(\rho_A)
=\widehat{\mathcal F}^{0B}_1(2/\rho_A)\).

## Folder contents

- `Code/`: the purpose-organized Type 0B Python project.  Central-charge
  recursion, fixed-weight recursion, double-Virasoro code, independent PBW
  cross-checks, and the frozen older SCFT genus-two snapshot have separate
  subfolders; see `Code/README.md` for the map.  The code filenames described
  below are basenames within those purpose folders.
- `Data Set/`: generated JSON/SVG outputs, archived cluster results, and
  retained visual-QA artifacts.  The large genus-two production dataset is
  distributed through Google Drive rather than Git; follow
  `Data Set/genus2/README.md` to download and verify it.
  `Code/genus_2_cross_channel/data` is a relative compatibility link so the
  frozen snapshot keeps its original data paths after extraction.
- `Machine Notes/conventions.md`: the authoritative BRY convention ledger, physical
  vertices, PCO, structure constants, and matrix dictionary.
- `References/literature_genus1_genus2.md`: source-to-formula map for the genus-one and
  genus-two ingredients.
- `Machine Notes/roadmap.md`: benchmark ladder and completion criteria.
- `Machine Notes/TORUS_MODULARITY_CODE_REVIEW.md`: review index and archived
  cluster-run ledger for the torus modularity checks.
- `Machine Notes/type0b_matrix_model_setup.tex` and its PDF: standalone
  research note.
- `Machine Notes/h-Recursion/`: private working derivation and build artifacts
  for the general $h$-recursion review.
- `Machine Notes/c-Recursion/super_zamolodchikov_recursion.tex` and its PDF:
  separate derivation note for
  the arbitrary-graph NS/R $c$-recursion, its plumbing and spin-structure
  ledger, the higher-genus vacuum-seed problem, and the convention-locked
  four-R sphere calibration.
- `Machine Notes/c-Recursion/ramond_channel_c_recursion.tex` and its PDF:
  fixed-$\beta$ sphere Ramond-channel $c$-recursion working note.
- `Machine Notes/c-Recursion/ramond_sphere_level4_exact_expansion.md`: exact
  symbolic long-R expansion accompanying the recursion implementation.
- baseline.json: executable \(\alpha'=2\) smoke-test configuration.
- benchmark_genus_one.py: analytic genus-one benchmark.
- test_benchmark_genus_one.py: convention, anchor, and radius checks.
- superconformal_blocks.py: BRY \(c\)-recursion and elliptic-\(q\)
  evaluation for all eight NS sphere four-point blocks.  The low-level
  `order` counts coefficients; `bry_elliptic_block(..., L, ...)` instead
  implements BRY's parity-dependent truncation through displayed order
  \(q^L\).
- two_virasoro_fusion.py: arbitrary-precision all-NS
  \(\mathsf{Vir}\oplus\mathsf{Vir}\) branching coefficients from the
  free-field Fermi-sea prescription, including the parity-dependent blow-up
  products, identity-specialized norms, and the canonical \(B_a^2\).
- test_two_virasoro_fusion.py: low-level product formulas, negative-label
  continuation, norm, trinion-ordering, and \(b=1\) complex-momentum checks.
- ns_genus2_three_way_symbolic_check.py: exact coefficientwise comparison of
  direct NS PBW sewing, NS c-recursion, and the two-Virasoro branching
  expansion with an ordinary Virasoro c-recursion in each factor. It follows
  the human note's ordinary product with the auxiliary fermion block: direct
  PBW and NS recursion agree, while the present diagonal double-Virasoro sum
  has a convention mismatch beginning at total level 3/2.
- test_ns_genus2_three_way_symbolic_check.py: branching-weight, ordinary
  series-division, SCA-recursion, and Virasoro (2,1)-kernel regressions.
- test_superconformal_blocks.py: BRY ancillary-notebook regression,
  endpoint-pole, and independent \(z\)-versus-\(q\) checks.
- ramond_sphere_blocks.py: Hadasz--Jaskolski--Suchanek elliptic recursion for
  four Ramond external primaries with NS exchange, translated to the
  ordinary-\(c\), \(\beta=iP/\sqrt2\) convention used by Xi Yin and BRY.
- ramond_c_recursive_sphere_blocks.py: legacy fixed-weight \(c\)-pole
  validation layer for RRRR and the NS-exchange channel of NSNSRR. It checks the
  \((3,1)\) and \((2,2)\) residues against an independent local-\(z\)
  extraction of the HJS recursion. This remains a local residue regression;
  the production Ramond \(c\)-recursion instead holds \(\beta\), not
  \(h_{\mathrm R}\), fixed.
- mixed_ns_ramond_descendant_blocks.py: independent NS PBW/Gram evaluator,
  closed NSNS three-form, and open-ground-index RRNS Ward matrices for the
  NSNSRR block with NS exchange. It reproduces both HJS sign branches
  directly through arbitrary practical low level.
- ramond_c_regular_seed.py: pole-subtracted regular part derived from that
  direct Ward/Gram evaluator. It completes the legacy low-level NSNSRR
  fixed-weight identity without using the HJS \(h\)-recursion. Its regular
  term is not the seed for the corrected fixed-\(\beta\) recursion.
- test_ramond_c_recursion.py: leading Ward/Gram anchors, all-sign RRRR
  \(c\)-residue tests, mixed NSNSRR residue tests, and the regular-seed
  safety guard.
- test_mixed_ns_ramond_descendant_blocks.py: direct-versus-\(h\)-recursion,
  completed \(c\)-recursion, and large-\(c\) global-seed power-counting
  checks.
- evaluate_ramond_sphere_four_point.py: numerical evaluation of all four HJS
  chiral sign branches at the regulated \(b=1\) benchmark point.
- test_ramond_sphere_blocks.py: \(G_0\)-phase, fusion-polynomial,
  level-\(1/2\), level-1, numerical-anchor, and cutoff checks for that layer.
- mixed_ramond_sphere_blocks.py: NS- and long-R-exchange elliptic recursions
  for the two-NS/two-R sphere block, including an independent level-one
  Ramond Gram/Ward evaluator.  The Ramond inverse-norm product uses the even
  Kac sublattice fixed by that direct check.  The long-R class deliberately
  rejects the shortened \(P=0\) ground state.
- ramond_descendant_blocks.py: independent long-R PBW bases, direct
  super-Virasoro Gram matrices, N--R--R Ward vectors, and descendant sewing
  through arbitrary low integer level.  It converts the resulting local
  \(z\)-series to the elliptic \(H(q)\) convention without using Kac
  recursion.
- compare_ramond_q3.py: exact-\(b=1\) coefficient ledger comparing direct
  long-R descendant sewing and collision-aware recursion for all four HJS
  sign components through \(q^3\).
- test_ramond_descendant_blocks.py: PBW-dimension, Gram-symmetry, analytic
  level-one, and all-sign \(q^3\) direct-versus-recursion tests.
- self_dual_superconformal_blocks.py: collision-aware finite-part projection
  of the assembled RRRR and RRNN elliptic coefficients at the exact Type-0B
  point \(b=1\), including the generic long-R exchange channel.  It uses the
  local uniformizer \(b=e^t\) and a two-radius Cauchy projection, avoiding a
  one-sided \(c=27/2+\epsilon\) evaluation.
- test_self_dual_superconformal_blocks.py: Laurent-projection, exact-\(c\)
  order-12, and internal-\(G_0\) level-one regressions.
- ramond_sphere_correlators.py: positive-momentum BRY assemblies for the
  physical RRRR correlator and the RRNSNS correlator in its NS and crossed-R
  channels.
- test_ramond_mixed_crossing.py: all-sign level-one direct-versus-recursion
  check and reduced-cost physical RRRR and RRNSNS crossing regressions.
- stress_rrrr_crossing.py: reproducible physical RRRR crossing scans using
  either the symmetric displaced-\(c\) diagnostic or the production
  coefficient-wise finite part directly at \(c=27/2\).
- stress_ns_crossing.py: independent \(s\)- and \(t\)-channel continuum
  decompositions of the four-bottom-component NS Liouville sphere correlator
  at \(\widehat c=9\), evaluated with the direct exact-\(c\) recursion and
  exact hypergeometric global blocks at its leaves.  It records recursion-order
  convergence and renders the final channel curves without imposing equality
  or truncating a local-\(z\) or elliptic-\(q\) series.
- stress_ns_torus_modularity_c_recursion.py: ordinary-NS torus one-point
  modularity from two independently assembled super-Liouville continuum
  integrals.  Each chiral block uses the functional central-charge recursion
  with exact hypergeometric global and infinite-product vacuum leaves, so the
  numerical order controls only nested Kac residues and never a \(q\)-series
  cutoff.  The default audit is the asymmetric point \(\tau=i/4\), where
  \(|q|=0.20788\) but \(|\widetilde q|=1.22\times10^{-11}\); the earlier scan
  in which both nomes were small is preserved explicitly as a calibration.
- superconformal_torus_blocks.py: genus-one NS and generic long-R toric
  recursions, lifted NS plumbing, explicit unnormalized Ramond ground-fiber
  matrices, cycle projections, and exact-\(c=27/2\) finite-part wrappers.
- evaluate_superconformal_torus_block.py: command-line coefficient ledger for
  the exact Type-0B NS- or R-handle one-point block.
- test_superconformal_torus_blocks.py: NS lift/character checks, explicit
  \(G_0\) ground sewing, direct-versus-recursive NS/R leading coefficients,
  HJS level-one anchors, and exact-\(c\) coefficient tests.
- superconformal_torus_descendants.py: independent Gram/Ward sewing through
  NS level \(1\) and long-R level \(1\), including the two-state Ramond
  descendant matrix.
- compare_superconformal_torus_leading.py: prints the low-level Gram and
  vertex matrices and compares their contractions with the recursion.
- superconformal_torus_two_point.py: bivariate NS necklace h-recursion with
  an explicit two-state fermion routing, independent NS plumbing lifts, and
  nested finite-part projections for both the self-dual \(b=1\) Kac
  collisions and coincident internal weights.  It also contains the
  two-edge long-R beta-pole recursion, its positive/negative Kac-pole sign
  routing, and a direct open-ground-fiber regular-seed oracle validated
  through level one on each R edge.  It also fixes the mixed NS--R
  ground-fiber contraction for two external Ramond punctures.
- superconformal_torus_blocks.py: includes
  `TorusTwoPointSpinStructure`, which validates the edge/puncture sectors
  and represents NS, NS-tilde, R, R-tilde, and all four mixed NS--R sewing
  assignments.
- super_liouville_torus_two_point.py: BRY-normalized
  \(dP_1\,dP_2/\pi^2\) spectral assembly of the NS h-recursive block, the
  direct leading NS block, or the sign-summed R-handle beta-recursive block.
- superconformal_torus_two_point_ope.py: complementary
  sphere-three-point--torus-one-point sewing channel.  It reuses the
  all-level NS/R torus one-point recursions on the handle and contracts an
  independent NS Ward/Gram bridge exactly through level one in the collision
  coordinate \(x=1-z\).
- super_liouville_torus_two_point_ope.py: double spectral assembly of the
  even NS three-form OPE contribution with either an NS or ordinary-R
  handle.  This is a pilot crossing channel; the odd bridge family and full
  spin-structure pairing are still required for a complete certificate.
- test_superconformal_torus_two_point_ope.py: bridge Ward identity,
  torus-one-point degeneration, NS/R handle factorization, cutoff, and
  exact-\(c=27/2\) wrapper checks.
- test_superconformal_torus_two_point.py: leading Ward identities, fermion
  parity, cyclic exchange, identity-puncture reduction to the torus
  one-point block, positive and negative R-pole residues, all nonzero
  Ramond sign/cycle sectors, self-dual collision stability, and
  recursive-correlator tests.
- super_liouville_structure_constants.py: the normalized \(b=1\)
  \(C\), \(\widetilde C\), and R--R--NS even/odd pants coefficients.
- sphere_four_point.py: the BRY nonchiral correlators \(G,H,J\), their
  exchanged-momentum integrands, and the reduced four-tachyon moduli density.
- bry_one_to_three.py: BRY's first regulated four-punctured-sphere benchmark,
  with the complex-energy family (4.15), the leading t-channel OPE
  counterterm, the three-region \(z\)-integration, an (h)-recursive default
  for the interior sphere block, and comparison with the matrix-model answer
  (2.13).
- recover_bry_one_to_three_h_recursion.py: production architecture check that
  performs the full regulated integral with (h)-recursion, audits generic
  block densities against (c)-recursion without using it in the integral,
  and writes the three boundary/bulk contributions separately.
- evaluate_sphere_four_point.py: command-line evaluation on a chosen
  cross-ratio grid.
- plot_bry_figure4.py: dependency-free reproduction of BRY's Figure 4
  $VWWV$ crossing test, comparing the direct $q^8$ reference to a selectable
  crossed-channel order ($q^{12}$ by default), with full and zoom panels.
  The default 24-point momentum quadrature is stable against a 32-point
  refinement at the few-parts-in-$10^7$ level for each curve at $z=0.01$.
  The blocks and the crossing exponent use the same regulated
  $c=27/2+10^{-5}$ convention.
- `../Data Set/bry_figure4_crossing.svg`: the rendered BRY Figure 4 comparison.
- test_sphere_four_point.py: structure-constant, high-precision threshold,
  pointwise-correlator, and crossing checks.
- test_bry_one_to_three.py: fast complex-energy, OPE, threshold, local-block,
  and amplitude-normalization checks for the regulated \(1\to3\) layer.
- requirements-sphere.txt: the sole third-party dependency of the sphere
  layer, pinned to the tested version.
- `../References/references.bib`: primary literature.
- `Code/genus_2_cross_channel/`: copied genus-two block, period-map, and Monte
  Carlo implementation, together with the production data needed for later
  adaptation.  See its `README.md` and `SNAPSHOT_MANIFEST.md` before
  modifying it.

Run the code from the workspace root with `Code` on `PYTHONPATH` so imports
can span the purpose folders:

    export PYTHONPATH="$PWD/Code${PYTHONPATH:+:$PYTHONPATH}"

The genus-one checks are:

    python3 -m unittest test_benchmark_genus_one
    python3 Code/benchmark_genus_one.py
    python3 -m unittest h_recursion.test_superconformal_torus_blocks
    python3 -m h_recursion.compare_superconformal_torus_leading

Inspect the exact-\(c=27/2\) NS- and R-handle coefficient ledgers:

    python3 -m h_recursion.evaluate_superconformal_torus_block --sector NS --internal-momentum 0.61 --external-momentum 0.33 --q 0.03 --order 2 --lift-sign -1
    python3 -m h_recursion.evaluate_superconformal_torus_block --sector R --internal-momentum 0.60 --external-momentum 0.33 --q 0.03 --order 3 --r-sign 1 --cycle-insertion identity

Evaluate the self-dual NS torus two-point block through level two on both
necklace edges:

    python3 -m h_recursion.evaluate_superconformal_torus_two_point --block-method h-recursion --spin-structure NS --max-twice-level-1 4 --max-twice-level-2 4

Flip the temporal fermion holonomy without changing the reduced plumbing
parameters:

    python3 -m h_recursion.evaluate_superconformal_torus_two_point --block-method h-recursion --spin-structure NS_tilde --max-twice-level-1 4 --max-twice-level-2 4

Assemble it into the truncated Type-0B two-point correlator:

    python3 -m h_recursion.evaluate_super_liouville_torus_two_point --internal-sector NS --max-twice-level-1 4 --max-twice-level-2 4 --quadrature-order 8

Evaluate the self-dual R-handle block and its sign-summed spectral
contribution through the currently validated level-one layer:

    python3 -m h_recursion.evaluate_ramond_torus_two_point --block-method beta-recursion --spin-structure R --max-level-1 1 --max-level-2 1
    python3 -m h_recursion.evaluate_super_liouville_torus_two_point --internal-sector R --max-level-1 1 --max-level-2 1 --quadrature-order 6

Inspect the mixed NS--R ground-fiber normalization for the twice-R-punctured
torus and any of its four spin assignments:

    python3 -m h_recursion.evaluate_mixed_torus_two_point --ns-lift -1 --r-cycle parity

Evaluate the complementary sphere--torus OPE block and its pilot even-form
spectral contribution.  Here \(z\) is the annulus-plane insertion position
and the bridge expansion parameter is \(x=1-z\):

    python3 -m h_recursion.evaluate_superconformal_torus_two_point_ope --handle-sector NS --z 0.8 --q 0.004 --max-bridge-twice-level 2 --max-handle-twice-level 8
    python3 -m h_recursion.evaluate_super_liouville_torus_two_point_ope --handle-sector NS --z 0.8 --q 0.004 --max-bridge-twice-level 2 --max-handle-twice-level 8 --quadrature-order 8

The handle recursion may be increased independently.  The bridge cutoff is
currently restricted to level one; higher levels need torus one-point Ward
amplitudes with general external NS descendants.

The two level cutoffs and the momentum cutoff/quadrature must be increased
independently.  The bridge recursion beyond level one requires the coupled
even/odd toric three-form seed; both NS- and R-handle production evaluators
therefore retain the verified direct bridge cutoff for now.

Run the sphere-block checks with:

    python3 -m unittest c_Recursion.test_superconformal_blocks h_recursion.test_ramond_sphere_blocks c_Recursion.test_ramond_c_recursion c_Recursion.test_mixed_ns_ramond_descendant_blocks h_recursion.test_ramond_descendant_blocks h_recursion.test_self_dual_superconformal_blocks

Compare the generic long-R block coefficient by coefficient through \(q^3\):

    python3 -m h_recursion.compare_ramond_q3

Evaluate the four-R sphere-block benchmark:

    python3 -m h_recursion.evaluate_ramond_sphere_four_point

Evaluate the BRY Figure 4 momentum assignment as a function of \(z\):

    python3 -m c_Recursion.evaluate_sphere_four_point --four-tachyon

Reproduce BRY's Figure 4 crossing plot:

    python3 -m c_Recursion.plot_bry_figure4

Evaluate the first nontrivial BRY \(1_R\to3_R\) benchmark at
\(\omega=1/3+0.6i\), \(\omega_i=\omega/3\):

    python3 -m c_Recursion.bry_one_to_three --block-backend h --p-order 24 --angular-order 14 --radial-order 14 --cap-angular-order 14 --cap-radial-order 10 --block-q-order 8

Run the matched-grid recovery check, including the independent pointwise
(h)-versus-(c) audit and the recorded-(q^8) regression gate:

    python3 c_Recursion/recover_bry_one_to_three_h_recursion.py

This product-quadrature implementation currently reproduces the complex
matrix-model coefficient to about 1.8% at that setting.  The \(q^6\) cutoff is
not converged: its displacement from \(q^8\) is \(4.08\%\) of the matrix-model
target.  The \(q^8\)-to-\(q^{10}\) control changes the result by only
approximately \(0.09\%\), while raising the moduli orders from 12 to 14 at \(q^8\) changes
it by \(0.074\%\).  Production therefore starts at \(q^8\); resolving the
narrow \(z=1\) boundary layer and the sign-changing final \(P\) integral is
the remaining numerical limitation.

Run the complete sphere test suite with:

    python3 -m unittest c_Recursion.test_superconformal_blocks h_recursion.test_ramond_sphere_blocks c_Recursion.test_ramond_c_recursion c_Recursion.test_mixed_ns_ramond_descendant_blocks h_recursion.test_ramond_descendant_blocks h_recursion.test_self_dual_superconformal_blocks c_Recursion.test_sphere_four_point c_Recursion.test_bry_one_to_three

## Scope boundary

The baseline is fluxless and symmetrically filled. Unequal fillings, finite
R--R flux, 0A/0B T-duality, nonperturbative contour choices, and a complete
genus-two integration cycle remain later milestones. In particular, the
sphere relation \(g_s=4/(\pi\mu_{\rm F})\) converts the expected power of
\(\mu_{\rm F}\),
but does not by itself determine the absolute genus-two vacuum normalization.
