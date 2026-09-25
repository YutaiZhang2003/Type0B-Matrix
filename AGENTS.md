# Working conventions for superconformal blocks

This file records the **current** convention for work on the higher-genus
superconformal-block algorithm. It supersedes older convention passages in
dated machine notes and experiment reports. Keep historical results as
historical evidence; do not silently transplant their phases or timings into
the current calculation. The derivation and directed checks are in
`Machine Notes/bpz_one_over_z_sewing_2026-09-25.tex`.

## BPZ, plumbing, and local forms

- At the first slot of a three-point form use the BPZ inversion
  \(I(z)=1/z\) with spin lift \(\lambda_I(z)=-i/z\). An edge uses
  \(I_q(z)=q/z\) with lift \(\lambda_{I_q}(z)=-i\sqrt q/z\). Take
  \(\sqrt{-1}=i\) and the analytic branch of \(\sqrt q\) fixed by the
  plumbing coordinate. The edge map is the BPZ map followed by scaling
  by \(q\), inducing \(q^{L_0}\); there is no separate edge half-turn
  or BPZ conversion.
- BPZ is linear and graded:
  \((AB)^\dagger=(-1)^{|A||B|}B^\dagger A^\dagger\),
  \(L_n^\dagger=L_{-n}\), \(G_r^\dagger=iG_{-r}\), and
  \(\psi_r^\dagger=-i\psi_{-r}\). Thus \(L_n^\dagger=L_n\) only at
  \(n=0\). The bilinear pairing obeys
  \(\langle Ax,y\rangle=(-1)^{|A||x|}\langle x,A^\dagger y\rangle\).
- With the paper's \(G_0w^\pm=i\beta e^{\mp i\pi/4}w^\mp\) and
  \(\psi_0u^a=u^{1-a}/\sqrt2\), the physical and auxiliary Ramond
  ground metrics are respectively \(\operatorname{diag}(1,-1)\) and
  \(\operatorname{diag}(1,-i)\). Inverse Gram matrices always mean the
  inverses of these BPZ pairings. The ordered three-point ground values
  in the long paper, including \(\rho_1^{(\eta)}(\phi,w^-,w^+)=i\eta\)
  and \(\rho_{\mathsf F}(\mathbf1,u^1,u^1)=i\), remain unchanged for
  even NS primaries. For an odd first-slot NS primary the BPZ ground
  norm and ordered ground vertex acquire \(i\). A vertex value is not
  a Gram entry.
- Keep the existing \(\sqrt{z(z-1)}\) contour branch. Use the *new*
  four fermionic Ward identities in the September 25 derivation; the
  Virasoro Ward identity is unchanged. Do not use the earlier
  \(G_r^\dagger=G_{-r}\), \(\psi_r^\dagger=-\psi_{-r}\) Ward solver.

## Enlarged blocks and convolution

- Order each tensor state auxiliary first. Its BPZ pairing is
  \(\langle a\otimes x,b\otimes y\rangle
    =(-1)^{|a||x|}\langle a,b\rangle_{\mathsf F}
      \langle x,y\rangle_{\mathsf{SCA}}\). Sew with the inverse of this
  **full graded tensor Gram matrix**. Do not append a separate
  \((-1)^{\epsilon'_e}\) edge factor.
- The ordered enlarged vertex is
  \(\widehat\rho(a_1\otimes x_1,a_2\otimes x_2,a_3\otimes x_3)
   =(-1)^{|a_1||x_1|+|x_2||a_3|}
     \rho_{\mathsf F}(a_1,a_2,a_3)\rho(x_1,x_2,x_3)\).
  The first sign is BPZ dualization at infinity; the second reorders
  the middle physical state past the third auxiliary state. Any
  oscillator-to-ordered-form conversion is an internal basis change,
  not an additional three-point constant.
- On an arbitrary ordered graph, the mixed convolution exponent is
  \(K_\Gamma(\epsilon+\epsilon')+K_\Gamma(\epsilon)
   +K_\Gamma(\epsilon')+\sum_e\epsilon_e\epsilon'_e
   +\sum_v(\epsilon_{v_1}\epsilon'_{v_1}
            +\epsilon_{v_2}\epsilon'_{v_3})\pmod2\).
  The direct free-fermion factor includes its own BPZ inverse Gram and
  \(\eta_e^{\epsilon'_e}\); the old \((-\eta_e)^{\epsilon'_e}\)
  shorthand is not the current definition. Recover the physical block
  coefficientwise in the allowed sector.
- For a \(\Theta\psi\) insertion, construct the enlarged and auxiliary
  blocks on the **split graph** with its two edge parameters and spin
  signs. Use the same BPZ pairing and vertex rule. Only then extract
  equal left/right plumbing powers and identify their product with the
  original edge parameter. Do not invent split parameters inside a
  convolution defined solely on the unsplit graph.
- The two commuting Virasoro algebras, their CCY recursion, and the
  Schottky seed keep their algebraic form. Branching norms and vertices
  are evaluated with the current BPZ pairing. Recheck any closed
  branching-norm or phase formula inherited from older notes before use.

## Evidence and manuscript boundary

- The directed new-convention tests include local Ward identities,
  BPZ/tensor Gram checks, direct PBW versus double-Virasoro
  coefficients through total level three with and without the odd-loop
  insertion, and ten fixed-spin cross-channel comparisons at the
  stated source and target cutoffs. Read the precise residuals in the
  September 25 derivation; do not turn them into claims at higher levels.
- Yutai's compared data are sphere four-point and torus two-point
  blocks. The theta-channel oracle is a separate local check, not a
  Yutai theta comparison. No complete integrated Type 0B amplitude was
  established by these block tests.
- The user has explicitly reserved edits to `draft/` and `draft-long/`.
  Do not change either manuscript without a new explicit instruction.
  The current paper is `draft-long/scblock_long.tex`; proposed changes
  to it should be given in chat when no edit is authorized.

## Reading older notes

- Machine Notes/arbitrary_plumbing_superconformal_blocks.tex and
  its included conventions and partition appendices are the
  comprehensive current derivation. Its numerical validation
  section separates September 25 BPZ-native checks from older
  test tables. Machine Notes/README.md gives the reading order.
  The separate Ramond timing note remains a historical frame
  record and is not an alternative current block definition.
- Machine Notes/conventions.md fixes physical BRY amplitude and
  matrix-model normalizations. Its older BRY-to-Human-Note sewing
  dictionary must be checked in the current chiral BPZ frame before
  use. Distinguish this physical normalization question from the
  auxiliary free-fermion sewing convention.
