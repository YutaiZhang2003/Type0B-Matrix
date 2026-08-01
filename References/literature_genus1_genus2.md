# Literature map: genus-one and genus-two type 0B amplitudes

Search date: 2026-07-22

## 0. Convention policy and scope of arXiv:2201.05621

Balthazar--Rodriguez--Yin (BRY),
[arXiv:2201.05621](https://arxiv.org/abs/2201.05621), is the master
convention for this project. In particular, use

\[
\alpha'=2,\qquad h=\frac12(p^2-x^2),\qquad
\omega_{\rm MM}=2\omega,\qquad
g_s=\frac4{\pi\mu_{\rm F}},\qquad C_{S^2}=\frac{\pi}{g_s^2},
\]

and

\[
\mathcal L=T-A,\qquad \mathcal R=T+A.
\]

Use their delta-normalized physical vertices, phases, PCO, and
\(C,\widetilde C,C_{\rm even},C_{\rm odd}\) without adding an independent
leg-pole factor.

Dimensionally, BRY's \(\alpha'=2\) means

\[
\ell_{\rm B}\equiv\sqrt{\alpha'/2},\qquad
\alpha'=2\ell_{\rm B}^2,\qquad [\ell_{\rm B}]_{\rm BRY}=1.
\]

We write \(R_{\rm phys}\) for the dimensionful circle radius and
\(\rho=R_{\rm phys}/\ell_{\rm B}\) for its BRY numerical value.

The paper prints \(\mu\) both in the super-Liouville interaction and as the
matrix Fermi depth. We write these as \(\mu_{\rm L}\) and
\(\mu_{\rm F}\). The genus-one derivation is carried out in
\(\mu_{\rm L}\); the relation
\(\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}\) is introduced only during the
matrix-model/literature comparison.

The scope must remain explicit: BRY verifies tree-level \(1\to2\) and
\(1\to3\) amplitudes. It does not contain a torus vacuum computation, a
genus-two free energy, or a completed higher-genus supermoduli measure; its
discussion lists higher-genus blocks and loop amplitudes as future work.
Thus the paper fixes the convention in which we formulate the desired
genus-one and genus-two checks, while the actual higher-genus measure comes
from the other sources below.

This note maps the ingredients needed for the fluxless two-dimensional type
0B vacuum amplitude to primary literature.  The central conclusion is:

- the genus-one spectrum and vacuum measure are already treated rather
  explicitly in the original 0B matrix-model paper, including the odd spin
  structure and its superghost zero modes;
- the genus-two calculation is not available as a finished noncritical-0B
  formula in the literature located here.  It must be assembled from
  super-Liouville structure constants and sewing blocks, a genus-two
  superstring measure, the diagonal type-0B spin-structure sum, and a
  prescription for the odd-spin sector.

## 1. Requirement-to-source map

| Required ingredient | Best starting source | What it supplies | Status for this project |
|---|---|---|---|
| BRY master normalization | Balthazar--Rodriguez--Yin, [arXiv:2201.05621](https://arxiv.org/abs/2201.05621), secs. 2--4 | \(\alpha'=2\), physical \(T/A\) vertices, PCO, delta-normalized Liouville states, \(h=(p^2-x^2)/2\), \(\omega_{\rm MM}=2\omega\), and sphere normalization | Authoritative convention for every imported formula |
| 2D type 0B spectrum | Douglas et al., [hep-th/0307195](https://arxiv.org/abs/hep-th/0307195), secs. 2.2--2.3 | GSO sectors, NS and R vertex operators, momentum/winding spectrum, discrete states and ground ring | Translate all state names and normalizations into the BRY ledger |
| Full BRST/discrete-state spectrum | Itoh--Ohta, [hep-th/9110013](https://arxiv.org/abs/hep-th/9110013) | NS and R BRST cohomology, including nonzero ghost-number discrete states | Needed for a factorization-complete state ledger |
| Type 0B torus measure | Douglas et al., [hep-th/0307195](https://arxiv.org/abs/hep-th/0307195), sec. 4.2 and app. A | Even-spin determinant cancellation, odd-spin \(\beta\gamma\) and fermion zero modes, direct integration of the odd supermodulus | Direct genus-one anchor |
| All-genus 0B GSO phase | Kaidi et al., [arXiv:1911.11780](https://arxiv.org/abs/1911.11780), sec. 3.1 | Type 0B as the diagonal spin-structure sum with no Arf sign; type 0A differs by \((-1)^{\mathrm{Arf}}\) | Fixes the genus-two spin sum conceptually |
| Complete sphere super-Liouville three-point data | Poghossian, [hep-th/9607120](https://arxiv.org/abs/hep-th/9607120); independently Rashkov--Stanishkov, [hep-th/9602148](https://arxiv.org/abs/hep-th/9602148) | NS--NS--NS and R--R--NS structure constants and reflection amplitudes | Foundational derivations |
| Type-0B-normalized \(b=1\) constants | Balthazar--Rodriguez--Yin, [arXiv:2201.05621](https://arxiv.org/abs/2201.05621), sec. 3.1 | Implementation-ready \(C,\widetilde C,C_{\rm even},C_{\rm odd}\), two-point normalization, and \(\Upsilon_{\rm NS/R}\) conventions | Required numerical convention |
| NS super-Virasoro blocks | Belavin et al., [hep-th/0703084](https://arxiv.org/abs/hep-th/0703084); Belavin, [arXiv:0705.1983](https://arxiv.org/abs/0705.1983); Suchanek, [arXiv:1012.2974](https://arxiv.org/abs/1012.2974) | Elliptic recursions, including blocks involving top components | Building blocks for sewing |
| Ramond super-Virasoro blocks | Hadasz--Jaskolski--Suchanek, [arXiv:0810.1203](https://arxiv.org/abs/0810.1203) | Elliptic recurrence for four-point blocks with external Ramond states and NS factorization | Needed for Ramond channels; not itself a genus-two vacuum block |
| Supermoduli and PCO geometry | Witten, [arXiv:1209.2459](https://arxiv.org/abs/1209.2459) and [arXiv:1209.5461](https://arxiv.org/abs/1209.5461) | Super-Riemann-surface moduli, low genus, integration cycles, geometric picture changing | Foundational framework |
| Genus-two even-spin measure | D'Hoker--Phong, [hep-th/0110283](https://arxiv.org/abs/hep-th/0110283) and [hep-th/0211111](https://arxiv.org/abs/hep-th/0211111) | Slice-independent chiral measure, super-period-matrix projection, and a formula for general \(c=15\), \(N=1\) matter | Structurally applicable; Liouville correlators must replace flat-space matter data |
| Vertical integration | Sen, [arXiv:1408.0571](https://arxiv.org/abs/1408.0571); Sen--Witten, [arXiv:1504.00609](https://arxiv.org/abs/1504.00609) | Local PCO sections, avoidance of spurious poles, and corrections on cell boundaries and corners | Practical fallback/consistency prescription |

## 2. Genus one

### 2.0 Douglas worldsheet route translated to BRY variables

The primary genus-one derivation now follows Douglas et al., sec. 4.2 and
appendix A, while retaining the BRY field and radius ledger. Write
\(y=R_{\rm phys}/\sqrt{\alpha'}=\rho/\sqrt2\). The three even structures
give

\[
\widehat{\mathcal F}_{1,\rm even}
=\frac{3\rho}{16\pi}
\int_{\mathcal F}\frac{d^2\tau}{\tau_2^2}\Theta_\rho
=\frac\rho{16}+\frac1{8\rho}.
\]

The lattice integral is evaluated by modular-orbit unfolding:

\[
\int_{\mathcal F}\frac{d^2\tau}{\tau_2^2}\Theta_\rho
=\frac\pi3+\frac4{\pi\rho^2}\sum_{k\ge1}\frac1{k^2}
=\frac\pi3\left(1+\frac2{\rho^2}\right).
\]

This uses an absolutely convergent sum. It does not use the formal
zeta-regularized zero-point-energy expression discussed later in Douglas et
al. For the odd structure, direct integration of the global odd modulus and
the associated supercurrent contact term gives

\[
\widehat{\mathcal F}^{(\varepsilon)}_{1,\rm odd}
=\frac{\varepsilon}{24\sqrt2}\left(y-\frac1y\right)
=\varepsilon\left(\frac\rho{48}-\frac1{24\rho}\right).
\]

Douglas et al. fix its magnitude by the large-radius spectrum. The
massless-field count fixes \(\varepsilon_B=-1\) and
\(\varepsilon_A=+1\). The complete densities are therefore

\[
\widehat{\mathcal F}^{0B}_1(\rho_B)
=\frac1{24}\left(\rho_B+\frac4{\rho_B}\right),
\qquad
\widehat{\mathcal F}^{0A}_1(\rho_A)
=\frac1{12}\left(\rho_A+\frac1{\rho_A}\right).
\]

The spectrum multiplicities are retained as checks, not as the regulated
definition of the answer. Independently, shifting the BRY Liouville action
gives

\[
V_\phi^{\rm reg}
=-\log(\mu_{\rm L}/\Lambda_{\rm L})+V_{\rm scheme}
\qquad (b=1).
\]

Thus

\[
\frac{\partial\mathcal F_1^{0B}}
{\partial\log\mu_{\rm L}}
=-\frac1{24}\left(\rho+\frac4\rho\right).
\]

Douglas et al., Eq. (4.8), write the dimensionful result as

\[
\mathcal F^{0A}_1\big|_{\rm univ}
=-\frac{\log(\mu_{\rm L}/\Lambda_{\rm L})}{12\sqrt2}
\left(\frac{2R_A}{\sqrt{\alpha'}}
+\frac{\sqrt{\alpha'}}{R_A}\right).
\]

Their type-changing T-duality is
\(R_A=\alpha'/R_B\), or \(\rho_A=2/\rho_B\), and maps the 0A expression
to the 0B expression exactly.

Their Liouville kinetic normalization agrees with the BRY coordinate at
\(\alpha'=2\), and their \(V_{\rm D}=-\log|\mu_{\rm D}|\) agrees with the
BRY wall displacement up to an additive cutoff constant and a constant
rescaling of the interaction parameter.

The translation ledger is:

| Convention | Coordinate/parameter | Universal regulated length |
|---|---|---|
| BRY action | \(\phi,\ \mu_{\rm L}e^{b\phi}\) | \(-b^{-1}\log(\mu_{\rm L}/\Lambda_{\rm L})\) |
| Rescaled source | \(\Phi=a\phi,\ \mu_{\rm L}=\kappa\lambda^p\) | \(-(ap/b)\log\lambda\) |
| Douglas et al. at \(\alpha'=2\) | same kinetic coordinate, \(\mu_{\rm D}\) | \(-\log|\mu_{\rm D}|\) |
| Matrix model | Fermi depth \(\mu_{\rm F}\) | no geometric volume; use \(\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}\) only at comparison |

Accordingly, bare quantities called \(V_L\) are never compared directly.
The invariant comparison object is the logarithmic derivative after the
field and coupling map has been stated.

### 2.1 Spectrum

At \(b=1\), \(Q=2\), the worldsheet matter theory is the free \(N=1\)
time multiplet together with \(N=1\) super-Liouville theory.  Type 0B keeps

\[
(\mathrm{NS}+ ,\mathrm{NS}+)
\oplus(\mathrm{NS}-,\mathrm{NS}-)
\oplus(\mathrm{R}+ ,\mathrm{R}+)
\oplus(\mathrm{R}- ,\mathrm{R}-).
\]

For generic momentum the asymptotic propagating spectrum contains the NS--NS
tachyon and the R--R axion.  On the circle, 0B retains R--R momentum modes
and projects out R--R winding modes; 0A does the converse.  Discrete BRST
states occur at special momenta.  They are not needed to reproduce the
ordinary thermodynamic interpretation of the universal torus logarithm, but
they must be restored in a factorization-complete state sum and in any audit
of resonant/contact contributions.

Use two complementary sources:

1. Douglas et al., secs. 2.2--2.3, for the physical 0B spectrum in the same
   conventions as the matrix-model proposal.
2. Itoh--Ohta for the systematic NS/R BRST cohomology and its discrete
   states.  Di Francesco--Kutasov,
   [hep-th/9109005](https://arxiv.org/abs/hep-th/9109005), is useful background
   for the relation between worldsheet discrete states and 2D string
   scattering.

### 2.2 Even spin structures: combined ghost measure

Section 4.2 of Douglas et al. gives a compact determinant audit.  If
\(D_{r,s}\) is the fermion determinant divided by the square root of the
scalar determinant, the three factors in an even spin structure are

\[
Z_X=\frac{y}{\sqrt{\tau_2}}
 |D_{r,s}|^2\sum_{m,n}e^{-S_{m,n}},
\qquad
S_{m,n}=\frac{\pi\rho^2}{2\tau_2}|n-m\tau|^2,
\]

\[
Z_{\rm SL}=\frac{V_{\rm D}}{2\pi\sqrt{2\tau_2}}|D_{r,s}|^2,
\qquad
Z_{\rm sgh}=\frac{1}{2\tau_2}|D_{r,s}|^{-4}.
\]

Thus the oscillator determinants cancel and the modulus integral has the
standard \(d^2\tau/\tau_2^2\) form.  The paper denotes the last factor as the
superghost contribution; before coding, its normalization should be unpacked
into the \(bc\) and \(\beta\gamma\) determinants in one fixed convention.
The combined formula above is the normalization anchor that the unpacked
calculation must reproduce.

An unpunctured even-spin supertorus has moduli dimension \(1|0\), so no PCO
is required.  There is one bosonic modulus and the corresponding \(b\)-ghost
pairing, together with division by the conformal Killing translation.

### 2.3 Odd spin structure: zero modes and the odd supermodulus

The odd torus is the essential subtlety.  Douglas et al., appendix A, finds:

- one \(\gamma\) and one \(\bar\gamma\) zero mode from the conformal Killing
  spinors;
- one \(\beta\) and one \(\bar\beta\) zero mode from the gravitino components
  that cannot be gauged away;
- zero modes for both the matter and Liouville Majorana fermions;
- integration over the \(\beta,\bar\beta\) zero modes inserts
  \(G(z)\bar G(\bar w)\);
- the Liouville-fermion zero modes cancel the \(\gamma,\bar\gamma\) zero-mode
  divergence, while the supercurrents soak the remaining matter-fermion zero
  modes.

The calculation reduces to the matter insertion
\(\langle\partial X(z)\bar\partial X(\bar w)\rangle\), including its contact
term.  Equivalently, the odd supertorus is written globally as

\[
(z,\theta)\sim(z+1,\theta)
\sim(z+\tau+\lambda\theta,\theta+\lambda),
\]

and the single odd modulus \(\lambda\) is integrated directly.  This agrees
with Witten's low-genus description: the odd supertorus has one odd
deformation, with a jumping automorphism at the split locus.

The result is the indispensable odd-spin contribution

\[
\frac{Z^{0B}_{1,\rm odd}}{V_{\rm D}}
=-\frac{\rho}{48}+\frac{1}{24\rho},
\]

with the 0B sign.  Combining it with the three even structures gives

\[
Z^{0B}_1\big|_{\log}
=-\frac{\log(\mu_{\rm L}/\Lambda_{\rm L})}{24}
\left(\rho+\frac4\rho\right).
\]

This is now a comparison result. The odd-spin sign is fixed by the 0B GSO
choice: it raises the \(1/\rho\) coefficient to that of two massless fields
and lowers the \(\rho\) coefficient. It vanishes at \(\rho=\sqrt2\),
equivalently \(R_{\rm phys}=\sqrt{\alpha'}\).

### 2.4 Does genus one require vertical integration?

For this vacuum amplitude, the cleanest answer is: use the global
supermoduli treatment above.  Even spin structures have no odd modulus, and
the odd structure has a global one-odd-parameter supertorus description.
There is therefore no separate vertical-integration correction to add to
the Douglas et al. result.

If the same calculation is reformulated with a local PCO, or if punctures
and off-shell continuations are introduced, the PCO section must reproduce
the direct odd-modulus answer, including the contact term.  Sen--Witten then
provides the correct patching prescription.  Vertical integration is a way
to represent the supermoduli integral consistently, not an additional
physical term on top of a correct direct supermoduli integral.

## 3. Genus two

### 3.1 Moduli and spin structures

For an unpunctured genus-\(g\ge2\) super-Riemann surface,

\[
\dim\mathfrak M_g=(3g-3\,|\,2g-2).
\]

Hence each chiral half at genus two has dimension \(3|2\).  A PCO gauge
therefore uses two holomorphic PCOs and two antiholomorphic PCOs.  Genus two
has ten even and six odd spin structures.

The oriented type-0 theories gauge the diagonal worldsheet fermion parity,
so the left and right spin structures are correlated.  Kaidi et al. express
the two choices as

\[
Z_{0(n)}\propto\sum_\sigma(-1)^{n\,\mathrm{Arf}(\sigma)}
Z_L[\sigma]Z_R[\sigma],
\qquad n=0,1.
\]

Type 0B is \(n=0\): all diagonal spin structures have the same topological
phase.  Type 0A is \(n=1\).  The overall gauging normalization and any
orientation conventions must still be fixed by factorization to the torus
anchor.

### 3.2 The required super-Liouville three-point data

There is not a single three-point coefficient.  For NS fields one needs the
two independent superconformal structures

\[
C(P_1,P_2,P_3),\qquad \widetilde C(P_1,P_2,P_3),
\]

corresponding, in component language, to bottom- and top-component
couplings.  For two Ramond fields and one NS field one needs the two
independent combinations

\[
C_{\rm even}(P_1,P_2;P_3),\qquad
C_{\rm odd}(P_1,P_2;P_3),
\]

or equivalently the \(C_\pm\) combinations appropriate to the two 0B Ramond
families.  Correlators with an odd number of Ramond fields are not allowed on
the sphere, so these NS--NS--NS and R--R--NS vertices are the pants vertices
needed in a genus-two sewing description.

Poghossian and Rashkov--Stanishkov are the original bootstrap sources.  For
implementation at the physical 0B point \(b=1\), use the normalization and
explicit \(\Upsilon_{\rm NS/R}\) formulae in sec. 3.1 of
Balthazar--Rodriguez--Yin.  Their equations (3.4) and (3.9) give all four
functions above, normalized so that the two-point functions are delta
functions in \(P\) and the constants are real for \(P_i\ge0\).

The PCO supercurrents couple to top components and descendants.  Keeping
only the bottom-component constant \(C\) is therefore insufficient even in
a purely NS channel.

### 3.3 Three-point constants are necessary but not sufficient

A genus-two CFT integrand also needs the descendant contractions, or
equivalently genus-two super-Virasoro conformal blocks, in each allowed
sewing channel.  The located literature provides mature sphere four-point
recursions in the NS sector and Ramond-sector recursions for important
classes of blocks, but not a turnkey genus-two vacuum block for this
noncompact theory.

A practical route is to implement plumbing-fixture sewing directly:

1. choose a trivalent pants decomposition;
2. sum/integrate over NS or R internal representations consistent with the
   spin structure;
3. build descendant Gram matrices and three-point descendant matrix
   elements from the super-Virasoro algebra;
4. contract the two pants vertices as a power series in the three plumbing
   parameters;
5. validate the result under changes of pants decomposition and in all
   separating and nonseparating degenerations.

The NS recursions of Belavin et al. and Suchanek, and the Ramond analysis of
Hadasz--Jaskolski--Suchanek, are benchmarks for this sewing engine rather
than complete replacements for it.

### 3.4 Even-spin measure: super period matrix versus naive PCOs

D'Hoker--Phong construct a slice-independent genus-two chiral measure by
projecting with the super period matrix and integrating the two odd moduli.
Their general \(c=15\), \(N=1\) matter formula has the schematic structure

\[
d\mu_{\mathcal C}[\delta]
=\frac{Z_{\mathcal C}[\delta]}{Z_M[\delta]}
\left[
\frac{\Xi_6[\delta]\,\vartheta[\delta]^4}{4\Psi_{10}}
-\mathcal Z\,\langle S_{\mathcal C}(q_1)S_{\mathcal C}(q_2)\rangle
\right]\frac{d^3\Omega}{4\pi^6},
\]

in split gauge \(S_\delta(q_1,q_2)=0\).  Here \(\mathcal C\) is the matter
SCFT and the formula also contains the finite-dimensional \(b\)- and
\(\beta\)-ghost determinants through \(\mathcal Z\).

This is the best starting point for the even-spin 0B sector because the
time-plus-super-Liouville system is an \(N=1\), \(c=15\) worldsheet SCFT.
However, the closed theta-functional answer for flat ten-dimensional matter
cannot simply be copied.  One must insert the noncompact time and interacting
super-Liouville partition functions and their supercurrent correlator, keep
the Liouville zero-mode regulator, and recheck degeneration boundary terms.

D'Hoker--Phong also explain why the old expression made only from

\[
S(q_1)\delta(\beta(q_1))\,
S(q_2)\delta(\beta(q_2))\,
\prod_{a=1}^{3}b(p_a)
\]

is incomplete: its PCO collision singularity is cancelled by an equal and
opposite singularity in the associated finite-dimensional determinants.
This cancellation must be retained in any implementation.

### 3.5 Odd spin structures are a separate work package

The standard explicit D'Hoker--Phong genus-two measure is an even-spin
construction.  Type 0B cannot discard the six odd structures merely because
critical type-II vacuum amplitudes often do: the odd torus already
contributes essentially in 2D type 0B.  Generic odd genus-two spin structures
carry fermion zero modes, and the two supercurrent insertions produced by the
odd-modulus integral can soak them.

The literature located here does not provide a finished odd-spin,
super-Liouville genus-two vacuum measure.  This sector should therefore be
treated as a named research problem:

- count matter, Liouville, \(bc\), and \(\beta\gamma\) zero modes for each
  diagonal odd spin structure;
- derive the odd-modulus insertions directly on supermoduli space;
- determine all contact terms and Liouville zero-mode contributions;
- verify both separating degeneration to two tori and nonseparating
  degeneration to a punctured torus;
- only then translate the result to a PCO/vertical-integration description.

### 3.6 What vertical integration means at genus two

With two PCOs per chiral half, a global smooth choice avoiding spurious poles
need not exist.  Sen--Witten prescribe:

1. cover ordinary moduli space by cells with safe local choices
   \((y_1,y_2)\) of PCO positions;
2. integrate the ordinary PCO measure over each cell;
3. on each codimension-one shared face, add a vertical segment that moves
   the PCO configuration between the two local choices at fixed bosonic
   moduli;
4. on codimension-two intersections, add the corresponding two-dimensional
   vertical cell so that the face corrections close consistently.

The hierarchy terminates at codimension two because there are two PCOs.
PCO choices and vertical chains must also be compatible with plumbing
degenerations.  This construction is not an extra correction to the
super-period-matrix integral; it is an alternative representation of the
same integration over odd directions when a global projection is not used.

## 4. Recommended calculation order

1. **Genus-one spectrum ledger.** Encode the continuous NS/R states, circle
   momentum/winding projections, reflection identification \(P\sim-P\), and
   discrete BRST states in a machine-readable table.
2. **Genus-one measure audit.** Reproduce the three even determinant factors
   separately as \(bc\), \(\beta\gamma\), matter, and Liouville pieces, then
   reproduce appendix A's odd-spin answer by direct odd-modulus integration.
3. **Super-Liouville data module.** Implement the \(b=1\) functions
   \(C,\widetilde C,C_{\rm even},C_{\rm odd}\) in the 2022 normalization and
   test permutation, reflection, reality, and special-momentum limits.
4. **Genus-two sewing prototype.** Start near the separating degeneration,
   where factorization onto two genus-one components gives the strongest
   normalization check.  Build the NS channel before adding Ramond channels.
5. **Even-spin measure.** Adapt the D'Hoker--Phong general-\(c=15\) formula,
   preserving its finite-dimensional determinant corrections.
6. **Odd-spin measure.** Derive it independently rather than extrapolating
   the even-spin formula.
7. **Vertical-integration cross-check.** On a finite cell decomposition,
   verify invariance under changing local PCO sections, including both face
   and corner terms.
8. **Matrix-model comparison.** Compare only after the separating and
   nonseparating degeneration residues, GSO phase, and Liouville-volume
   normalization are fixed without fitting.

## 5. Main gap statement

The literature supplies all sphere three-point constants and a powerful
even-spin genus-two gauge-fixing framework.  It does **not** appear to supply
the full object needed here: the diagonal type-0B genus-two vacuum integrand
for time \(\times\) \(N=1\) super-Liouville, including all six odd spin
structures, contact terms, and a degeneration-compatible integration cycle.
That missing assembly—not the availability of the basic Liouville structure
constants—is the central worldsheet research problem.
