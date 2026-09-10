# Minimal-model blocks at higher genus: literature scope audit

Checked September 10, 2026. This is a literature audit; no numerical
cross-checks or production algorithm changes were made.

## Explicit Ising one-point blocks: Choi–Kim (1990)

Jae Hoon Choi and Jae Kwan Kim, *Higher-genus characters of the Ising
model*, Phys. Rev. D **41** (1990) 484–491,
[DOI](https://doi.org/10.1103/PhysRevD.41.484),
[original full text](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevD.41.484/fulltext).

The original PDF was retrieved and Eq. (26), p. 488, inspected visually.
It explicitly gives chiral Ising weight-1/2 one-point blocks at arbitrary
genus as sums of

\[
 Z_1^{-1/4}
 \left[\sum_{i=1}^g v_i(z)\partial_i
 \theta[\delta](0\mid\Omega)\right]^{1/2},
 \qquad \delta\text{ odd}.
\]

Its Eqs. (14)–(17), with the modification after Eq. (26), specify the
pants-label-to-characteristic map, coefficients and signs. Thus it
covers the full six-dimensional space of genus-two fermion one-point
blocks, not just vacuum partition functions or a special one-parameter
family. The odd theta derivative is the square of the holomorphic
zero-mode half-differential.

The source uses a chain/separating pants basis and a common scalar
factor. Applying it to our ordered \(\Theta\psi\) auxiliary requires
the period/Abel map, sewing normalization, local half-differential frame
and basis/spin-frame transport. The formula is explicit irreducible
Ising data, but it is not CCY central-charge recursion. Detailed formula
transcription and the six genus-two specializations are in
[LITERATURE_ISING_ONEPOINT.md](LITERATURE_ISING_ONEPOINT.md).

## Direct genus-two predecessor: Behera–Malik–Kaul (1989)

N. Behera, R. P. Malik and R. K. Kaul, *Genus-two correlators for
critical Ising model*, Phys. Rev. D **40** (1989) 1993–2003,
[DOI](https://doi.org/10.1103/PhysRevD.40.1993),
[original full text](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevD.40.1993/fulltext).

Figure 4, p. 1996, identifies six energy one-point blocks; Eq. (14),
p. 1997, expresses their nonchiral contributions as squared linear
combinations of square roots of odd theta derivatives. Equations
(4)–(6) display the scalar factor and anomaly-free normalization.
This directly establishes that the relevant genus-two chiral blocks
were treated in the literature. The printed Eq. (14) has apparent
transcription defects, including a repeated characteristic and an
incorrect upper summation limit. The later Choi–Kim formula is the
cleaner starting point for a faithful implementation.

## Ares–Santachiara–Viti (2021)

F. Ares, R. Santachiara and J. Viti, *Crossing-symmetric twist field
correlators and entanglement negativity in minimal CFTs*,
JHEP **10** (2021) 175, [primary full text](https://arxiv.org/html/2107.13925).

Section 3.4, Eq. (39), uses **different central charges in vertices and
Gram matrices**:

\[
 \rho\longmapsto\rho(c_{p,q}+\varepsilon^2),\qquad
 G\longmapsto G(c_{p,q}+\varepsilon).
\]

Equation (40) analogously separates the vacuum vertex weight
\(\delta^2\) from its Gram weight \(\delta\). This is not a limit of a
single ordinary CCY block. Section 4 reports computing the direct
combinatorial series through level six. It then matches that series to
diagonal-Virasoro sphere blocks with central charge \(Nc\), Eq. (46),
and uses elliptic recursion for those sphere blocks. Equation (51)
distinguishes the resummed known primary families from the still-missing
higher families. The geometry is a one-parameter cyclic-cover family;
the paper does not give an arbitrary-plumbing, punctured-genus-two CCY
quotient algorithm.

**Implication for our calculation:** this supports the need to distinguish
vertex and propagator regularizations, but does not provide the missing
irreducible Ising implementation.

## Felder–Silvotti (1989, 1992)

G. Felder and R. Silvotti, *Free field representation of minimal models
on a Riemann surface*, Phys. Lett. B **231** (1989) 411–416,
[DOI](https://doi.org/10.1016/0370-2693(89)90685-0).
Its abstract describes a BRST-invariant screened three-point vertex
whose sewing reduces higher-genus minimal-model correlators to
free-field calculations.

G. Felder and R. Silvotti, *Conformal blocks of minimal models on a
Riemann surface*, Commun. Math. Phys. **144** (1992) 17–40,
[publisher record](https://link.springer.com/article/10.1007/BF02099189).
The publication year is **1992**, although the manuscript was received
in March 1991. The publisher explicitly describes integral
representations on arbitrary compact Riemann surfaces. The original
full text was not retrieved during this audit; Springer exposed only
the abstract and metadata, and the attempted Project Euclid route failed.
No claim about a particular equation or recursion in that paper is made
here.

## Accessible all-genus BRST construction

H. Konno, *SU(2)k × SU(2)l / SU(2)k+l coset conformal field theory and
topological minimal model on higher genus Riemann surface*,
[primary full text, hep-th/9212118](https://arxiv.org/pdf/hep-th/9212118).

Sections 4–6 construct screened three-string vertices, sew BRST-invariant
loop operators, and express higher-genus blocks through those operators.
The BPZ minimal series is included at \(k=1\). The introduction explicitly
requires traces over the irreducible subspace to decouple null states.
This is a genuine higher-genus minimal-model construction, but its
computational mechanism is screened free-field sewing and BRST
cohomology, rather than CCY central-charge recursion.

## Burge restrictions and the trifundamental gap

M. Bershtein and O. Foda, *AGT, Burge pairs and minimal models*,
JHEP **06** (2014) 177,
[primary full text, arXiv:1404.7075](https://arxiv.org/pdf/1404.7075).

The partition restrictions remove null states from the AGT basis and
produce irreducible minimal-model block sums. The paper explicitly
limits its scope to **linear conformal blocks**, p. 2, footnote 3.
Its stated checks are a torus identity insertion and a six-spin Ising
block on the sphere. It does not itself establish a genus-two
trifundamental Burge prescription.

L. Hollands, C. A. Keller and J. Song, *Towards a 4d/2d correspondence
for Sicilian quivers*, JHEP **10** (2011) 100,
[primary record, arXiv:1107.0973](https://arxiv.org/abs/1107.0973).
This addresses generic AGT on arbitrary-genus curves and the
trifundamental building block. Its existence does not establish the
additional minimal-model restrictions needed here. The searches in
this audit did not identify a published combined genus-two
trifundamental/Burge quotient construction.

The negative finding is limited to the sources and searches inspected;
it is not a proof that no such construction exists.

## Zamolodchikov recursion at minimal-model parameters: precise scope

N. Javerzat, R. Santachiara and O. Foda, *Notes on the solutions of
Zamolodchikov-type recursion relations in Virasoro minimal models*,
JHEP **08** (2018) 183,
[primary full text, arXiv:1806.02790](https://arxiv.org/html/1806.02790).

This paper studies **sphere four-point blocks and torus characters**.
Sections 3.3–3.5 regulate the internal and external dimensions at the
same order; at rational central charge, the regulator of \(b^2\) is also
of that order. Crucially, §3.5 labels the resulting minimal-model limit
as a **conjecture**. Section 3.6 analyzes cancellation between resonant
recursion paths, with explicit examples at levels 7 and 20; it does not
supply a closed prescription for every finite resonance contribution.

For characters, §§5.2–5.3 correlate the external identity limit, the
internal degenerate limit, and the central-charge limit so the two
primitive null residues tend to \(-1\). Taking the external identity
limit first would instead produce a Verma character. Section 5.5 and
Appendix B organize the remaining resonance chains. These results
justify using correlated regulators and summing resonant paths before
limits, but do not establish a punctured higher-genus quotient rule.

S. Ribault, *On 2d CFTs that interpolate between minimal models*,
SciPost Phys. **6** (2019) 075,
[primary full text, arXiv:1809.03722](https://arxiv.org/pdf/1809.03722).

Section 3.3 gives a more structured **sphere four-point** continuation:
keep external fields on degenerate Kac families with compatible generic
fusion, and continue the allowed internal channel accordingly. At
nonrational central charge, fusion zeros remove its null residue.
Conjecture 3.1, Eq. (3.32), proposes that taking the rational limit then
recovers the minimal block. The text explicitly leaves proving this
conjecture and finding a recursion manifesting these limits unresolved.
Section 3.2 exhibits cancellations that produce higher-order poles at
rational central charge. It therefore does not justify deleting each
apparently singular recursion term separately. Its results do not cover
our punctured genus-two graph, whose internal edges all meet sewn
vertices.

## Additional qualifications on the AGT route

In Bershtein–Foda, Proposition 4.1 proves the denominator criterion for
Burge pairs. Section 7's identification with irreducible blocks uses
Conjecture 7.1 on regularity of the AGT basis. Its elementary matrix
element has one primary and two descendant legs; a vertex with three
descendant legs requires additional input.

K. B. Alkalaev and V. A. Belavin, *Conformal blocks of W_N minimal models
and AGT correspondence*, JHEP **07** (2014) 024,
[primary full text, arXiv:1404.7094](https://arxiv.org/pdf/1404.7094).
Proposition 3.2 and Eq. (3.11) give the corresponding restricted
Young-diagram expression for Virasoro **four-point** minimal blocks.
This is an explicit combinatorial quotient, not a CCY recursion for a
trivalent higher-genus graph.

V. A. Fateev and A. V. Litvinov, *On AGT conjecture*, JHEP **02** (2010)
014, [primary full text, arXiv:0912.0504](https://arxiv.org/pdf/0912.0504).
Section 1, Eqs. (1.22)–(1.24), derives the generic torus one-point
weight recursion. This supplies the recursion used in the character
analysis above; it does not prescribe the irreducible minimal-model
limit itself.

The inference for the present implementation is narrow: these papers
supply useful regularization principles, sphere/torus results, and
explicit alternative quotients. They do not yet provide a ready-made
irreducible Ising version of our punctured genus-two CCY block.
