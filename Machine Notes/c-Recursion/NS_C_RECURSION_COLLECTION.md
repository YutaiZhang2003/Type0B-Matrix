# NS \(c\)-recursion collection

This collection records the current derivation, implementation, and
consistency checks for the all-Neveu--Schwarz central-charge recursion at
generic internal weights.  Exceptional coincident Kac loci are not
implemented.  The arbitrary-graph endpoint and BPZ/spin-frame signs are now
compiled from graded slot permutations, the NS reflection identity, and the
literal odd-coordinate frame and transition signs of the lifted plumbing
maps.
The files are copied into the archive with their repository-relative paths,
so the TeX bibliography and Python imports continue to work after extraction.

## Type 0B application convention

The conformal-block recursion itself is independent of Liouville pants data.
When a Type 0B correlator is assembled, use the global convention boundary in
**Machine Notes/conventions.md**, section 4.1.  In particular, a BRY-native
sphere formula keeps the real \(\widetilde C_{\rm BRY}\) directly.  Only an
import into the separate Human-Note graded sewing basis uses

\[
C_{\rm HN}^{(1)}=\sigma i\,\widetilde C_{\rm BRY},\qquad
(C_{\rm HN}^{(1)})^2=-\widetilde C_{\rm BRY}^{2}.
\]

Apply this map once at that interface, not inside the recursion and not to a
BRY-native \(G,H,J\) or PCO amplitude.  The branch \(\sigma=\pm1\) is not
fixed by a two-pants genus-two vacuum graph.

## Main document

- **Machine Notes/c-Recursion/ns_genus_c_recursion.tex** is the source.
- **Machine Notes/c-Recursion/ns_genus_c_recursion.pdf** is the compiled
  manuscript.
- **References/references.bib** is the bibliography database.

The consistency-check section includes:

1. the sphere four-point specialization;
2. the order-ten sphere \(c\)- versus \(h\)-recursion comparison at
   \(\widehat c=9\);
3. the order-ten torus one-point \(c\)- versus \(h\)-recursion comparison
   at \(\widehat c=9\), for both torus spin lifts;
4. direct genus-one and genus-two PBW sewing;
5. the exact symbolic genus-two comparison through total physical order
   three; and
6. the numerical genus-two stress test through total physical order eight.

## Core implementation

- **Code/superconformal_blocks.py**: NS sphere four-point \(c\)-recursion.
- **Code/superconformal_torus_blocks.py**: toric NS \(h\)-recursion and
  spin-lifted torus evaluation.
- **Code/self_dual_superconformal_blocks.py**: coefficient-wise finite parts
  at the self-dual point.
- **Code/ns_genus_c_recursion_checks.py**: Kac poles, null norms, fusion
  polynomials, and low-level representation-theoretic checks.
- **Code/ns_global_osp_block.py**: global \(\mathfrak{osp}(1|2)\) module and
  trinion kernels.
- **Code/ns_regular_block.py**: endpoint reflection/Koszul signs,
  frame-derived graph orientation, and polarized vacuum/global assembly.
- **Code/ns_vacuum_schottky.py**: lifted primitive Schottky product.

The archive also contains the local modules imported transitively by these
files.  The definitive inventory is
**Machine Notes/c-Recursion/ns_c_recursion_collection_files.txt**.

## Main consistency checks

- **Code/compare_ns_sphere_c_h_recursion.py** compares the independent
  sphere \(c\)- and \(h\)-recursions through \(q^{10}\).
- **Code/compare_ns_torus_c_h_recursion.py** compares the independent
  torus one-point \(c\)- and \(h\)-recursions through \(q^{10}\).
- **Code/ns_genus2_symbolic_low_order.py** checks all 84 genus-two
  coefficients through total physical order three as exact rational
  identities.
- **Code/ns_genus12_finite_c_check.py** performs direct finite-\(c\) PBW
  sewing and the genus-two order-eight stress test.
- **Code/ns_grassmann_sewing.py**, **Code/ns_osp_superspace.py**, and
  **Code/ns_supermoduli_plumbing.py** check the sign, superspace, and
  supermoduli reformulations.

The numerical ledgers are:

- **Machine Notes/c-Recursion/ns_sphere_c_h_order10.json**;
- **Machine Notes/c-Recursion/ns_torus_c_h_order10.json**; and
- **Machine Notes/Genus 2/ns_genus2_order8_stress_test.md**.

## Reproduction

Run commands from the extracted collection root.  Python 3.10 or newer is
recommended.

Install the small numerical dependency set:

    python3 -m pip install -r "Machine Notes/c-Recursion/requirements-ns-c-recursion.txt"

Compile the manuscript:

    cd "Machine Notes/c-Recursion"
    latexmk -pdf -interaction=nonstopmode -halt-on-error ns_genus_c_recursion.tex
    cd ../..

Run the quick representation-theoretic and unit checks:

    PYTHONPATH=Code python3 Code/ns_genus_c_recursion_checks.py
    PYTHONPATH=Code python3 Code/ns_global_osp_block.py
    PYTHONPATH=Code python3 Code/ns_regular_block.py
    PYTHONPATH=Code python3 Code/ns_grassmann_sewing.py
    PYTHONPATH=Code python3 -m unittest \
      Code.test_superconformal_blocks \
      Code.test_self_dual_superconformal_blocks \
      Code.test_superconformal_torus_blocks

Regenerate the two order-ten ledgers:

    PYTHONPATH=Code python3 Code/compare_ns_sphere_c_h_recursion.py \
      --order 10 \
      --output "Machine Notes/c-Recursion/ns_sphere_c_h_order10.json"

    PYTHONPATH=Code python3 Code/compare_ns_torus_c_h_recursion.py \
      --order 10 \
      --samples 32 \
      --output "Machine Notes/c-Recursion/ns_torus_c_h_order10.json"

Run the exact and higher-genus checks:

    PYTHONPATH=Code python3 Code/ns_genus2_symbolic_low_order.py
    PYTHONPATH=Code python3 Code/ns_genus12_finite_c_check.py \
      --c 41.3 \
      --weights 0.731 0.913 1.173 \
      --genus-one-order 6 \
      --genus-two-order 8
    PYTHONPATH=Code python3 Code/ns_vacuum_schottky.py

The order-ten finite-part calculations use multiprecision contour averages
and can take a few minutes.

## Recorded benchmark results

- Sphere four-point block, ten \(z\)-points: maximum pointwise relative
  discrepancies \(3.30\times10^{-59}\) and \(2.72\times10^{-59}\) in the
  even and odd families, with maximum coefficient discrepancy
  \(9.53\times10^{-17}\).
- Torus one-point block, ten \(q\)-points and both spin lifts: maximum
  pointwise relative discrepancy \(6.22\times10^{-61}\), with maximum
  coefficient discrepancy \(5.19\times10^{-58}\).
- Genus-two symbolic check: 84/84 exact identities through total physical
  order three.
- Genus-two numerical stress test: 969/969 coefficients through total
  physical order eight, with maximum scaled discrepancy
  \(9.124\times10^{-11}\).

The sphere and torus comparisons concern normalized conformal blocks.  They
do not use Liouville structure constants, reflection amplitudes, spectral
measures, or momentum integrals.

For an arbitrary ordered graph, supply the lifted local plumbing maps rather
than a separate sign table.  The packaged sign compiler derives the endpoint
Koszul/reflection phase and the linear BPZ/spin-frame bits.  In the theta
convention it derives \((\beta_0,\beta_1,\beta_\infty)=(0,0,1)\) from frame
signs \((+,+,+,+,+,-)\), and reproduces the combined odd-null incidence
pattern \((-1,+1,+1)\) on the reflected trinion.
