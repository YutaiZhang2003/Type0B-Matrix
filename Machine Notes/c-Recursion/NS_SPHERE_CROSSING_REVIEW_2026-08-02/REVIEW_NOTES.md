# NS Liouville sphere-crossing review snapshot

This snapshot records the direct central-charge-recursion check of the
nonchiral Neveu--Schwarz sphere four-point function at
\(\widehat c=9\), or \(c=27/2\).  The two channel sums were evaluated
independently; equality was not imposed, averaged, or used to normalize the
data.

## What is computed

For the ordered correlator
\[
  \langle V_{P_4}(\infty)V_{P_3}(1)V_{P_2}(z,\bar z)V_{P_1}(0)\rangle,
\]
the direct channel uses \(G_{4321}(z,\bar z)\).  The crossed calculation uses
\(G_{4123}(1-z,1-\bar z)\), obtained by the permutation
\((P_1,P_2,P_3,P_4)\mapsto(P_3,P_2,P_1,P_4)\).  Each continuum integral
contains the BRY self-dual structure constants, the \(dP/\pi\) measure, and
both the even and odd NS block families.

The chiral block is evaluated from the functional Zamolodchikov
\(c\)-recursion itself.  Recursion order \(N\) keeps every nested Kac-residue
path whose accumulated physical null level is at most \(N\).  Every terminal
node is the exact hypergeometric global \(\mathfrak{osp}(1|2)\) block.  There
is no descendant-Gram-matrix truncation, local-\(z\) truncation, elliptic-\(q\)
truncation, displaced-\(c\) regulator, or \(h\)-recursion in this run.

Parameters are
\[
  b=1,\qquad (P_1,P_2,P_3,P_4)=(1/2,1/3,1/4,3/5),
\]
with \(P\leq5\), 40-point Gauss--Legendre quadrature, 80-digit block
arithmetic, and 35-digit structure-constant arithmetic.

## Numerical conclusion and its limitation

At recursion order 12, over \(z=0.30,0.35,\ldots,0.70\), the largest observed
relative channel mismatch is
\[
  3.0846\times10^{-8}.
\]
This number is faithful as the residual of the recorded order-12 computation,
but it is **not** a certified error bound for the infinite-recursion
correlator.  At the same momentum nodes, the order-10 residual is
\(3.7697\times10^{-7}\), while the largest order-10-to-order-12 drift of an
individual channel is \(3.4612\times10^{-7}\).  Raising the recursion order is
deliberately postponed.

Crossing also cannot diagnose an error shared by both channel implementations.
The independent sphere \(c\)- versus \(h\)-recursion comparison described in
the note supports the chiral kernel, but the nonchiral structure constants and
continuum normalization remain shared inputs here.

## Reproduction

Run from this snapshot's top directory:

```sh
PYTHONPATH=Code python3 Code/stress_ns_crossing.py \
  --orders 12 \
  --z 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
  --p-max 5 --quadrature-order 40 \
  --structure-precision 35 --block-working-precision 80 \
  --output 'Data Set/ns_liouville_crossing_direct_c_recursion_order12.json' \
  --plot 'Data Set/ns_liouville_crossing_direct_c_recursion.svg'

PYTHONPATH=Code python3 -m unittest \
  Code/test_superconformal_blocks.py Code/test_sphere_four_point.py

cd 'Machine Notes/c-Recursion'
latexmk -pdf -interaction=nonstopmode -halt-on-error ns_genus_c_recursion.tex
```

The production order-12 run is computationally substantial.  The bundled JSON
is the immutable numerical ledger used for the table and plot.

## Files

- `Code/stress_ns_crossing.py`: crossing driver and SVG generation.
- `Code/superconformal_blocks.py`: functional direct \(c\)-recursion.
- `Code/sphere_four_point.py`: nonchiral continuum assembly.
- `Code/super_liouville_structure_constants.py`: BRY constants at \(b=1\).
- `Code/test_*.py`: targeted regression checks.
- `Data Set/*order10.json` and `*order12.json`: convergence ledgers.
- `Data Set/*.svg` and `*.png`: plotted result and TeX rasterization.
- `Machine Notes/c-Recursion/ns_genus_c_recursion.tex`: updated note.
- `Machine Notes/c-Recursion/ns_genus_c_recursion.pdf`: compiled review copy.
- `README.md`: repository-level entry for the crossing driver.
- `SHA256SUMS`: checksums of all collected files except itself.
