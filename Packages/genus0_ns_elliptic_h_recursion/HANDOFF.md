# Receiving-project handoff

This archive is self-contained apart from ordinary Python dependencies.
The runtime does not depend on files outside this directory. The human
note is not included and was not edited. Its conventions are reproduced
in the standalone machine note. The existing bosonic package is unchanged.

## Integration checklist

1. Install the wheel or install this directory; Python >=3.9 and mpmath
   are sufficient for numerical use.
2. Run the portable tests before integration. If SymPy is available, run
   the independent audit tests in VALIDATION.md as well.
3. Supply weights in (0,z,t1,...,1,infinity) order and internal weights
   from left to right. Use the ordinary charge c, not Belavin--Geiko's
   rescaled charge.
4. Explicitly choose the parity sector when evaluating. Do not interpret
   the collection of all coefficient sectors as a physical sector sum.
5. Keep the endpoint factors (16q for four points; 4p at each cap for
   higher points), the full Lambda^(c), and C_NS. Never substitute the
   bosonic prefactor or an unprojected torus character.
6. Build once per weight tuple and reuse across moduli. Persist tables
   with save/load_table; never reuse a table for different weights.
7. Check convergence by order and precision at the actual moduli. Degree
   ten is the validation cutoff, not an accuracy guarantee for every point.

## Scientific status

Supported: bottom-component, intrinsically even all-NS sphere comb blocks,
all internal parity sectors, generic nonconfluent parameters.

Not supplied: Ramond or upper-component formulas, full correlators/CFT
structure constants, automatic complex sphere continuation, torus
reconstruction, confluent residues, or arbitrary-n proof of the seed.

The candidate new ingredient is the multipoint NS elliptic construction
and its validated implementation. Literature priority for that general
statement has not been exhaustively established. The machine note
credits the established four-point NS and bosonic multipoint ingredients.

The independent c-recursion follows Belavin--Geiko's linear-channel
sewing. In arXiv v1, the interior indices in equation (3.18) are offset
from Figure 3; the implementation uses that figure's actual adjacency.
This is documented explicitly, not hidden as a normalization adjustment.

## Provenance and distribution

MANIFEST.json records the SHA-256 of every source, document, and audit
file in the handoff. Run tools/verify_manifest.py from the extracted
directory. Original audit ledgers are copied byte-for-byte; historical
absolute paths in their metadata are provenance only. Runners use the
bundled relative paths, never those historical paths.

This is a private research handoff. No new open-source license is assigned
by packaging. Reference papers are linked/cited, not redistributed as PDFs.
Only the project-authored machine note is included as a PDF.

The archive and wheel checksums are supplied beside the release artifacts.
Keep this bundle and its validation ledgers together when handing it to
another project.
