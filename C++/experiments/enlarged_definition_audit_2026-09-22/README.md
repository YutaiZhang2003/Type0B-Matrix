# Enlarged PBW definition: correspondence with the implementation

This directed audit checks machine-notes Eq. (2.10), including its local
forms (2.7), (2.9), against the existing code. It does not rerun a full block.

## Correspondence

- `../total_level6_2026-09-16/graph_ccy.hpp`, `Graph::sign_of`: the permutation
  from edge-end pairs to vertex triples is exactly the notes' exponent K.
  `Graph::kernel` adds the local middle-to-ket mixed parity term.
- `../total_level6_2026-09-16/graph_validate.cpp`, `Sewing::edge`: physical
  edges use full inverse Gram matrices; auxiliary edges use diagonal entries
  `sign(p)`, namely the inverse auxiliary metric `(-1)^epsilon'`.
- `../total_level6_2026-09-16/ns_local.hpp`, `NSBranches::raw`: the local
  all-NS form contains `(-1)^(B mathsfC)`, as in (2.7) for the even intrinsic
  primaries used by this graph driver.
- `../../include/ramond/anchors.hpp`, `LowAnchors::raw`: the auxiliary/physical
  crossing is `(-1)^[A mathsfA + (B+alpha+p1)(mathsfC+mathsfc)]`.
  `PhysicalForm` supplies the contour convention. The boundary call in
  `../../include/ramond/branching.hpp`, `OuterBranching::prepare`, uses
  `sign(f)*eta`; together these give the full local form (2.9).
- `../total_level6_2026-09-16/graph_validate.cpp`,
  `DoubleVirasoro::compute`: the branch sum explicitly multiplies the graph
  sign, one `i` to the total NS descendant parity at each NS-R-R vertex,
  the local branching forms, and one inverse primary norm per edge.
  `../mercedes_all_ns_level10_2026-09-16/dv_engine.hpp` uses the same assembly.
- `../unphased_vertex_2026-09-15/include/ramond/literal_enlarged.hpp`,
  `LiteralEnlargedPBW::coefficient`: the theta reference contracts both
  physical endpoint words through their inverse Gram matrices and sums a
  common auxiliary state per edge. Its simplified sign is `(-1)^mathsfA`
  times the total-parity theta reordering sign. This equals (2.10): the
  NS-R-R vertex phases occur twice, auxiliary parity conservation makes
  the three edge-metric signs multiply to one, and the surviving local
  factor is `(-1)^mathsfA`.

## Ground-state basis conversion

The notes use physical Ramond grounds `(w+, w-)` with metric `diag(1,i)`.
The code's native basis is `(w+, exp(3*pi*i/4) w-)` with metric `diag(1,1)`.
This is a basis change, not an additional sewing phase. It multiplies a
vertex by `exp(3*pi*i/4)` for each odd ground label and transforms the
inverse Gram entries inversely. All internal factors cancel in the block.
External state coordinates must be converted when comparing the two bases.

## Fresh directed checks

`local_conventions.cpp` includes the actual production headers and compares
`PhysicalForm(f,(-1)^f eta)` against the human-convention `ScaWard(f,eta)`
multiplied by the factor in (2.9) and the native ground-basis conversion.
The physical contour factor is
`(-i)^(A mod 2) (-1)^[f(A+B+alpha)]`.
The check includes both intrinsic NS primary parities, both f values,
both eta signs, NS words `1,G_-1/2,L_-1`, and Ramond words `1,G_-1,L_-1`
with both ground labels. It also compares the Gram entries after the same
basis conversion.

At 40 digits and the established generic momenta:

| Check | Comparisons | Maximum scaled error | Failures |
|---|---:|---:|---:|
| Physical contour convention | 864 | 7.825273386683346e-32 | 0 |
| Ground-basis Gram conversion | 43 | 2.295887403949780e-41 | 0 |

`parity_checks.py` additionally checks all 64 choices of local physical,
auxiliary, form, and primary parities for the phase cancellation and the
theta simplification, with exact powers of i.

These are local convention and source checks. The previously saved
total-level-6 theta/glasses/Mercedes block comparisons and all-NS Mercedes
level-10 comparison were inspected, not rerun. The general definition with
arbitrary external states is not an implemented arbitrary-genus frontend.

From the repository root, reproduce the fresh local checks with:

```sh
clang++ -O2 -std=c++17 -IC++/include -I/opt/homebrew/include C++/experiments/enlarged_definition_audit_2026-09-22/local_conventions.cpp -L/opt/homebrew/lib -lmpc -lmpfr -lgmpxx -lgmp -framework Accelerate -o /tmp/scblock_local_conventions
/tmp/scblock_local_conventions
python3 C++/experiments/enlarged_definition_audit_2026-09-22/parity_checks.py
```
