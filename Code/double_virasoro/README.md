# Double Virasoro

This folder contains the two-Virasoro branching/fusion implementations and
their focused unit tests.

* `two_virasoro_fusion.py` implements the all-NS branching data.
* `nsrr_genus2_block.py` implements the low-order genus-two NS--R--R theta
  block, the auxiliary-Majorana star sewing, and the independent NS--R--R PBW
  comparison.  Integral NS branch labels are supplied by the internally checked
  recursion, while `../ramond_branching_recursion/half_ns_anchor.py` constructs
  the half-integral NS anchors directly from free-field chi strings, converts
  the endpoints to PBW states, and evaluates the three-point function by Ward
  identities.

All NS--R--R numerical paths use only the convention written in the Human Note;
there is no external Ramond-frame conversion. Ground resolution and total
twice-level one close, but the PBW/double-Virasoro comparison has a reproducible
first mismatch at total twice-level two. The module therefore sets
`PBW_DOUBLE_VIRASORO_MATCH_VERIFIED = False`.

Broader exploratory comparisons remain in
`../PBW_c_recursion_double_virasoro crosscheck/`.
