# Fixed-spin partition check in the paper's convention

The source and target plumbings give the same two **marked spin surfaces**.
Using the saved plumbing parameters, independent holomorphic-one-form
collocation and the charged-fermion sewing calculation agree on each period
matrix to `3.39e-15` after its recorded integer period shift. The saved
symplectic map takes the source period to the target period with residual
`6.3524e-11`. It transports `[11|00]` to `[00|00]` and `[11|11]` to
`[00|10]`. The target values are attached to these marked characteristics;
an integer shift of the period matrix in the bosonization routine does not
create a second physical spin structure.

The all-NS failure is already visible at total level one. If the paper's
explicit Koszul sign `(-1)^K` is combined with ordinary algebraic Grams and
ordinary infinity-slot forms, the three two-fermion terms in `F` have
relative signs `-eta_1 eta_2`, `-eta_1 eta_3`, `-eta_2 eta_3`.
Their product is `-1`, whereas the product for any assignment of three
tube signs is `+1`. Thus changing `eta_e` on either side cannot fix the
fixed-spin identification. This test uses only nonzero free-Majorana
coefficients and does not depend on Liouville structure constants.

The missing physical information is the BPZ spin lift of `w=q/z` and its
action on odd inverse Grams and infinity-slot forms. In the paper's *same*
PBW and `w^+,w^-` bases, these local factors remove the three relative minus
signs in the free-Majorana test. They do **not** arise by changing any
`eta_e`. However, inserting these factors into the current printed PBW
formula changes its coefficients. The current driver therefore computes
the paper's literal `F` without that insertion. Its diagonal sum is marked
as a diagnostic trial, not a partition function. There is no four-by-four
matrix in the current code.

This identifies a **necessary correction to the draft**, which has not
been edited: the displayed PBW formula cannot simultaneously denote the
currently computed `F` and the fixed-spin BPZ amplitude used in a diagonal
partition decomposition. The exact sewing definition must state the BPZ
spin-lift factors if diagonal sewing is intended. Calling a coefficientwise
adjusted result the unchanged paper `F` would conceal the error, so the
current code does not do that.

The direct Majorana Wick sum through total level 8 contains 228 nonzero
monomials. Diagonal absolute squares of the literal paper `F` differ from
independent bosonization by `5.53%` to `31.86%` across the four target
marked spins. The paper-basis Ramond ground Gram and exact NS parity
identities pass at machine and 40-digit precision. A directed rerun of one
target momentum node confirms that the current `F` equals the **original
literal** block saved before the BPZ adjustment, and that its diagonal
trial differs from the adjusted result.

As an additional diagnostic, summing the saved literal `F` at all `10^3`
target momentum nodes and comparing with the saved `7^3`-node source at
total levels 8 and 5 gives ratios `0.987817` and `0.931222` for the two
marked spins, and `0.959054` for their sum. The earlier near-one ratios
`1.000007836` and `1.000003799` used coefficientwise BPZ-adjusted target
blocks, so they cannot be cited as a cross-channel test of the paper's
unchanged `F`. Moreover, the full Ramond source contraction still needs a
direct test in the paper's ground-state basis; an old test of that
contraction used rephased kets and has been removed. Neither issue affects
the exact all-NS sign obstruction or the period and spin identification.

Run `make -C C++ check-partition` and `tests/partition_spin_geometry.py`
for the directed checks. The old full-run data are in
`results/partition_no_M_2026-09-24/`; their former adjusted-block array
is historical and is not written by the current driver.
