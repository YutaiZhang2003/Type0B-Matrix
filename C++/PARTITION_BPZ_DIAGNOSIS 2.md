# BPZ sewing in the paper's PBW and Ramond ground-state conventions

The current paper block is defined by its printed PBW formula, including
the theta-channel Koszul factor `(-1)^K`. The C++ recursion and direct PBW
calculations agree for this literal block through total level 3. The
question is whether its diagonal absolute square has one fixed-spin
partition interpretation. It does not, already in the free-Majorana
factor at total level one.

For plumbing inversion `w=q/z`, the spin lift is
`sqrt(dw/dz)=±i sqrt(q)/z`. With the paper's ordinary algebraic adjoint
`G_r^dagger=G_{-r}`, a geometrically BPZ-sewn odd inverse Gram contributes
`-i`; the two infinity-slot forms and the first tube lift contribute
their corresponding signs. At edge parities `epsilon_e`, their combined
effect relative to the printed theta coefficient is
`(-i)^a (-1)^K`, where `a=sum_e epsilon_e mod 2`.
The first factor is common within a three-point form, but `(-1)^K`
depends on which edges carry odd descendants. Thus including the BPZ
factors changes the paper's literal block coefficient by coefficient.
It cannot be described as a change of `eta_e` or as an overall phase.

For example, the three level-one two-fermion terms of the literal block
carry `-eta_1 eta_2`, `-eta_1 eta_3`, and `-eta_2 eta_3`. Their product
is `-1`, whereas any three genuine tube signs give product `+1`.
The free-Majorana Wick check through total level 8 finds 228 nonzero
monomials and a 5.53–31.86% mismatch between the literal diagonal
square and independent fixed-spin bosonization. This is an exact sign
obstruction with a numerical illustration, not a normalization fit.

The physical Ramond kets are kept exactly as in the paper:
`w^+=(|++>-i|-->)/sqrt(2)` and
`w^-=(|+->+|-+>)/sqrt(2)`. In the ordered product basis
`(++,+-,-+,--)`, the ordinary product Gram has diagonal
`(1,-i,i,-1)` and restricts to `(1,0)` on these two kets. The BPZ
product Gram in the same basis has diagonal `(1,-1,-1,-1)` and restricts
to `(1,-1)`. These four direct ground-state checks are in
`tests/partition_bpz.cpp` and `tests/partition_sewing.cpp`; neither test
rephases the kets. They do **not** yet prove the full Ramond source
three-point contraction, so the saved near-one cross-channel ratios
cannot be treated as a completed physical partition check.

The independently reconstructed source and target periods and their
symplectically transported marked spin characteristics agree; see
[PARTITION_NO_M_AUDIT.md](PARTITION_NO_M_AUDIT.md). The draft has not been
edited. A diagonal fixed-spin decomposition would require stating the
BPZ spin-lift factors in its block definition; until then, the native
driver keeps the literal paper block and labels its diagonal sum as a
diagnostic trial.
