# What the BPZ test does and does not establish

`Human Notes/SCblock.tex` derives a diagonal theta-channel decomposition
from its own graded full-state prescription. That reduction is correct:
the mixed holomorphic–antiholomorphic inverse-Gram signs combine with
the full reordering sign to leave `(-1)^(a+p_1+p_2+p_3)` and one
Koszul factor in each chiral block. The exact identity is recorded in
[PARTITION_NO_M_AUDIT.md](PARTITION_NO_M_AUDIT.md).

The remaining question is geometric. In the Human Note, the chiral
super-Gram is defined with the algebraic adjoint `G_r^dagger=G_{-r}`.
In a sewn spin surface, inversion `w=q/z` acts on half-integral-weight
fields through a chosen square-root branch. Its action on the odd duals
and infinity-slot forms must be reconciled with the note's `q_e,eta_e`.
The affine map between period-matrix characteristics is not enough to
fix that local sewing choice. The explicit fermionic sewing prescription
in [Tuite–Zuevsky](https://arxiv.org/pdf/1007.5203) illustrates the
need to record the inversion lift.

The code's direct paper-basis Ramond ground states are
`w^+=(|++>-i|-->)/sqrt(2)` and
`w^-=(|+->+|-+>)/sqrt(2)`. Its ground-Gram and exact NS parity tests
pass in machine and 40-digit arithmetic. A full nonchiral Ramond
three-point contraction in these kets has not yet been independently
verified. The free-Majorana comparison currently rejects a **particular
linear** map from the paper's `eta_e` to theta-characteristic signs at
fixed `q_e`; it does not disprove the Human Note's algebraic factorization.
