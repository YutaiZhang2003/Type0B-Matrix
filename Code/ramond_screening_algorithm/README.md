# Ramond screening algorithm

> **Current status (August 2026).** Read
> [`TWO_CHART_ALGORITHM.md`](TWO_CHART_ALGORITHM.md) first.  Endpoint `Z_0`
> gives an exact identity between the coefficients of the two *raw chi-path
> expansions*, but it is not an identity between the corresponding abstract
> SCA states.  Starting at `n=3/4`, the two free-field-to-SCA transition maps
> differ by nonzero momentum-dependent `L_-1` terms.  A signed nonzero Coulomb
> value therefore still requires the 2013 reflection operator.  A genuine
> native positive hard-channel evaluator is checked through `N=3`, including
> its natural maximal-screening node, but neither the remaining positive
> nodes nor a signed `D+1`-node callback is certified.  Consequently this
> directory still does not provide an end-to-end all-level branching
> coefficient.

This directory develops a state-free backend for the Ramond branching
problem.  The proposed production path does not construct `v_n`,
`W_n^epsilon`, a super-Virasoro PBW basis, or a Gram matrix.  A separate
exact finite-level reflection oracle and an exact modular transition backend
are included for checks and practical finite-level fallbacks.

The intended reconstruction is the Ramond analogue of the 2013 NS proof.

1. Keep the 2016 ordered chi strings as contour-mode insertions.  For a
   Ramond branch `n`, the string contains the consecutive modes
   `0,-1,...,-M`, with `M=2|n|-1/2`, and at most one additional zero mode
   for the other copy.
2. For the special consecutive strings, use the coefficient identity

   ```text
   coeff_path[-n,P] = (-1)^g coeff_path[+n,-P].
   ```

   This organizes the raw paths and their ground labels.  It must not be
   promoted to an SCA-state identity or used as a signed value callback.
   Convert the negative branch with the exact reflection recurrence

   ```text
   A_-(X_m;l) R_l = R_(l-m) A_+(X_m;l),  X=L,G, m>0,  R_0=1.
   ```

   `reflection/intertwiner_recurrence.py` builds this map directly from sparse
   oscillator actions.  The reflected representatives in the 2013 argument
   may still supply homogeneous fusion zeros; that use does not require a
   nonzero signed value.
3. Evaluate a positive and a signed Coulomb chart independently at
   same-parity neutrality nodes.  Each node must use the full native
   ground-resolved fermion kernel and exact screening integral.  Only after
   both analytic chart polynomials have been reconstructed may the constant
   two-ground-state map be used to resolve `eta=+/-`.
4. Convert each resulting symmetric Laurent polynomial to the finite
   Schur/Jack basis and integrate it with the exact generalized Selberg
   products.  Even and odd screening number select the two Ising channels.
5. Divide by the corresponding primary three-point form and branch norms,
   and reconstruct the generic momentum polynomial from enough exact
   charge-neutral samples.  The Ward degree bound is
   `D=(2n1)^2+(2n2)^2+(2n3)^2-1/2`.

There is a separate, exact obstruction to completing this program using
only the eight 2013 zero charts and two normalization constants.  For a
sign triple `sigma`, set

```text
m_sigma = 2*(sigma1*n1 + sigma2*n2 + sigma3*n3).
```

The `Psi_-A` counting argument proves one scalar chart equation at each
allowed `r+s<m_sigma` with `m_sigma>0`.  Since
`m_(-sigma)=-m_sigma`, the union of all eight representations contains
exactly `D` such equations.  The Ramond problem has two polynomials of
degree at most `D`, hence `2*(D+1)` unknown coefficients.  The zero system
therefore has nullity at least `D+2`, not two.

Even granting, without claiming an evaluator, every equality-plane value
`r+s=m_sigma` does not repair the count.  The exact two-covector
Vandermonde audit gives

```text
labels                 unknowns   zero rank   zero+saturation rank   nullity
(0,3/4,3/4)               10          4                 9                1
(1,5/4,5/4)               34         16                25                9
(3/2,7/4,7/4)             68         33                46               22
```

Along `(k/2,(2k+1)/4,(2k+1)/4)`, `D=3*k^2+2*k` while the number of
equality-plane rows is only linear in `k`; the residual nullity is
quadratic.  Values at a full lattice of *interior* mixed-reflection nodes
could change this conclusion, but obtaining those values is precisely the
missing reflected Ramond chiral-vertex oracle.  The 2013 paper uses those
representations to infer homogeneous zeros, not as nonzero Pfaffian--Selberg
callbacks.  Run the source-independent rank audit with

```bash
python3 -m python.ramond_screening_algorithm.eight_chart_constraint_audit
```

The same-parity surplus nodes also invalidate the simplest fixed-width
Selberg story.  At the natural screening number `N=N0`, the projected
insertion is `C*Delta^2`.  Already for `(0,1/4,1/4)`, the next node `N=3`
is not divisible by `Delta^2`; its exact Schur support is
`(2,1),(1,1,1),(2,2),(2,1,1)`.  For the hard labels, the cleared
one-variable degrees are `4` at `N=3` and `6` at `N=5`, while
`Delta_5^2` has degree `8`.  Thus the width-zero reduction does not persist
along interpolation nodes.  The smallest surplus happens to equal the
standard BFL `(1,1,0)` polynomial up to a constant, so this check does not
rule out a future Uglov or holonomic oracle; it does rule out treating the
minimal determinant formula as the general callback.  The exact audit is

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.audit_surplus_screening_width
```

The earlier claim that a ground two-by-two rotation could be applied to an
excited same-plane Majorana Pfaffian was false.  Such a rotation is valid
only after two independently evaluated analytic SCA forms are known.  The
generic level-one reflection block also shows the obstruction directly:
`psi_-1|g>` mixes with `c_-1|1-g>` by a momentum-dependent coefficient.

The hard evaluator in `pfaffian/native_hard_screening.py` is a genuine
positive-chart Coulomb calculation: literal consecutive chi strings,
ground-resolved auxiliary and physical Pfaffians, the bosonic screening
weight, and exact Selberg averages.  Its independent audit checks both form
parities at two rational samples for every `N=0,1,2,3`; in particular the
natural hard node `N=3` is noncircular.  Symbolic expansion at `N=5` is not a
viable continuation of this implementation, and the endpoint-`Z_0`
diagnostic is not a signed callback.  Thus the required remaining positive
nodes and all signed nonzero nodes are still absent.  Nor is an all-level
polynomial bound claimed: a uniform level-independent width bound for the
two full Schur reductions remains to be proved.

The auxiliary Virasoro sewing evaluator was also repaired.  Besides retaining
negative-mode commutators and the descendant contribution to `L_0`, the
conversion requires the BPZ sign of the NS fermion primary, the
infinity--middle pair cocycle, and the middle-chart transport
`(-1)^(N1*N2+ell2*(1+N2))`.  With these corrections the Virasoro expansion
and the native Ising OPE Pfaffian agree on all 512/512 stored auxiliary
endpoints.  Run the exact endpoint and impact audits below.

In particular, it has not been proved that `R_l` may be moved through a
Ramond chiral vertex as a scalar reflection amplitude that cancels after
primary normalization.  The naive version of that chart shortcut fails the
mixed-sheet `H` channel.  Any future chart-changing acceleration must first
derive the full two-ground-index vertex reflection law and reproduce that
channel; it is not an established part of the algorithm above.

Run the source-formula tests and the difficult-case work estimate with

```bash
python3 python/ramond_screening_algorithm/n2_selberg.py
python3 python/ramond_screening_algorithm/complexity.py 2 7/4 7/4
python3 python/ramond_screening_algorithm/profile/modular_transition.py
python3 -m python.ramond_screening_algorithm.signed_chart_reduction
python3 -m python.ramond_screening_algorithm.reflection.audit_native_spin_kernel
python3 -m python.ramond_screening_algorithm.reflection.audit_signed_state_obstruction
python3 -m python.ramond_screening_algorithm.eight_chart_constraint_audit
python3 -m python.ramond_screening_algorithm.pfaffian.audit_surplus_screening_width
python3 -m python.ramond_screening_algorithm.pfaffian.auxiliary_ising_kernel
python3 -m python.ramond_screening_algorithm.pfaffian.audit_native_hard_screening
python3 -m python.ramond_screening_algorithm.pfaffian.audit_auxiliary_virasoro_repair
python3 -m python.ramond_screening_algorithm.reflection.intertwiner_recurrence
python3 -m python.ramond_screening_algorithm.pfaffian.hard_two_chart_certificate
python3 -m python.ramond_screening_algorithm.two_chart_interpolation
```
