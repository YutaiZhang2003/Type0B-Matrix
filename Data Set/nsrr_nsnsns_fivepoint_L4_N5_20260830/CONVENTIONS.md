# Coefficient-phase and decomposition-sign audit

This local five-point refinement uses the same numerical assembly as the
three-point toy run. No sign or normalization was changed to improve agreement.

## All-NS channel: both factors are explicit

The physical NS primaries in this experiment have intrinsic parities
`(p1,p2,p3)=(0,0,0)`. The chiral block label `a` is relative descendant
parity, and the absolute trilinear parity is `a+p1+p2+p3 mod 2`.

The BRY-to-Human-Note conversion is applied once:

```text
C_HN^(0) = C_BRY
C_HN^(1) = i * C_BRY_tilde
```

The separate nonchiral decomposition factor is
`(-1)^(a+p1+p2+p3)`. The odd term in the present even-primary sector is
therefore

```text
(-1) * (i*C_BRY_tilde)^2 * |primary*F_1|^2
    = +C_BRY_tilde^2 * |primary*F_1|^2.
```

The code does NOT replace the coefficient square by an absolute square.
These two minus signs are kept at separate boundaries: the coefficient
conversion in `compare_nsrr_nsnsns_theta.all_ns_node`, and the decomposition
sign in `theta_partition.theta_diagonal_sector_contribution`.

The authoritative references are Human Notes/SCblock.tex, the all-NS theta
block definition and subsequent partition formula, and section 4.1 of
Machine Notes/conventions.md. The latter derives the squared phase map from
the BPZ normalization, not from fitting modular agreement. The branch choice
is +i; the two-pants vacuum observable fixes only its square.

## NSRR channel: do not identify distinct labels

The HJS chiral-form sign eta=+/- is different from total form parity f=0/1
and different from a nonchiral Ramond-family label. The established RRNS
dictionary is C_+=C_even and C_-=C_odd. In particular, an additional i must
not be attached to C_- merely because its BRY name contains “odd.”

The component convention is

```text
rho_0^(eta) = rho^(++) + eta*rho^(--)
rho_1^(eta) = rho^(+-) + i*eta*rho^(-+).
```

The computational odd Ramond basis carries exp(3*pi*i/4); this basis phase
is included in `PhysicalThreePoint.base_value`. The double-Virasoro chiral
contraction includes the literal Human-Note factor

```text
(-1)^[2*n1 + (2*n1+p1)*(alpha2+alpha3) + alpha2*alpha3]
```

and multiplies the left/right branching coefficients without conjugation.
The nonchiral assembly retains the established HJS 1/2 completeness and
the two closed-R ground-state sums. The physical scalar+Majorana denominator
is separate from the auxiliary Majorana quotient used in the block.

These checks verify consistency with the implemented HJS/Human-Note ledger.
They do not by themselves prove the full nonchiral NSRR sewing prescription
or its modular covariance; that is what this comparison is testing.

## Regression tests

`Code/genus_2/test_nsrr_nsnsns_conventions.py` checks:

- The actual all-NS caller passes coefficient squares `[4,-9]` for mock
  BRY constants `[2,3]`, while the assembled terms are `[4,+9]`. This catches
  omission of either sign, including the deceptive case where both are lost.
- The decomposition sign uses absolute parity for all intrinsic parity triples.
- The HJS sign dictionary and odd/even ground-component phases remain distinct.
- The actual double-Virasoro contraction has the note's extra `2*n1` sign,
  the full graded quadratic sign, and the nonconjugated left/right product.
- The original toy design remains unchanged; the one-order refinement must
  be selected explicitly with `--design fivepoint-l4`.

No Human Note or production numerical-kernel edit was made in this audit.
