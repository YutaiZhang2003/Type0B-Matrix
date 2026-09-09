# Recovering the physical Ramond block

The global inverse written in draft Eq. (23) is undefined, but this does not
prevent recovery of the equal-structure physical blocks. In the conventions
implemented here, the physical block satisfies an additional parity identity.
It restricts the recovery problem to an ideal where the auxiliary constant
is invertible. This note concerns the NS–R–R genus-two theta channel and
generic modules with invertible Gram matrices.

## The physical restriction

Write `x = eta_2 eta_3` for the **tube-sign monomial**, and `eta, eta_prime`
for the two **vertex-form signs**. These are different labels. Under the
draft's star product,

```math
x\star x=1,\qquad e_\pm=\frac{1\pm x}{2},\qquad
e_\pm\star e_\pm=e_\pm,\quad e_+\star e_-=0.
```

The physical sewing coefficients obey

```math
x\star F_f^{(\eta,\eta')}=\eta\eta'F_f^{(\eta,\eta')}.
```

Consequently an equal-structure block belongs entirely to `I_+=e_+ A`.
Setting its complementary components to zero uses a representation-theoretic
constraint on the desired answer; it is not an arbitrary choice of a
pseudoinverse solution. Both intrinsic NS parities and both form parities
obey this identity. The current production branching grid computes `f=0`
and `eta=eta_prime=+1`.

### Derivation from the Ramond parity exchange

The code's Ramond ground metric is `diag(1,i)`. Define a level-preserving map

```math
S w^+=e^{-i\pi/4}w^-,\qquad
S w^-=e^{i\pi/4}w^+,
```

and extend it by commuting it through the PBW modes. It commutes with all
`L_n,G_r` as an ordinary linear map, satisfies `S^2=1`, reverses parity,
and preserves the BPZ form. These facts follow on the ground doublet and
extend by the mode algebra and BPZ invariance. This is ordinary commutation,
not the graded commutation rule for an odd superintertwiner.

Let `P=(-1)^F` on a Ramond module and `K=(SP)_2 tensor S_3`. For an NS
descendant with `a` supercurrent modes modulo two, the Ward forms obey

```math
\rho_f^{(\eta)}(\xi,SP\zeta_2,S\zeta_3)
=-i\eta(-1)^{f+a}\rho_f^{(\eta)}(\xi,\zeta_2,\zeta_3).
```

To see this, first evaluate the four ground components. Extending to
descendants, `SP` anticommutes with the Ramond supercurrent in slot 2,
`S` commutes with it in slot 3, and flipping slot 3 changes the sign in the
Ramond Ward contour. The factor `(-1)^a` compensates an NS supercurrent
insertion. Thus the transformed form obeys the original Ward system;
uniqueness from the ground normalization gives the displayed identity.
The intrinsic NS parity is included in the Ward contour convention and
does not change this relative-`f` formula.

At fixed levels let `C_{b,c}` be the contraction of two vertex forms with
inverse Gram matrices, before the theta sewing sign. Since `K` is an
isometry, applying it to both vertices gives

```math
C_{b\oplus1,c\oplus1}=-\eta\eta' C_{b,c}.
```

The ratio of the two theta sewing signs is `(-1)^(b+c+1)`. The parity
coefficients including that sign therefore satisfy

```math
F_{b\oplus1,c\oplus1}=\eta\eta'(-1)^{b+c}F_{b,c}.
```

But `x star e_p = (-1)^(p_2+p_3) e_(p xor 6)`, proving the restriction.
The auxiliary Majorana Ward form gives the corresponding `+` identity:
`x star A=A`, where `A=F_F`. Its ground coefficient is `A_0=1+x=2e_+`.

## Restricted inverse and finite computation

The algebra `I_+` has identity `e_+`, not the identity of the full algebra.
There is a unique series `B` in this ideal satisfying

```math
B\star A=e_+,\qquad B_0=\frac{e_+}{2}=\frac{1+x}{4}.
```

For equal vertex signs the correct replacement of the global inverse is

```math
F=B\star\widehat F.
```

No explicit inverse series is required. In ascending total degree, recover
each coefficient by

```math
F_{\boldsymbol n}=\frac12\left[
\widehat F_{\boldsymbol n}
-\sum_{\substack{\boldsymbol 0<\boldsymbol m\le\boldsymbol n}}
A_{\boldsymbol m}\star F_{\boldsymbol n-\boldsymbol m}
\right].
```

Here `0<m` means that the multi-index is nonzero, and `m<=n` is componentwise.
The implementation uses twice-levels to accommodate NS half-integer powers.
Every term on the right has already been determined. It uses the auxiliary
free-fermion coefficients and the double-Virasoro output, with no physical
PBW block as input. Its normalization gives `F_0=1+x`, rather than the
enlarged ground coefficient `2(1+x)`.

The implementation checks both input series against the ideal. Only residuals
within the specified tolerance are projected away. Larger violations are
errors: they cannot be dismissed by choosing a pseudoinverse. The tolerance
measures distance from the ideal, not the full numerical error inside it.

## Opposite vertex signs

For `eta*eta_prime=-1`, the physical block lies in `I_-`. The present auxiliary
factor annihilates this entire ideal, so the forward product is identically
zero and supplies no information about that block. An additional auxiliary
observable/prescription with nonzero action there is needed. It is not
implemented by this change. The recovery API explicitly rejects those signs.
The current construction retains all eight **tube-sign** assignments for
the supported equal-structure block; this does not supply the opposite
**vertex-structure** blocks.

## Implementation and commands

- `../double_virasoro/nsrr/nsrr_genus2_block.py` contains
  `recover_same_structure_nsrr_series`, the projector, and its residual check.
- `recover_physical_block.py` reads an enlarged coefficient record and writes
  an explicitly labeled physical coefficient record.
- `compute_q_expansion.py --physical-json PATH` computes the enlarged series
  and then writes the recovered physical series in the same invocation.
- `--direct-pbw-check` on the producer now compares the recovered physical
  series with the independent physical PBW series as well as checking the
  forward identity. This validation is not part of reconstruction.

From the repository root, with the bundled Virasoro implementation:

```sh
TYPE0B_STRINGMC_ROOT="$PWD/Code/bosonic_c1_one_to_n_reference/reference_implementation" \
python3 Code/full_ramond_block_runtime/compute_q_expansion.py \
  --cutoff 6 --direct-pbw-check \
  --json /tmp/enlarged_level6.json \
  --physical-json /tmp/physical_level6.json
```

To recover from an existing enlarged record without rerunning its branching
calculation:

```sh
python3 Code/full_ramond_block_runtime/recover_physical_block.py \
  /tmp/enlarged_level6.json --json /tmp/physical_level6.json
```

The test module `test_nsrr_physical_recovery.py` checks 256 exact descendant
Ward-form parity exchanges, verifies the physical sector for both intrinsic
parities, both form parities, and all vertex-sign pairs through total level
one, and checks rejection of opposite signs and uncontrolled projection.

Fresh level-three double-Virasoro output recovered the physical block with
maximum absolute difference `1.45e-12` and maximum scaled difference
`5.14e-13` from independent direct PBW sewing. The level-three auxiliary and
recovery stages took about 0.0046 and 0.0023 seconds, respectively. These are
separate from the cost of producing the enlarged series.

The full level-six physical coefficients are saved in
[physical_level6_q_expansion.json](physical_level6_q_expansion.json). They were
recovered from fresh double-Virasoro output and then independently compared
with physical PBW sewing. The maximum absolute discrepancy is `1.1555e-9` and
the maximum scaled discrepancy is `3.9752e-10`. Auxiliary generation took
0.0354 seconds and recovery took 0.0295 seconds, excluding the enlarged-block
calculation and the optional PBW validation. All 280 nonzero physical parity
coefficients retain the three independent plumbing variables.

Both the historical saved `level10_q_expansion.json` and fresh level-ten
double-precision output violate the sector identity at high order. The fresh
input's maximum scaled distance from `I_+` is approximately `1.872e-2`, and
the default `1e-8` check rejects it. No level-ten physical output is claimed
as validated. This is an upstream numerical consistency issue; raising the
projection tolerance does not establish the accuracy of coefficients inside
the ideal. Improving that calculation is separate from the now-defined
physical recovery map.
