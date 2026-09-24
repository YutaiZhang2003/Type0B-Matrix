# Resolution of the NSRR factor four

The mismatch came from dividing the imported NSRR Upsilon coefficients by two
when expressing a physical vertex in the normalized ordered forms. With the
two-point metrics used here, the coefficients are **`c_+=E, c_-=O`**, and the
two physical Ramond-family ground amplitudes are **`E+O, E-O`**. The previous
adapter used `E/2, O/2`. Two vertices therefore suppressed the source partition
function by four. This correction changes the structure-constant dictionary,
not any chiral block, Ward identity, Gram matrix, measure, or free-frame factor.

## How the normalization is fixed without the genus-two comparison

Both NS primaries and each physical Ramond family have the same continuum
two-point normalization, `pi delta(p-p')`. Inserting the NS identity must
reproduce that metric in either sector. Take its Liouville charge to be
`alpha_NS=epsilon`, tending to zero from above, and let the other two momenta
approach the same positive value `p`.

Write `C_NS` for the all-NS structure constant at these same momenta. The
external NS leg and the common cosmological factor cancel from the ratios
`E/C_NS` and `O/C_NS`. In the notation of the implemented Upsilon formulas,

\[
\begin{aligned}
x_0&=\epsilon+i(p_1+p_2),&x_3&=Q-\epsilon+i(p_1+p_2),\\
x_1&=\epsilon+i(p_2-p_1),&x_2&=\epsilon-i(p_2-p_1),
\end{aligned}
\]

and the exact ratios are

\[
\frac{E}{C_{\mathrm{NS}}}
=\frac{N_{\mathrm R}(p_1)N_{\mathrm R}(p_2)}
 {b^2N_{\mathrm{NS}}(p_1)N_{\mathrm{NS}}(p_2)}
 \frac{\Upsilon_{\mathrm{NS}}(x_0)\Upsilon_{\mathrm{NS}}(x_3)}
 {\Upsilon_{\mathrm R}(x_0)\Upsilon_{\mathrm R}(x_3)},
\qquad
\frac{O}{C_{\mathrm{NS}}}
=\frac{N_{\mathrm R}(p_1)N_{\mathrm R}(p_2)}
 {b^2N_{\mathrm{NS}}(p_1)N_{\mathrm{NS}}(p_2)}
 \frac{\Upsilon_{\mathrm{NS}}(x_1)\Upsilon_{\mathrm{NS}}(x_2)}
 {\Upsilon_{\mathrm R}(x_1)\Upsilon_{\mathrm R}(x_2)}.
\]

Here `N_s` denotes the positive internal-leg factors already used by
`GenericSuperLiouvilleConstants`. For positive real momenta their gamma-function
metrics simplify exactly to `g_NS=1/b`, `g_R=1`. Hence

\[
N_{\mathrm{NS}}(p)^2=b^{-2}\Upsilon_{\mathrm{NS}}(2ip)
 \Upsilon_{\mathrm{NS}}(-2ip),\qquad
N_{\mathrm R}(p)^2=\Upsilon_{\mathrm R}(2ip)\Upsilon_{\mathrm R}(-2ip).
\]

Reflection `Upsilon_s(Q-x)=Upsilon_s(x)` makes the first ratio tend to one.
The simple zero of `Upsilon_NS` at zero makes the second tend to zero:

\[
\boxed{\quad E/C_{\mathrm{NS}}\longrightarrow1,\qquad
 O/C_{\mathrm{NS}}\longrightarrow0.\quad}
\]

These limits also hold with `p_2-p_1=epsilon*t` at fixed finite `t`. Thus they
compare the residues of the same delta-function peak, not just unrelated
pointwise values at coincident momenta. The singular denominator near that
peak is `Upsilon_NS(epsilon+i(p_2-p_1))` times its conjugate in both `E` and
`C_NS`; away from coincidence the vanishing external identity leg removes the
regular terms. The two Ramond-family amplitudes `E+O` and `E-O` consequently
have the same identity residue as `C_NS`. Taking their half-sums instead
gives **half the required two-point metric**.

The old module identity checks assigned `E=2, O=0` by hand. Those checks tested
the Clifford contraction, but did not test the identity limit of the actual
Liouville inputs. The new test connects these two parts of the calculation.
After factoring out the NS identity residue, the correct inputs are `E=1,
O=0`; the independent finite Clifford contraction gives Ramond trace two.

## Relation to the literature

The equal NS/R two-point normalization and the relative NSRR prefactor follow
[Poghossian, hep-th/9607120](https://arxiv.org/pdf/hep-th/9607120), Eqs. (14)–(15),
(51), and (56). In Eq. (56) the two Upsilon terms occur with a plus/minus sign
and no additional half relative to the NS prefactor; the relative `b^-2` is
already included in our input generator. Equations (68) and (70) give the
corresponding exponential-field formulas with the same common prefactor.

The previous adapter combined our numerical Upsilon terms with the literal
half-sum dictionary in [Balthazar–Rodriguez–Yin, §3.1, Eq. (3.8)](https://arxiv.org/html/2201.05621#S3.SS1).
That identification does not pass the equal-metric identity test above. In
particular, merely citing their half-sum formula does not establish that its
inputs can be identified with ours while retaining our two-point metrics.
The current implementation specifies the identity-normalized Poghossian
convention explicitly. We do not claim to have reconciled every normalization
statement in that other presentation, or to fix the mismatch by a rephasing.

## Native implementation

The production inputs use the paper's `C_a` and `C_f_eta` in edge order
`(infinity,1,0)`. `C_1` includes its factor of `i`.
`C_{f,+}=E`, `C_{f,-}=O`, with no extra half; both f values are equal
reduced coefficients. The current partition driver stores the paper's
literal `F` and applies the geometric BPZ phase coefficientwise before
evaluating each fixed-spin partition contribution, as explained in
[PARTITION.md](PARTITION.md).

Outputs carry `coefficient_convention: unit_identity_upsilon_coefficients_2026-09-24`
and `sewing_convention: fixed_tube_sign_geometric_BPZ_both_channels_2026-09-24`.
The reducer rejects old or mixed conventions. `normalization_factor: 1`
means no multiplier is applied after sewing.

## Directed checks

* The actual special-function ratios were evaluated at `b=0.8,1,1.4`, two
  momenta, and three identity regulators, including offsets across the
  delta-function peak. All 30 cases passed; at regulator `10^-6` the corrected
  family ratios are within `10^-5` of one. The independent Barnes-G formulas
  at `b=1` agree within `5.47e-17` (their public return values are binary64).
  This test used 40-digit internal arithmetic and took 27.92 seconds.
* A historical finite Clifford reduction used rephased Ramond kets and
  reported small algebraic residuals. It does not certify the full
  nonchiral contraction in the paper's `w^+,w^-` basis. The current native
  test checks the four paper-basis ground Gram entries and the exact
  all-NS sign obstruction.

The test code is `Code/c_Recursion/test_nsrr_identity_normalization.py` and
`C++/tests/partition_sewing.cpp`. Numerical evidence is saved under
`C++/results/partition_normalization_resolution_2026-09-24/`.

From the repository root, reproduce the normalization checks with:

```sh
make -C C++ partition bin/partition_sewing
C++/bin/partition_sewing
python3 Code/c_Recursion/test_nsrr_identity_normalization.py --output /tmp/nsrr-identity.json
```

The special-function test requires `mpmath` and `numpy`. See
[PARTITION.md](PARTITION.md) for the fixed-spin sewing audit.

The structure-constant normalization result is separate from the
fixed-spin sewing check. The literal paper block without geometric BPZ
conversion fails the free-fermion spin-geometry test; see
[PARTITION.md](PARTITION.md) for the converted prescription and its limits.
