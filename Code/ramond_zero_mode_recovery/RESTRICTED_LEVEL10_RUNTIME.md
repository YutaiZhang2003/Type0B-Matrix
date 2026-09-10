# Restricted-inverse level-10 runtime

One complete physical NSRR theta-channel block was computed through total
level 10 in **116.4794 seconds** at **384-bit complex precision**. The
observed command wall time, including imports and output writing, was
117.1690 seconds. This is a fresh computation, with no previously stored
branching or block coefficients loaded.

The block is \(\mathbb F_0^{(+,+)}\) with intrinsic NS parity \(p_1=0\),
at \(b=7/5\) and \((P_1,P_2,P_3)=(11/23,13/29,17/31)\). All three
plumbing variables and all eight parity components are retained: 506
monomials and 4,048 coefficient slots, including zeros.

| Stage | Seconds |
| --- | ---: |
| Setup | 0.0090 |
| Branching primary data | 88.1086 |
| 776 ordinary Virasoro blocks | 15.7283 |
| Products of the two Virasoro series | 2.0436 |
| Assembly over 388 branch-label triples | 0.1014 |
| Enlarged block, total including loop overhead | **105.9918** |
| Ordinary auxiliary block | 4.6211 |
| Restricted-inverse domain requirement | 0.0371 |
| Restricted convolution recovery | 5.8293 |
| **Total computation** | **116.4794** |

Excluding branching-primary preparation, the measured remaining stages
total **28.3708 seconds**. This is a subtotal of this run, not a separately
timed run with loaded coefficients. For context, the earlier zero-mode
implementation took 299.1915 seconds at the same precision and parameter
point; other work overlapped part of that earlier measurement.

## Method and scope

The numerator is computed by the ordinary diagonal double-Virasoro branch
sum. Branching primary data come from explicit chi-string states and
three-point forms. Each ordinary Virasoro block is evaluated separately
using Virasoro Ward identities and inverse Virasoro Gram matrices. This
avoids forming the large tensor-product insertion matrices. This run
does **not** use the older binary64 c-recursion/branching-Ward-grid driver.

With \(x=\eta_2\eta_3\), the inputs are required to lie in
\(I_+=\{v:x\star v=v\}\), and the auxiliary constant is \(1+x\).
The physical coefficients are recovered in increasing total degree by

\[
\mathbb F_{\boldsymbol n}
=\frac12\left[
\widehat{\mathbb F}_{\boldsymbol n}
-\sum_{\boldsymbol0\ne\boldsymbol m\le\boldsymbol n}
(\mathbb F_{\mathsf F})_{\boldsymbol m}\star
\mathbb F_{\boldsymbol n-\boldsymbol m}\right].
\]

The inverse's domain requirement passed at tolerance 1e-30. The enlarged
input's maximum scaled distance from \(I_+\), measured before projection,
was 8.3677e-109; the auxiliary residual was zero. This is an algebraic
applicability requirement, not an independent error bound on the block.

During this timing run, **no independent cross-checks were run**: no direct physical
PBW block, zero-mode comparison, second cut, or extra parity/sign cases.
The arithmetic uses FLINT midpoints and discards scalar terms below 1e-80;
the result is not a certified interval enclosure.

The subsequently requested comparison with the saved zero-mode result
passed through level 10, with maximum absolute difference 2.29682e-106 and
maximum scaled difference 3.1067634e-108. See
[RESTRICTED_VS_ZERO_MODE_LEVEL10.md](RESTRICTED_VS_ZERO_MODE_LEVEL10.md).
This comparison is separate from the computation timed above.

## Saved outputs and reproduction

- [Physical block and full timing record](restricted_level10.json)
- [Enlarged block](restricted_level10.enlarged.json)
- [Auxiliary block](restricted_level10.auxiliary.json)
- [Branching prefactors](restricted_level10.branching.json), including
  normalization and theta sewing signs for this particular block
- [Computation script](compute_restricted_block.py)

The stored coefficients use high-precision decimal strings and specify the
twice-level and parity-index conventions in their metadata.

```sh
python3 Code/ramond_zero_mode_recovery/compute_restricted_block.py \
  --level 10 \
  --json Code/ramond_zero_mode_recovery/restricted_level10.json
```

The interpreter needs python-flint and the existing workspace dependencies.
The measured run used `/private/tmp/zero_mode_precision_env/bin/python`.
