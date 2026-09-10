# Direct check of the unsaturated convolution kernel

The user-requested check confirms numerically, through total plumbing level
3, that opposite vertex signs give a nonzero physical block but an
unsaturated enlarged block consistent with zero.

Parameters: `b=7/5`, `(P1,P2,P3)=(11/23,13/29,17/31)`, in the notes'
algebraic momentum convention. Arithmetic: 384-bit FLINT complex midpoints.
Scalar clipping was disabled (`THRESHOLD=0`); the calculation uses no sector
projection, inverse convolution, or zero-mode insertion. These are numerical
residuals, not certified interval bounds or a proof at arbitrary level.

## Computations

- Physical F: direct PBW sewing using physical Gram matrices and Ward forms.
- Enlarged Fhat: separately evaluated ordinary double-Virasoro branch sum,
  using ordinary Virasoro Gram matrices and Ward forms.
- Auxiliary F_F: direct auxiliary oscillator sewing with the identity
  propagator, summing both Ramond ground labels.
- Convolution: multiply the separately computed F and F_F with the draft's
  star product and compare with the directly computed Fhat.

All `p1=0,1`, `f=0,1`, and opposite pairs `(+,-),(-,+)` were evaluated.
Each of these eight cases retains 30 plumbing monomials and eight parity
coefficients, for 1,920 coefficient slots. Retaining the parity coefficients
covers all eight evaluations of the plumbing signs. One equal-sign control,
`p1=f=0`, `(eta,eta')=(+,+)`, was also computed by the same routines.

## Results

| Quantity, maximum over the opposite-sign cases | Absolute value |
|---|---:|
| Direct enlarged coefficient | 6.857817e-114 |
| Convolution coefficient | 5.175022e-115 |
| Direct enlarged minus convolution | 6.387108e-114 |
| Physical sector-identity residual | 2.494321e-115 |

The auxiliary sector residual was zero in the computation. Every
opposite-sign physical block had coefficients of magnitude about 1.02.
For example, at `p1=0`, `f=1`, `(eta,eta')=(+,-)` and all plumbing signs
equal to +1, the physical ground coefficient is -2i, whereas the enlarged
ground coefficient is zero.

The equal-sign control has physical ground coefficient 2 and enlarged
ground coefficient 4 at all positive plumbing signs. Its full convolution
agrees with its direct enlarged series to 4.403330e-113.

## Runtime and saved data

The successful run took **3.264242 seconds**, including backend loading:
auxiliary sewing 0.074930 s, physical sewing 0.261853 s, enlarged sewing
2.086277 s, and convolution 0.372449 s. Remaining time includes loading,
diagnostics, and in-memory serialization. The final file write is excluded.

All coefficients and per-case residuals are saved in
[unsaturated_kernel_level3.json](unsaturated_kernel_level3.json).
The focused runner is [check_unsaturated_kernel.py](check_unsaturated_kernel.py).
An initial invocation stopped on a runner argument mismatch before any
enlarged-block calculation; the mismatch was corrected before this run.

Reproduce from the repository root:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /private/tmp/zero_mode_precision_env/bin/python \
  Code/ramond_zero_mode_recovery/check_unsaturated_kernel.py --level 3 \
  --json Code/ramond_zero_mode_recovery/unsaturated_kernel_level3.json
```

No other validation suite was run for this request.
