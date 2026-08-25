# Genus-one 1→1 hybrid-recursion exercise

## Block prescription

The Liouville torus two-point function now supports the following production
split at `c=25`:

- necklace region: simultaneous internal-weight (`h`) recursion;
- OPE discs: fixed-weight central-charge (`c`) recursion in `(q,v)`, followed
  by the existing `v=exp(-i z)-1` re-expansion;
- direct descendant sewing: independent validation backend and a precision
  fallback for unstable resonant `h`-recursion momentum nodes.

The flat-frame normalization is unchanged. In particular, the OPE block still
receives

```text
q^(h_loop-c/24) v^h_ope (2 sin(z/2))^(-2 d)
```

outside the descendant recursion. No additional Weyl factor is introduced by
switching the coefficient generator.

At `b=1`, individual necklace `h`-recursion terms are resonant. The code uses
a three-point `c→25` Richardson limit and a symmetric generic-weight limit at
confluent poles. A second central-charge regulator audits every momentum node.
Nodes whose coefficient tensors disagree by more than `1e-7` use exact
finite-level sewing. In the audited ten-point run, 216–220 of the 256 necklace
momentum pairs required this conservative fallback; the OPE block remained
entirely `c`-recursive.

## Validation

- OPE `c`-recursion versus direct descendant coefficients through bidegree
  `(q^4,v^5)`: `1.35e-14` maximum relative error at generic weights and
  `3.6e-12` including coincident internal weights.
- Momentum-integrated fixed-modulus comparison: `1.0e-9` relative difference
  in the audited necklace channel and `4.4e-15` in the OPE channel.
- Fresh all-descendant versus hybrid rerun at `t=0.45`: absolute difference
  `2.07e-13` in the final native integral, or `1.80e-11` relatively.

## Audited amplitude scan

The v2 scan now uses fifty points below the first residue at `t=1`.  It
combines the earlier twenty-point grid with thirty additional stratified
points: `t=0.02,0.04,0.07`, followed by the same offsets in each successive
interval of width `0.1`, through `t=0.92,0.94,0.97`.  All fifty points use
momentum order 16,
necklace orders `(6,3)`, OPE orders `(3,8)`, four shared Sobol scrambles, and
the same collision-disc and cusp-tail treatment.  Every point records its
backend and regulator audit in its frozen JSON file.

The post-freeze weighted fit

```text
-i A^(1)(i t) = (-a t^2 + 2 b t^4 - c t^5)/24
```

gives

```text
a = 1.0016588 ± 0.0565665
b = 1.0053852 ± 0.0666207
c = 1.0093253 ± 0.0764546
```

where the displayed uncertainties are the shared-scramble RQMC errors. The
common-shape fit `a=b=c=kappa` gives

```text
kappa = 1.0082693 ± 0.0240434.
```

For comparison, the twenty-point fit gave
`(a,b,c)=(1.0018614,1.0069916,1.0123599)` and
`kappa=1.0112973 ± 0.0158314`.  The fifty-point central values again move
closer to `(1,1,1)`.  The nominal independent-point errors decrease to
`(0.0106241,0.0205939,0.0318074)`, but those errors are not the production
uncertainties because the same four scrambles are shared across all values of
`t`.  With only four shared replicates, adding points does not provide fifty
independent Monte Carlo measurements; the displayed production errors are the
scatter of the four complete fitted curves and can fluctuate upward.  The
`t^4` and `t^5` columns also remain strongly correlated on `0<t<1`.

The pointwise worldsheet/analytic ratios range from `0.99165` to `1.01474`.
The largest excursion is the new near-residue point `t=0.99`; its native value
is `-8.3699617e-4 ± 5.1325e-6`, compared with the analytic
`-8.2483583e-4`.  The independent-error chi-squared of the unconstrained
three-parameter fit is `0.4842` for 47 degrees of freedom; the common-shape
fit gives `3.1001` for 49 degrees of freedom.

Artifacts:

- `results/genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2/`
- `bry_postfreeze_fit.json`
- `bry_postfreeze_fit_10point.json`
- `bry_postfreeze_fit_20point.json`
- `bry_postfreeze_fit_50point.json`
- `genus1_two_point_hybrid_recursion.svg`
- `genus1_two_point_hybrid_recursion.png`
