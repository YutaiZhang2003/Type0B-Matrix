# Reproduction runbook

Run every command below from the bundle root.  The frozen data are included so
the inexpensive analysis and comparison stages do not require rerunning the
Monte Carlo integrations.

## 1. Sphere normalization and block kernels

```bash
python plumbing/audit_genus0_one_to_two_amplitude.py
python plumbing/audit_genus0_one_to_two_amplitude_checks.py
python plumbing/ccy_sphere_four_point_checks.py
python plumbing/ccy_sphere_five_point_checks.py
python plumbing/ccy_sphere_six_point_checks.py
python plumbing/ccy_sphere_six_point_star_checks.py
```

## 2. Sphere 1 -> 3

The expensive target-blind production driver is
`plumbing/sphere_four_point_worldsheet_scan.py`.  With the packaged frozen
inputs, rerun the target-free fit and only then its separate comparison:

```bash
python plumbing/sphere_four_point_imaginary_ray_fit.py
python plumbing/sphere_four_point_30point_matrix_comparison.py
```

## 3. Sphere 1 -> 4

The worldsheet kernel is `plumbing/sphere_five_point_equal_energy.py`; the
thirty-point target-blind extension is
`plumbing/sphere_five_point_30point_worldsheet_extension.py`.  Reproduce the
frozen fit and later comparison with:

```bash
python plumbing/sphere_five_point_30point_worldsheet_fit.py
python plumbing/sphere_five_point_30point_audit_summary.py
python plumbing/sphere_five_point_30point_matrix_comparison.py
```

## 4. Sphere 1 -> 5

The production driver is `plumbing/sphere_six_point_worldsheet_scan.py` and
the completed Cannon orchestration is in
`plumbing/sphere_six_point_cannon_blind.py`.  The current paired order-eight
analysis and its deliberately separate comparison are:

```bash
python plumbing/sphere_six_point_order8_current_fit.py
python plumbing/sphere_six_point_order8_current_comparison.py
```

## 5. Torus 1 -> 1

Run the kernel/block checks first:

```bash
python plumbing/torus_two_point_blocks_checks.py
python plumbing/genus1_two_point_worldsheet_checks.py
```

The target-blind production scan is
`plumbing/run_genus1_two_point_imaginary_scan.py`.  Reproduce the post-freeze
BRY fit from the packaged fifty-point scan with:

```bash
python plumbing/fit_genus1_two_point_bry_scan.py \
  --scan-dir plumbing/results/genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2
```

## 6. Torus 1 -> 2

The full chain is the three-point block, worldsheet kernel, channel atlas,
stratified bulk/direct-tail integrator, and h-dominant scan driver:

```text
torus_three_point_blocks.py
genus1_three_point_worldsheet.py
genus1_three_point_channel_atlas.py
smoke_genus1_three_point_channel_atlas.py
run_genus1_three_point_hdominant_scan.py
```

Run the kernel checks with:

```bash
python plumbing/genus1_three_point_worldsheet_checks.py
python plumbing/genus1_three_point_channel_atlas_checks.py
```

Revalidate the blind ten-point freeze, recompute the target-free shape fit,
freeze it, and only afterward regenerate the BRY-normalized comparison:

```bash
python plumbing/analyze_genus1_three_point_hdominant_scan.py \
  --scan-dir plumbing/results/genus1_three_point_worldsheet/hdominant_scan10_p8_h8l3_q030_007_n256_r4_v1 \
  --reused-t075 plumbing/results/genus1_three_point_worldsheet/channel_atlas_hdominant_t075_p8_h8l3_q030_007_n256_r4_v1.json
```

This last command requires Matplotlib to regenerate the comparison plot.  It
checks every blind input hash before writing the target-free analysis.

## Cluster scripts

The packaged Slurm files preserve the exact current Cannon orchestration, but
site paths and account/partition settings must be adapted before submission on
another cluster.  They are not needed for local kernel checks or postprocessing.
