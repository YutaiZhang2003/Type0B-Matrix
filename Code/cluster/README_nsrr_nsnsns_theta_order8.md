# NSRR theta / NSNSNS theta order-eight Cannon run

This workflow evaluates the same marked genus-two surface in two theta
plumbing charts related by the partial modular `S` transformation on the
second handle (including the integer period-basis adjustment of the target
chart).

The source NSRR integrand is constructed in two stages:

1. generate the NS--R--R branching coefficients with their finite-cutoff
   Ward/branching recursion, using the certified low-state data only as
   anchors;
2. compute each factor in the double-Virasoro product with the ordinary
   genus-two Virasoro `c`-recursion.

The direct PBW genus-two block contraction is not used in the production
integrand. The target all-NS block uses collision-aware, 70-digit N=1
`c`-recursion. Both block constructions are truncated at total order 8.

The convergence axis uses per-edge Gauss--Laguerre orders 8, 10, and 12. It
contains 6,480 immutable shards:

- 3,240 source NSRR momentum nodes;
- 3,240 target NSNSNS momentum nodes.

Source and target nodes are separate Slurm arrays, with many immutable shards
evaluated sequentially per array element to stay below Cannon's account-level
job submission limit. The source defaults to 768 shards per element, `8G`, a
five-hour time limit, and five array elements. The target defaults to 1,024
shards per element, `4G`, a five-hour time limit, and four elements. A deterministic
reducer runs only after both arrays finish and refuses stale, missing, or
fingerprint-mismatched shards.

The physical scalar-plus-Majorana denominator is evaluated separately in the
same local theta frame as each numerator and raised to

```text
kappa = c_super-Liouville/(3/2) = 1 + 2(b + 1/b)^2.
```

The source determinant selected directly by its plumbing lifts is the NS
reference `[00|10]`; the required NSRR spin `[01|10]` is obtained by the
explicit theta-constant ratio. The modular characteristic transport gives
the target `[00|00]` spin.

## Local preflight

```bash
export TYPE0B_STRINGMC_ROOT=/Users/yutaizhang/Desktop/Project/StringMC
export PYTHONPATH="$PWD/Code:$PWD/Code/genus_2:$PWD/Code/c_Recursion:$PWD/Code/full_ramond_block_runtime:$PWD/Code/genus_2_cross_channel:$PWD/Code/ramond_branching_recursion:$PWD/Code/double_virasoro/nsrr:$TYPE0B_STRINGMC_ROOT"
python3 Code/genus_2/nsrr_nsnsns_theta_cannon.py \
  --config Code/config/nsrr_nsnsns_theta_order8_cannon_20260829.json plan
python3 -m unittest Code/genus_2/test_nsrr_nsnsns_theta_cannon.py
```

An actual order-eight source-node preflight on the development machine took
about 11 seconds. Its maximum normalized Ward residual was
`2.7e-10`. The largest-momentum order-12 node took about 14 seconds and had
residual `4.5e-10`; production fails closed above the configured `1e-7`
limit. A corresponding target node took about 2 seconds.

## Stage and submit

```bash
Code/cluster/stage_submit_nsrr_nsnsns_theta_order8.sh \
  cannon \
  /n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/nsrr_nsnsns_theta_r8_n8_10_12_20260829_v1 \
  /n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus2_period_table_20260718_176ab14d/.venv/bin/python
```

The wrapper stages a code snapshot, including the exact six StringMC plumbing
modules and pure-Python SymPy package used by the local double-Virasoro
preflight, checks the numerical
environment and unit tests remotely, validates the period/spin ledger,
submits the two arrays, then submits the dependent reducer. It records all job IDs locally in
`Data Set/nsrr_nsnsns_theta_order8_cannon_submission.json`.

Resource defaults can be overridden with `NSRR_G2_SOURCE_MEMORY`,
`NSRR_G2_SOURCE_TIME`, `NSRR_G2_SOURCE_ARRAY_CAP`, and their `TARGET`
counterparts. The moderate production defaults keep both time limits at five
hours.
