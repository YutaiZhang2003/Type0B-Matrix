# Reproduce the new Type0B calculation

Run these commands from `/Users/yutaizhang/Desktop/Type0B-Matrix`.
They evaluate Type0B code only. The accepted Het numerical review stays
closed. The archive was extracted into its own reference directory; do
not copy its old Type0B checkout over this project.

## Runtime

The calculations used Python 3.11 and the versions in `requirements.txt`.
The existing interpreter used during this run is
`/private/tmp/hetso23-logic-review-py311/bin/python`; it is an environment,
not a scientific dependency on Het results. For a durable new environment:

```sh
python3.11 -m venv .venv-type0b-genus1
.venv-type0b-genus1/bin/python -m pip install -r Code/type0b_genus1/requirements.txt
```

For the following commands choose the interpreter explicitly:

```sh
TYPE0B_PY=/private/tmp/hetso23-logic-review-py311/bin/python
export OPENBLAS_NUM_THREADS=1
TYPE0B_DATA='Data Set/type0b_genus1_20260920'
TYPE0B_REF='handoffs/reference/HetSO23_1to1_20260920/data_exports/spin23_genus1_vv'
```

The delivered manifest records platform, Python and library versions.
Numerical last bits can vary with the platform. No Git reset is part of
any preparation command.

## Read and verify existing results

These commands do not perform a new integration:

```sh
"$TYPE0B_PY" Code/type0b_genus1/provenance.py verify "$TYPE0B_DATA/MANIFEST.json"
"$TYPE0B_PY" Code/type0b_genus1/report.py
```

Read `RESULTS.md` and `$TYPE0B_DATA/RESULTS.json`. The manifest is a final
delivery snapshot, not a retroactively asserted preparation-time source
signature. Every NPZ bank independently records and checks hashes of its
arrays. Integration JSONs record the exact bank file hashes and the
Type0B module source hashes present when the integration was run.

The underlying current-recursion and frozen-PBW source closure is listed
in `MANIFEST.json`. The original ZIP SHA-256 is
`89aca730573414061aeb5bf9bb34981e23202019739c52497c9d806521d966d8`.

## Prepare compact-region banks

The supplied banks are sufficient for evaluation. To regenerate them:

```sh
"$TYPE0B_PY" Code/type0b_genus1/prepare.py --order 24 --cutoff 8 \
  --omega .25j --output "$TYPE0B_DATA/power24.npz"
"$TYPE0B_PY" Code/type0b_genus1/odd.py \
  "$TYPE0B_DATA/power24.npz" "$TYPE0B_DATA/odd_power24.npz" --cutoff 8
```

Orders 16/17 are prepared by replacing 24 with 16 and using the `power16`
output names. `prepare.py` uses the current NS recursion and current BRY
structure constants. Odd preparation computes new G−3/2 data; it does
not populate the odd sector by relabeling ordinary Ramond traces.

The ordinary collision bank contains only reusable Liouville blocks and
couplings. Its heterotic free-field factors are not used. Generate the
new odd-supertrace contact bank by calling the public preparation function:

```sh
"$TYPE0B_PY" - <<'PY'
import sys
sys.path.insert(0, 'Code/type0b_genus1')
from even import ROOT, BANK_ROOT
from collision import prepare_odd_contact
prepare_odd_contact(BANK_ROOT/'completion/ope_24_16.npz',
    ROOT/'Data Set/type0b_genus1_20260920/odd_contact.npz', cutoff=8)
PY
```

This calculation is currently assembled only for `.25j`. The generic
bulk-bank preparer accepts other complex energies, but the full evaluator
rejects an energy mismatch with the collision banks. New energies require
new collision couplings and their own continuation checks; changing only
`--omega` is not a valid full-amplitude energy scan.

## Evaluate the compact integral

```sh
"$TYPE0B_PY" Code/type0b_genus1/integrate.py \
  --height 3 --radius .12 --order 8 \
  --bulk-bank "$TYPE0B_DATA/power24.npz" \
  --odd-bank "$TYPE0B_DATA/odd_power24.npz" \
  --output "$TYPE0B_DATA/integral_power24_Y3_r012_n8.json"
"$TYPE0B_PY" Code/type0b_genus1/integrate.py \
  --height 3 --radius .08 --order 8 \
  --bulk-bank "$TYPE0B_DATA/power24.npz" \
  --odd-bank "$TYPE0B_DATA/odd_power24.npz" \
  --output "$TYPE0B_DATA/integral_power24_Y3_r008_n8.json"
```

The modular cap is included; it is not approximated by a rectangular
lower boundary. The puncture region is half of the centered torus,
weighted by two, with the collision disk removed. The analytical disk
is restored once. Identity and S-related charts are selected according
to their smaller cylinder gap. Geometry orders and descendant levels
are independent controls.

## Prepare and evaluate a matched cusp

Use the collision bank's momentum grid to keep the Ramond infrared
regulator common to the bulk tail and its disk. A ground-handle bank is
marked `leading-cusp` and cannot be passed to the full-torus evaluator.

```sh
"$TYPE0B_PY" Code/type0b_genus1/prepare.py \
  --collision-grid "$TYPE0B_REF/completion/ope_24_16.npz" \
  --long-ground --cutoff 10 --omega .25j \
  --output "$TYPE0B_DATA/cusp_24_16_L5.npz"
"$TYPE0B_PY" Code/type0b_genus1/odd.py \
  "$TYPE0B_DATA/cusp_24_16_L5.npz" \
  "$TYPE0B_DATA/odd_cusp_24_16_L5.npz" --cutoff 10
"$TYPE0B_PY" Code/type0b_genus1/cusp.py \
  --even-bank "$TYPE0B_DATA/cusp_24_16_L5.npz" \
  --odd-bank "$TYPE0B_DATA/odd_cusp_24_16_L5.npz" \
  --height 3 --radius .12 --order 24 --smax 128 \
  --output "$TYPE0B_DATA/cusp_matched_L5_Y3_r012_n24.json"
"$TYPE0B_PY" Code/type0b_genus1/cusp.py \
  --even-bank "$TYPE0B_DATA/cusp_24_16_L5.npz" \
  --odd-bank "$TYPE0B_DATA/odd_cusp_24_16_L5.npz" \
  --height 3 --radius .08 --order 24 --smax 128 \
  --output "$TYPE0B_DATA/cusp_matched_L5_Y3_r008_n24.json"
```

These runs return every even twice-level through 10, so their lower-order
rows provide the level-4 comparison without rerunning coefficient
preparation. The earlier matched level-4 bank was made with the same
`--collision-grid` but without `--long-ground`, through cutoff 8, under
the names `collision_grid_24_16.npz` and `odd_collision_grid_24_16.npz`.
Those names are used in the saved height-6 and geometry-order controls.

The `cusp_power16_*` failure diagnostics used independent infrared
quadratures. They can only be rerun with `--allow-unmatched-ir`; they
must not be added to a physical result. The saved output is retained
to document why the shared-regulator requirement was introduced.

## Component checks and final manifest

```sh
"$TYPE0B_PY" Code/type0b_genus1/validate.py \
  --even-bank "$TYPE0B_DATA/power24.npz" \
  --odd-bank "$TYPE0B_DATA/odd_power24.npz" \
  --output "$TYPE0B_DATA/validation_power24.json"
"$TYPE0B_PY" Code/type0b_genus1/report.py
```

`validate.py` saves spin-resolved modular S/T tests, independent collision
comparisons, exact-height-primitive comparisons with adaptive quadrature,
and puncture-domain area checks. The program tests those individual
identities; it does not assert convergence of the final physical answer.

If intentionally regenerating or editing outputs, save a new final
manifest **after** the calculations and report are complete:

```sh
"$TYPE0B_PY" Code/type0b_genus1/provenance.py write "$TYPE0B_DATA/MANIFEST.json"
"$TYPE0B_PY" Code/type0b_genus1/provenance.py verify "$TYPE0B_DATA/MANIFEST.json"
```

The displayed numerical precision is for reproducibility. It is not an
error estimate, and none of these commands currently produces a
certified physical amplitude.
