# NS–R–R chi finite evaluator

This directory does **not** contain a closed Ramond branching formula.
It implements a finite expansion of the 2016 chi strings into descendant
three-point functions, which are then evaluated by Ward identities. The
audit is an implementation regression test against the stored Ward data; it
is not an independent verification of a product or determinant formula.

`nsrr_chi_formula.py` implements the exact finite binary-path expansion of
the 2016 Ramond chi strings.  It returns raw and normalized NS–R–R branching
three-point functions using exact symbolic arithmetic.

Run a fast structural check with

```bash
python3 python/nsrr_chi_branching/audit_stored_values.py
```

Run the exhaustive comparison with every stored restriction at both exact
momentum samples with

```bash
python3 -u python/nsrr_chi_branching/audit_stored_values.py --full --samples 2
```

The exhaustive audit checks 864 stored low-grid values, the twelve direct
Ramond norms, sixteen additional values at NS branch `n=3/2`, and 64
independent checks with signed Ramond branch labels.
