# Exploratory Type-0B NS sphere five-point amplitude

This directory contains the attempted genus-zero (1\to4) all-NS Type-0B
worldsheet computation: the PCO integrand, boundary-domain ledger, numerical
drivers, plotter, and tests.  It is deliberately separated from the reusable
`c_Recursion` library because the integrated five-point amplitude has not
been numerically certified or frozen.

The implementation imports the general multipoint NS (c)-recursion and
BRY-normalized Liouville utilities from `Code/c_Recursion/`, and the plumbing
atlas from the supplied bosonic (c=1) reference implementation.

Run the attempt-specific tests from the repository root with:

```bash
PYTHONPATH='Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:Code/c_Recursion:Code' \
python3 -m unittest \
  Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_type0b_ns_five_tachyon.py \
  Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_type0b_ns_five_tachyon_domain.py
```

The generic multipoint-recursion tests remain in `Code/c_Recursion/`.
