# Fixed-spin genus-two partition comparison

The historical numerical inputs and outputs have been removed from this
code-only checkout. The retained driver accepts a supplied configuration and
momentum-node directory; the spin assignments and previous test setup are
documented here for reference.

In the recorded run, `bin/partition` read the two sets of plumbing parameters from
`results/partition_geometric_fixed_2026-09-24/inputs/config.json`. Each row
specifies **one** source spin and its transported target spin, in paper edge
order `(infinity,1,0)`:

| case | NS-R-R `eta_e` | NS-NS-NS `eta_e` |
|---|---|---|
| first | `(1,-1,1)` | `(-1,1,-1)` |
| second | `(-1,-1,1)` | `(-1,1,1)` |

The raw paper blocks contain their printed `(-1)^K` Koszul sign. For each
parity coefficient `epsilon=(epsilon_1,epsilon_2,epsilon_3)`, the driver
multiplies that coefficient by

`(-i)^(epsilon_1+epsilon_2+epsilon_3) (-1)^epsilon_1`

before evaluating it at the **single** specified `eta_e`. This is the
geometric BPZ conversion from the odd inverse Grams and the two
infinity-slot vertices. It is applied to both channels. No other tube-sign
assignment enters either contribution. The Ramond family sum over `eta=+,-`
labels three-point forms, not surface spin structures.

With primary propagation powers and continuum measure restored, the
implemented contractions are

```
Z_NS   = integral [prod_e dp_e/pi |q_e^h_e|^2]
         sum_{a=0,1} (-1)^a C_a^2 |F_geom^(a)(q;eta_e)|^2,

Z_NSRR = (1/2) integral [prod_e dp_e/pi |q_e^h_e|^2]
         sum_{eta=+,-} C_{0,eta}^2 |F_geom,0^(eta,eta)(q;eta_e)|^2.
```

The source computes both `f` blocks; at each fixed spin it records the
residual of `F_geom,1 + F_geom,0 = 0`. The factor `1/2` is the physical
Ramond ground-pairing normalization with the paper's `w^+,w^-` kets. The
structure constants obey `C_{0,eta}=C_{1,eta}` in this spectrum, which the
driver requires. The target also stores the unconverted paper blocks as a
separate diagnostic. Their literal diagonal squares are **not** substituted
for the geometric partition contribution.

The period and spin transport check is in `tests/partition_spin_geometry.py`.
The independent free-Majorana BPZ check in `tests/partition_bpz_geometry.py`
uses exactly the above raw fixed signs and reaches `3.1e-8` relative error
at total level 8. The native low-level PBW tests are in `make check-partition`.
The free-Majorana test and ground-state Clifford metric do not by themselves
constitute a full independent interacting NS-R-R descendant-pairing proof;
that scope limit must be retained when citing the numerical comparison.

To repeat the calculation with supplied physical input nodes, build with
`make -C C++ partition`, then run `bin/partition` separately for
`--channel source` and `--channel target`. The fixed geometry configurations
can be regenerated from the retained ten-point geometry with
`tools/prepare_partition_ten_points.py`. Use
`--level 5` for source, `--level 8` for target, `--dps 40`, and fresh output
directories. Reduce the two complete runs with `--reduce-source` and
`--reduce-target`. The driver rejects incomplete grids and mismatched sewing
conventions. The source and target nodes are recomputed from physical input
momenta, measures and three-point constants; no saved block value is read.
