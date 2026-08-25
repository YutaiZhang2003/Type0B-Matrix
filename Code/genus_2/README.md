# Type-0B genus two: theta and glasses channels

This folder contains the parity-sensitive assembly of the all-NS genus-two
theta- and glasses-channel contributions.  The theta sewing convention comes
from `Human Notes/SCblock.tex`, especially `NSblockThetaDefinition`; the
derived glasses recursion and its PBW audit are recorded in
`Machine Notes/c-Recursion/ns_genus_c_recursion.tex`.

The chiral block uses the relative label

```text
a = A + C + E mod 2,
```

while the nonchiral partition term carries

```text
(-1)^(a + p1 + p2 + p3).
```

The antiholomorphic block is selected by

```text
a + sum(p_i) = a_tilde + sum(p_tilde_i) mod 2.
```

For the diagonal propagating Type-0B NS continuum, the internal primaries are
even in both chiral halves.  The theta numerator at each momentum node is
therefore assembled as

```text
C_0^2 |q^h F_0|^2 - C_1^2 |q^h F_1|^2,
```

not as the old unsigned sum.  `theta_partition.py` also records the discrete
transport in the chiral `c`-recursion: an odd level-`rs/2` null flips the
relative sector and the two spectator plumbing lifts, and contributes the
lift on the null edge.

The finite-c blocks, fusion polynomials, Schottky vacuum seed, and numerical
quadrature remain in `Code/c_Recursion`.  Both the local evaluator and Cannon
worker import the assembly functions from this folder.  Existing production
*totals* use the old unsigned formula and must not be mixed with new totals.
An archive that retains separate sector contributions can, however, be
corrected exactly without reevaluating its conformal blocks:

```bash
PYTHONPATH=Code python3 Code/genus_2/recombine_theta_parity.py \
  Machine\ Notes/Genus\ 2/Archives/ns-genus2-fivepoint-r24-n10-production-repro.tar.gz \
  --output Data\ Set/ns_genus2_fivepoint_r24_n10_theta_parity_corrected.json
```

The utility verifies the archive hash-independent provenance fields, all
10,000 task indices, the sector decomposition of every theta node, and the
reconstruction of each old unsigned summary row before applying the sign.

Run the focused tests from the repository root with

```bash
PYTHONPATH=Code python3 -m unittest discover -s Code/genus_2 -p 'test_*.py' -v
```

This is the corrected all-NS contribution in one theta plumbing spin sector.
It is not, by itself, the full BRST-complete genus-two Type-0B amplitude: the
remaining NS/R channels, GSO sum, superghost measure, and odd-moduli treatment
must be supplied before comparing a complete genus-two free energy with the
matrix model.

## Glasses channel

At either glasses trinion the handle primary occurs twice, so its intrinsic
parity cancels.  The absolute parity and nonchiral sign are

```text
a_abs = a + p_bridge mod 2,
sign  = (-1)^a_abs.
```

Thus the Type-0B even-primary continuum again uses the even-minus-odd sector
sum.  `glasses_partition.py` is the single source of truth for this sewing
sign and for odd-null transport.  An odd handle null leaves the sector fixed
and flips the bridge lift; an odd bridge null toggles the sector and leaves
all lifts fixed.

`glasses_c_recursion_pbw.py` implements the independently testable
coefficient recursion.  It also fixes the glasses large-c seed: the two
self-loop trace-factorization signs cancel the graph polarization, so the
vacuum and global glasses functions multiply ordinarily.  The old extra
odd-sector flips of the vacuum handle lifts were a double counting.

Run the complete PBW audit through total physical level 4 with

```bash
PYTHONPATH=Code/c_Recursion:Code python3 \
  Code/genus_2/glasses_c_recursion_pbw.py
```

## Cross-sewing production check

`prepare_cross_sewing_config.py` derives a one-design production config from
the five-point convergence scan and certifies the branch-composed spin
transport before any numerical work is submitted.  The matched physical
lifts are `(+,-,+)` in the human-note theta edge order
`(zero,one,infinity)` and `(+,+,+)` in glasses.  Both represent `[00|00]`
after the affine genus-two characteristic transport.  The theta second beta
bit carries an affine shift; using `(+,+,+)` in both channels selects
different physical Majorana spin structures.

The denominator of

```text
Q_L = Z_L / Z_(X+psi)^9
```

is the physical partition function of one noncompact real scalar plus one
physical NS Majorana.  It is now evaluated entirely in the plumbing frame,

```text
Z_(X+psi)^pl = G_X^pl |P_X^pl|^2 |F_psi^pl|^2.
```

Here `F_psi^pl` is the physical-Majorana Fredholm determinant with the Human
Note descendant sign, and `G_X^pl` is the two-loop charge Gaussian derived
from charged free-boson pants sewing with `h(alpha)=alpha^2/2` and measure
`d alpha_1 d alpha_2`.  No period matrix or Riemann theta constant defines
this result.  This denominator is unrelated to the auxiliary Majorana block
`F_F` whose star inverse appears only inside the double-Virasoro computation
of an NS superconformal block.

```bash
PYTHONPATH=Code/c_Recursion:Code:Code/genus_2_cross_channel python3 \
  Code/genus_2/prepare_cross_sewing_config.py \
  Code/config/ns_genus2_cannon_fivepoint_r20_24_n8_12_axis.json \
  --output Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json
```

For the focused local rerun used to check signs before launching fresh
order-24 shards:

```bash
PYTHONPATH=Code:Code/c_Recursion:Code/genus_2_cross_channel python3 \
  Code/genus_2/rerun_human_note_genus2.py
```

To reuse an already completed numerator and recompute only the independent
physical free denominator at several mode cutoffs:

```bash
PYTHONPATH=Code:Code/c_Recursion:Code/genus_2_cross_channel python3 \
  Code/genus_2/recompute_ql_plumbing_free.py
```

After reducing fresh theta and glasses shards, use
`summarize_cross_sewing.py` to compare the result with both the old unsigned
theta assembly and the intermediate theta-sign-only correction.  The audit
fails closed if either channel does not carry `[00|00]`.
