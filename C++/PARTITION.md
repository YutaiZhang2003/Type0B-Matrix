# Fixed-spin genus-two partition comparison

`bin/partition` evaluates the block returned by each chiral algorithm at
one specified tube-sign triple. The Ramond source uses the double-Virasoro
restricted inverse; the all-NS target uses superconformal c-recursion and
the Schottky large-c seed. Both return coefficients in the BPZ sewing
convention of the theta chart. The partition driver applies no further
coefficient-dependent phase or sewing matrix.

The code's paper edge order is `(infinity,1,0)`. One matched surface has
two separately transported marked spins:

| spin | NS-R-R tube signs | all-NS tube signs |
| --- | --- | --- |
| first | `(1,-1,1)` | `(-1,-1,-1)` |
| second | `(-1,-1,1)` | `(-1,-1,1)` |

The block state sum contains the Koszul sign `(-1)^K`. Its BPZ sewing
also contains the tube inverse-Gram factors `(-i)^epsilon_e` and the
infinity-slot vertex factors `i^epsilon_(v,1)`. In the theta chart their
product is `(-i)^(epsilon_1+epsilon_2+epsilon_3)(-1)^epsilon_1`.
The double-Virasoro code includes it before returning `F`; the NS
recursion and direct PBW code use the same convention. The independent
free-Majorana test fixes this local spin lift: tube-sign relabeling alone
cannot change all three first pair terms.

The local theta PBW oracle based on `Human Notes/SCblock.tex` returns the linear-pairing
coefficients before this BPZ factor. With identical `f`, vertex-sign and
primary-parity labels, multiplying its components by the displayed
factor agrees with the current C++ PBW block through independent edge
level 1: 1536 components, maximum absolute difference `8.89e-16`.
Run `python3 C++/tools/check_theta_pbw_oracle.py --level 1` to rebuild
the executable and repeat that comparison. This is an independent
theta implementation in the local package, not a comparison with
Yutai's sphere four-point or torus two-point blocks. The amplitude package uses
half-sized NSRR coefficients together with a half-sized Ramond
continuum metric; this is the Ramond-field rescaling described in the
machine notes.

Run python3 C++/tools/check_yutai_local_forms.py for the local NSRR
form comparison: it checks 224 components against the torus production
Ward solver and 168 components in both orientations on the actual
sphere four-point support.
These tests compare local forms in the same momentum frame, not sphere
or torus block arrays.
The direct auxiliary factor comparison is
python3 C++/tools/check_theta_auxiliary.py; all 22 nonzero components
through total twice-level 4 agree with the local theta oracle to 1.2e-16.
The Type 0B sphere block check is
python3 C++/tools/check_yutai_sphere_blocks.py. Across all three
families, 864 C++ PBW coefficients through internal level 2 agree
with Yutai's PBW arrays to 3.9e-16 after the defined local form and
metric frame conversion. His PBW and double-Virasoro sphere arrays
agree on 1600 components to 1.3e-15 at c=3. At physical c=27/2,
our PBW coefficients agree with one stored Type 0B bank node in each
sphere family on 256 supported components through internal level 2;
the maximum absolute difference is 7.2e-9. These banks use a finite-b
extrapolation, so the comparison is not an error bound at other nodes.
The native sphere and theta
raw arrays use different local form frames; the partition driver
applies no such conversion.

The actual Type 0B torus two-point open-R-edge block is checked by
`python3 C++/tools/check_yutai_torus_open_bpz.py`. A separate C++ PBW
probe uses the (1/z) BPZ metric on the two sewn edges, leaving the
external R punctures open. Through NS twice-level 2 and R level 1,
including an external (G_{-1}) on either puncture, 576 coefficients
agree with Yutai's `MixedNSRamondPlumbingBlock` to (2.78\times10^{-16})
after the explicit sewn-edge BPZ phase. This compares chiral block
coefficients, not the integrated torus amplitude.

For each fixed spin, the all-NS contraction is

```
Z_NS = integral [prod_e dp_e/pi |q_e^h_e|^2]
       sum_(a=0,1) (-1)^a C_a^2 |F^(a)(q;eta_e)|^2.
```

For the Ramond source, with the ordered odd three-point form and
`C_(1,eta)=C_(0,eta)`, the full contraction is

```
Z_NSRR = (1/4) integral [prod_e dp_e/pi |q_e^h_e|^2]
         sum_(f=0,1; eta=+,-) C_(f,eta)^2 |F_f^(eta,eta)(q;eta_e)|^2.
```

The source calculates both `f` blocks and verifies
`F_1^(eta,eta) = -F_0^(eta,eta)` for the same `eta`. It therefore also
equals the reduced `f=0` formula with prefactor `1/2`. The sum over
`eta` labels three-point forms within one surface spin, not a sum of
surface spin structures. The structure constants and their identity
normalization are given in the machine notes' fixed-spin appendix.
The long draft currently states `C_(1,eta)=C_(0,-eta)`, which conflicts
with its own odd-form ground values and this fixed-spin prescription.
Reweighting the 20 saved spin comparisons with that exchanged coupling
raises the maximum ratio deviation from `1.59e-4` to `2.94e-2`.
Run `python3 C++/tools/check_nsrr_coupling_relation.py RUN_DIRECTORY`
to audit completed node outputs without recomputing blocks.

For the fresh 40-digit block run on one matched surface, the source
used a `7^3` momentum grid and total level 5; the target used `10^3`
and level 8. With three workers, their fresh wall times were 92.84 s and
71.64 s. The two fixed-spin ratios, after the specified free-frame
factor, were `0.9999726372542113` and `1.0000141626411249`. The
source odd-form identity residual was at most `6.44e-39`. Geometry,
quadrature and structure-constant inputs have approximately machine
accuracy, so these are finite-cutoff comparisons. The period transport
and independent free-Majorana checks are described in the machine
notes. Generated node and result JSON lives outside this code-only
checkout.

The same fixed-spin comparison on all ten matched surfaces completed
in 704.27 s with three surfaces and three workers per surface in
parallel. All 20 individually transported spin ratios lie between
0.9998405584 and 1.0000233972; the maximum deviation from one is
1.59442e-4. The archived period-transport residual is at most
2.49e-16. The ten-point driver is C++/tools/run_partition_ten_points.py.
The new BPZ-native convolution path was then run fresh with the same
cutoffs and physical inputs. It completed in 732.60 s and returned the
same twenty ratios to the reported digits. Its retained summary is
`/private/tmp/native_bpz_20260925/ten/summary.json`; generated results
remain outside the code checkout.

Build with `make -C C++ partition`. Supply the matched configuration
and physical momentum nodes to `bin/partition` separately for
`--channel source` and `--channel target`, then combine complete runs
with `--reduce-source` and `--reduce-target`. The driver rejects
incomplete grids and mixed conventions. `make -C C++ check-partition`
runs the directed sewing and low-level PBW comparisons.
