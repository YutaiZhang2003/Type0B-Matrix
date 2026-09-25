# SO(7) × E8 four-point campaign: 32 momentum nodes

The user increased the budget from eight to **32 total nodes per internal
NS/R channel** after the eight-node pilot failed the fixed-coordinate crossing
check. The previous eight-node release is preserved in its original archive.

The existing sector-dependent threshold quadrature is unchanged: endpoint
0.18, tail 3.2, beta=2 for NS and beta=0 for R, a=1, s=0, and the same bulk
breakpoints. The 32-node rule allocates **5 endpoint, 19 bulk, and 8 infinite-tail
nodes**. These are dP weights; physical amplitude integration applies 1/pi once.

The block backend is the literature-convention double-Virasoro assembly, with
ordinary sphere CCY c-recursion for both Virasoro factors. The assembled tables
at b=1.01 and 1.02 use the requested Type0B endpoint extrapolation

    [4*(1+epsilon)*F(1+epsilon)-(1+2*epsilon)*F(1+2*epsilon)]/(3+2*epsilon), epsilon=.01.

Exact b=1 primary and elliptic prefactors multiply the resulting coefficients.
Physical block order starts at four, compares four with five, and increases
until the adjacent complex block values change by at most 2%, or order ten
is reached. A cap is recorded as unconverged. Branching and CCY arithmetic are
complex128; Upsilon constants use 70 digits. Signed complex physical momenta
are never conjugated in the analytically continued correlator.

The release includes the existing eight energies and reserves SS/SV/VS/VV
mixed amplitudes plus the existing four-R family assignment: 40 amplitude
cases. The four-NS moduli settings are preserved. The current executable
stages build **1,024 coefficient banks** and compare both channels for every
physical multiplet component at z=0.35+0.1i, 0.5+0.1i, 0.65+0.1i. The crossed
coordinate is 1-z with its actual opposite imaginary part; Möbius/exchange
phases and the NS descendant weight shifts are retained. Only internal
momentum is integrated for this crossing test.

The full-component gate is rerun with 32 nodes; the older four-component
32-node diagnostic alone does not pass this gate. A 2% crossing tolerance is
recorded separately from the user's 2% block stopping rule. Successful pure
super-Liouville crossing must be followed by verification of the physical
heterotic GSO/PCO/free-field assembly. The old failed mixed assembler is not
enabled by a successful block test.

Prepare a new immutable release:

```sh
PYTHONPATH=Codes python Codes/prepare_so7e8_literature_cannon.py --p-nodes 32 \
  --output data_exports/so7e8_literature_cannon_N32_20260920
```

Extract the source archive into a new Cannon directory, then run:

```sh
export PYTHONPATH="$PWD/Codes"
python cluster/submit_so7e8_literature.py --dry-run
python cluster/submit_so7e8_literature.py
```

Jobs use `yin_lab/shared`, one CPU and 8 GB each, with at most 128 concurrent
bank tasks. The crossing reducer depends on successful completion of every
bank task. Source, manifest and checkpoint hashes prevent mixing releases.
The submission journal prevents duplicate dispatch after uncertain sbatch
responses. Results and submission status are recorded beside the prepared
archive; preparing alone never submits a job.
