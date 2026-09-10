# Ramond recovery with a punctured CCY block

The requested end-to-end checks pass through total level five for
`b=7/5`, `P=(11/23,13/29,17/31)`, `p=f=0`, `(eta,eta')=(+,-)`.
The user selected direct fermion sewing for the auxiliary factor after
its level-ten benchmark. The full level-ten physical calculation is now
being timed; no level-ten runtime is claimed until that run completes.

The production path is:

1. Solve the middle primary matrix elements by the physical L1 Ward identity,
   using reusable action coefficients and low primary anchors.
2. Solve the outer branching coefficients by the existing L1 Ward system.
3. Evaluate the two genus-two one-point Virasoro blocks by CCY c-recursion.
4. Evaluate the full auxiliary factor from fermion mode Ward identities and
   sparse Fock-state sewing, in exact rational arithmetic before restoring Q/sqrt(2).
5. Divide in the minus ideal of the theta parity algebra to recover the
   physical superconformal block, restoring both universal CCY vacuum factors.

An exponent `(a,l,r,d)` means
`q1**(a/2) * q2_left**l * q2_right**r * q3**(d/2)`.
Here `d` is even. The cutoff is `a+l+r+d <= 2*N`, corresponding to
`q2_left = u*sqrt(q2), q2_right = sqrt(q2)/u`. The dependence on the split
parameter is retained throughout the quotient; off-diagonal terms must not
be deleted to enforce the desired answer.

## Required evidence

- [x] Derivation and implemented recurrence for the middle matrix element.
- [x] Genuine punctured CCY c-recursion, including its large-c seed.
- [x] User-selected direct auxiliary factor, including the full inserted
      fermion and all four-variable coefficients through physical level ten.
- [x] Full numerator and restricted convolution division.
- [x] Final physical coefficients compared to independent PBW through total
      q-level 5 (the first authorized cross-check).
- [x] Recovered block evaluated at distinct split parameters at fixed product
      (the second authorized cross-check).
- [ ] After both checks pass, cold runtime for an eta*eta'=-1 physical block
      through total q-level 10, including all production stages.
- [x] Separate detailed LaTeX notes under Machine Notes.
- [x] Renewed adversarial mathematical review of the complete selected algorithm.
- [x] Renewed adversarial presentation review, followed by revision.

The user explicitly allows only the two listed numerical cross-checks.
Component comparison suites must not be added or run. Mathematical review,
source inspection, and errors that expose invalid production states remain
part of implementing the requested algorithm.

Existing draft and Human Notes modifications predate this work and are not
part of this deliverable. Historical zero-mode Ward/Gram timings are not
timings of this algorithm.

## Saved level-five result

`results/ccy_direct_reflected_level5.json` contains the numerator, auxiliary,
recovered physical series, precisions, stage timings and source hashes.
`results/validation_ccy_direct_reflected_level5.json` records the two requested
checks. Across 4,648 parity-coefficient slots the maximum scaled PBW
difference is `4.71318e-52`. At fixed `q2=0.007`, changing the split by
`u=0.5,1,2` gives maximum scaled variation `2.95169e-60` over all spin signs.
The full level-five run took **82.1968 seconds**. This is not a level-ten
runtime or an estimate of one.

The reported pipeline computation time includes all mathematical stages
and intermediate checkpoint writes. It starts after imports and argument
parsing and ends before encoding and saving the final physical series.
It is therefore distinct from a full-process wall time.

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/pipeline.py \
  --level 5 --dps 100 --ward-dps 70 --auxiliary direct \
  --json Code/theta_fermion_ccy/results/ccy_direct_reflected_level5.json
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/validate_requested.py \
  --production Code/theta_fermion_ccy/results/ccy_direct_reflected_level5.json \
  --pbw Code/theta_fermion_ccy/results/pbw_level5_p0_f0_eta_plus_minus.json \
  --json Code/theta_fermion_ccy/results/validation_ccy_direct_reflected_level5.json
```

The adapter includes the native odd-form eta transport
`eta_native=(-1)^f*eta_physical`. Negative-label Ramond action coefficients
are computed on the reflected positive-label chart, avoiding a high-level
PBW reflection solve. The two requested checks above use this implementation.
Earlier Ising-backend results remain saved under their original names;
that backend is available through `--auxiliary ising` only through level five.

## Subsequently requested direct-definition auxiliary benchmark

The user separately requested the level-ten runtime of the auxiliary
fermion block computed directly from its definition, using ingredients
that generalize to arbitrary genus. `direct_fermion.py` implements exact
fermion mode Ward forms and sparse Fock-state sewing for the full ordered
`Q Theta psi(1)` insertion. The coefficient construction took **0.186281209 s**
in a fresh process; imports and the initial result save brought the elapsed
time to **0.290214667 s**, with **46.77 MiB** peak resident memory.

The run retained all unequal split-edge powers through balanced physical
level ten, and saved 9,586 nonzero rational parity coefficients after
factoring out `Q/sqrt(2)`. This was a benchmark without numerical comparisons.
The user subsequently selected this auxiliary method for production. Its
standalone runtime remains distinct from the full physical-block timing.

See [DIRECT_FERMION_BENCHMARK.md](DIRECT_FERMION_BENCHMARK.md) for the definition,
truncation and command, and
[results/direct_fermion_level10.json](results/direct_fermion_level10.json)
for the complete reusable series and timing metadata.
