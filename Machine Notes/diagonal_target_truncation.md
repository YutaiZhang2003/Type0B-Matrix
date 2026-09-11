# Avoiding doubled middle-edge levels in the punctured construction

This note derives the diagonal-target truncation. It is now implemented together with the [forward CCY recurrence](../Code/theta_fermion_ccy/FORWARD_CCY.md). A complete revised level-ten machine pipeline has been timed; no new PBW run was performed. The earlier estimates in [the runtime-estimate note](diagonal_truncation_runtime_estimates.md) predate the forward recurrence.

## Convolution closes on equal middle powers

The physical block depends on the split plumbing parameters through q2 = q2_left*q2_right. Consequently its two middle exponents are equal. In the modified enlarged-block factorization, extraction of equal middle powers gives

\[
[\hat q_{2,1}^{k}\hat q_{2,2}^{k}]\widehat{\mathbf F}
=\sum_{j=0}^{k}
([q_2^j]\mathbf F)\star
([\hat q_{2,1}^{k-j}\hat q_{2,2}^{k-j}]\mathbf F_{\mathsf F}).
\]

The coefficients in this equation remain series in q1 and q3, and the star includes their convolution and the existing parity algebra. Thus the diagonal enlarged numerator and diagonal auxiliary factor suffice to recover the physical block with the same invertible constant on the relevant sector ideal. Off-diagonal physical coefficients are identically zero in the exact factorization, so they do not enter this restricted convolution.

The universal Schottky factor and its square also have equal middle exponents; removing or restoring them preserves the statement. This applies to the auxiliary-fermion insertion used by the current pipeline, whose SCA insertion is the identity. It is not a claim about a general punctured block with a nontrivial physical insertion.

Equal-power extraction must occur **after multiplying the two Virasoro series and including the branching shift**. Their individual off-diagonal coefficients can combine into an equal-power coefficient. Discarding those individual coefficients would be incorrect.

## Required Virasoro index set

Write the accumulated descendant indices in the product as (a,b,c,d), corresponding to (1,L,R,3). The fixed branching shift in the pipeline is

\[
(4n_1^2,\ 2n_2^2-\tfrac18,\ 2(n_2')^2-\tfrac18,\ 4n_3^2-\tfrac14).
\]

To contribute to an equal-power coefficient through physical total level N, a lower descendant index must be extendable to equal middle powers within that budget. The required downward-closed set is therefore

\[
a+d+\max\left(2n_2^2-\tfrac18+b,\ 2(n_2')^2-\tfrac18+c\right)
\le N-2n_1^2-\left(2n_3^2-\tfrac18\right).
\]

For integer descendant indices the bound on the right is rounded downward. This is precisely the coordinatewise downward closure of the desired product coefficients: any permitted lower index can be extended by increasing the smaller shifted middle exponent until both are equal. Conversely, every lower index of a desired coefficient satisfies this inequality.

Each individual Virasoro series needs this set because the other series has a level-zero term. CCY lowers one level and therefore stays inside the set. Scalar products only need to assemble the final equal-power targets; individual Virasoro inputs must retain the required unequal powers inside the closure.

For the lowest branch the condition is a+d+max(b,c)<=N. In particular **b<=N and c<=N**, instead of allowing either middle level to reach 2N. Nonzero branching shifts tighten the bounds further.

## Direct structural counts

These counts use the same pole-expansion caching as the current Python engine and assume no additional exactly-zero residue or branching skips. They are obtained by integer generating-function sums, without evaluating any block.

| Level | Current full four-variable cutoff: pole additions | Proposed diagonal-target closure: pole additions | Reduction factor |
| --- | ---: | ---: | ---: |
| 10 | 87,397,160 | 17,977,704 | 4.86 |
| 15 | 7,402,804,416 | 1,608,644,832 | 4.60 |

At level 15, the number of computed Virasoro series decreases from 1,416 to 1,248; cached nonterminal expansions from 148,938,956 to 52,192,468; and stored child terms from 1,261,927,272 to 335,907,712. The largest individual series requires 725,280 nonterminal expansions and 5,913,188 stored terms. These are workload reductions, not measured runtime speedups.

This removes the unnecessary doubled middle-edge cutoff, but leaves a substantial four-edge recursion workload. It does not by itself establish that CCY will outperform PBW.

The former machine-precision computation allowed numerical off-diagonal residuals to propagate during division. Restricting recovery to the exact diagonal support changes that floating-point computation, so bitwise equality with the old output is not expected. Subsequent validation and timings are recorded in the forward CCY note linked above.

## Reproduce the count

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/count_diagonal_target_work.py --levels 10 15 --json Code/theta_fermion_ccy/results/diagonal_target_work_counts_level10_level15.json
```

- [Counting script](../Code/theta_fermion_ccy/count_diagonal_target_work.py)
- [Results](../Code/theta_fermion_ccy/results/diagonal_target_work_counts_level10_level15.json)
