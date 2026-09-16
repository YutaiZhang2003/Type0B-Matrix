# Directed low-level glasses checks

Run from the repository root:

```sh
python3 C++/experiments/arbitrary_plumbing_2026-09-16/glasses_run.py
```

This builds four small C++ test drivers, exports their local Ramond tensors for the mixed-channel Python check, and reruns the directed checks. All arithmetic is complex machine precision. `glasses_manifest.json` records commands, source hashes, and elapsed time. No production header or manuscript is modified.

## Graph and conventions

The separating NS edge is the first edge, called B; left/right handle edges are L,R. Each original trinion has B at infinity and the two occurrences of its handle at 1 and 0. Edge-pair to vertex regrouping has no residual global Grassmann sign in this order. Auxiliary/physical regrouping **still has** the self-loop sign

\[
(-1)^{\epsilon_L\delta_L+\epsilon_R\delta_R}.
\]

The transported NSRR vertex introduces, in addition,

\[
(-1)^{\sum_{v\text{ NSRR}}f_v(A_v+\epsilon_{v,2})}.
\]

Both primary intrinsic parities are zero in these tests. For two Ramond handles this transport reduces to \((-1)^{f(\epsilon_L+\epsilon_R)}\). For a single right Ramond handle it is \((-1)^{f(A_B+\epsilon_R)}\). The \(fA_B\) term is essential in the mixed test: omitting it reverses every nonzero odd-bridge coefficient.

The enlarged sewing weight is \(i^{2n_B\bmod2}\) for each NSRR vertex incident on the bridge. Thus the two-Ramond case gives \((-1)^{2n_B}\), while the mixed case retains a single \(i\) for an odd branch. Ramond loops are split when \((-1)^f\eta_v=-1\). The numerical insertion is the production-normalized \(Q\Theta\psi\), i.e. the same normalization used for the auxiliary factor, so it cancels out of recovery.

The two segment lifts enter through their product because the inserted operator is even. Each segment power is retained separately in `glasses_full_split.cpp`.

## Independent computations

The double-Virasoro numerator is **not** defined by multiplying a physical SCA block by the auxiliary factor.

* R/R numerators use production `LowAnchors::raw`, `MiddleBranching::raw`, branch norms, and the exact product of two Virasoro blocks sewn on the ordinary or split glasses graph.
* NS/NS numerators use the independently implemented human-note NNN branching tensors from `ns_fusion_data` and the same two Virasoro networks.
* Mixed NS/R numerators use one NNN tensor, one production NSRR tensor, the split vertex when needed, and the explicit single-NSRR bridge phase.
* Physical references use `ScaWard` and SCA Gram inversion for Ramond handles; all-NS tensors and Grams use `NSDescendantThreeForm` and `NumericNSVermaModule`. They do not call branching or Virasoro algorithms.
* Auxiliary references are direct fermion Ward forms and inverse auxiliary norms. The full split test evaluates the mode matrix of \(Q\Theta\psi(1)\), including the modes that change the auxiliary level. It is not a diagonal zero-mode-only test.
* Diagonal recovery uses only the computed numerator and auxiliary factor, performs triangular division in the known loop sector, and then compares the recovered physical series to PBW.

At the chosen cutoff each **ordinary or split Virasoro edge has descendant level at most 1**. Thus the CCY answer is exactly its global term: no Kac residue or nonconstant Schottky vacuum seed can enter. These checks test the new graph assembly, branch normalization, Grassmann signs, insertion, convolution, and recovery. They do not retest the higher-level CCY residue implementation.

The one-copy global coefficient uses the standard Ward value

\[
\rho(L_{-1}^{i}v_1,L_{-1}^{j}v_2,L_{-1}^{k}v_3)
\]

and edge norms \(1\) or \(2h\). In a self-loop, the two local slots use the same descendant label; in a split loop, the inserted middle vertex joins two separately summed descendants. An original physical loop coefficient pulls back to equal powers on both split segments, while auxiliary powers need not be equal.

## Saved results

Parameters are \(b=7/5\), \(P_B=11/23\), \(P_L=13/29\), \(P_R=17/31\).

| Check | Cases/components | Maximum scaled discrepancy |
|---|---:|---:|
| R/R diagonal numerator vs independent PBW convolution | 8 / 768 | 7.02e-13 |
| R/R recovered physical block vs PBW | 8 / 768 | 8.71e-14 |
| R/R full split numerator, including unequal segment levels | 8 / 1728 | 8.30e-13 |
| NS/NS numerator vs PBW convolution | 2 / 432 | 1.45e-15 |
| NS/NS recovered block vs PBW | 2 / 432 | 1.90e-15 |
| NS/R numerator vs PBW convolution | 4 / 576 | 3.17e-14 |
| NS/R recovered block vs PBW | 4 / 576 | 2.79e-14 |

The scaled error is \(|x-y|/\max(1,|x|,|y|)\). Counts include constrained zero coefficients; 960 of the full split R/R components have genuinely unequal powers on at least one split pair. R/R tests run both \(f=0,1\), every \(\eta_L,\eta_R=\pm1\), and every spin-lift parity slot. This includes no insertion, one inserted loop, and two inserted loops. The uninserted enlarged block in each obstructed case is separately checked to vanish.

Ordinary/split coefficient limits are individual physical level 1 on each edge, with the bridge also containing its half-integer level. The NS/R assignment is represented with the R handle on the right; exchanging the named handles gives the isomorphic R/NS assignment. No claim is made of testing external Ramond punctures, arbitrary genus numerically, intrinsic-odd primaries, or every higher-level coefficient.

Authoritative output files:

* `glasses_results.json`: R/R diagonal numerator, recovered PBW, and all coefficients.
* `glasses_full_split_results.json`: full split R/R comparison and component counts.
* `glasses_ns_assignments_results.json`: NS/NS and mixed NS/R comparisons and recovery.
* `glasses_ramond_local_data.json`, `glasses_fermion_loop_data.json`: freshly exported local tensors used by the mixed test.
* `glasses_manifest.json`: exact build/reproduction commands and source hashes.
