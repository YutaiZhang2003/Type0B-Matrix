# RR reflection at omega = i/4: bulk trace and normalization audit

The requested point is recorded in
[`RR_OMEGA_I4_POINT.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_OMEGA_I4_POINT.json).
**Its worldsheet amplitude and error estimate are null.** The bulk
calculation exposed a relative NS/R normalization mismatch before the
moduli integral. This is now addressed by overall correlator prefactors:
two in the radial channel and `(2,2,1,1)` in the OPE theta sectors,
applied after descendant summation. Primary norms and inverse Grams are
unchanged. See [the implementation and factorization prescription](INTERNAL_NORMALIZATION.md).
The original raw results below are retained as the derivation of that
choice. The common physical amplitude constant remains free until
factorization. The matrix comparison value is

\[
a_{AA}^{(1)}(i/4)=-\frac{5i\pi^2}{49152}
=-0.00100398807791029445585\,i.
\]

This is the coefficient of `g_s^2 delta(omega-omega_prime)`. It is not a
measured RR amplitude. Neither this value nor a TT integral enters the
radial trace, OPE, or normalization tests.

**Quadrature follow-up:** the earlier small uniform grids did not separate
spectral resolution from the cutoff. The
[Ramond-specific momentum study](MOMENTUM_QUADRATURE.md) now checks the
full coupled endpoint powers, pole distances, independent edge refinement,
and finite tail shells. Its two independent modular rules agree within
`8.95e-8` at fixed block cutoffs and reproduce ratios `(1,2,1,1/2)` to
about `1.1e-6` relative. The discrepancy survives this quadrature check.
The normalized copies of those banks now satisfy this modular comparison
within `1.10e-6`. The raised overlap below is still a coarse diagnostic,
not a converged result; `normalized/overlap.py` is prepared to refine it
with correlator prefactors applied after both original state sums. The
interrupted inverse-metric trial is archived and is not a completed run.

## New bulk construction

`numeric_nrr.py` evaluates the generalized NS-R-R Ward system in complex
arithmetic. `radial_trace.py` constructs a second channel for the torus
pair directly from physical state sums. Its three-point matrix has slots

\[
T_e(N,R)=\langle N|R_e(1)|R\rangle,
\qquad e=+,-.
\]

The external Ramond state is at one and the loop Ramond state is at zero.
The two coupling coefficients are the stored `C_even/2,C_odd/2`.
The physical Ramond Gram matrix is induced from the full graded ket
embedding before inversion, with unit reference ground-state Hermitian
norm. The NS Gram is the ordinary Hermitian Gram. Both are unchanged by
the overall correlator prefactor. For real internal
momenta, analytic continuation uses

\[
T^\sharp(\omega)=T(\overline\omega)^*,
\]

where the second operation conjugates coefficients, not the physical
energy. Geometry is conjugated in the antiholomorphic factor.

With \(K=2\pi i\), the two cylinder intervals are
\((t_{\rm NS},t_R)=(z,\tau-z)\) for theta 1 and 2 and
\((\tau-z,z)\) for theta 3 and 4. A term of levels
\((n/2,\bar n/2;r,\bar r)\) carries

\[
(2\pi)^{4h_E}
 e^{K t_{\rm NS}(h_{\rm NS}+n/2-c/24)
       -K\bar t_{\rm NS}(h_{\rm NS}+\bar n/2-c/24)}
 e^{K t_R(h_R+r-c/24)-K\bar t_R(h_R+\bar r-c/24)}.
\]

Theta 1 inserts the complete Ramond state parity. Theta 4 inserts
`(-1)^(n+bar_n)`. The other two sectors use ordinary traces. The raw
spectral convention is `dP_NS dP_R/pi^2`. Unlike the marked-pants
construction, this trace does not assign a theta characteristic to an
unidentified plumbing lift.

For the raised puncture, the plane-to-cylinder transformation uses
`K*(G_-1+3 G0/4)` and its antiholomorphic counterpart. Radial reflection
reverses the two odd operators. The implementation retains this sign,
including at `G0 Gbar0`. In particular,

\[
L^{++}_{00}=-i\omega^2 L^{++}/2,\qquad
L^{--}_{00}=+i\omega^2 L^{--}/2.
\]

The generalized Ward and small-representation conventions can be compared
with [Suchanek, equations (26)–(31)](https://arxiv.org/pdf/1012.2974).
The full physical adjoint used here is checked separately; it is not an
assertion that the paper's individual NR and RN chiral forms are complex
conjugates.

## Checks that passed

[`RADIAL_TRACE_CHECKS.json`](../../Data%20Set/type0b_rr_genus1_20260921/RADIAL_TRACE_CHECKS.json)
records:

- 384 comparisons with the symbolic, nonchiral Ward construction at
  imaginary and genuinely complex external energy: maximum absolute
  residual `1.80e-15`.
- Both free charge branches and all four spin structures compared with
  independent boson/Majorana theta functions: maximum bare relative
  error `2.15e-8` at the higher tested cutoff.
- All four raised components compared with independently differentiated
  free correlators: maximum relative error `3.77e-6`, decreasing with
  the cutoff.
- Eight physical zero-mode checks: maximum residual `4.87e-17`.
- A complex-energy Cauchy–Riemann check with fresh couplings: relative
  residual `3.79e-9`.

These checks validate the numerical vertices and the free radial frame.
They do not fix the interacting continuum normalization.

## Historical raw interacting discrepancies

The independent channel comparison is made **after** integrating both
spectral momenta; the fixed-momentum blocks are different channels.
At `tau=0.13+1.2i`, `z=0.04+0.18i`, and `omega=i/4`, an 8-by-8 Gauss
grid on `[0,2.5]^2` gives the following bare `R+ R+` components:

| Sector | Radial trace | Local OPE | Radial/OPE |
|---|---:|---:|---:|
| theta 1 | 0.03060059735 | 0.03060496334 | 0.9998573 |
| theta 2 | 0.03407461563 | 0.03408072823 | 0.9998206 |
| theta 3 | 0.03744258757 | 0.07502136553 | 0.4990923 |
| theta 4 | 0.03626668990 | 0.07265993850 | 0.4991291 |

The disorder components show the same relative factor. The bare
Ramond-handle comparison has maximum discrepancy `7.64e-4`; the
NS-handle comparison fails by about 50 percent. These are finite-cutoff
diagnostics, not amplitude error bars.

A second test uses only the local OPE at `tau=i` and compares `z` with
`z/i`, where `z=0.04+0.1i`. Scalar S transport fixes theta 1 and 3 and
exchanges theta 2 and 4. Refining to 12 nodes per edge and momentum
cutoff 3.5 gives `R+ R+` transported/original ratios

\[
(1.0000044,\ 2.0004666,\ 1.0000131,\ 0.4998888).
\]

This makes the relative NS/R discrepancy visible independently of the
bulk adjoint. The earlier v4 physical-trace and local-sphere checks
remain valid algebraic checks; they did not certify this interacting
normalization or modular covariance.

## Identity-limit evidence and the adopted convention

The audit evaluates the **implemented** coupling formulas at
`P_identity=i*(1-epsilon)` and compares

\[
\frac{C^\pm(P,P+u\epsilon;P_{\rm identity})}
 {C(P,P_{\rm identity},P+u\epsilon)}.
\]

For three P values, three fixed u values, and decreasing epsilon, both
ratios tend to `1/2`; linear extrapolation has maximum residual `2.11e-8`
from that value. This includes off-diagonal approaches to the identity
peak, not just its diagonal value. These pointwise ratios alone do not
establish a delta-function normalization; the quadrature follow-up also
integrates the peaks against smooth test functions.

That integrated test is saved in
[`RR_IDENTITY_DISTRIBUTION_CHECKS.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_IDENTITY_DISTRIBUTION_CHECKS.json).
It uses the common NS-identity rescaling `V_I/(2*i*epsilon)` and a tangent
coordinate that resolves the peak. At three real loop momenta, the NS
areas divided by `pi*test(P)` tend to one, while both R-family areas tend
to one half. The largest 64-to-128-node change is `6.67e-6`.
This records a relative identity-peak normalization in the implemented
couplings. It does not by itself identify a primary-state metric or
determine the absolute string amplitude.

The equal two-point norms quoted in
[BRY I, (3.1) and (3.6)](https://arxiv.org/html/2201.05621v2#S3.SS1)
refer to its field conventions. The adopted correction leaves the
primary metrics and descendant construction unchanged; the factor is
applied at the separate partition/correlation-function layer. The
remaining common conversion to the absolute amplitude convention is
fixed by string factorization.
The Type 0B GSO choice itself is fixed and is not adjusted here.

In the original audit, multiplying the raw radial density by two and
the local R-handle density by two leaves the local NS-handle normalization
unchanged. This reduces the bare channel discrepancy to `1.82e-3` and
the refined modular discrepancy to `1.17e-3`. It does not finish the
calculation: the raised channel comparison still differs by as much as
`5.74e-2` at these cutoffs. That audit preserved the raw data. The new
normalization layer now applies these factors to completed correlators,
outside the unchanged descendant sums, and writes normalized banks;
it does not alter the historical reports or claim that their remaining
finite-cutoff error is resolved.

The full record is
[`RR_NORMALIZATION_AUDIT.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_NORMALIZATION_AUDIT.json).
The remaining steps are convergence of the raised overlap, completion of
the transported cusp/collision contributions, the matched torus integral,
and determination of the single common constant from a fully dressed
lower-genus factorization coefficient. The relative correlator
prefactor prescription is fixed; no primary metric is modified.

## Reproduce

Run from the repository root, with the project Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B Code/type0b_rr_genus1/check_radial_trace.py --output /tmp/rr/RADIAL_TRACE_CHECKS.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B Code/type0b_rr_genus1/check_radial_overlap.py --nodes 8 --output /tmp/rr/RR_RADIAL_OVERLAP_N8.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B Code/type0b_rr_genus1/check_radial_overlap.py --nodes 8 --raised --output /tmp/rr/RR_RADIAL_RAISED_OVERLAP_N8.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B Code/type0b_rr_genus1/check_local_modular.py --nodes 8 --output /tmp/rr/RR_LOCAL_MODULAR_N8.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B Code/type0b_rr_genus1/check_local_modular.py --nodes 12 --pmax 3.5 --output /tmp/rr/RR_LOCAL_MODULAR_N12.json
python -B Code/type0b_rr_genus1/audit_rr_normalization.py --directory /tmp/rr
```

The raw reports record failures without fitting or asserting that the
interacting comparison passed. The old local banks retain their original
metadata. Their checkpoint and documentation are archived under
`history/LOCAL_DENSITY_*`.
