# RR genus-one integration layers

The TT strategy is now implemented for **supplied physical RR components**:
finite bulk minus its own long-handle term, independent elliptic cusp
restoration, and analytic collision disks. The RR OPE, marked bulk/cusp
blocks, free collision factors, height moments, and integration bookkeeping
have separate checks. The [GSO layer](GSO_PROJECTION.md) now implements the
combined-state projection in the marked sewing basis, including picture
grading and BRY pants couplings. The [local density layer](LOCAL_DENSITY.md)
now assembles the interacting theta-frame density and collision coefficients
in a contractible OPE patch. **Global bulk/cusp transport and an independently
integrated RR string amplitude remain pending.** The theory's GSO choice
is fixed.

The frozen TT production is unchanged. Its NS/R one-point Ward engines,
modular-lambda coordinate, height-moment routine and geometry quadrature
are reused. RR external vertices, RR couplings and picture-changing factors
are evaluated in the new code.

## Implemented pieces

| File | Implemented calculation | Validation scope |
| --- | --- | --- |
| `local_density.py`, `prepare_local_bank.py` | Four theta-frame OPE densities, BRY spin/disorder contraction, normalized NS/R handle pairing, full PCO/free factors, and actual coupled collision polynomials | Independent full sphere states, physical R traces, zero modes, complex-energy analyticity, local polynomial and truncation checks; contractible patch only |
| `gso_projection.py`, `gso_sewing.py` | Combined matter/ghost grading, BRY external cocycles, projected NS-R sewing, pants couplings, and diagonal spin-character conventions | Exact Clifford/PCO grading; three equivalent projections of finite interacting tensors; full flat-frame transport still pending |
| `ope_channel.py` | RR pair fuses to an NS bridge; separate NS and R handle cutoffs; ordinary traces and supertraces; flat `G_-1/G0` transport | Sphere coefficients through bridge grade 4.5, saved TT handle traces, exponential coordinate map |
| `torus_blocks.py` | Flat primary and raised-puncture marked blocks; both NS-long and R-long degeneration charts | Finite-torus approach to the analytic leading term |
| `elliptic_series.py` | RR short-channel series in the elliptic nome, with integer and half-integer powers and an inverse expansion | Interacting coefficient roundtrips; independent free-theory cusp |
| `collision_kernel.py` | All four spin-dependent free Taylor kernels, four PCO components, finite Gaussian contact, analytic disk and annulus primitives | Exact theta functions and independent angular/radial quadrature |
| `cusp_kernel.py` | RR free factors on the cylinder and integration of its height to infinity | Direct, exponentially scaled high-precision quadrature |
| `matched_integration.py` | Cap, subtracted strip, restored cusp and compact/tail disk ledger, preserving four spin entries | Manufactured density; radius and matching-height cancellation |
| `prepare_ope_bank.py` | Fresh RR chiral coefficient banks with separate sphere and handle couplings | Finite arrays, source hashes, archive roundtrip; **no physical spin contraction** |

The finite-state Ward implementation is a checked prototype. These checks
do not establish the cost or convergence of an RR production run at the
TT bulk cutoff of level eight.

## RR OPE and coordinate transport

At fixed bridge momentum p, set
\(h=(1+p^2)/2\) and \(h_R=9/16+\omega^2/2\).
For bridge family f=0,1 and relative level j, the new chiral coefficient is

\[
b_{fjk}=\sum_{a,b}
\rho^{\eta}(N_a,R_z,R_0)
(B_{\rm NS}^{-1})_{ab}\,T_{\delta,k}(N_b),
\qquad |N_a|=f/2+j.
\]

Both external Liouville momenta are omega, including on the imaginary
energy ray. No complex conjugate energy is substituted into a second
physical factor. The bank stores chiral coefficients; construction of the
antiholomorphic product and physical projection remains explicit.

The leading odd sphere coefficient is
\(e^{-i\pi/4}(\beta_0-\eta\beta_z)/(2h)\), with
\(\beta=i\omega/\sqrt2\). The inverse norm `1/(2h)` is already included.
The independent mixed sphere comparison uses the signed analytic beta,
rather than resetting its sign through a square root of the weight.
The mixed RR/NS block conventions are described in
[Suchanek, section 3](https://arxiv.org/pdf/1012.2974).

In the OPE chart, x=exp(Kz), K=2 pi i. Raising the puncture at zero gives

\[
(G_{-1}R)_{\rm flat}
=K^{h_R+1}\left[(G_{-1}R)_{x=1}
 +\frac34(G_0R)_{x=1}\right].
\]

The local series therefore includes the shifted `3 K z / 4` zero-mode
polynomial. The torus handle uses theta3/4 for the NS trace/supertrace
and theta2/1 for the R trace/supertrace. This handle labeling follows the
[torus one-point trace definitions](https://arxiv.org/pdf/1207.5740);
it does not identify the two lifts in the marked mixed necklace.

## Collision disks

The collision module computes `z*d_delta` and `zbar*dbar_delta` directly
from theta-function Taylor series. Their leading terms are `-3/2` in the
three even spin structures and `+1/2` in the odd one. All four Liouville
components must be supplied. The finite time-boson contraction is retained.

After factoring the individual Liouville powers, the nominal radial power is

\[
\lambda=2h+f-4-\mathbf1_{\delta=1}.
\]

Physical component cancellations can remove the nominal leading term;
this formula alone is **not** a claim about the leading physical divergence.
For a Taylor coefficient a_mn, angular integration gives

\[
\operatorname{FP}\int_{|z|<r}d^2z\,|z|^\lambda
 \sum_{m,n}a_{mn}z^m\bar z^n
=2\pi\sum_m\frac{a_{mm}r^{\lambda+2m+2}}{\lambda+2m+2}.
\]

An exact radial pole requires an explicit logarithmic subtraction scale.
Annuli use an `expm1` expression that remains stable near such a pole.
The local layer now supplies those physical coefficients in
`RR_LOCAL_DENSITY_PILOT.npz`, including both bridge families and all four
spin structures. Their spectral convergence and the global subtraction
prescription remain to be established.

## Cusp and matched integral

For the R-long chart retain r=0 in the mixed series and divide out
\(Q^{h_R(P_R)-c/24}\). For the NS-long chart use the marked puncture
`tau-z`, retain n=0, and divide out \(Q^{h_{NS}(P_{NS})-c/24}\).
The two charts have separately implemented limiting coordinate derivatives.
Their identification with one physical puncture/spin frame is still needed.

The elliptic conversion re-expands only the actual RR short coefficients,
using s=lambda(qhat). It imports no TT external block prefactor.
The finite nome series has a checked inverse expansion for collision
matching. Its truncation requires an interacting convergence study.

Once physical components in the same flat spin frame are supplied, the
height dependence is

\[
T^{-1/2}e^{-aT-b/T}(u_0+u_1/T+u_2/T^2),\quad
a=2\pi P_{\rm long}^2,\quad
b=2\pi\omega^2(\operatorname{Im}z)^2.
\]

`cusp_kernel.py` integrates it from `max(Y,2 Im z)` to infinity. The
NS Casimir is canceled by the corresponding free determinant. The PCO
connection and finite Gaussian contraction determine all three u_k.

The integration ledger is

\[
I=I_{\rm cap,out}
 +I_{1<T<Y,\,bulk-own\ cusp}
 +I_{T>1,\,restored\ cusp,out}
 +I_{T<Y,\,disk}+I_{T>Y,\,cusp\ disk}.
\]

As in the accepted TT design, the excited long-handle remainder above Y is
omitted and must be tested by varying Y. RR puncture transport permutes
spin labels, so the new quadrature evaluates both z and tau-z explicitly.
It does not double each fixed-spin value or apply an unverified modular
chart permutation.

## Saved evidence and pilot

The new reports are in `Data Set/type0b_rr_genus1_20260921`:

- `LOCAL_DENSITY_CHECKS.json`: 256 independent nonchiral sphere comparisons,
  468 physical Ramond traces, zero-mode and energy-analyticity checks, and
  actual coupled collision-polynomial/annulus comparisons.
- `RR_LOCAL_DENSITY_PILOT.npz` and `RR_LOCAL_DENSITY_POINTS.json`: a fresh
  16-node coupled spectral pilot in the collision neighborhood, at omega=i/4.
  These are local densities, with no moduli integral or spectral convergence
  claim. [Reproduction and conventions](LOCAL_DENSITY.md) are recorded separately.

- `OPE_LAYER_CHECKS.json`: 60 independent sphere coefficients, maximum
  residual `1.25e-16`; saved handle agreement `4.75e-13`.
- `COLLISION_LAYER_CHECKS.json`: all four spin structures at imaginary and
  complex energies; maximum annulus residual `3.28e-16`.
- `CUSP_LAYER_CHECKS.json`: both marked degenerations and 48 height integrals;
  maximum height-integral relative residual `5.35e-14`.
- `ELLIPTIC_CUSP_CHECKS.json`: 16 interacting coefficient roundtrips; eight
  free-theory cases improve monotonically through nome order four, with
  maximum relative error `2.42e-6`.
- `MATCHED_LAYER_CHECKS.json`: a manufactured density at radii `.12,.08`
  and heights `3,5`; agreement `1.15e-16`. This is an integration test,
  **not an RR amplitude or its error estimate**.
- `RR_OPE_PILOT.npz`: omega=i/4, four Gauss nodes on `[0,2]^2`, bridge
  relative order two, NS/R handle level one. All four ambient ground pairs
  and both raised modes are retained. The BRY RR sphere constants are
  evaluated afresh and kept separate from the NS/R handle constants.

To reproduce, run the five `check_*` scripts named by these reports, each
with `--output /tmp/type0b-rr/REPORT_NAME.json`. Generate the pilot with

```sh
python3.11 -B Code/type0b_rr_genus1/prepare_ope_bank.py \
  --output /tmp/type0b-rr/RR_OPE_PILOT.npz
```

The local runtime used for the saved checks is
`/private/tmp/hetso23-logic-review-py311/bin/python` with NumPy, SciPy,
SymPy and mpmath.

## Still required for the reflection amplitude

The next dependency is transport of the implemented local OPE density
into the mixed bulk and cusp charts. It must pass modular and puncture
transport and mixed/OPE overlap after the spectral integral. The local
contractible-cut calculation does not establish that global identification.
Afterward, prepare converged collision/cusp banks and run the spectral,
block, radius, height and quadrature studies, followed by the overall
S-matrix phase and normalization audit.

The matrix prediction remains an independent comparison target.
`MATRIX_TARGET.json` still contains zero RR worldsheet datapoints.
