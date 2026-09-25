# Type 0B RR scalar: genus-one reflection calculation

The selected observable is the reflection of the Type 0B RR axion, denoted
**A** in the project convention ledger. The background is noncompact time,
zero RR flux and the same NSNS Liouville wall as the tachyon calculation.

**Latest omega=i/4 result: relative correlator normalization implemented;
no worldsheet amplitude yet.** The factor two is an overall
partition/correlation-function prefactor, applied after the original
descendant sums. Primary norms, inverse Grams, trinion couplings and
conformal-block coefficients remain unchanged. The radial prefactor is
two and the OPE theta prefactors are `(2,2,1,1)`. The two independent
quadrature banks now pass the bare modular comparison within `1.10e-6`
at the tested block cutoffs. See
[the correlator normalization and factorization prescription](INTERNAL_NORMALIZATION.md).
One common amplitude constant is left for an independently normalized
string factorization limit. The raised overlap still requires convergence,
and the local banks remain pilots. The requested amplitude point is unset in
[`RR_OMEGA_I4_POINT.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_OMEGA_I4_POINT.json).

The [Ramond momentum-quadrature study](MOMENTUM_QUADRATURE.md) fixes
separate NS/R endpoint rules and checks moving poles and finite tail
shells. Its raw reports and [earlier normalization audit](RADIAL_TRACE.md)
are retained to document how the relative convention was selected.

**Status: the matrix prediction, local chiral/nonchiral sewing,
RR OPE, marked bulk/cusp blocks, free collision kernels, elliptic
re-expansion, matched integration machinery and GSO projection of the
combined states in marked sewing are implemented and checked. The local
OPE layer now assembles all four interacting spin densities, their BRY
external-state contraction, picture changing, and coupled collision data.
An independently integrated RR worldsheet amplitude has not been obtained.**
No saved tachyon integral is relabeled as an RR result.

The new [integration layers and saved checks](INTEGRATION_LAYERS.md) follow
the TT bulk/OPE/cusp strategy. A fresh RR chiral pilot bank is available;
the [GSO layer](GSO_PROJECTION.md) now fixes the product-state projector,
relative ghost picture grading, BRY product cocycles and global spin weights.
The [local density derivation](LOCAL_DENSITY.md) fixes the contractible-cut
theta frame in the OPE chart and documents the fresh 16-node pilot.
Transport to the mixed bulk/cusp charts and the global integration remain.

## External state and matrix target

In the BRY basis,

\[
A^\pm_\omega=g_s\omega c\widetilde c\,
e^{-\varphi/2-\widetilde\varphi/2}e^{\pm i\omega X^0}
\frac{\sigma^0V^{R,+}_\omega\pm\mu^0V^{R,-}_\omega}{\sqrt2}.
\]

The time momenta are opposite at the two punctures; both Liouville momenta
are the same continued value \(P=\omega\). The in/out sign is distinct from
the Ramond ground label or the HJS three-form sign.
[BRY I, equations (2.6)–(2.10)](https://arxiv.org/pdf/2201.05621)
fix the state phases and \(\mathcal R=T+A,\mathcal L=T-A\).

The two matrix-model sides decouple perturbatively. Consequently,

\[
a_{AA}^{(g)}=\frac14(a_{RR}^{(g)}+a_{LL}^{(g)})=a_{TT}^{(g)},
\qquad a_{TA}^{(g)}=0.
\]

With \(\Omega=2\omega\), the energy-delta Jacobian \(1/2\), and
\(g_s=4/(\pi\mu_B)\), the same-side particle-hole kernel gives

\[
S_{AA}=\delta(\omega-\omega')\left[\frac\omega2
 +g_s^2a_{AA}^{(1)}(\omega)+O(g_s^4)\right],\qquad
\boxed{a_{AA}^{(1)}(\omega)=\frac{i\pi^2\omega^2}{384}
 (1+8\omega^2+8i\omega^3).}
\]

The tree-normalized elastic factor is
\(1+2g_s^2a_{AA}^{(1)}/\omega+O(g_s^4)\). These expressions absorb the
RR leg phase in the external state; equality to the NSNS coefficient does
not assert equality of unstripped Liouville reflection phases.
The same-side genus-one coefficient is independently given by
[BRY II, equation (2.33) at zero temperature](https://arxiv.org/pdf/2204.01747).

At \(\omega=i/4\), the matrix target is
\(a_{AA}^{(1)}=-5i\pi^2/49152\), or
\(I_{AA}=-5\pi/98304\) if the existing conditional \(a=2\pi iI\) conversion
is used. `kinematics.py` derives the coefficient from the kernel and exports
twenty prediction points. Every RR worldsheet value in that file is null.

## Ramond punctures change the sewing problem

An external R intertwiner changes the sector of a state propagated along
the necklace. The two internal edges are **NS and R**, with independent
momenta \(P_{\rm NS},P_{\rm R}\); they are not the two R edges used for
NSNS external operators in the earlier R-handle calculation.

The implementation cuts one R edge of the checked NS-R-R theta sewing.
Its open puncture states are \(u,v\), including both R grounds and their
descendants. At fixed internal levels \(n/2,m\), the marked tensor is

\[
F^{f_1\eta_1,f_2\eta_2}_{uv}[n,m]
=\sum_{p_R=0,1}(-1)^{(n\bmod2)p_R}\ell_{\rm NS}^{n\bmod2}\ell_R^{p_R}
\sum_{i,j,a,b}
\rho_{f_1}^{\eta_1}(N_i,R_a,u)
(B_{\rm NS}^{-1})_{ij}(B_R^{-1})_{ab}
\rho_{f_2}^{\eta_2}(N_j,R_b,v).
\]

Here each three-form has slot order \((\mathrm{NS},R,R)\) at
\((\infty,1,0)\). The sign displayed is the crossing of the two sewn
odd edges in this marked frame. \(\ell_R=-1\) inserts actual R state
parity, including its ground parity. HJS signs \(\eta_i\), form parities
\(f_i\) and spin lifts \(\ell_i\) remain separate.
No complex conjugation of momenta, structure constants or coefficients is
performed when continuing the antiholomorphic block.

`mixed_blocks.py` evaluates this bilinear tensor by generalized NRR Ward
identities and inverse Gram matrices. It supports a descendant on either
open R puncture, including \(G_{-1}\). This direct implementation is a
finite-level check and prototype, not a replacement of the production
recursion by an untested high-level PBW sum.

The open tensor is checked by sewing the R punctures back together through
both ground and level-one states and comparing with the existing theta
oracle, for both homogeneous forms and all HJS sign pairs. These are tests
of the open contraction and its signs; they share the pre-existing Ward
matrix-element backend. A separate global \(L_{-1}\) coefficient and a
complex-energy check test normalization and analytic bilinear sewing.
The saved run includes 800 closures through total descendant level three,
with maximum absolute residual `6.67e-16`.

## Local nonchiral spin/disorder contraction

In the ambient ground basis `(++, +-, -+, --)`, the two physical
Liouville Ramond kets are the columns of

\[
E_R=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&1\\0&1\\-i&0\end{pmatrix}.
\]

For the canonical state \(W_hW_a|R^\epsilon\rangle\), each ambient
coefficient acquires \((-1)^{|W_a|g_h}\). At the trinion, regrouping
holomorphic and antiholomorphic entries contributes

\[
(-1)^{b(a+\alpha+\gamma)+\beta\gamma},
\]

where \(a,b\) are the NS descendant parities and
\((\alpha,\beta),(\gamma,\delta)\) the full chiral/antichiral parities
of the two R states. Contract the embedded states with the holomorphic
even HJS form and its antiholomorphic counterpart, both of sign eta.
The antiholomorphic convention phases are conjugated, while momenta
remain analytically independent of conjugation.

This gives unit ground amplitudes `(1,eta)` in the two physical families.
Multiplication by \(c_+=C^{\rm even}/2\) and
\(c_-=C^{\rm odd}/2\) therefore reproduces BRY's
\(C^\pm=(C^{\rm even}\pm C^{\rm odd})/2\).
`physical_components.py` implements this construction and opens the
external R edge using the physical bilinear BPZ metric. The saved tests
compare 576 vertices exactly with successive nonchiral Ward reduction
and perform 360 physical edge closures, with maximum residual `5.84e-16`.
They include both imaginary and genuinely complex external energies.

These are **local** product-algebra and BPZ tests. The subsequent
`gso_sewing.py` layer applies the full product-state projection to these
tensors and compares projected Grams, restricted bases and parity-insertion
sums. A marked tube parity insertion is not yet identified with a
flat-torus theta characteristic.
The provisional factor four in an unrelated genus-two comparison is
not an input to this calculation.

## Uniformization and descendant transport

On the two pants use coordinates \(w,v\), with sewing
\((1/w)(1/v)=q_{\rm NS}\), \((w-1)(v-1)=q_R\).
The closed-cycle transformation is

\[
M(w)=\frac{1+q_{\rm NS}(q_R-1)w}{1-q_{\rm NS}w}.
\]

For \(Q=e^{2\pi i\tau}\) and \(s=e^{2\pi iz}\), its multiplier is Q when

\[
q_{\rm NS}=\frac{s(1-Q)^2}{(1-Qs)^2},\qquad
q_R=\frac{Q(1-s)^2}{s(1-Q)^2}.
\]

The fixed points are \(a=(1-Qs)/(1-Q)\), \(b=a/s\).
The logarithmic coordinate maps \(w=0\) to z and \(v=0\) to zero.
`geometry.py` gives both derivatives at the punctures. For ground
insertions, the local transport used in `evaluate_flat_ground` is

\[
\mathcal F_{\rm flat}
=Q^{-c/24}(f'_w f'_v)^{-h_E}
q_{\rm NS}^{h_{\rm NS}}q_R^{h_R}
\sum_{n,m}q_{\rm NS}^{n/2}q_R^m F[n,m].
\]

There is **one torus Casimir factor**, not a separate subtraction of
`c/24` on both plane-plumbing edges. The sewing maps before the logarithmic
uniformization are Möbius transformations; the cylinder vacuum shift
belongs to the closed-cycle multiplier Q. Branches remain tied to the
marked patch.

This normalization and its phase have an independent free-theory test.
At \(b=i,c=3/2\), set \(p_{\rm NS}=k-\eta p_R\). The SCA module becomes
a free boson times a Majorana. With

\[
\Pi=\prod_{n\ge1}(1-Q^n),\quad
\mathscr D=(1-s)\prod_{n\ge1}
\frac{(1-Q^ns)(1-Q^n/s)}{(1-Q^n)^2},
\]
\[
H_\ell=\sum_{n\in\mathbb Z}(-\ell)^{n+1}
Q^{n(n+1)/2}s^{(n+1)/2},
\]

the ground-0 block is compared to the independent theta-function answer

\[
(2\pi)^{k^2+1/8}Q^{p_R^2/2}
s^{k^2/2-\eta p_Rk-1/16}
\mathscr D^{-k^2-1/8}\Pi^{-3/2}\sqrt{H_\ell}.
\]

\(\ell=+1\) is the theta-1 branch with its leading phase removed;
\(\ell=-1\) is theta-2. Both roots are continued from small positive
Q and s. Across three complex-coordinate patches and both charge/spin
choices, the discrepancy decreases at levels one, two and three.
Level-three errors range from `5.1e-12` to `3.6e-7`; this is a truncation
test of the free chiral block, not an error estimate for a string amplitude.

For a coordinate map f, the level-one R descendant transforms as

\[
(G_{-1}R)_{\rm local}
 =f'^{h_R+1}(G_{-1}R)_{\rm flat}
 +\frac34f''f'^{h_R-1}(G_0R)_{\rm flat}.
\]

The coefficient follows from \([L_1,G_{-1}]=\tfrac32G_0\).
It must be retained for the separate Liouville and timelike factors.
The combined physical matter state obeys \(G_0^{\rm m}|\mathcal W\rangle=0\),
so its complete \(G_{-1}^{\rm m}|\mathcal W\rangle\) is Virasoro primary.

## Picture changing and the ghost factor

Two R punctures at genus one have one odd modulus per chirality:
\(2g-2+n_{\rm NS}+n_R/2=1\). Start in \((-1/2,-1/2)\) picture at both
punctures and insert one PCO per chirality. Raising one puncture gives
pictures \((+1/2,-1/2)\) in each chirality. There is one factor \(-1/2\)
from each chiral PCO, rather than the four PCO factors in the NSNS pair.
The RR vertex normalization also supplies \(\omega^2\).

For periods \(1,\tau\), define
\(E(z)=\theta_1(z|\tau)/\partial_z\theta_1(0|\tau)\).
With R punctures at zero and z and PCO at u, a fixed branch patch has

\[
B_\delta(u;z)=\frac{\eta(\tau)E(-z)^{-1/4}
 E(u)^{1/2}E(u-z)^{1/2}}{\theta_\delta(u-z/2|\tau)}.
\]

Since \(B_\delta\sim u^{1/2}\), the potentially divergent \(G_0\) term
in the matter OPE cancels only after using the physical RR combination.
The finite local raised-vertex factor is

\[
\lim_{u\to0}\frac{B_\delta(u;z)}{u^{1/2}}
=\frac{\eta(\tau)E(-z)^{1/4}}{\theta_\delta(-z/2|\tau)},
\]

and multiplies the full \(G_{-1}^{\rm m}\) insertion. `kinematics.py`
checks this limit for all four theta characteristics and verifies the
Ramond zero-mode constraint in a Clifford basis.
The source is the supplied [String Notes, “One-loop amplitudes involving Ramond states”](../../handoffs/reference/HetSO23_1to1_20260920/references/string%20notes.tex).
Transporting a Ramond puncture around a torus cycle changes its spin
characteristic. It is therefore insufficient to reuse four fixed NSNS
determinant weights or to take absolute values of all chiral factors.

The [picture-changing derivation](PICTURE_CHANGING.md) further reduces
the time supercurrent to a connection multiplying the Liouville zero mode:

\[
D_\delta=-\frac32\frac{E'}{E}
 +\frac{\theta'_\delta(z/2)}{\theta_\delta(z/2)}
 -\frac{2\pi i\operatorname{Im}z}{\tau_2}.
\]

Given the four Liouville components `C_pq` for `D_1=G_-1`, `D_0=G_0`
on the raised puncture, the reduced expression is

\[
C_{11}+D_\delta C_{01}+\overline D_\delta C_{10}
+\left(D_\delta\overline D_\delta
 -\frac{\pi}{\omega^2\tau_2}\right)C_{00}.
\]

`picture_changed.py` implements this expression and the free prefactor.
The finite Gaussian term is independently checked by differentiating the
time-boson Green function. Exact Clifford checks now use BRY's actual
incoming/outgoing spin/disorder combination and its graded pairing.
The free factors pass the S and T modular transformations and both
puncture transports for all four characteristics at a complex energy.
These checks do not identify the interacting `C_pq` with a particular
theta characteristic or validate the boundary subtraction. The subsequent
[local OPE density layer](LOCAL_DENSITY.md) now supplies the interacting
`C_pq` in a contractible theta-frame patch and its collision polynomials.

## Remaining work before an RR numerical amplitude

0. Use the fixed overall correlator prefactors without changing primary
   norms or descendant sums. Converge the picture-changed channel
   comparison independently of the successful bare modular test.
1. Extend the implemented local OPE spin frame to the mixed bulk and cusp
   charts, including the transported disorder line and raised components.
2. Establish spectral convergence of the coupled local density, and test
   modular/puncture transport and mixed/OPE overlap after the spectral
   integral. The existing free-field checks do not replace these checks.
3. Prepare converged physical collision/cusp banks from the local coefficients
   and transported cusp tensors. Check the remaining distributional/boundary
   contributions and the overall S-matrix phase/normalization.
4. Integrate at \(\omega=i/4\), then on the shared grid, with radius,
   channel-overlap and level checks. Compare to the independent matrix target.

The completed local, free-field and integration-machinery checks do not
certify these completed-density steps. The existing NSNS production remains
the control calculation, rather than the source of alleged RR datapoints.

## Reproduce the completed checks

From the repository root, using NumPy, SymPy and mpmath:

```sh
python3.11 -B Code/type0b_rr_genus1/check_mixed_blocks.py \
  --max-twice-level 6 --output /tmp/type0b-rr/BLOCK_CHECKS.json
python3.11 -B Code/type0b_rr_genus1/kinematics.py --output-dir /tmp/type0b-rr
python3.11 -B Code/type0b_rr_genus1/check_geometry.py --output /tmp/type0b-rr/GEOMETRY_CHECKS.json
python3.11 -B Code/type0b_rr_genus1/check_free_field.py --output /tmp/type0b-rr/FREE_FIELD_CHECKS.json
python3.11 -B Code/type0b_rr_genus1/check_physical_components.py --output /tmp/type0b-rr/PHYSICAL_COMPONENT_CHECKS.json
python3.11 -B Code/type0b_rr_genus1/check_picture_changed.py --output /tmp/type0b-rr/PICTURE_CHANGING_CHECKS.json
python3.11 -B Code/type0b_rr_genus1/check_gso_projection.py --output /tmp/type0b-rr/GSO_LAYER_CHECKS.json --bank /tmp/type0b-rr/GSO_COEFFICIENT_FIXTURES.npz
python3.11 -B Code/type0b_rr_genus1/check_local_density.py --output /tmp/type0b-rr/LOCAL_DENSITY_CHECKS.json
```

[Saved checks and matrix targets](../../Data%20Set/type0b_rr_genus1_20260921/)
record the actual validation scope. No complete RR worldsheet result exists yet.

## Why the target was changed from 0A

The user selected 0B after the physical-spectrum check. In noncompact
two-dimensional 0A, opposite R chiralities and equal left/right target
momenta allow only zero-energy RR flux insertions; there is no RR particle
to reflect at nonzero energy. Equivalently, a two-dimensional RR one-form
has source-free equation \(d[f(T)*F]=0\), fixing a conserved electric flux.
The two RR gauge fields and their allowed background modes are described
in [Douglas et al., section 2, equation (2.9)](https://arxiv.org/pdf/hep-th/0307195);
the chiral vertex obstruction is explicit in
[Takayanagi–Toumbas, section 2, equation (2.3)](https://arxiv.org/pdf/hep-th/0307083).
