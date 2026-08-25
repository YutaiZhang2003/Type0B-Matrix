# Pfaffian/Selberg screening prototype

This directory contains no super-Virasoro state construction.  Its two
independent descriptions of the Ising screening polynomial are:

- `spin_pfaffian_polynomial`: one cubic Pfaffian of the two-spin kernel;
- `bfl_110_polynomial`: the symmetrized polynomial of arXiv:1011.4090.

The punctures in the project are `(0,1,infinity)=(R,R,NS)`.  The correct
BFL order labels are therefore `(1,1,0)`.  Odd screening number uses the
primal `k=1` product and even screening number uses the dual `k=0` product.

For `s` screenings the radical-free Pfaffian and the normalized BFL
polynomial differ only by the odd-spin-field normalization,

```text
s=2m-1: 1/sqrt(2)
s=2m:   1.
```

The signs which make this identity constant are the projective-limit
signs in equations (3.6) and (A.35) of arXiv:1011.4090.  Dropping them
would incorrectly produce alternating signs.

This is checked symbolically through five screenings by the default quick
audit (passing `audit(6)` performs the slower six-screening check):

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.ising_polynomial
```

## Ground-resolved consecutive-string pipeline

`boundary_zero_modes.py` implements the literal 2016 consecutive `chi`
strings without constructing an SCA PBW state.  The Ramond zero mode is
not put into the bulk Gaussian covariance.  It is acted on explicitly in
each two-dimensional ground space, giving exactly four boundary sectors
for the two natural Ramond strings.  All remaining auxiliary/physical
assignments are summed by one Pfaffian per sector.

For `N=2*(n1+n2+n3)` maximal screenings there are `N-1` nonzero external
rows.  The physical ground matrix is

```text
B = Gamma_f^eta X^(N mod 2),       X = [[0,1],[1,0]],
```

and the auxiliary form parity is the external parity minus `f`.  The
Fock-to-SCblock conversion of each physical minus ground is
`-(1-i)/sqrt(2)`.  These choices are not fitted: the independent literal
Fock-path audit agrees identically with the compressed Pfaffian in every
`(f,eta)` component:

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.audit_boundary_paths
```

In full detail, ground indices are ordered `(0,1)`, and the four matrices
used by the code are

```text
Gamma_0^F   = [[1,0],[0,-1]],    Gamma_1^F   = [[0,1],[-1,0]],
Gamma_0^eta = [[1,0],[0, eta]],  Gamma_1^eta = [[0,1],[i*eta,0]].
```

For positive `n`, the natural Ramond copy has

```text
epsilon_nat = (2*n+1/2) mod 2,
W_n^epsilon_nat = chi_0^- chi_-1^- ... chi_-M^-,
M = 2*n-1/2,                    chi^- = psi-i*eta_free.
```

For a fixed auxiliary endpoint `(a2,a3)`, the physical endpoint is first
`(1-a2,1-a3)` and the `X^N` twist toggles the third physical index.  The
auxiliary form is

```text
g = (2*n1 + epsilon2 + epsilon3 - f) mod 2.
```

The reference coefficient on Ramond leg `j` is

```text
(-1)^(M_j*(M_j+1)/2) / sqrt(2) * (1 if a_j=1 else -i),
```

multiplied by the cross-leg Koszul sign
`(-1)^((1-a2)*(M3+a3))` and by one Fock-to-SCblock conversion for each
physical minus ground.  These formulas specify the phase convention of
every exported value; there is no remaining proportionality constant.

The charge-preserving projection selects the physical screening border.
Whenever its external--external block vanishes, every perfect matching
pairs the `N-1` external rows and the one border row with the `N`
screenings.  Thus the Pfaffian is one `N by N` determinant.  After clearing
the endpoint powers, its rows have degree at most `N-1`, so

```text
clearing(t) * projected_correlator(t) = C * Delta(t).
```

The full insertion is therefore `C*Delta(t)^2`, and its integral is

```text
A = -b*(Q/2+P3)-1/2,  B = -b*(Q/2+P2)-1/2,  g = -b*Q/2,
numerator = C * Selberg_N(A-M3, B-M2, g+1),
denominator = J_(1,1,0)(N;A,B,g) / sqrt(2)^(N mod 2).
```

Here odd `N` uses the primal BFL product and even `N` the dual product.
The coefficient matrix is formed in quadratic work and its determinant
costs cubic work.  The public entry points are `projected_determinant_constant` and
`projected_selberg_ratio`.  They raise `NotImplementedError` instead of
silently applying the determinant reduction when the external block is
nonzero.

The requested eleven-screening example is small:

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.benchmark_boundary --large --selberg
```

On the development machine the coefficient determinant for
`(v_2,W_(7/4),W_(7/4))` takes about `0.6s`; a direct exact Pfaffian sample
takes about `15s`.  Both give
`C=-sqrt(2)*i*(1-i)/4`; the subsequent exact Selberg product takes about
`0.4s` at the benchmark point.  No Ward evaluator is imported by this
benchmark.

This is a certified *Gaussian/factorized screening channel*, not a closed
formula for either raw fixed-`eta` Ramond three-point form.  It matches all
sixteen ground components and the hard `(0,3/4,3/4)` factorized component.
At `(0,5/4,5/4)`, however, even the raw `eta=+` form differs from it by a
momentum-dependent rational function.  The short-level rule
`eta=(-1)^(2*n1+M2+M3)` therefore labels a coincident factorized direction;
it is not an all-level identification with a fixed physical chiral form.
The deliberately separate Ward harness verifies both statements at exact
rational momenta:

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.audit_projected_ward
```

When the projected external block is nonzero, the same state-free
Pfaffian remains an exact evaluator of the Gaussian screening functional
at chosen screening positions.
`finite_schur_reconstruction.py` supplies the bounded-width Schur
interpolation layer, with `binomial(N+k,k)` calls for width `k`, and the
Jack/Kadell modules integrate the recovered coefficients.  That route is
polynomial at fixed `k`, but a uniform level-independent width has not been
proved for every charge ordering.  It must not be advertised as an
all-level polynomial algorithm yet.

The fermionic compression itself is never exponential: all `2^K`
auxiliary/physical assignments are already contained in constant-many
Pfaffians.  What remains missing for the physical raw branching
coefficient is described below.

For a fixed Ramond zero-mode channel, `chi_block.py` makes the collapse
explicit.  If `A` is the auxiliary covariance, `B` the physical covariance
on the external contours, `C` the external--screening covariance and `D`
the screening covariance, the complete matrix is

```text
[[A-B, -i C],
 [i C^T,   D]].
```

Its Pfaffian already sums every auxiliary/physical choice in every chi.
The two-dimensional ground space gives a fixed finite sum of these blocks.
It does not introduce a level-dependent state basis.

## Exact missing interface

Changing `P` to `-P` in a Ramond leg is not implemented by replacing a
`chi^-` row with `chi^+` while keeping the ordinary two-spin covariance.
That replacement passes every ground-state component but fails already for
the reflected excited pair `(3/4,-3/4)`: the ordinary answer is momentum
independent, whereas the Ward form depends on `P2,P3` on the reflected
neutrality plane.  The needed operator is the reflected free-field
fermion, conventionally denoted `psi^R`; it includes the Ramond reflection
intertwiner and is momentum dependent from mode one onward.

The reflection intertwiner itself is now available in
`../reflection/intertwiner_recurrence.py`.  Its exact level-one block, in
the basis

```text
(c_-1|0>, c_-1|1>, psi_-1|0>, psi_-1|1>),
```

is

```text
R_1 = [[a,0,0,u], [0,a,u,0], [0,u,c,0], [u,0,0,c]],
d = 4*P^2-6*P*Q+2*Q^2+1,
a = -(4*P^2-2*P*Q-2*Q^2+1)/d,
u = 2*sqrt(2)*i*Q/d,
c = -(4*P^2+2*P*Q-2*Q^2+1)/d.
```

This passes both the independent `S_- S_+^(-1)` audit and all sixteen
hard mixed-sheet restrictions.  It also identifies why a modified Ising
covariance cannot be the answer: `psi_-1^R` has a bosonic-current
`c_-1` component with a ground flip.  Reflection is not closed on the
fermionic rows of the Pfaffian.

```bash
python3 -m python.ramond_screening_algorithm.reflection.intertwiner_recurrence
python3 -m python.ramond_screening_algorithm.reflection.audit_hard_channel
```

The precise remaining input needed to complete the scalable screening
algorithm is therefore a chart-adapted, state-free call of the form

```text
reflected_mode_form(
    fermion_and_current_modes_at_infinity,
    fermion_and_current_modes_at_one, ground_at_one, sheet_at_one,
    fermion_and_current_modes_at_zero, ground_at_zero, sheet_at_zero,
    screening_positions,
    f, eta, Q, P1, P2, P3,
) -> normalized exact correlator.
```

Here a `sheet` says whether the modes are ordinary or reflected.  The
result must include the two-by-two Ramond ground matrix, the
Fock-to-SCblock phase, and the radial-order Koszul sign.  Fermion parts can
still be Wick-compressed, while the current parts act by logarithmic
derivatives of the bosonic Selberg weight.  What is not yet implemented is
their combined consecutive-string reduction to a determinant or a
bounded-width symmetric polynomial.

The existing finite-field reflection recurrence gives exact blocks through
any requested level and reaches the complete level-six endpoint needed by
`W_(7/4)` in about `0.04s` in the development benchmark.  Its parity-block
dimension is nevertheless `exp(O(sqrt(level)))`, so it is an excellent
oracle and fallback, but not the requested all-level polynomial algorithm.
The next production step is to apply its reflected `chi` string directly
to the screening chart, commuting every generated `c_-m` into power-sum
insertions before any Fock block is materialized.

This is a sharp boundary, not an unknown phase.  The reflection identity
locates the second physical direction, and the exact low-level
intertwiner is known, but the ordinary Ising kernel does not evaluate it.
Until the chart-adapted `reflected_mode_form` is supplied, the code exports
the proven factorized screening projection and explicitly refuses to call
it the crossed Ramond branching coefficient.

## Native ground spin frame and the hard two-chart calibration

There is already a useful exact correction at the level of the two Ramond
zero modes.  Put

```text
C = -(1-i)/sqrt(2),       D = diag(1,C),
K = [[1,0],[0,-1]],       J = [[0,1],[-1,0]].
```

Here `K,J` are the native Majorana identity and fermion ground forms in the
normalized free-field ground basis, while the notes use
`(|+>,|->)=(w^+,C w^-)`.  Transporting a bilinear form to the `w` frame
therefore gives

```text
Khat = D^(-1) K D^(-1) = diag(1,-i),
Jhat = D^(-1) J D^(-1)
     = [[0,-exp(i*pi/4)],[exp(i*pi/4),0]].
```

They are not individual fixed-`eta` ground forms.  With the matrices
`Gamma_f^eta` displayed above,

```text
Khat = (1-i)/2 Gamma_0^+ + (1+i)/2 Gamma_0^-,
Jhat = -i/sqrt(2) Gamma_1^+ - 1/sqrt(2) Gamma_1^-.
```

Equivalently, the parity rows `f=0,1` and fixed-spin columns `eta=+,-`
form the exact ground coefficient table

```text
A = [[(1-i)/2, (1+i)/2],
     [-i/sqrt(2), -1/sqrt(2)]],
det(A)=(-1+i)/sqrt(2).
```

This finite zero-mode change of frame is derived directly from `K,J` and
the ground conversion; it uses no three-point data.  It also explains why
assigning one native Coulomb form to one fixed `eta` before combining the
two charge charts is unsafe.  It is only a ground statement: it does not
determine the nonzero-mode covariance or odd functional.  The executable
identity check is

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.native_ground_change
```

For the first reflected pair `(n_1,n_2,n_3)=(0,3/4,-3/4)`, the two
zero-screening charts can nevertheless be completed explicitly.  On

```text
Q/2+P1+P2+P3=0       the current is i(Q/2+P2),
Q/2+P1-P2+P3=0       the current is i(Q/2-P2).
```

The second equality is precisely the complementary vertex
`E^(Q/2-P2)`.  Merely making this current replacement while keeping the
ordinary Majorana pair kernel gives zero of eight correct complementary
hard components.  The hard oracle therefore retains a second mode-one
pair multiplier.  With `r=-P3`,

```text
e2=Q+2P2, e3=Q+2r, dj=ej^2+Q ej+1,
x=-2Q(P2+r), H=x^2+2x(e2 e3+1)+d2 d3,
D=4r^2+6Qr+2Q^2+1,
B0=18Q^2+50Qr+28r^2+7,
A0=9(2Q^2+2Qr-4r^2-1),
lambda_comp = (16 D H/(d2 d3)-B0)/A0.
```

This `lambda_comp` is exact but is calibrated by solving the known hard
crossed polynomial `H`; it is not an independent prediction of `H`.
The resulting state-free calculation checks the direct level-one
reflection, both Coulomb currents, all ground phases, both parity copies,
and both form parities.  At two independent rational points it gives
`32/32` exact Ward values:

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.audit_reflected_hard_oracle
```

No extension of `lambda_comp` beyond this mode-one pair is claimed.  At
the next reflected level, `R_2 psi_-2|g>` already contains
`psi_-2|g>`, `c_-1 psi_-1|g>`, `c_-1^2|1-g>`, and `c_-2|1-g>`.
Consequently a general evaluator needs current--fermion and
current--current forms as well as the two-spin covariance; a scalar
replacement of the pair kernel cannot supply them.

## Colored two-core audit

`colored_staircase.py` corrects an important earlier identification.  The
two Ramond holonomies are not a common pair `(delta_M,delta_M)` with the
corner colors exchanged.  They are the two distinct consecutive-GKO
two-core paths.  If `k=2*x` is the level-one affine shift, its core is

```text
k>0  : delta_(k-1) with corner color k mod 2,
k<=0 : delta_(-k)  with corner color k mod 2.
```

For example, at `n=3/4` the two Ramond paths are

```text
(empty^1,empty^0),  ((1)^0,(1)^1).
```

With the exact Coulomb dictionary

```text
alpha_col=Q/2+P1,  p_R,col=P_R+Q/2,
```

the Ward-free hard matrix is

```text
Z = [[1, (x_-- - Q)(x_++ + Q)],
     [x_++(x_-- - 2Q),          K]],
```

where `K` is exactly the factorized hard numerator.  Thus the colored
construction does determine the known diagonal component, including its
correct two-core representative.  It does not determine the crossed
numerator `H`: neither `det(Z)` nor `perm(Z)` equals `H`.  The file writes
the unique hard momentum-dependent kernel (under a symmetric
off-diagonal convention) for which `H=sum(R_Racah .* Z)`.  This kernel is
an obstruction certificate, not a fit promoted to an all-level formula.

The reason is visible already in arXiv:1210.7454: `Z_bif` is a matrix
element in `H + sl(2)_2 + NSR`; the paper obtains NSR blocks only after a
convolution with the two `sl(2)_2` WZW blocks.  The desired fixed chiral
branch coefficient needs the inverse trivalent/WZW-Racah projection.
The ordinary spin-1/2 KZ fusion matrix cannot provide it, because the two
holonomies label distinct ordered external tensor products, rather than
two intermediate weights of one fixed tensor product.

The colored product visits each selected box once for each of four diagram
pairs.  With precomputed column heights its arithmetic cost is `O(B)` and
memory is `O(B)`, where `B` is the total number of boxes (so it is safely
within the requested `O(B^2)` bound):

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.colored_staircase
```

`audit_colored_108.py` is the separate validation harness which imports
the old Ward oracle.  It tested all 108 independent masters at both stored
exact samples.  Only 6/108 are a momentum-independent multiple of a single
allowed published holonomy entry: the four ground masters and the two
parity copies of the hard factorized direction.  The other 102/108 fail,
including every mixed or longer-staircase triple:

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.audit_colored_108
```

Consequently the colored-Nekrasov product is a fast ingredient, but not a
closed Ramond branching oracle by itself.  The exact missing datum is the
momentum-dependent trivalent projection which removes the Heisenberg and
`sl(2)_2` factors and maps the two colored paths to the repository's
`(epsilon_2,eta)` spin frame.  No formula for that projection is given in
arXiv:1210.7454 or arXiv:1211.2788; the latter explicitly constructs only
the NS vertex commuting with the affine Cartan charge and lists the
Ramond-changing vertex as future work.
