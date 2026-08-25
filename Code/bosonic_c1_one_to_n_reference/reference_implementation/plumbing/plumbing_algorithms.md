# Plumbing Algorithms

This document is the companion note for `plumbing_algorithms.py`. It records the
two numerical routes now kept in the repository:

1. the small-$q$ Schottky/Poincare-series method;
2. the direct boundary-collocation method used when one wants a solver that does
   not depend on Schottky word suppression.

The plumbing parameters are treated as the primary moduli. Schottky data are
derived from them.

## 1. Plumbing Data

Each three-punctured sphere has coordinate $z$ and punctures

$$
z^{(1)}=0,\qquad z^{(2)}=1,\qquad z^{(3)}=\infty.
$$

The local puncture coordinates are

$$
u_0=z,\qquad u_1=z-1,\qquad u_\infty={1\over z}.
$$

For a plumbing edge with local coordinates $u,v$ and parameter $q$, the seam
relation is

$$
uv=q.
$$

The genus-two glasses channel uses two spheres and three plumbings:

$$
(z_1^{(1)},z_1^{(3)}),\qquad
(z_2^{(1)},z_2^{(3)}),\qquad
(z_1^{(2)},z_2^{(2)}).
$$

The parameters are $q_1,q_2,q_3$. The first two are self-plumbing handles on
$S_1$ and $S_2$. The third is the bridge between the two $1$-punctures.

The sunrise channel also uses two spheres and three plumbings, but all three
edges go between the spheres:

$$
(z_1^{(1)},z_2^{(1)}),\qquad
(z_1^{(2)},z_2^{(2)}),\qquad
(z_1^{(3)},z_2^{(3)}).
$$

The implementation chooses the first edge as the spanning-tree edge; the other
two edges give the two Schottky generators.

## 2. Schottky Group From Plumbing

Choose the bridge as the spanning-tree edge and use the $S_1$ coordinate as the
root coordinate. The bridge map is

$$
T(z)=1+{q_3\over z-1}.
$$

This map is an involution, $T(T(z))=z$. The first Schottky generator is the
contracting dilation

$$
\gamma_1(z)=q_1 z.
$$

The second self-plumbing is a dilation in the $S_2$ coordinate. Expressed in the
root coordinate, it is

$$
\gamma_2(z)=T(q_2T(z)).
$$

Writing $p=q_2$ and $s=q_3$, the explicit Mobius matrix is

$$
\gamma_2(z)
=
{(p+s-1)z+(p-1)(s-1)
\over
(p-1)z+p(s-1)+1}.
$$

Thus the fixed points and multipliers are

$$
(a_1,b_1,k_1)=(0,\infty,q_1),
$$

and

$$
(a_2,b_2,k_2)=(1-q_3,1,q_2).
$$

For the sunrise channel, the code uses the general plumbing-transition recipe.
For a source puncture $\alpha$ and target puncture $\beta$, the transition is

$$
P_{\alpha\to\beta}(z)
=
\psi_\beta\left({q\over \phi_\alpha(z)}\right),
$$

where $\phi_\alpha$ is the local coordinate and $\psi_\beta=\phi_\beta^{-1}$.
After choosing the zero-zero edge as the tree edge, each non-tree edge produces

$$
\Gamma_e=C_2^{-1}\circ P_e\circ C_1.
$$

The code then finds the fixed points of $\Gamma_e$ and orients the generator so
the stored multiplier is attracting.

The code stores Mobius maps projectively. Multiplying all four matrix entries by
a common nonzero scalar does not change the transformation.

## 3. Small-$q$ Schottky Series

Given Schottky generators $\gamma_1,\ldots,\gamma_g$, the normalized
holomorphic one-form is evaluated by the Poincare series

$$
\omega_j(z)
=
{1\over 2\pi i}
\sum_{\gamma\in \Gamma/\langle\gamma_j\rangle}
\left(
{1\over z-\gamma(a_j)}
-
{1\over z-\gamma(b_j)}
\right)\,dz.
$$

The implementation enumerates reduced words up to a finite word length. For
small multipliers, long words are suppressed and the truncation converges
quickly. The method is therefore the preferred small-$q$ solver.

The period matrix is computed by integrating the normalized forms along
Schottky $B$ paths:

$$
B_j:\quad z_0\longrightarrow \gamma_j(z_0).
$$

The implementation returns the matrix in forms-by-cycles convention:

$$
\Omega_{Ij}=\int_{B_j}\omega_I.
$$

## 4. Why Schottky Is Not The Whole-Moduli Solver

The Schottky series is not safe when active multipliers are large. If
$|q|\sim1$, long words are not suppressed, image poles can approach integration
paths, and a fixed truncation is not reliable.

For genus one this is a chart issue:

$$
q=e^{2\pi i\tau}.
$$

When $|q|$ is large, one should reduce $\tau$ by an $SL(2,\mathbb Z)$
transformation and compute in the reduced chart. In higher genus the analogous
operation is changing pants decompositions or switching to a direct solver.

The genus-one reducer records the matrix

$$
\begin{pmatrix}a&b\\c&d\end{pmatrix}
$$

such that

$$
\tau_{\rm red}={a\tau+b\over c\tau+d}.
$$

This metadata is now checked before returning.  In particular, translations
after inversions are composed on the left, so the stored matrix reproduces the
reported reduced modulus.

The Schottky health check remains conservative, but it now rejects nonfinite,
zero, and unit-disk-violating plumbing parameters, and for the glasses channel
it also checks a basic fixed-point separation diagnostic.  The hard-coded
Schottky \(B\)-period paths now reject paths whose sampled segments pass too
close to finite Schottky image poles.

## 5. Boundary Collocation

The direct solver represents the surface as punctured spheres with circular
seams. It writes a candidate one-form on each component as

$$
\omega_i=f_i(z)\,dz.
$$

For the glasses channel, the basis on each sphere is the independent rational
ansatz

$$
f_i(z)
=
\sum_{n=1}^{N} a^{(0)}_{i,n}z^{-n}
+
\sum_{n=1}^{N} a^{(1)}_{i,n}(z-1)^{-n}
-
\sum_{n=2}^{N} a^{(\infty)}_{i,n}z^{n-2}.
$$

The infinity block starts at $n=2$ because

$$
{du_\infty\over u_\infty}=-{dz\over z},
$$

which duplicates the $z=0$ simple-pole mode. Keeping this duplicate mode gives
an overcomplete basis and can hide conditioning problems.

## 6. Seam Matching

For a plumbing relation $uv=q$, sample

$$
u(\theta)=r e^{i\theta},
\qquad
v(\theta)={q\over u(\theta)}.
$$

The one-form must agree after pullback:

$$
f_L(z_L(\theta))
=
f_R(z_R(\theta)){dz_R\over dz_L}(\theta).
$$

For the two self-plumbing seams,

$$
z_R={z_L\over q},
\qquad
{dz_R\over dz_L}={1\over q},
$$

so each row imposes

$$
f(z_L)-{1\over q}f(z_R)=0.
$$

For the bridge seam,

$$
z_2=1+{q_3\over z_1-1},
\qquad
{dz_2\over dz_1}=-{q_3\over (z_1-1)^2},
$$

so each row imposes

$$
f_1(z_1)-f_2(z_2){dz_2\over dz_1}=0.
$$

All seam rows form an overdetermined matrix $M$.

## 7. Normalization

The two $A$ periods in the glasses channel are residues around the two
self-plumbing handles. With the duplicate infinity mode removed, the constraint
matrix $P$ simply selects the $z^{-1}$ coefficient on each sphere:

$$
\int_{A_I}\omega_J=\delta_{IJ}.
$$

For each normalized form, the solver minimizes the seam residual subject to the
exact period constraint:

$$
\min_c \|Mc\|_2
\qquad
\text{subject to}
\qquad
Pc=e_I.
$$

The implementation column-scales the joint system before solving. This is
essential at high cutoff because rational basis columns can differ by many
orders of magnitude.

## 8. Period Extraction For Collocation

The $B$ periods must be integrated along paths where the rational approximation
is controlled. Straight Schottky segments can pass through removed disks. At
high cutoff this can produce enormous wrong periods even when the seam residual
is tiny.

The corrected direct-collocation period extraction uses annular paths inside
the self-plumbing collars:

$$
z(t)=z_{\rm out}\exp(t\log q),
\qquad
0\le t\le1.
$$

Then

$$
dz=z(t)\log(q)\,dt.
$$

The default glasses collocation run uses $N=100$, $512$ collocation samples per
seam, and Schottky word length $8$ for the comparison target.

Collocation inputs are validated before solving.  The implementation rejects
zero or nonfinite plumbing parameters, under-sampled systems, nonpositive seam
radii, and radii outside the basic annular condition

$$
|q|<r<1.
$$

The seam residual reported by the fast period-matrix helper is now evaluated
on an independently oversampled seam matrix rather than only on the rows used
for the least-squares fit.

## 8.1 Local Inverse Solver

The module now includes a local glasses-chart inverse,
`solve_glasses_inverse_from_omega`.  It solves

$$
\Omega_{\rm glasses}(q_1,q_2,q_3)=\Omega_{\rm target}
$$

by real nonlinear least squares using the existing forward Schottky period
matrix.  The diagonal multipliers are parameterized as

$$
q_I=\exp(2\pi i\tau_I),
$$

so the logarithmic leading behavior of the diagonal periods is solved in
well-conditioned $\tau_I$ variables rather than directly in $q_I$.

The default seed is

$$
\tau_1^{(0)}=\Omega_{11},\qquad
\tau_2^{(0)}=\Omega_{22},\qquad
q_3^{(0)}=-2\pi i\,\Omega_{12},
$$

with clipping only to keep the first iterate inside the local chart bounds.
The final result should always be judged by the returned residual

$$
\Omega_{\rm glasses}(q_{\rm solved})-\Omega_{\rm target}.
$$

This is not a global inverse on Siegel space.  The caller must put
`Omega_target` in the same symplectic homology frame as the glasses plumbing
chart, and must choose another plumbing chart near other boundary components.

## 9. Genus-One Lookup And Refined Search

The genus-one theta graph is the clean test case for the table-plus-refinement
strategy.  The function `build_genus1_lookup_table` builds a fixed-perimeter
atlas

$$
(\ell_1,\ell_2,\ell_3)\longmapsto \tau(\ell)
$$

using random positive integer triples with

$$
\ell_1+\ell_2+\ell_3=L.
$$

The diagnostic table uses \(L=500\) and \(1000\) rows.  Given a plumbing
coordinate

$$
q=e^{2\pi i\tau},
$$

the inverse `genus1_lookup_refined_inverse` first selects the table row
minimizing the direct reconstruction error

$$
\left|q-\exp(2\pi i\tau(\ell))\right|.
$$

It then performs projected finite-difference descent on the fixed-perimeter
integer simplex.  The elementary move is

$$
(\ell_i,\ell_j)\mapsto(\ell_i-s,\ell_j+s),
$$

with all other edge lengths fixed.  The default step schedule is

$$
s=64,32,16,8,4,2,1.
$$

The direct \(q\)-metric is deliberate.  If the lookup uses only modularly
reduced \(\tau\)-distance, it can return a permuted theta graph that is
equivalent as a torus but does not reproduce the same plumbing coordinate
\(q\).

The check

```bash
python3 plumbing/plumbing_checks.py --check-genus1-lookup-refinement
```

generates `plumbing/generated/genus1_lookup_L500_s1000.npz`, tests \(10\)
random target \(q\) values, and writes
`plumbing/generated/genus1_lookup_refinement_10_samples.csv`.

## 10. Plumbing Data To Genus-Two Ribbon Lengths

The current repository already contains two forward maps:

1. plumbing data

$$
(q_1,q_2,q_3)
$$

to a normalized genus-two period matrix

$$
\Omega_{\rm plumb};
$$

2. one-face genus-two ribbon graph lengths

$$
\ell=(\ell_1,\ldots,\ell_9)
$$

to a normalized genus-two period matrix

$$
\Omega_{\rm rib}(\ell).
$$

The implemented inverse uses these two maps as a shooting method:

$$
(q_1,q_2,q_3)
\longmapsto
\Omega_{\rm plumb}
\quad\hbox{then choose}\quad
\ell
\quad\hbox{so that}\quad
\Omega_{\rm rib}(\ell)\approx \Omega_{\rm plumb}.
$$

This is not a new closed-form Strebel solver. It is an inverse-by-forward-map
algorithm built from the numerical technology already present in the codebase.

### 10.1 Target Period Matrix

For the glasses channel, the target matrix is obtained by
`plumbing_genus2_period_matrix`. With `algorithm="auto"` it uses the shared
adaptive policy in `genus2_hybrid_period_map.py`:

- if every `|q_e|` is below the topology threshold (`0.15` for theta, `0.20`
  for glasses) and collocation is comfortable: evaluate both methods and
  require agreement modulo an integral symmetric B-period shift;
- if every `|q_e|` is below that threshold but one tube is beyond the
  collocation conditioning floor: adaptive Schottky words with a measured
  word-tail error;
- if one tube is extremely long while another `q` remains outside the
  all-small region: rescaled multiprecision holomorphic one-forms;
- otherwise: normalized holomorphic one-forms by boundary collocation.

The Laurent basis and Schottky word length are raised adaptively to the
requested numerical bar. There is no deliberately unsupported scalar-q band.
A chart
whose standard sewing disks overlap is geometrically invalid and must be
replaced by another atlas marking.  The theta channel uses the same policy.
The returned matrix is symmetrized:

$$
\Omega_{\rm plumb}
\leftarrow
{1\over2}\left(\Omega_{\rm plumb}+\Omega_{\rm plumb}^T\right).
$$

For the sunrise channel, the current target solver is the Schottky series. A
direct sunrise boundary-collocation solver is not implemented yet.

### 10.2 Ribbon Forward Map

For each stored one-face genus-two topology, the code reconstructs the ribbon
rotation system from `compact_partition.get_stored_genus2_graph(topology)`.
The forward map is

```python
ribbon_genus2_period_matrix(edge_lengths, topology)
```

which calls `riemann_surface_tools.build_surface_from_ribbon_graph`.

The important implementation detail is that this forward map coerces every
edge length to an integer. Therefore the inverse problem is treated as a
discrete search problem, not a smooth Newton problem.

### 10.3 Gauge Choice, Forgetting The Marked Point, And Non-Uniqueness

A genus-two unpunctured period matrix has six real moduli. A one-face
genus-two metric ribbon graph has nine positive edge lengths; after fixing the
total perimeter it still has eight real coordinates. The extra two real
coordinates are the marked puncture position on the Riemann surface.

Equivalently, the Strebel graph naturally parameterizes the fixed-perimeter
version of

$$
{\mathcal M}_{2,1},
$$

while the plumbing period matrix only remembers the image in

$$
{\mathcal M}_{2,0}.
$$

The map being inverted is therefore really the projection

$$
{\mathcal M}_{2,1}\longrightarrow{\mathcal M}_{2,0},
$$

followed by the period map. The fiber has two real dimensions. The plumbing
data alone cannot determine those two coordinates.

This is the same issue one worries about in genus one, but genus one is
special. After fixing the theta-graph perimeter, the graph has two independent
length ratios, and the forgetful map

$$
{\mathcal M}_{1,1}\longrightarrow{\mathcal M}_{1,0}
$$

does not leave an extra continuous modulus because translations move the
marked point on an elliptic curve. The genus-one fixed-perimeter theta graph
therefore gives a clean two-real-dimensional test case for the inverse
strategy. Genus two does not have this simplification.

Therefore the genus-two inverse

$$
\Omega\longmapsto \ell
$$

is not unique unless a puncture/perimeter gauge is chosen. The implemented
routine chooses a reproducible representative by imposing

$$
\sum_{a=1}^9 \ell_a=L
$$

and adding a small balance penalty

$$
\lambda
\sqrt{
{1\over9}\sum_{a=1}^9
\left(\log\ell_a-{1\over9}\sum_{b=1}^9\log\ell_b\right)^2
}.
$$

The default is `L=72` and `lambda=10^{-3}`. This penalty is only a gauge
selector. The period-matrix residual is always reported separately.

Thus the function `plumbing_to_genus2_ribbon_lengths` should be read as
constructing a numerical section of the projection

$$
{\mathcal M}_{2,1}\to{\mathcal M}_{2,0},
$$

not as returning a unique Strebel graph canonically determined by the plumbing
parameters.

### 10.4 Residual

The target and ribbon period matrices are compared in Siegel coordinates. The
real parts are periodic under integral shifts of the $B$-cycle basis, so the
real residual is wrapped to the nearest integer:

$$
\Delta_R
=
\operatorname{Re}(\Omega_{\rm rib}-\Omega_{\rm plumb})
-
\operatorname{round}
\left[
\operatorname{Re}(\Omega_{\rm rib}-\Omega_{\rm plumb})
\right].
$$

The imaginary residual is compared directly:

$$
\Delta_I
=
\operatorname{Im}(\Omega_{\rm rib}-\Omega_{\rm plumb}).
$$

The scalar diagnostic is

$$
\epsilon_\Omega
=
\max\left(
|(\Delta_R)_{11}|,
|(\Delta_R)_{12}|,
|(\Delta_R)_{22}|,
|(\Delta_I)_{11}|,
|(\Delta_I)_{12}|,
|(\Delta_I)_{22}|
\right).
$$

The search objective is

$$
\epsilon_\Omega+\lambda\,{\rm balance}(\ell).
$$

The returned result stores both numbers as `period_residual` and
`balance_penalty`.

### 10.5 Search Strategy

The routine `plumbing_to_genus2_ribbon_lengths` performs:

1. compute $\Omega_{\rm plumb}$ from the selected plumbing solver;
2. choose one or more stored genus-two topologies;
3. generate fixed-perimeter integer candidates from Dirichlet-distributed
   edge-weight vectors, always including the uniform vector;
4. discard candidates whose ribbon period matrix is not in the Siegel upper
   half-space;
5. keep the best candidate by the objective above;
6. locally refine it by moving a small integer amount from one edge to another
   while preserving the total perimeter and positivity;
7. return the best topology, the nine edge lengths, both period matrices, and
   all diagnostics.

The local move is

$$
\ell_i\mapsto \ell_i-s,
\qquad
\ell_j\mapsto \ell_j+s,
$$

with all other edge lengths fixed. The default step schedule is

$$
s=2,\quad s=1.
$$

The default search is intentionally light. For a serious scan, increase
`topologies`, `random_candidates_per_topology`, and the local step schedule.

### 10.6 Large-$L$ Ribbon Evaluation

The ribbon period solver is an asymptotic large-edge-length method. A total
length such as

$$
\sum_a \ell_a=108
$$

is useful for a cheap cell/ratiometric search, but it is not a reliable
accuracy diagnostic for the ribbon period matrix. The code therefore separates
two operations:

1. search at a coarse fixed perimeter to find a candidate ratio;
2. rescale that ratio and recompute $\Omega_{\rm rib}$ at a large total length.

This is controlled by `evaluation_total_edge_length`. For example, setting

```python
evaluation_total_edge_length=3006
```

searches at the requested `total_edge_length`, rescales the best coarse ratios
to total length $3006$, and reports the period residual using the best
large-$L$ period matrix.

The result keeps both pieces of information:

- `search_edge_lengths`: the coarse integer ratio found by the search;
- `edge_lengths`: the scaled large-$L$ lengths used for the reported
  $\Omega_{\rm rib}$.

The coarse-to-large reranking is controlled by:

- `coarse_refine_count_per_topology`: how many coarse candidates per topology
  are locally refined;
- `large_evaluation_count`: how many refined coarse candidates are rescaled and
  re-evaluated at `evaluation_total_edge_length`.

This is a necessary correction to the first version of the inverse. The first
version scaled only the single best coarse candidate, which can choose the
wrong ratio because the ribbon solver is not accurate at small total length.

### 10.7 Current Weak Points

The residual is a diagnostic, not a proof of success. A large residual can mean
any of the following:

- the selected ribbon topology/cell does not contain the desired Strebel
  representative;
- the fixed perimeter or balance gauge chose a poor puncture representative;
- the period-matrix marking differs by a nontrivial symplectic transformation
  not covered by nearest-integer wrapping of real parts;
- the discrete integer cutoff is too coarse;
- the plumbing target matrix itself is inaccurate, especially outside the
  small-$q$ Schottky regime.

The next required refinement is a symplectic-marking search, or an invariant
comparison layer, before this inverse can be treated as a high-accuracy global
map from plumbing coordinates to Strebel lengths.

The high-accuracy diagnostics currently suggest the following repair order:

1. add the missing two real inputs, or an explicit section condition, for the
   fiber of ${\mathcal M}_{2,1}\to{\mathcal M}_{2,0}$;
2. optimize the ribbon ratios with a real global optimizer or surrogate model,
   not only sparse random Dirichlet samples;
3. compare period matrices after a serious symplectic reduction/search, not
   only after wrapping real parts by integers;
4. once a candidate ratio is found, validate it only at large $L$, preferably
   $\sum_a\ell_a\ge 5000$.

## 11. Output

The Schottky method and the boundary-collocation method both output the
normalized period matrix

$$
\Omega_{IJ}=\int_{B_J}\omega_I.
$$

The check file verifies seam residuals, $A$-period normalization, symmetry of
$\Omega$, positivity of $\operatorname{Im}\Omega$, Schottky convergence,
collocation convergence against Schottky at small $q$, chart diagnostics, and
an opt-in plumbing-to-ribbon inverse diagnostic.
