# NS upper components in the elliptic h-recursion

30 August 2026. Research extension; only the human-note convention is used.
The companion `.tex` file gives the derivation and full conformal factor.
The research extension itself did not change the human note, production
amplitudes, subtraction code, or released bottom-component handoff package.
The subsequent production decision is **all c-recursion**, recorded in
`../c-Recursion/NS_SPHERE_ALL_C_PRODUCTION_POLICY_2026-08-30.md`.

## Result

For an external marking `beta_j=0,1`, insert
`(G_{-1/2})^beta_j phi_(d_j)`. There are three distinct ingredients:

1. **Geometry:** replace `d_j` by `D_j=d_j+beta_j/2` in Lambda. The
   upper component is a Virasoro primary, but not an NS superprimary.
2. **Null-vector residues:** retain the original `d_j` in the fusion
   polynomials and change the three-form parity labels by incidence.
3. **Regular part:** retain all nonnegative common-weight powers.
   General cap components do **not** retain the all-bottom unit seed.

Let `epsilon_i` be the internal level parity, `m=n-3`, and use one-based
indices in the following formulas:

    alpha_1 = epsilon_1 XOR beta_1 XOR beta_2
    alpha_v = epsilon_(v-1) XOR epsilon_v XOR beta_(v+1), 2 <= v <= m
    alpha_(m+1) = epsilon_m XOR beta_(n-1) XOR beta_n

The h-recursion is

\[
H_{\beta,\epsilon}(h;p)=S_{\beta,\epsilon}(h_1,a;p)
+\sum_{k,r,s}\frac{\varrho_k^{rs/2}(-1)^{rs}A^c_{rs}
 P^{\alpha_k}_{rs}(L_k;c)P^{\alpha_{k+1}}_{rs}(R_k;c)}{h_k-h_{rs}}
 H_{\beta,\epsilon'}(h';p).
\]

Here `a_i=h_i-h_1`, the adjacent ordered pairs are unchanged from the
bottom formula, and

    hhat_j = h_j-h_k+h_rs
    h'_j = hhat_j + (rs/2) delta_(j,k)
    epsilon'_j = epsilon_j XOR ((rs mod 2) delta_(j,k))

Adjacent internal weights in `L_k,R_k` are evaluated at `hhat`. Every
child recomputes its differences; external component markings stay fixed.

The normalization is exactly

\[
\mathcal F_{\beta,\epsilon}
=\Lambda_n^{(c)}(D)\prod_i\varrho_i^{h_i-c/24}
 C_{\rm NS}(q)H_{\beta,\epsilon},\qquad
C_{\rm NS}(q)=\theta_3(q^2)\prod_{n\ge1}(1-q^{2n})^{-3/4}.
\]

No alternative H normalization, scalar BRY rephasing, or implicit parity
sum has been introduced. The `16q` / endpoint `4p_i` factors are unchanged.

## Why the new seed matters

For four points with upper components at z and 1, `beta=0110`, the odd
leading plane coefficient is

\[
-\frac{(h+d_2-d_1)(h+d_3-d_4)}{2h}.
\]

Its elliptic regular part therefore contains

\[
S_{0110,1}=2(-h+d_1-d_2-d_3+d_4)q^{1/2}+O(q^{3/2}).
\]

For five points with `beta=01110`, write `a=h_2-h_1`. Through total
degree one, the four seeds are

\[
\begin{aligned}
S_{00}&=1+O_{>1},\\
S_{10}&=(a-d_3)\sqrt{p_1}+O_{>1},\\
S_{01}&=-(a+d_3)\sqrt{p_2}+O_{>1},\\
S_{11}&=[2h_1+a+d_3-\tfrac12+2(d_2+d_4-d_1-d_5)]\sqrt{p_1p_2}+O_{>1}.
\end{aligned}
\]

The higher regular coefficients are not being set to zero. The JSON
symbolic ledgers retain every polynomial through their stated cutoff.

## Constructive finite-order seed computation

At fixed elliptic degree, the PBW pullback is rational in the internal
weights, with only the Kac poles. Substitute `h_i=H+a_i` and use exact
polynomial division in H:

\[
S_{\beta,N}(H,a)=\operatorname{Pol}_{H=\infty}
H_{\beta,N}(H,H+a_2,\ldots).
\]

This keeps the entire quotient, including its constant term. There is no
large-H numerical fit. The resulting polynomials depend on the internal
differences and must be evaluated at the *child* differences as well.

`ExactPBWSeeds` implements this prescription. Its compiled numerical
form can be reused at different internal weights for fixed c, external
weights, component marking, and cutoff. The code rejects missing seeds,
the wrong marking or c/external parameters, an exceeded seed cutoff, or
evaluation above the precision at which numeric seeds were compiled.

This is an executable reference method, **not** a new independent closed
seed formula. Since PBW supplies these general seeds, testing the ensuing
recursion against the same PBW coefficients verifies pole transport and
rational reconstruction; it does not independently validate the seed.
General-cap production speedups are not claimed.

## Independently tested fast special case

When all four cap fields remain bottom components,

    beta_1 = beta_2 = beta_(n-1) = beta_n = 0,

upper markings may be placed at mobile insertions. Under the same
coefficientwise cap-asymptotic assumptions as the bottom proposal, the
normalized diagonal upper insertion contributes `(-1)^F` at large h.
Only even F survives from bottom caps, so the sign is +1 and the proposed
regular seed remains

\[
S_{\beta,\epsilon}=\delta_{\epsilon,0}.
\]

`InteriorUnitSeed` explicitly opts into this fast proposal. It refuses
any upper marking at a cap. No PBW or c-recursion coefficients enter its
h-recursion. In particular, `beta=00100` is the five-point block with
one upper component at t, and `beta=001100` has both six-point mobile
insertions upper. The finite tests below support this proposal, not an
arbitrary-n theorem or a moduli-uniform convergence bound.

## Checks completed

Physical degree N includes all twice-level keys with sum <= 2N, across
every internal parity sector.

| Check | Coverage | Result |
|---|---|---|
| Human three-form table | All 8 component triples, symbolic weights | Exact agreement |
| Four-point h-pole certificates | All 16 markings, degree 3; arbitrary internal h at fixed exact fixture A | 112 coefficients, 336 residue identities pass |
| Five-point h-pole certificates | All 32 markings, degree 3; arbitrary h1,h2 at fixed exact fixture A | 896 coefficients, 3,392 residue identities pass |
| Compiled seed reuse | Three new internal-weight tuples for every marking in the preceding two rows | All pass |
| Fully symbolic four-point certificate | Marking 0110, degree 3; b and every external/internal weight symbolic | 7 coefficients, 21 residue identities pass |
| Fully symbolic five-point certificate | Marking 01110, degree 3; b and every external/internal weight symbolic | 28 coefficients, 106 residue identities pass |
| Independent PBW/c comparison | All 32 five-point markings, degree 4, fixture B | 1,440 coefficients; maximum scaled discrepancy below 2.0e-78 |
| Independent PBW/c comparison | All 64 six-point markings, degree 3, fixture C | 5,376 coefficients; maximum scaled discrepancy below 4.1e-79 |
| Fast interior seed, h/PBW/c | Five points, 00100, degree 10, four generic fixtures A-D | 924 coefficients; maximum h/PBW discrepancy below 4.6e-70 |
| Fast interior seed, h/PBW/c | Six points, 001100, degree 6, fixture B | 455 coefficients; maximum h/PBW discrepancy below 1.7e-72 |
| Unit/regression suite | 10 new component tests + 10 existing pillow tests | 20 pass |
| Released bottom-component package regression suite | Existing package, unchanged | 10 pass |

The degree-3 polynomial-seed h audit for five-point marking 01110 was
also repeated with fixture B. Numerical comparisons use 80 digits
(compiled-seed reuse tests use 70). Scaled error means
`abs(x-y)/max(1,abs(x),abs(y))`.

The degree-10 fast-seed fixtures, in zero-to-infinity order, are:

| Case | b | d1,d2,d3,d4,d5 | h1,h2 |
|---|---|---|---|
| A | 1.27 | .31,.42,.53,.47,.28 | .73,1.10 |
| B | 1.43 | .22,.61,.39,.74,.45 | 1.13,.85 |
| C | .83 | .17,.83,1.21,.34,.67 | .19,1.47 |
| D | 1.61 | 1.12,.26,.79,.58,.91 | 2.31,.64 |

## Using the research code

Run from the repository root with
`PYTHONPATH=Code/sphere_five_kummer_h_recursion`. It uses mpmath and SymPy.
Always construct and evaluate numerical recursion inside a fixed
`mp.workdps(...)` context. The returned values are coefficients of H,
not the complete sphere block or the physical PCO sum.

Fast mobile-insertion case:

```python
import mpmath as mp
from ns_pillow_components import ComponentEllipticRecursion, InteriorUnitSeed
from ns_pillow_elliptic_audit import indices

with mp.workdps(80):
    beta = (0, 0, 1, 0, 0)
    d = tuple(map(mp.mpf, ('.31', '.42', '.53', '.47', '.28')))
    h = tuple(map(mp.mpf, ('.73', '1.10')))
    block = ComponentEllipticRecursion(
        mp.mpf('1.27'), d, h, beta, seed=InteriorUnitSeed(beta))
    coefficients = {k: block.coefficient(k) for k in indices(2, 20)}
    p = (mp.mpf('.03'), mp.mpf('.07'))
    epsilon = (1, 0)
    H = mp.fsum(value * mp.fprod(x**(mp.mpf(n)/2) for x,n in zip(p,key))
                for key,value in coefficients.items()
                if tuple(n % 2 for n in key) == epsilon)
```

General cap components, with exact finite-order seeds:

```python
import sympy as sp
import mpmath as mp
from ns_pillow_components import ExactPBWSeeds, ComponentEllipticRecursion

b_exact = sp.Rational(127, 100)
c_exact = sp.Rational(3, 2) + 3*(b_exact + 1/b_exact)**2
d_exact = tuple(map(sp.Rational, ('.31', '.42', '.53', '.47', '.28')))
beta = (0, 1, 1, 1, 0)
seed_reference = ExactPBWSeeds(c_exact, d_exact, beta, degree=3).build()

with mp.workdps(80):
    seed = seed_reference.numeric(dps=80)
    d = tuple(mp.mpf(str(x.p))/x.q for x in d_exact)
    block = ComponentEllipticRecursion(
        mp.mpf('1.27'), d, (mp.mpf('.73'), mp.mpf('1.10')), beta, seed=seed)
    coefficient = block.coefficient((1, 1))
```

The same compiled `seed` may be passed to a new block at different
internal weights. It must not be reused for different c or external
weights. Nothing here creates the missing complex-moduli branch adapter.

## Reproducing the audits

From the repository root:

```sh
python3 -m unittest discover -s Code/sphere_five_kummer_h_recursion -p 'test_ns_pillow*.py' -v
python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_components.py --mode symbolic --points 5 --degree 3 --output /tmp/ns-upper-symbolic.json
python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_components.py --mode symbolic --points 5 --degree 3 --markings 01110 --fully-generic --output /tmp/ns-upper-fully-generic.json
python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_components.py --mode unit-seed --points 5 --degree 10 --markings 00100 --case D --output /tmp/ns-upper-unit-seed.json
python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_components.py --mode plane --points 6 --degree 3 --case C --output /tmp/ns-upper-pbw-c.json
```

The stored ledgers are `Data Set/h-Recursion/ns_components_*.json` and
include source hashes and parameter/truncation scopes. The exact
regular polynomials appear in the symbolic ledgers.
The fully generic five-point degree-3 audit took about 16.5 minutes on
the local machine; the fixed-external-parameter and numerical audits are
substantially cheaper. This is verification time, not h-recursion timing.

## Remaining h-recursion research (not the production route)

- Derive an efficient all-orders seed representation for upper cap fields;
  PBW seed extraction is presently the finite-order reference route.
- Validate a confluent b=1 prescription, without silently detuning the
  physical theory.
- Implement consistent complex-coordinate and spin/logarithm transport.
- The former proposal to enable an elliptic-nome gate at `abs(q)<0.3`
  is superseded: production remains all-c. Any future h-recursion
  adoption requires a separate decision and amplitude-level validation.
- Keep polynomial subtraction on top of the c-recursive evaluator. The existing
  `q1,q2` plane ratios are not the elliptic segment nomes.

The earlier bottom-component package is deliberately not repackaged as
though these additional production ingredients were finished.
