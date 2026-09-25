# Type 0B torus NSNS two-point calculation — 2026-09-20

This is a record of a new calculation, not a claim of a finished amplitude.
The spin sum and even-spin sewing are implemented. An odd-spin assembly,
collision finite parts and analytic leading cusp are executable and have
been evaluated. The numerical boundary tests and the remaining physical
normalization audit are recorded separately. In particular, `trial_sum` in
a JSON file is **not** a BRY-normalized genus-one S-matrix coefficient.

## External states and continuation

The external states are the project's BRY NSNS tachyons,

\[
T_\omega^\pm=g_s c\bar c e^{-\varphi-\bar\varphi}
 e^{\pm i\omega X^0}V_\omega,
\qquad \alpha'=2,\quad b=1,\quad Q=2,\quad c_L=27/2.
\]

The two **Liouville** momenta are both \(\omega\); the target-time
momenta are \(+\omega,-\omega\). We use
\(h=(1+P^2)/2\), \(h_R=9/16+P^2/2\), and \(dP/\pi\) for every
Liouville sewing edge. The primary two-point coefficient is
\(\pi\delta(P-P')\). No change to the \(\mathcal R,\mathcal L\) basis is
made. If needed later, that basis is \(\mathcal R=T+A,\mathcal L=T-A\),
with no square root of two.

The computed energy is \(\omega=i/4\). At imaginary energy the time
correlator is continued from the Lorentzian formulas; it is not replaced
by a correlator with oppositely continued Liouville momenta. The
antiholomorphic conformal block uses the **same complex weights and
structure constants** as the holomorphic block. Only the geometric nome
is conjugated. Thus `Fbar` below does not mean conjugating all the
coefficients of `F`.

The real momentum contours are retained at this energy. The elementary
external-pair poles have not crossed them for \(0<\operatorname{Im}
\omega<1/2\). This statement does not supply a Lorentzian \(i\epsilon\)
prescription for the integrated answer. No Lorentzian value, polynomial
fit in energy, or full resummation is claimed here.

## Spin pairing and projector

Let \(\delta=3,4,2,1\) denote \(\theta_3,\theta_4,\theta_2,\theta_1\),
respectively. These are the NS trace, NS supertrace, R trace and R
supertrace. Type 0 uses a **single common** left/right spin structure:
there is no independent sum over sixteen pairs.

The diagonal projection retains

\[
(NS+,NS+)\oplus(NS-,NS-)\oplus(R+,R+)\oplus(R-,R-).
\]

In the conventional Arf orientation the formal worldsheet sum is

\[
\boxed{\mathcal I_{0B}=\tfrac12
 (\mathcal I_{3,3}+\mathcal I_{4,4}+
  \mathcal I_{2,2}+\mathcal I_{1,1}).}
\]

Each spin structure has weight \(1/2\). The Ramond ground doublet is
retained inside the sewing traces; it is not an additional factor to put
in front of the R terms. Changing to 0A inserts
\((-1)^{\operatorname{Arf}(\delta)}\), reversing only the odd term in
this orientation. This agrees with the oriented-type-0 discussion in
[Kaidi–Parra-Martinez–Tachikawa, p. 3](https://arxiv.org/pdf/1908.04805).
This abstract GSO sign must still be distinguished from signs in a
specific evaluated ghost/Clifford trace. The numerical odd assembly
below has not completed that independent convention audit.

The heterotic effective weights \((+,-,-)\) do not apply. The minus
signs of a chiral superghost trace occur in both chiralities here. In
particular, using a heterotic chiral sign only once changes the theory.

## Even spin: pictures and free fields

Two NS punctures at genus one require two PCOs in each chirality. Use

\[
\chi=-\tfrac12 e^\varphi(G_L-i\psi^0\partial X^0)+\cdots.
\]

For an even spin structure both PCOs can approach their corresponding
punctures. The resulting vertices are in picture zero. There are four
Liouville components, denoted GG/GG, GG/PP, PP/GG and PP/PP, where GG
means \(G_{-1/2}\) at each of the two punctures in that chirality.
The product of all four PCO coefficients is \(1/16\).

The numerical coordinates have periods \(1,\tau\). Its integration
measures are explicitly \(d\tau_1 d\tau_2\) and \(dx\,dy\), not the
twice-area convention sometimes denoted \(d^2z\) in string notes.
Set

\[
E(z)=\frac{\theta_1(z|\tau)}{\theta_1'(0|\tau)},\qquad
G(z)=-2\log|E(z)|+\frac{2\pi(\operatorname{Im}z)^2}{\tau_2},
\]
\[
S_\delta(-z)=\frac{\theta_\delta(-z|\tau)}
 {\theta_\delta(0|\tau)E(-z)}.
\]

The timelike bosonic oscillator factor is \(e^{-\omega^2G}\). Stripping
its time zero-mode conservation factor gives the free-boson determinant
\(1/[\sqrt{8\pi^2\tau_2}|\eta|^2]\). One Majorana together with the
superghosts gives \(|\eta/\theta_\delta|\). The working
\(bc\)-measure/automorphism convention is \(|\eta|^4/2\). Together
with the GSO projector and the PCO coefficients this gives

\[
B_\delta(\tau,z)=
\frac{|\eta|^3 e^{-\omega^2 G(z)}}
 {64\sqrt{8\pi^2\tau_2}|\theta_\delta(0|\tau)|}.
\]

An independent comparison of this stripped convention with the complete
BRY punctured-torus measure, including the Wick-rotation phase and the
energy delta function, remains necessary before quoting a physical
S-matrix coefficient. The sphere constant \(C_{S^2}=\pi/g_s^2\) alone
does not perform that comparison.

### Explicit sewn even-spin expression

Write \(q_0=e^{2\pi i z}\), \(q_1=e^{2\pi i(\tau-z)}\) and keep the
unwrapped logarithms. For each homogeneous three-form assignment
\(f=(0,0),(1,1)\), let \(F_{G,f},F_{P,f}\) be the stripped chiral
necklace blocks supplied by the current NS recursion or the checked
Ramond PBW sewing. Define

\[
D_{L,f}=-(-1)^{f_1}(2\pi i)F_{G,f}
              -\omega^2 S_\delta(-z)F_{P,f},
\]
\[
D_{R,f}=-(-1)^{f_1}(-2\pi i)\bar F_{G,f}
              -\omega^2\overline{S_\delta(-z)}\bar F_{P,f}.
\]

Then the implemented even density is

\[
\mathcal J_\delta=(2\pi)^{2(1+\omega^2)}B_\delta
 \int\frac{dP_0dP_1}{\pi^2}\,
 e^{-2\pi[P_0^2y+P_1^2(\tau_2-y)]}
 \sum_f K_fD_{L,f}D_{R,f},
\]

with the NS Casimir factor \(e^{\pi\tau_2/4}\), when applicable, and
the Ramond HJS sign sum included. This expression already includes the
spin's GSO factor. In NS, \(K_f\) is the product of the two \(C\) or
\(\widetilde C\) coefficients. In R, sum over the two HJS signs at
each vertex, with coefficient \(C_{\rm even}/2\) for sign + and
\(C_{\rm odd}/2\) for sign −, and the PP/PP homogeneous-form phase.
These HJS sign labels are not the physical \((C_{\rm even}\pm
C_{\rm odd})/2\) Ramond-family labels.

For \(\theta_4\), insert full NS parity on the long edge:
\((-1)^{2\ell_1}\). For R, do not replace parity by integer level.

The fixed-form tensor phase used in the code is, for external parity
words \(a,b\),

\[
\Phi(f;a,b)=(-i)^{|b|+\epsilon |f|}
 (-1)^{f\cdot b+\sum_{i<j}(f_i+b_i)(f_j+a_j)},
\quad \epsilon=1\ (NS),\quad0\ (R),
\]

where the exponent of −1 is reduced modulo two. The GG conversion is
performed once at this sewing boundary. The current BRY constants are
not multiplied by a second blanket odd-pants factor.

## Odd spin: it does not vanish by the Het argument

There are a timelike fermion zero mode and a super-Liouville fermion
zero mode in each chirality. There are no 23 spectator fermions. The
available insertions can saturate the actual zero modes.

Raising both punctures to picture zero in odd spin gives an inadmissible
zero-times-infinity expression: the superghost zero modes have not been
treated. Leave vertex 1 in picture −1, raise vertex 2, and retain a
generic PCO. In period-one coordinates,

\[
\langle\delta(\beta)(x)\delta(\gamma)(0)\rangle_{\rm odd}
 =-\frac{1}{2\pi\eta(\tau)^2}.
\]

It is independent of the PCO position. The constant-term contour
\(\oint dx/(2\pi i x)\) replaces the matter supercurrent by
\(G^m_{-3/2}\) at vertex 1. Thus the relevant matter correlator contains
\(G^m_{-3/2}\bar G^m_{-3/2}V_1^m\) and
\(G^m_{-1/2}\bar G^m_{-1/2}V_2^m\). This prescription follows the
one-loop NSNS section of the supplied `references/string notes.tex`,
equations labeled `oddscont` and `oddscontbb`; pure ghost terms drop by
the separate ghost-number selection rules.

Exactly one timelike fermion must remain in each chirality. The two
Liouville words are therefore

\[
A=(G_{-3/2}V_1)V_2,\qquad B=V_1(G_{-1/2}V_2).
\]

The \(\partial\psi^0\) part has no zero-mode contribution. The map
\(w=e^{2\pi i z}\) sends the first word to
\((2\pi i)^{2h_E+3/2}(G_{-3/2}+G_{-1/2})\) in plane sewing.
That additional \(G_{-1/2}\) is essential; merely attaching a primary
weight to \(G_{-3/2}V\) is incorrect.

New coefficients are generated for words (3,0), (1,0), (0,1), where 3
means \(G_{-3/2}\). At the trace closure, every actual Ramond basis state
is multiplied by \((-1)^{\mathrm{parity(state)}}\). The implementation
never uses \((-1)^{\mathrm{integer\ level}}\) as a substitute.

### Executed odd-spin assembly and its status

Set

\[
d=\frac{\theta_1'(-z|\tau)}{\theta_1(-z|\tau)}
                  -\frac{2\pi i y}{\tau_2}.
\]

With the fixed-form phase above, the implemented coefficients multiplying
the sewn AA, AB, BA, BB blocks are

\[
(4\pi^2\omega^2,\;-2\pi i\omega^2\bar d,\;
 -2\pi i\omega^2 d,\;\omega^2d\bar d).
\]

The opposite ordering of the two cross components includes the graded
reordering of the timelike and Liouville fermions; it cannot be obtained
by treating both factors as ordinary commuting scalar fields.

There is also a BB contact-contraction component
\(-\pi/\tau_2\). It comes from the regular
\(\langle\partial X^0\bar\partial X^0\rangle\) contraction and the
two factors of −i in the time supercurrent. Omitting it fails the local
collision comparison.

The common factor in this assembly is

\[
B_o^{\rm trial}=
\frac{(2\pi)^{3+2\omega^2}e^{-\omega^2G}}
 {128\pi\sqrt{8\pi^2\tau_2}}.
\]

The superghost, timelike-fermion and bosonic nonzero-mode determinants
cancel against the \(bc\) determinants. Sum the two allowed homogeneous
forms (0,1), (1,0) and all four HJS sign assignments with the explicit
Ramond parity insertion. This is implemented in `odd.py`, which
deliberately exposes `trial_components`, not a certified amplitude API.

Modular covariance and a leading contact-collision check have been
performed. They **cannot fix a constant relative sign**, nor do they
certify that this local PCO section has the required completed boundary
prescription. A convention check against a fully ordered physical
Ramond factorization matrix element remains open. The opposite-sign
diagnostic is saved, not chosen by comparison to a matrix-model answer.

## Collision continuation

For an NS bridge of weight \(h=(1+P^2)/2\), the two even-spin chiral
Ward vectors (Liouville GG, timelike PP) are

\[
(1+\omega^2-h,-\omega^2),\qquad
(h+\tfrac12+\omega^2,-\omega^2).
\]

Their sums are \(1-h\) and \(h+1/2\). Type 0B has a PCO sum in both
chiralities, so the collision coefficients involve their **squares**.
The powers are \(r^{2h-4}\) and \(r^{2h-3}\). With the corresponding
Liouville one-point coefficients denoted \(K_e,K_o\), the leading
continued disk terms are

\[
\pi K_e(h-1)\rho^{2h-2},\qquad
\pi K_o\frac{(h+1/2)^2}{h-1/2}\rho^{2h-1}.
\]

The second-family sewing coefficient includes the norm denominator
\((2h)^2\). The apparent \(h=1\) pole cancels in the first expression.
The \(h=1/2\) endpoint must be taken together with its structure-constant
zeros. The code never sets a singular denominator to zero by hand.
This defines a local radial analytic finite part; it is not by itself a
derivation of the global string integration cycle.

The known angular Gaussian is also integrated through power four using
\(\langle\sin^{2m}\theta\rangle=\binom{2m}{m}/4^m\).
The collision disk is added once, after excluding that disk from the
bulk puncture integration. Cusp restoration uses the complementary
height interval, so its disk is not added to the finite-height disk a
second time.

For the odd-spin BB contact term, the leading collision involves a new
Ramond-supertrace one-point function of the NS primary. It is generated
independently, not copied from the ordinary Ramond one-point trace. Its
ground coefficient is 2 in the odd HJS structure, rather than the even
structure selected by the ordinary trace. This gives the saved
`odd_contact.npz` and a radial primitive
\(\pi\rho^{2h-1}/(h-1/2)\). The remaining odd-spin local terms have not
yet been supplied with a complete independently derived disk expansion.
Their omission is an explicit limitation of the reported integrals.

## Cusp height integration and the shared infrared regulator

The long-edge ground state is retained; the short-edge descendant series
is kept through the recorded cutoff. No heterotic tail coefficients are
reused. At \(\tau_2=Y\to\infty\),

\[
E(-z)\to-\sin(\pi z)/\pi,\quad
S_{3,4}(-z)\to-\pi/\sin(\pi z),\quad
S_2(-z)\to-\pi\cot(\pi z).
\]

After cancelling Casimir factors the even-spin determinant weights are
1, 1, 1/2. For \(z=x+is\) in half of the punctured torus, integrate
\(Y\ge L=\max(Y_0,2s)\). The other half supplies a factor of two.
Define

\[
I_k(A,B,L)=\int_L^\infty dY\,Y^{-k-1/2}e^{-AY-B/Y},
\quad A=2\pi P_{\rm long}^2,\quad B=2\pi\omega^2s^2.
\]

The even term needs \(I_0\). In odd spin,
\(d=d_0+d_1/Y\), with \(d_0=-\pi\cot\pi z\),
\(d_1=-2\pi i s\), so the four noncontact terms need \(I_0,I_1,I_2\),
and the contact term needs \(I_1\). `cusp.py` uses scaled complementary
error functions and a removable-singularity expansion around \(B=0\).
It combines the exponentials before evaluating, avoiding a separately
overflowing \(e^{A s}\). These primitives were compared to independent
adaptive quadrature, including negative B at imaginary external energy.

In the Ramond term, separately height-integrated bulk and disk pieces
contain an infrared \(1/P_{\rm long}\) behavior. Their cancellation
must use the same long-momentum regulator. Using unrelated endpoint
quadratures produced a large false remainder. The current cusp entry
point rejects mismatched long-momentum grids unless the explicit
failure-diagnostic option `--allow-unmatched-ir` is provided.

Matching the grids reduces the collision mismatch drastically. It does
not prove convergence of the joint zero-momentum/collision limit:
the short-channel block remainder is nonuniform near the boundary.
Descendant-order, collision-radius, momentum-rule, and matching-height
checks are separate requirements.

## What remains before a physical result

1. Complete and verify the odd-spin collision prescription, including
   the noncontact terms, and its physical Clifford/ghost orientation.
2. Demonstrate convergence of the matched Ramond infrared/collision
   cancellation. The leading cusp formula is an approximation; its
   fixed-z overlap check is not a uniform error bound over the integral.
3. Fix the complete punctured-torus normalization and Wick/delta-function
   conversion by worldsheet factorization in the BRY conventions.
4. Only after these steps, compare with the matrix-model coefficient.

The current numerical values and convergence evidence are in `RESULTS.md`
and the data JSONs. None of the integrator modules imports a matrix-model
prediction or a fitted energy dependence.
