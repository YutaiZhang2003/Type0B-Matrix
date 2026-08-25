# BRY Convention and Duality Ledger

The master convention for this target is Balthazar--Rodriguez--Yin (BRY),
[arXiv:2201.05621](https://arxiv.org/abs/2201.05621), especially sections
2--4. Their paper fixes the on-shell worldsheet states, super-Liouville
normalization, sphere amplitudes, and matrix-model scattering dictionary.
It does **not** compute the torus vacuum amplitude or the genus-two vacuum
measure. Those higher-genus ingredients are imported separately and must be
translated into this ledger before use.

## 1. Symbols that must not be conflated

- \(\phi\): super-Liouville boson.
- \(\varphi,\widetilde\varphi\): bosonized \(\beta\gamma\) superghosts.
- \(\mu_{\rm L}\): coefficient of the BRY super-Liouville interaction.
- \(\mu_{\rm F}\): BRY matrix-model Fermi depth, fixed below by
  \(g_s=4/(\pi\mu_{\rm F})\).
- BRY print the same glyph \(\mu\) for these two roles. This project keeps
  them distinct until the torus comparison tests a map
  \(\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}\).
- \(A\): spacetime R--R axion. We do not rename it \(C\), because
  \(C,\widetilde C,C_{\rm even},C_{\rm odd}\) denote super-Liouville
  structure constants.
- \(R_{\rm phys}\): dimensionful Euclidean target-time circle radius.
- \(\ell_{\rm B}=\sqrt{\alpha'/2}\): BRY reference length.
- \(\rho=R_{\rm phys}/\ell_{\rm B}\): dimensionless numerical radius used
  in BRY-unit formulas.
- \(\mathcal L,\mathcal R\): left- and right-side matrix-model scattering
  modes.
- \(\omega\): worldsheet energy; \(\omega_{\rm MM}\): matrix-model energy.

## 2. Worldsheet theory in the BRY normalization

BRY use the shorthand

\[
\boxed{\alpha'=2}.
\]

Dimensionally, this means

\[
\ell_{\rm B}\equiv\sqrt{\alpha'/2},\qquad
\alpha'=2\ell_{\rm B}^2,\qquad [\ell_{\rm B}]_{\rm BRY}=1.
\]

Thus a length written as the number \(R\) in BRY-style formulas is really
the dimensionless \(\rho=R_{\rm phys}/\ell_{\rm B}\).

The Lorentzian worldsheet CFT is a timelike free boson \(X^0\) and Majorana
fermions \(\psi^0,\widetilde\psi^0\), tensor \(N=1\) super-Liouville theory,
the \(bc\) ghosts, and the \(\beta\gamma\) superghosts. In BRY component
normalization,

\[
S_{\rm SL}=\int d^2z\left[
\frac{1}{2\pi}\left(
\partial\phi\,\bar\partial\phi+
\psi\,\bar\partial\psi+
\widetilde\psi\,\partial\widetilde\psi\right)
+2i\mu_{\rm L} b^2\psi\widetilde\psi e^{b\phi}
+2\pi b^2\mu_{\rm L}^2e^{2b\phi}\right].
\]

\[
Q=b+b^{-1},\qquad
c_{\rm SL}=\frac32(1+2Q^2),\qquad
b=1,\quad Q=2,\quad c_{\rm SL}=\frac{27}{2}.
\]

The diagonal type 0B GSO projection is

\[
(\mathrm{NS}+,\mathrm{NS}+)\oplus
(\mathrm{NS}-,\mathrm{NS}-)\oplus
(\mathrm{R}+,\mathrm{R}+)\oplus
(\mathrm{R}-,\mathrm{R}-).
\]

For \(P\ge0\), the super-Liouville weights are

\[
h_{\rm NS}(P)=\frac12\left(\frac{Q^2}{4}+P^2\right)
=\frac{1+P^2}{2},
\]

\[
h_{\rm R}(P)=\frac1{16}
+\frac12\left(\frac{Q^2}{4}+P^2\right)
=\frac1{16}+\frac{1+P^2}{2}.
\]

The primary two-point normalization is

\[
\langle V_{P_1}(z,\bar z)V_{P_2}(0)\rangle
=\frac{\pi\,\delta(P_1-P_2)}
{|z|^{2(h_1+h_2)}},
\]

and the same delta normalization is used for \(V_P^{\mathrm R,\pm}\).
Define

\[
\Lambda_P=G_{-1/2}V_P,\qquad
\widetilde\Lambda_P=\widetilde G_{-1/2}V_P,\qquad
W_P=G_{-1/2}\widetilde G_{-1/2}V_P.
\]

Then

\[
\langle W_{P_1}(z,\bar z)W_{P_2}(0)\rangle
=\frac{\pi(2h_1)^2\delta(P_1-P_2)}
{|z|^{2(h_1+h_2+1)}}.
\]

Descendants use

\[
\{G_r,G_s\}=2L_{r+s}
+\frac{c}{12}(4r^2-1)\delta_{r,-s},
\]

and the corresponding antiholomorphic algebra.

## 3. Physical states, pictures, and PCO

BRY's incoming/outgoing BRST representatives are

\[
T_\omega^\pm
=g_s\,c\widetilde c\,
e^{-\varphi-\widetilde\varphi}
e^{\pm i\omega X^0}V_{P=\omega},
\]

\[
A_\omega^\pm
=g_s\omega\,c\widetilde c\,
e^{-\varphi/2-\widetilde\varphi/2}
e^{\pm i\omega X^0}\mathcal A_{P=\omega}^{\pm},
\]

where

\[
\mathcal A_P^\pm
=\frac1{\sqrt2}\left(
\sigma^0V_P^{\mathrm R,+}
\pm\mu^0V_P^{\mathrm R,-}\right).
\]

Here the superscript on \(T_\omega^\pm,A_\omega^\pm\) labels outgoing or
incoming energy sign, while \(\sigma^0,\mu^0\) are the timelike-fermion spin
and disorder fields. BRY's phases make the two-point functions
delta-normalized and the physical three-point coefficients real.

The holomorphic PCO used for the sphere calculation is

\[
\chi=-\frac12e^\varphi
\left(-i\psi^0\partial X^0+G^{\rm L}\right)
+\hbox{ghost terms},
\]

with an analogous antiholomorphic operator. At higher genus, this local
operator is only a representative of odd-modulus integration. Spurious-pole
avoidance and vertical corrections are governed by the chosen
supermoduli prescription, not fixed by BRY.

## 4. Super-Liouville three-point data

At \(b=1\), define

\[
\Upsilon_{\rm NS}(x)
=\frac{\Gamma(x/2)}{\Gamma(1-x/2)}
\Upsilon_1(x/2)^2,
\qquad
\Upsilon_{\rm R}(x)=
\Upsilon_1\!\left(\frac{x+1}{2}\right)^2,
\]

where \(\Upsilon_1\) is the \(b=1\) Barnes double-gamma combination used by
BRY. Also define

\[
N_{\rm NS}(P)=
\frac{\Gamma(1+iP)}{\Gamma(1-iP)}
\Upsilon_{\rm NS}(2iP),
\qquad
N_{\rm R}(P)=
\frac{\Gamma(\tfrac12+iP)}{\Gamma(\tfrac12-iP)}
\Upsilon_{\rm R}(2iP).
\]

With \(P_\Sigma=P_1+P_2+P_3\) and
\(\Delta_i=P_j+P_k-P_i\) for \(\{i,j,k\}=\{1,2,3\}\), the two NS
structures are

\[
C(P_1,P_2,P_3)=
\frac{i}{2}
\frac{\prod_{j=1}^3N_{\rm NS}(P_j)}
{\Upsilon_{\rm NS}(1+iP_\Sigma)
 \prod_{i=1}^3\Upsilon_{\rm NS}(1+i\Delta_i)},
\]

\[
\widetilde C(P_1,P_2,P_3)=
i\,
\frac{\prod_{j=1}^3N_{\rm NS}(P_j)}
{\Upsilon_{\rm R}(1+iP_\Sigma)
 \prod_{i=1}^3\Upsilon_{\rm R}(1+i\Delta_i)}.
\]

For \(P_1,P_2\) Ramond and \(P_3\) NS, set
\(\mathcal N=N_{\rm R}(P_1)N_{\rm R}(P_2)N_{\rm NS}(P_3)\). Then

\[
C_{\rm even}=
-\frac{i}{2}\,
\frac{\mathcal N}
{\Upsilon_{\rm R}(1+iP_\Sigma)
 \Upsilon_{\rm R}(1+i\Delta_3)
 \Upsilon_{\rm NS}(1+i\Delta_1)
 \Upsilon_{\rm NS}(1+i\Delta_2)},
\]

\[
C_{\rm odd}=
-\frac{i}{2}\,
\frac{\mathcal N}
{\Upsilon_{\rm NS}(1+iP_\Sigma)
 \Upsilon_{\rm NS}(1+i\Delta_3)
 \Upsilon_{\rm R}(1+i\Delta_1)
 \Upsilon_{\rm R}(1+i\Delta_2)},
\qquad
C_\pm=\frac12(C_{\rm even}\pm C_{\rm odd}).
\]

These are the four pants coefficients required by NS and R sewing. They are
real for \(P_i\ge0\) in the BRY phase convention. The implementation must
also encode BRY's reflection rules: \(C,\widetilde C\) change sign when one
momentum is reflected; reflecting the NS momentum changes the sign of both
\(C_{\rm even},C_{\rm odd}\); reflecting a Ramond momentum exchanges the
even and odd coefficients.

### 4.1 Global scope rule: BRY coefficients versus the Human-Note graded basis

This distinction applies to every Type 0B computation in this repository.
BRY define the real number \(\widetilde C_{\rm BRY}\) as the coefficient of
their locally normalized nonchiral top component

\[
W_{\rm BRY}=G_{-1/2}\widetilde G_{-1/2}V,
\qquad
\langle W_{\rm BRY}W_{\rm BRY}\rangle
=+D(2h)^2,
\]

where \(D\) is the matched primary two-point normalization.  The Human Note
instead sews the ordered graded tensor state

\[
w_{\rm HN}=(G_{-1/2}V)\otimes
            (\widetilde G_{-1/2}\widetilde V).
\]

Its declaration that left- and right-moving odd operators anticommute,
together with its graded tensor-product BPZ rule, gives directly

\[
\langle w_{\rm HN},w_{\rm HN}\rangle_{\rm HN}
=-D(2h)^2.
\]

Thus, after matching the primary and chiral three-form normalizations, the
two odd pants coefficients obey

\[
\boxed{
C_{\rm HN}^{(0)}=C_{\rm BRY},\qquad
C_{\rm HN}^{(1)}=\sigma i\,\widetilde C_{\rm BRY},\qquad
\sigma\in\{+1,-1\}.}
\]

The BPZ comparison fixes the invariant statement

\[
\bigl(C_{\rm HN}^{(1)}\bigr)^2=-\widetilde C_{\rm BRY}^{2}.
\]

It does **not** select \(\sigma\).  The implementation uses \(\sigma=+1\)
as its square-root branch.  A process containing an odd number of odd pants
coefficients would require one explicitly ordered component matrix element
to fix that branch; a genus-two vacuum graph with two pants cannot do so.

Apply this conversion exactly once, and only at a BRY-to-Human-Note sewing
boundary:

- BRY-native sphere formulas, including \(G,H,J\), PCO amplitudes, and their
  real \(C,\widetilde C\) products, keep \(\widetilde C_{\rm BRY}\) with no
  additional \(i\).
- A Human-Note graded descendant sum first converts the imported odd pants
  coefficient by the boxed formula and then applies the Human Note's own
  descendant/Koszul signs unchanged.
- The phase has nothing to do with either the physical free Majorana in a
  Type 0B denominator or the auxiliary Majorana in the double-Virasoro
  construction.

The genus-two numerical agreement is a check of this dictionary, not its
derivation; the derivation is the direct BPZ-normalization comparison above.

## 5. Matrix quantum mechanics and exact scattering dictionary

BRY use

\[
H=\frac12\operatorname{Tr}(P^2-X^2),
\qquad
h=\frac12(p^2-x^2).
\]

All one-particle states with \(E\le-\mu_{\rm F}\),
\(\mu_{\rm F}>0\), are filled on both
sides of the inverted oscillator. There is no extra \(\alpha'\), oscillator
curvature, or matrix Planck constant in this convention.

The physical left/right scattering basis is

\[
\boxed{
\mathcal L_\omega^\pm=T_\omega^\pm-A_\omega^\pm,\qquad
\mathcal R_\omega^\pm=T_\omega^\pm+A_\omega^\pm.}
\]

There is no \(1/\sqrt2\) in this basis. With the phases and external-state
normalizations above, BRY compare directly to the matrix-model S-matrix:
do not insert an additional leg-pole factor into this benchmark.

For one incoming and \(k\) outgoing right- or left-side excitations,

\[
\mathcal A_{1\to k}
=\sum_{g\ge0}\mu_{\rm F}^{-(k-1+2g)}
\mathcal A_{\rm pert}^{(g)}
+O(e^{-2\pi\mu_{\rm F}}).
\]

The exact dictionary is

\[
\boxed{\omega_{\rm MM}=2\omega,\qquad
g_s=\frac4{\pi\mu_{\rm F}},\qquad
C_{S^2}=\frac{\pi}{g_s^2}.}
\]

The energy map includes the associated rescaling of energy-conservation
delta functions. Useful tree-level checks are

\[
\mathcal A_{\mathcal R\to2\mathcal R}^{(0)}
=4i\omega\omega_1\omega_2,
\]

\[
\mathcal A_{\mathcal R\to3\mathcal R}^{(0)}
=8i\omega\omega_1\omega_2\omega_3(1+2i\omega).
\]

## 6. Genus-one modular path integral in BRY variables

For the vacuum amplitude, Wick rotate \(X^0\) to a Euclidean scalar
\(X\sim X+2\pi R_{\rm phys}\). The dimensionless radii are

\[
x=\frac{R_{\rm phys}}{\sqrt{2\alpha'}}
=\frac{\rho}{2},
\qquad
y=\frac{R_{\rm phys}}{\sqrt{\alpha'}}
=\frac{\rho}{\sqrt2},
\qquad
\rho=\frac{R_{\rm phys}}{\ell_{\rm B}}.
\]

Here \(\mathcal F=\log Z_{\rm string}\) denotes the connected string vacuum
functional, not the thermodynamic convention
\(-\beta^{-1}\log Z\).

Follow Douglas et al. at the path-integral level. For each of the three even
spin structures, the compact matter, super-Liouville, and combined ghost
factors are

\[
Z^X_{r,s}=\frac{y}{\sqrt{\tau_2}}|D_{r,s}|^2\Theta_\rho,
\qquad
Z^{\rm SL}_{r,s}=\frac{V_\phi^{\rm reg}}{2\pi\sqrt{2\tau_2}}
|D_{r,s}|^2,
\qquad
Z^{\rm gh}_{r,s}=\frac1{2\tau_2}|D_{r,s}|^{-4},
\]

where

\[
\Theta_\rho(\tau)=\sum_{m,n\in\mathbb Z}
\exp\left[-\frac{\pi\rho^2}{2\tau_2}|n-m\tau|^2\right].
\]

The nonzero-mode determinants cancel. Including the GSO factor \(1/2\) for
each even structure gives

\[
\widehat{\mathcal F}_{1,\rm even}
=\frac{3\rho}{16\pi}
\int_{\mathcal F}\frac{d^2\tau}{\tau_2^2}\Theta_\rho(\tau).
\]

Separating the zero orbit and unfolding the nonzero orbits to the strip gives

\[
\begin{aligned}
\int_{\mathcal F}\frac{d^2\tau}{\tau_2^2}\Theta_\rho
&=\frac\pi3+2\sum_{k\ge1}\int_{-1/2}^{1/2}d\tau_1
\int_0^\infty\frac{d\tau_2}{\tau_2^2}
e^{-\pi\rho^2k^2/(2\tau_2)}\\
&=\frac\pi3+\frac4{\pi\rho^2}\sum_{k\ge1}\frac1{k^2}
=\frac\pi3\left(1+\frac2{\rho^2}\right).
\end{aligned}
\]

This uses only the convergent Basel sum, not zeta-function continuation at
\(s=-1\). Hence

\[
\widehat{\mathcal F}_{1,\rm even}
=\frac \rho{16}+\frac1{8\rho}.
\]

For the odd spin structure, the Liouville-fermion zero modes cancel the
\(\gamma,\bar\gamma\) zero-mode divergence. Integrating the
\(\beta,\bar\beta\) zero modes inserts \(G(z)\bar G(\bar w)\), so the
remaining matter factor is
\(\langle\partial X(z)\bar\partial X(\bar w)\rangle\), including its contact
term. Equivalently, integrate the global odd modulus in

\[
(z,\theta)\sim(z+1,\theta)
\sim(z+\tau+\lambda\theta,\theta+\lambda).
\]

This global odd-supermodulus integral introduces no arbitrary local PCO
position. A PCO reformulation must reproduce the same contact term; vertical
integration is not an extra term to add to the global genus-one result.

Douglas et al. fix the resulting large-radius normalization from the
spectrum. Before choosing the GSO sign,

\[
\widehat{\mathcal F}^{(\varepsilon)}_{1,\rm odd}
=\frac{\varepsilon}{24\sqrt2}\left(y-\frac1y\right)
=\varepsilon\left(\frac\rho{48}-\frac1{24\rho}\right).
\]

The massless-field normalization gives \(\varepsilon_B=-1\) and
\(\varepsilon_A=+1\). Therefore

\[
\widehat{\mathcal F}_{1,\rm odd}^{0B}
=-\frac \rho{48}+\frac1{24\rho}.
\]

The odd term vanishes at \(\rho=\sqrt2\), equivalently
\(R_{\rm phys}=\sqrt{\alpha'}\).

### 6.1 Fluxless type 0A cross-check

Adding the common even term to the two choices of odd sign gives

\[
\widehat{\mathcal F}^{0B}_1(\rho_B)
=\frac{\rho_B}{24}+\frac1{6\rho_B},
\qquad
\widehat{\mathcal F}^{0A}_1(\rho_A)
=\frac1{12}\left(\rho_A+\frac1{\rho_A}\right).
\]

In particular,

\[
\widehat{\mathcal F}^{0A}_{1,\rm odd}
=\frac{\rho_A}{48}-\frac1{24\rho_A}
=-\widehat{\mathcal F}^{0B}_{1,\rm odd}.
\]

The circle spectrum explains the result after the fact: 0B has two momentum
towers and one winding tower, whereas 0A has one momentum tower and two
winding towers. No zeta-regularized vacuum sum is used in the derivation.

The dimensionful T-duality relation and its BRY numerical form are

\[
R_A=\frac{\alpha'}{R_B},\qquad
\rho_A=\frac2{\rho_B},\qquad
\widehat{\mathcal F}^{0A}_1(\rho_A)
=\widehat{\mathcal F}^{0B}_1(2/\rho_A).
\]

Thus \(\rho_B=2\) maps to \(\rho_A=1\), or
\(R_A=\sqrt{\alpha'/2}\).

Now regulate the Liouville zero mode directly from the BRY action. Under

\[
\phi\longrightarrow\phi+c,\qquad
\mu_{\rm L}\longrightarrow e^{-bc}\mu_{\rm L},
\]

the interaction is unchanged. Hence the wall moves by
\(-b^{-1}\log\mu_{\rm L}\). With a fixed weak-coupling coordinate cutoff,

\[
V_\phi^{\rm reg}
=-\frac1b\log\frac{\mu_{\rm L}}{\Lambda_{\rm L}}
+V_{\rm scheme}.
\]

At \(b=1\), the universal BRY-coordinate result is

\[
\boxed{
\mathcal F_1^{0B}(\rho;\mu_{\rm L})\big|_{\rm univ}
=-\frac1{24}\left(\rho+\frac4\rho\right)
\log\frac{\mu_{\rm L}}{\Lambda_{\rm L}}.}
\]

For fluxless type 0A,

\[
\boxed{
\mathcal F^{0A}_1\big|_{\rm univ}
=-\frac1{12}\left(\rho_A+\frac1{\rho_A}\right)
\log\frac{\mu_{\rm L}}{\Lambda_{\rm L}}.}
\]

Equivalently,

\[
\mathcal F^{0A}_1\big|_{\rm univ}
=-\frac{\log(\mu_{\rm L}/\Lambda_{\rm L})}{12\sqrt2}
\left(\frac{2R_A}{\sqrt{\alpha'}}
+\frac{\sqrt{\alpha'}}{R_A}\right).
\]

The project compares logarithmic derivatives, not bare volumes:

\[
\frac{\partial\mathcal F_1^{0B}}
{\partial\log\mu_{\rm L}}
=-\frac1{24}\left(\rho+\frac4\rho\right).
\]

For a source using \(\Phi=a\phi\) and
\(\mu_{\rm L}=\kappa\lambda^p\),

\[
V_\Phi^{\rm reg}\big|_{\log\lambda}
=-\frac{ap}{b}\log\lambda.
\]

Thus additive cutoff changes and \(\kappa\) do not affect the logarithmic
coefficient, whereas a field rescaling \(a\) or power map \(p\) does.

Only after this derivation do we introduce the comparison ansatz

\[
\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}.
\]

It predicts

\[
\mathcal F_1^{0B}\big|_{\log\mu_{\rm F}}
=-\frac p{24}\left(\rho+\frac4\rho\right)\log\mu_{\rm F}.
\]

The standard string/matrix comparison corresponds to \(a=b=p=1\), up to
the irrelevant constant \(\kappa\). Agreement is therefore a test of
\(p=1\), not an input to the worldsheet modular integral.

## 7. Genus-two normalization boundary

BRY fixes the sphere normalization and hence the relation of matrix powers
of \(\mu_{\rm F}^{-1}\) to string perturbation theory. In particular,

\[
\mu_{\rm F}^{-2}=\frac{\pi^2}{16}g_s^2.
\]

Thus a matrix-model genus-two term \(A_2/\mu_{\rm F}^2\) maps to
\((\pi^2A_2/16)g_s^2\). This does **not** by itself fix the normalization of
the unpunctured genus-two worldsheet path integral: its \(bc\)-\(\beta\gamma\)
measure, odd-spin zero modes, contact terms, and integration cycle must be
fixed by factorization to the BRY-normalized sphere data and the torus
anchor.

BRY computes tree-level \(1\to2\) and \(1\to3\) amplitudes. It explicitly
leaves higher-genus super-Virasoro blocks and loop amplitudes for future
work. Therefore:

- BRY is the master convention and sphere-data source;
- Douglas et al. is the genus-one measure source;
- D'Hoker--Phong supplies the even-spin genus-two gauge-fixing framework;
- Sen--Witten supplies vertical integration when local PCO sections are
  used;
- the odd-spin genus-two 0B measure remains a separate derivation.
