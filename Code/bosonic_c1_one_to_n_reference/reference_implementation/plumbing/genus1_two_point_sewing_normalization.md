# Torus two-point normalization from target-free sewing

## Result

In the normalized \(c=1\) vertex convention

\[
\mathcal V_\omega^\pm
=g_s\,c\widetilde c\,
e^{\pm i\omega X^0/\sqrt{\alpha'}}V_{\omega/2},
\qquad
\langle V_PV_{P'}\rangle=\pi\delta(P-P'),
\]

the reduced genus-one two-point amplitude is

\[
\boxed{
\mathcal A_{1,2}^{\rm ws}(\omega)
=8\pi^2 i\,g_s^2\,\mathcal I_{1,2}(\omega).}
\]

This coefficient is obtained below from sphere tensors, inverse BPZ metrics,
plumbing forms, and the local period map. No torus amplitude from the string
notes, matrix model, or literature is used as an input or fitted target.

The executable audit is
[`audit_genus1_two_point_sewing_normalization.py`](audit_genus1_two_point_sewing_normalization.py).

## 1. Maximal stable graph

For \((g,n)=(1,2)\), a maximal pants decomposition has

\[
V=2g-2+n=2,
\qquad
E=3g-3+n=2,
\qquad
L=E-V+1=1.
\]

Thus the graph consists of two three-punctured spheres connected by two
parallel tubes, with one labelled external puncture on each sphere.

Each normalized sphere tensor carries the residual sphere-topology metric

\[
K_{S^2,\mathrm{res}}^{c=1}
=\sqrt{\alpha'}\,\widetilde K_{S^2}
=\sqrt{\alpha'}\frac{2}{\sqrt{\alpha'}}=2.
\]

Each internal edge contains the inverse of the same BPZ metric. Consequently

\[
\left(K_{S^2,\mathrm{res}}^{c=1}\right)^V
\left(K_{S^2,\mathrm{res}}^{c=1}\right)^{-E}
=2^2 2^{-2}=1.
\]

There is therefore no undetermined torus topology constant. The two sphere
three-point tensors leave the expected coupling power

\[
g_s^V=g_s^2=g_s^{2g-2+n}.
\]

This cancellation uses the *full inverse BPZ metric* on each edge. Replacing
it by the Liouville measure alone would not be a valid sewing computation.

## 2. Local plumbing-to-period map

Use the necklace plumbing coordinates already used by the torus block code:

\[
q_1=e^{iz},
\qquad
q_2=e^{i(2\pi\tau-z)},
\qquad
q_1q_2=e^{2\pi i\tau}.
\]

Then

\[
\frac{dq_1}{q_1}=i\,dz,
\qquad
\frac{dq_2}{q_2}=2\pi i\,d\tau-i\,dz,
\]

and hence

\[
\frac{dq_1}{q_1}\wedge\frac{dq_2}{q_2}
=2\pi\,d\tau\wedge dz.
\]

The antiholomorphic determinant is its conjugate, so the nonchiral Jacobian
is

\[
\left|\det\frac{\partial(\log q_1,\log q_2)}
{\partial(\tau,z)}\right|^2
=(2\pi)^2=4\pi^2.
\]

This \(4\pi^2\) is a local-coordinate result. It is not imported from a
known torus answer.

## 3. Form orientation and \(N_{1,2}\)

The natural product of the two tube forms is edgewise ordered:

\[
\Theta_q=
\left(\frac{dq_1}{q_1}\wedge
      \frac{d\bar q_1}{\bar q_1}\right)
\wedge
\left(\frac{dq_2}{q_2}\wedge
      \frac{d\bar q_2}{\bar q_2}\right).
\]

Moving \(d\bar q_1\) through \(dq_2\) to group holomorphic forms requires one
transposition. Therefore

\[
\Theta_q
=-4\pi^2\,
d\tau\wedge dz\wedge d\bar\tau\wedge d\bar z.
\]

The Polyakov phase is also derived by sewing. Starting with
\(N_{0,3}=1\), the one-particle residue and inverse physical BPZ metric give

\[
N_{0,3}N_{0,3}=-iN_{0,4},
\qquad
N_{0,4}=-iN_{1,2}.
\]

Thus

\[
N_{0,4}=i,
\qquad
\boxed{N_{1,2}=-1}.
\]

The two minus signs cancel:

\[
N_{1,2}\Theta_q
=4\pi^2\,
d\tau\wedge dz\wedge d\bar\tau\wedge d\bar z.
\]

Finally,

\[
d\tau\wedge dz\wedge d\bar\tau\wedge d\bar z
=4\,d^2\tau\,d^2z.
\]

Before the graph quotient, the positive-real geometric coefficient is

\[
4\pi^2\times4=16\pi^2.
\]

## 4. The double-edge quotient

With the two external labels fixed, the two parallel internal edges can still
be exchanged. The maximal graph therefore has an order-two automorphism.
In period coordinates it acts as

\[
z\longmapsto 2\pi\tau-z\equiv -z,
\]

which interchanges \(q_1\) and \(q_2\). Dividing the plumbing chart by this
\(\mathbb Z_2\) gives

\[
\frac{16\pi^2}{2}=8\pi^2.
\]

This is the graph/chart quotient. It is not an extra local state metric and
must be applied exactly once.

## 5. Lorentzian phase

The target-time contraction has two sphere Fourier tensors, two inverse
timelike BPZ metrics, and one independent loop-energy contour. Relative to
the positive-\(i\) tree-level Fourier convention, their phases are

\[
i^V\,i^{-E}\,i^L
=i^2 i^{-2}i^1=i.
\]

The plumbing-form phase has already become \(+1\) in Section 3, so the total
phase is \(+i\). Combining all factors gives

\[
\mathcal A_{1,2}^{\rm ws}
=\underbrace{8\pi^2}_{\text{geometry and quotient}}
\underbrace{i}_{X^0\text{ continuation}}
\underbrace{g_s^2}_{\text{two pants}}
\mathcal I_{1,2}.
\]

## 6. Independent CKV check

The same magnitude follows without the local \(q_i\) determinant. For a
torus with periods \(2\pi\) and \(2\pi\tau\),

\[
\int_{T^2}i\,dz_0\wedge d\bar z_0=8\pi^2\tau_2.
\]

Dividing by the translation CKV volume \(2\tau_2\) gives \(4\pi^2\). The
remaining modulus and relative-insertion forms each satisfy

\[
i\,du\wedge d\bar u=2\,d^2u,
\]

and the same order-two quotient is required. Hence

\[
4\pi^2\times2\times2\times\frac12=8\pi^2.
\]

This agrees with the maximal-plumbing derivation before any comparison with
another formulation.

## 7. Scope and remaining assumptions

The audit fixes the absolute coefficient multiplying the reduced torus CFT
integral in the declared vertex, BPZ, orientation, and necklace-coordinate
conventions. It does **not** evaluate the reduced integral
\(\mathcal I_{1,2}(\omega)\), and it does not assume a relation between \(g_s\)
and a matrix-model parameter.

The Liouville three-point coefficient and its
\(\Upsilon_1/\Gamma_2\) implementation are not redefined here. In particular,
no ordinary Gamma function is substituted for the Barnes double Gamma
function. Those local-CFT conventions remain the independently normalized
inputs stated above.

## 8. After-the-fact comparison in the common \(\mu\) convention

For an unambiguous comparison, define the common ordinary-area density

\[
\rho_{1,2}(\omega;\tau,z)
=
\frac{|\eta(\tau)|^2}{\sqrt{\tau_2}}\,
|\mathcal E(z|\tau)|^{\omega^2}\,
G_L(\omega;z,\tau),
\]

and the full-torus integral

\[
\mathcal I_{1,2}^{\rm full}
=
\int_{\mathcal F}d^2\tau
\int_{T^2(\tau)}d^2z\;\rho_{1,2}.
\]

The packaged determinant factor is

\[
\frac{|\eta|^2}{\sqrt{\tau_2}}
=
\underbrace{|\eta|^4}_{bc}
\underbrace{\frac{1}{\sqrt{\tau_2}|\eta|^2}}
_{X^0\ {\rm oscillators\ and\ loop\ Gaussian}}.
\]

The Liouville factor uses the same delta-normalized primaries, DOZZ
coefficient, and completeness measures \(dP/\pi\) in both calculations.
There is therefore no additional state-metric, leg, or momentum-measure
factor hidden inside \(G_L\).

Equation (4.1) of
[Balthazar--Rodriguez--Yin](https://arxiv.org/pdf/1705.07151#page=16)
gives

\[
\mathcal A_{1,2}^{\rm BRY}
=
i\frac{(2\pi)^2}{2}
(g_s^{\rm BRY})^2 C_{T^2}
\mathcal I_{1,2}^{\rm full}
=
2\pi^2i(g_s^{\rm BRY})^2C_{T^2}
\mathcal I_{1,2}^{\rm full}.
\]

Here \((2\pi)^2\) is the completed Beltrami--ghost coefficient, \(1/2\)
is the \(z\mapsto-z\) quotient, and \(C_{T^2}\) is a separate topology
constant. The paper fixes \(C_{T^2}=1\) by matching its resonance result.
The target-free sewing calculation above derives the same topology value
without using that match.

Our result is

\[
\mathcal A_{1,2}^{\Xi}
=
8\pi^2i(g_s^\Xi)^2
\mathcal I_{1,2}^{\rm full}.
\]

Using

\[
\mu^{-1}
=2\pi g_s^{\rm BRY}
=4\pi g_s^\Xi,
\qquad
g_s^{\rm BRY}=2g_s^\Xi,
\]

both formulas become

\[
\boxed{
\mathcal A_{1,2}(\omega)
=\frac{i}{2\mu^2}
\mathcal I_{1,2}^{\rm full}(\omega)}.
\]

At \(\omega=2i\), where
\(\mathcal I_{1,2}^{\rm full}(2i)=-1/3\), this gives

\[
\boxed{\mathcal A_{1,2}(2i)=-\frac{i}{6\mu^2}}.
\]

If a half-torus domain is used instead, then
\(\mathcal I^{\rm full}=2\mathcal I^{\rm half}\), so the coefficient
must be doubled. The numerical code samples a half-torus but explicitly
doubles the Jacobian and therefore returns the full-torus integral used in
the boxed formulas.
