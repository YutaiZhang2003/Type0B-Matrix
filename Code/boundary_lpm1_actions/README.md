# Boundary actions of the physical Virasoro modes

The calculation uses the conventions of `SCblock.tex`.  The NS vectors are
the normalized vectors \(v_n\) used in the main notes.  The Ramond vectors
are the chi-string vectors \(v_n^\alpha\), with \(\alpha=0,1\).

All boundary states are annihilated by the positive mode:

\[
L_1v_0=L_1v_{\pm1/2}=L_1v_{\pm1/4}^\alpha=0.
\]

For the NS vacuum,

\[
L_{-1}v_0=L_{-1}^{(1)}v_0+L_{-1}^{(2)}v_0.
\]

The two NS states at \(n=\pm1/2\) mix under \(L_{-1}\):

\[
\begin{aligned}
L_{-1}v_{1/2}={}&
-\frac{(2P+b)(-2Pb+b^2+1)}{2P(2Pb+b^2-1)}L_{-1}^{(1)}v_{1/2}
+\frac{(2Pb+1)(-2Pb+b^2+1)}{2Pb(-2Pb+b^2-1)}L_{-1}^{(2)}v_{1/2}\\
&+\frac{b(2Pb+b^2+1)}{2P(-2Pb+b^2-1)}L_{-1}^{(1)}v_{-1/2}
-\frac{2Pb+b^2+1}{2Pb(2Pb+b^2-1)}L_{-1}^{(2)}v_{-1/2},\\
L_{-1}v_{-1/2}={}&
-\frac{b(-2Pb+b^2+1)}{2P(2Pb+b^2-1)}L_{-1}^{(1)}v_{1/2}
+\frac{-2Pb+b^2+1}{2Pb(-2Pb+b^2-1)}L_{-1}^{(2)}v_{1/2}\\
&+\frac{(-2P+b)(2Pb+b^2+1)}{2P(-2Pb+b^2-1)}L_{-1}^{(1)}v_{-1/2}
+\frac{(2Pb-1)(2Pb+b^2+1)}{2Pb(2Pb+b^2-1)}L_{-1}^{(2)}v_{-1/2}.
\end{aligned}
\]

At the Ramond boundary, \(L_{-1}\) also reaches the new primaries at level
one:

\[
\begin{aligned}
L_{-1}v_{1/4}^\alpha={}&
-\frac{-4Pb+b^2+2}{2(2Pb-1)}L_{-1}^{(1)}v_{1/4}^\alpha
+\frac{-4Pb+2b^2+1}{2b(-2P+b)}L_{-1}^{(2)}v_{1/4}^\alpha\\
&-\frac{(-2Pb+b^2+2)(-2Pb+2b^2+1)}
{2^{\alpha+2}b(-2P+b)(2Pb-1)}v_{-3/4}^\alpha,\\
L_{-1}v_{-1/4}^\alpha={}&
\frac{4Pb+b^2+2}{2(2Pb+1)}L_{-1}^{(1)}v_{-1/4}^\alpha
+\frac{4Pb+2b^2+1}{2b(2P+b)}L_{-1}^{(2)}v_{-1/4}^\alpha\\
&+\frac{(2Pb+b^2+2)(2Pb+2b^2+1)}
{2^{\alpha+2}b(2P+b)(2Pb+1)}v_{3/4}^\alpha.
\end{aligned}
\]

The formulas are identities of rational functions at generic \((b,P)\).
At a zero of a displayed denominator they are understood by changing basis
or by analytic continuation when the limit exists.

## Verification and runtimes

`derive_symbolic.py` constructs the oscillator states, transports both
Ramond free-field realizations to one common abstract SCA module, solves an
independent square subsystem, and verifies every remaining oscillator
component symbolically.  Its final run took `4.82 s`.

`compute_boundary_actions.py` performs an independent double-precision
projection at \(b=3/2\), \(P=2/5\).  Every system had full column rank, all
\(L_1\) actions vanished exactly, and the largest relative residual was
\(6.242\times10^{-14}\).  Its final run took `0.53 s`.
