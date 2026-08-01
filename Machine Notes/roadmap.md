# Benchmark Roadmap

All stages use the BRY conventions of arXiv:2201.05621:
\(\alpha'=2\), \(h=(p^2-x^2)/2\),
\(\omega_{\rm MM}=2\omega\), \(g_s=4/(\pi\mu_{\rm F})\), and
\(\mathcal L=T-A,\ \mathcal R=T+A\). Any source written in another
normalization must be translated before its formulas enter code.

The notation \(\alpha'=2\) means
\(\ell_{\rm B}=\sqrt{\alpha'/2}=1\) in BRY units. Radius formulas use the
dimensionless number
\(\rho=R_{\rm phys}/\ell_{\rm B}\), never a dimensionful radius divided by
the pure number two.

The Liouville interaction \(\mu_{\rm L}\) and Fermi depth \(\mu_{\rm F}\)
remain distinct until comparison. The default comparison ansatz is
\(\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}\); neither \(p=1\) nor a bare
Liouville-volume formula is used as an input to the worldsheet derivation.

The project should climb this ladder in order.  Each stage fixes conventions
needed by the next one; later data must not be used to fit an earlier stage.

## A. Analytic genus-one anchor

1. Follow Douglas et al. at the worldsheet level: cancel the matter,
   super-Liouville, and combined ghost determinants in each even structure.
2. Evaluate the remaining circle-lattice integral by modular-orbit unfolding,
   using only the convergent \(\sum k^{-2}\) sum.
3. Integrate the odd supertorus modulus directly, including the supercurrent
   insertion and its contact term, and reproduce
   \[
   \widehat{\mathcal F}_{1,\rm odd}^{0B}
   =-\frac \rho{48}+\frac1{24\rho}.
   \]
4. Derive the Liouville wall displacement from the BRY action and compute
   \(\partial_{\log\mu_{\rm L}}\mathcal F_1\).
5. Use the BRST spectrum multiplicities and the formal zeta sum only as
   subsequent checks, then recover

   \[
   C_1(\rho)=-\frac1{12}\left(\frac \rho2+\frac2\rho\right).
   \]

Exit criterion: the modular path integral and free-fermion derivation agree;
the large-radius normalization input for the odd sector is explicit; and both
the \(R\) and \(1/R\) terms are separately accounted for.

## B. One-particle matrix-model data

1. Implement even and odd scattering states of the inverted oscillator.
2. Compute their reflection phases and regulated densities of states.
3. Verify independence of the cutoff-dependent constant in physical
   derivatives.
4. Recover the genus-one coefficient by a finite-temperature fermion
   calculation.

Exit criterion: increasing precision and moving the eigenvalue cutoff leave
the universal logarithmic coefficient stable within a declared tolerance.

## C. Collective fields and the NS-NS/R-R split

1. Construct left and right time-of-flight collective fields.
2. Form the even NS-NS and odd R-R combinations.
3. Check the odd-R-R selection rule at zero flux.
4. Reproduce the BRY-normalized \(1\to2\) and \(1\to3\) sphere amplitudes
   directly in the \(T,A,\mathcal L,\mathcal R\) basis.

Exit criterion: the energy delta function is translated with
\(\omega_{\rm MM}=2\omega\), the BRY coefficients are reproduced without
adding a separate leg-pole factor, and mixed-side connected amplitudes vanish
perturbatively.

## D. Flux and 0A/0B T-duality

1. Introduce left/right Fermi-level asymmetry with explicitly named flux
   variables.
2. Separate Lorentzian continuous-flux conventions from Euclidean quantized
   flux conventions.
3. Reproduce the finite-radius partition function with flux.
4. Verify the 0A/0B T-duality map.

Exit criterion: the flux-free result is recovered continuously and all radius,
(alpha'), and flux factors are fixed by the primary formulas rather than by
numerical fitting.

## E. Higher genus and relation to StringMC

1. Reproduce the genus-one even-spin determinant cancellation and the
   odd-spin direct-supermoduli calculation, including contact terms.
2. Implement the four independent families of super-Liouville pants data:
   (C), (C-tilde), (C-even), and (C-odd), in a fixed (b=1) normalization.
3. Build NS and Ramond plumbing-fixture sewing blocks and their degeneration
   checks before a full moduli integral.
4. Adapt the D'Hoker--Phong even-spin measure to time times super-Liouville;
   derive the six odd-spin contributions as a separate work package.
5. If local PCO sections are used, implement both codimension-one and
   codimension-two vertical-integration corrections.
6. Reuse generic sampling and error-estimation infrastructure from `plumbing/`
   only where the bosonic measure is not being imported implicitly.
7. Compare genus-expansion coefficients extracted from the matrix-model grand
   potential, using \(\mu_{\rm F}^{-2}=(\pi^2/16)g_s^2\) at genus two while keeping
   the absolute worldsheet vacuum normalization as a factorization check.

Exit criterion: boundary factorization, GSO phases, superghost normalization,
and the matrix-model coefficient are all independently audited.

## Immediate next calculation

Finish stage A at the measure level: reproduce the appendix-A odd-modulus
normalization, including the supercurrent contact term, and translate it to a
local-PCO description as a check that no extra vertical correction is needed
for the unpunctured torus. Then use the even/odd inverted-oscillator phase
shifts to test the comparison
map \(\mu_{\rm L}=\kappa\mu_{\rm F}^{\,p}\).
