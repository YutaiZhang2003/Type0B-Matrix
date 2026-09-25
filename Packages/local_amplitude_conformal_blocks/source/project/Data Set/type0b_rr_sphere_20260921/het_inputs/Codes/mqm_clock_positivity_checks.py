"""Independent finite-oscillator check of the supplied squeezed-clock test.

The product state gives a variational expectation, not the full clock's
lowest eigenvalue. No quantum logarithm or continuum MQM is asserted.
"""
import json
from pathlib import Path
import numpy as np
import sympy as s
from scipy.linalg import eigvalsh

checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

# Direct oscillator matrices, independent of the squeezed-state derivation.
# A single frequency contributes c*(2n + alpha*b^2 + alpha^* b^dagger^2).
spectra = []
for m in (1, 2, 5, 12):
    analytic = .5*m*(2*m/np.sqrt(1+4*m*m)-1)
    vals = []
    for dim in (24, 48):
        annih = np.diag(np.sqrt(np.arange(1, dim)), 1)
        number = np.diag(np.arange(dim))
        alpha = 1/(1+2j*m)
        h = .5*m*(2*number+alpha*(annih@annih)
                    +alpha.conjugate()*(annih.T@annih.T))
        value = float(eigvalsh(h, subset_by_index=[0, 0])[0])
        vals.append(value)
        check(f"one-mode oscillator m{m} dim{dim}",
              abs(value-analytic) < 2e-12)
    spectra.append({"m":m, "exact":analytic, "truncations":vals})

def expectation(N, g):
    modes = np.arange(1, N+1, dtype=float)
    return 1+23*g*g*np.sum(.5*modes*(2*modes/np.sqrt(1+4*modes*modes)-1))

check("negative product expectation N1 g1", expectation(1, 1) < -.214)
check("negative product expectation N100 g1/2", expectation(100, .5) < -.795)
m = s.symbols("m", positive=True)
term = m*(2*m/s.sqrt(1+4*m*m)-1)/2
check("negative logarithmic coefficient",
      s.limit(m*term, m, s.oo) == -s.Rational(1,16))

# Product squeezed states have zero one-point functions, so cross-mode
# terms vanish in expectation; 23 independent transverse copies add.
result = {"passed":True, "checks":len(checks), "named_checks":checks,
          "oscillator_spectra":spectra,
          "N1_g1":expectation(1,1), "N100_g_half":expectation(100,.5),
          "scope":"Variational positivity counterexample in the supplied fixed-frequency oscillator regulator."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_positivity_results.json').write_text(
    json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
