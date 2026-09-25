#!/usr/bin/env python3
"""Generate the figures used in the VVVV fit report."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parents[1]
DATA_DIR = WORKSPACE_DIR / "Data Set" / CODE_DIR.name
out = WORKSPACE_DIR / "Machine Note" / CODE_DIR.name

eq = json.loads((DATA_DIR / 'heterotic_equal_scan_segmented_complete.json').read_text())['rows']
a = np.array([r['a'] for r in eq])
v = np.array([complex(*r['value']) for r in eq])
aa = np.linspace(a.min(), a.max(), 500)
ww = 1 / 3 + 1j * aa
pp = -math.pi * ww**4 / (27 * (1 + 2j * ww / 3))

fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(aa, pp.real, label=r'conjecture $-\pi\omega^4/[27(1+2i\omega/3)]$')
ax.scatter(a, v.real, label='worldsheet block integral')
ax.set_xlabel(r'$a$ in $\omega_0=1/3+ia$')
ax.set_ylabel(r'$\mathrm{Re}\,A_{12|03}$')
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(out / 'heterotic_equal_fit_real.png', dpi=220)
fig.savefig(out / 'heterotic_equal_fit_real.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(aa, pp.imag, label=r'conjecture $-\pi\omega^4/[27(1+2i\omega/3)]$')
ax.scatter(a, v.imag, label='worldsheet block integral')
ax.set_xlabel(r'$a$ in $\omega_0=1/3+ia$')
ax.set_ylabel(r'$\mathrm{Im}\,A_{12|03}$')
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(out / 'heterotic_equal_fit_imag.png', dpi=220)
fig.savefig(out / 'heterotic_equal_fit_imag.pdf')
plt.close(fig)

rel = np.array([r['relative_error'] for r in eq])
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.scatter(a, rel)
ax.set_yscale('log')
ax.set_xlabel(r'$a$ in $\omega_0=1/3+ia$')
ax.set_ylabel('relative discrepancy')
ax.grid(True)
fig.tight_layout()
fig.savefig(out / 'heterotic_equal_fit_residual.png', dpi=220)
fig.savefig(out / 'heterotic_equal_fit_residual.pdf')
plt.close(fig)

D = json.loads((DATA_DIR / 'heterotic_vvvv_fit_data.json').read_text())
ys = []
pred = []
for row in D['rows']:
    w1, w2, w3, w0 = [complex(*z) for z in row['energies']]
    pair = [w1 + w2, w1 + w3, w2 + w3]
    common = -math.pi * w0 * w1 * w2 * w3
    for x, z in zip(pair, row['value']):
        ys.append(common / complex(*z))
        pred.append(1 + 1j * x)
ys = np.array(ys)
pred = np.array(pred)

# Parity plot avoids hiding nearly coincident points in the complex plane.
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.scatter(pred.real, ys.real, label='real parts')
ax.scatter(pred.imag, ys.imag, label='imaginary parts')
lo = min(pred.real.min(), pred.imag.min(), ys.real.min(), ys.imag.min())
hi = max(pred.real.max(), pred.imag.max(), ys.real.max(), ys.imag.max())
ax.plot([lo, hi], [lo, hi], label='identity line')
ax.set_xlabel(r'conjectured component of $1+i(\omega_i+\omega_j)$')
ax.set_ylabel('numerical normalized inverse')
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(out / 'heterotic_inverse_affine_fit.png', dpi=220)
fig.savefig(out / 'heterotic_inverse_affine_fit.pdf')
plt.close(fig)

err = np.abs((ys - pred) / pred)
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.scatter(np.arange(1, len(err) + 1), err)
ax.set_yscale('log')
ax.set_xlabel('complex coefficient in validation set')
ax.set_ylabel('relative discrepancy')
ax.grid(True)
fig.tight_layout()
fig.savefig(out / 'heterotic_asymmetric_fit_residual.png', dpi=220)
fig.savefig(out / 'heterotic_asymmetric_fit_residual.pdf')
plt.close(fig)

print('plots written')
