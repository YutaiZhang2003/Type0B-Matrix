"""Independent torus normalization/phase check in a free N=1 theory.

At Q_L=0 (b=i), the generic supermodule is a free boson and one Majorana.
Conserved trinion charges obey p_NS=k-eta*p_R. The exact reference below
uses their torus correlators, not sewn descendants or fitted constants.
It is continued from small positive Q and s, with 0<Q<s<1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mpmath as mp
import sympy as sp

from geometry import marked_coordinates
from mixed_blocks import MixedNSRamondPlumbingBlock


def reference(tau, z, p, k, eta, ns_lift):
    """Free chiral scalar x spin correlator in the marked ground-0 frame.

    D=(1-s) prod_n (1-Q^n s)(1-Q^n/s)/(1-Q^n)^2,
    H_lift=sum_n (-lift)^(n+1) Q^[n(n+1)/2] s^[(n+1)/2].
    H_+ is theta_1 with its leading i phase removed; H_- is theta_2.
    This leading phase is fixed by the plane spin-field OPE.
    """
    Q, s = mp.exp(2j*mp.pi*tau), mp.exp(2j*mp.pi*z)
    product = mp.qp(Q)
    D = (1-s)*mp.qp(Q*s, Q)*mp.qp(Q/s, Q)/product**2
    H = mp.fsum((-ns_lift)**(n+1)*Q**(n*(n+1)//2)
                *mp.exp(mp.pi*1j*z*(n+1)) for n in range(-30, 31))
    return ((2*mp.pi)**(k*k+mp.mpf(1)/8)
        *mp.exp(mp.pi*1j*tau*p*p)
        *mp.exp(2*mp.pi*1j*z*(k*k/2-eta*p*k-mp.mpf(1)/16))
        *D**(-k*k-mp.mpf(1)/8)*product**(-mp.mpf(3)/2)*mp.sqrt(H))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mp.mp.dps = 60
    start = time.monotonic()
    p, k = sp.Rational(2, 5), sp.Rational(3, 10)
    samples = ((.11+1.6j, .08+.7j), (.11+2j, .08+.9j),
               (.11+2.6j, .08+1.2j))
    rows = []
    for eta in (-1, 1):
        block = MixedNSRamondPlumbingBlock(b=sp.I, p_ns=k-eta*p, p_r=p, omega=k)
        for lift in (1, -1):
            for tau, z in samples:
                g = marked_coordinates(tau, z)
                wanted = reference(tau, z, mp.mpf(2)/5, mp.mpf(3)/10, eta, lift)
                errors, ratios = [], []
                for twice_level in (2, 4, 6):
                    actual = block.evaluate_flat_ground(g, twice_level,
                        etas=(eta, eta), ns_lift=lift)
                    ratio = complex(actual/wanted)
                    errors.append(abs(ratio-1))
                    ratios.append([ratio.real, ratio.imag])
                assert errors[2] < errors[1] < errors[0], (eta, lift, tau, errors)
                assert errors[2] < 5e-7, (eta, lift, tau, errors)
                rows.append(dict(eta=eta, ns_lift=lift, tau=[tau.real, tau.imag],
                    z=[z.real, z.imag], total_descendant_levels=[1, 2, 3],
                    complex_ratios=ratios, relative_errors=errors))
    result = dict(schema="type0b-rr-free-field-check-v1", c="3/2", b="i",
        p_r=str(p), external_charge=str(k),
        normalization="Q^(-c/24)*(f_w_prime*f_v_prime)^(-h_ext) times plane plumbing",
        spin_basis="ground-0; ns_lift=+1 gives theta_1, -1 gives theta_2, with OPE-fixed phases",
        scope="chiral ground tensors in one marked patch; not the nonchiral string integrand",
        check_count=len(rows)*3, rows=rows,
        seconds=time.monotonic()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
