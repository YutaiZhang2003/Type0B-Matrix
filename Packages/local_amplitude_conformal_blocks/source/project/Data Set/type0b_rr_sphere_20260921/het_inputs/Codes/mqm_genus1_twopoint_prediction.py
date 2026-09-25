"""Frozen one-loop elastic prediction of the existing Wick coupling flow.

The linear reflection phase is divided out. Unit-energy-delta states and
gamma=g/sqrt(2*pi) are used. No worldsheet amplitude enters this prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp


def prediction(energy, *, dps=60):
    with mp.workdps(dps):
        e = mp.mpf(energy)
        if not mp.isfinite(e) or e <= 0:
            raise ValueError('positive real energy required')
        vector_norm = e**2*mp.log1p(e**2)-2*e**2+2*e*mp.atan(e)
        vector_integral = 2*e*mp.quad(lambda x: x*(e-x)/(1+x*x), [0, e/2, e])
        ss = e/(1+e*e)*mp.quad(lambda x: x*(e-x)*
            (1+x*x+(e-x)**2+x*(e-x))**2/((1+x*x)*(1+(e-x)**2)), [0, e/2, e])
        vv = 23*e**4/(6*(1+e*e))
        return dict(energy=e, vector_splitting_norm=vector_norm,
            vector_quadrature_residual=abs(vector_integral-vector_norm),
            singlet_to_ss_norm=ss, singlet_to_vv_norm=vv,
            singlet_splitting_norm=ss+vv,
            vector_relative_g2=-vector_norm/(4*mp.pi),
            singlet_relative_g2=-(ss+vv)/(4*mp.pi),
            vector_relative_phase_g2=mp.mpf(0), singlet_relative_phase_g2=mp.mpf(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energies', default='0.25,0.5,0.75,1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = [('Codes/' + Path(__file__).name), 'docs/mqm/mqm_genus1_threepoint_prediction.md',
             'docs/mqm/mqm_clock_coupling_isotopy.md', 'docs/mqm/mqm_clock_continuum_cubic_audit.md']
    report = dict(schema='mqm-genus1-twopoint-prediction-v1',
        convention='R/R_tree=1+g^2*relative_g2+O(g^4); gamma=g/sqrt(2*pi)',
        prescription='Existing Wick-ordered coupling flow, fixed linear leg phases removed; no additional quadratic quantum counterterm',
        status='Candidate prediction, not a worldsheet result',
        source_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        results=[{k: mp.nstr(v, 45) for k, v in prediction(e).items()}
                 for e in args.energies.split(',')])
    rendered = json.dumps(report, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream: stream.write(rendered)
    else: print(rendered, end='')


if __name__ == '__main__': main()
