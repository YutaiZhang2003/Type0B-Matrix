"""Small action comparisons for indexed storage, exact phases, and fallback.

These exercise NS/R generators and complex momenta; no block is computed.
The scalar complex implementation supplies the reference action coefficients.
"""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mpmath as mp
from action_optimization import CachedActionModule, solve_ns_l1, solve_ramond_lminus
import compute_target as br


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    results = []
    cases = [('NS', F(3, 2), 0, False, 0),
             ('R', F(3, 4), 1, False, 0),
             ('R', F(1, 4), 0, False, 0),
             ('R', F(3, 4), 0, True, 0),
             ('R', F(3, 4), 0, True, 40)]
    for sector, label, parity, complex_momentum, dps in cases:
        br.set_multiprecision(dps)
        momentum = br.complex_number(F(13, 29))
        if complex_momentum:
            momentum += br.complex_number(1j)*br.real_number(F(1, 7))
        outputs, fits = [], []
        for optimized in (False, True):
            module = CachedActionModule(sector, F(7, 5), momentum,
                indexed_descendants=optimized, real_descendants=optimized,
                sparse_native=optimized)
            terms, fit = (solve_ns_l1(module, label) if sector == 'NS'
                          else solve_ramond_lminus(module, label, parity))
            assert module._descendant_cache is None and module._generator_matrices is None
            if complex_momentum:
                assert not module.real_descendants
                assert fit['descendant_cache']['real_descendant_primaries'] == 0
            if dps:
                assert not module.sparse_native
            outputs.append({(t.label, t.first, t.second): t.coefficient for t in terms})
            fits.append(fit)
        old, new = outputs
        assert old.keys() == new.keys()
        error = max(abs(new[k]-v)/max(1, abs(v)) for k, v in old.items())
        assert error < (mp.mpf('1e-20') if dps else 1e-10), str(error)
        results.append(dict(sector=sector, label=str(label), parity=parity, dps=dps,
            complex_momentum=complex_momentum, coefficients=len(old),
            maximum_scaled_difference=str(error),
            real_primaries=fits[1]['descendant_cache']['real_descendant_primaries'],
            relative_residuals=[fit['relative_residual'] for fit in fits]))
    report = dict(status='passed', scope='five small action solves, scalar versus optimized',
                  cases=results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
