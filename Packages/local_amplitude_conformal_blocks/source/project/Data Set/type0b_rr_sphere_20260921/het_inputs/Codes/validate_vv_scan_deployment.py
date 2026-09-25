"""Exercise auxiliary-bank preparation in the deployed RC environment."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from spin23_genus1_vv_energy_scan import load_run
from spin23_genus1_vv_matching import load_cusp_collision
from spin23_genus1_vv_ope import OPEBank
from spin23_genus1_vv_tail_bank import TailBank
from spin23_type0b_reference import reference_root


def main():
    started = time.perf_counter()
    manifest = load_run(Path('run'))
    reference = reference_root()
    results = []
    for kappa in (0.20, 0.40):
        folder = Path('deployment_check') / f'kappa_{kappa:.2f}'
        folder.mkdir(parents=True, exist_ok=True)
        for kind in ('ope', 'tail', 'cusp'):
            path = folder / f'{kind}.npz'
            subprocess.run([
                sys.executable, '-u', 'Codes/prepare_spin23_genus1_vv_completion.py',
                kind, '--omega', str(1j*kappa), '--order', '2', '--loop-order', '2',
                '--workers', '1', '--output', str(path),
            ], check=True)
            if kind == 'ope':
                bank = OPEBank.load(path)
                value = bank.disc(1.2j, .08, cutoff=4)
            elif kind == 'tail':
                bank = TailBank.load(path)
                value = bank.stripes(.1+.2j)
            else:
                bank = load_cusp_collision(path, [0., kappa])
                value = bank.disc_tail(3., .08)
            if not np.isfinite(value).all():
                raise ArithmeticError(f'nonfinite {kind} evaluation at {kappa}')
            results.append(dict(kappa=kappa, kind=kind, nodes=len(bank.momenta)))
    load_run(Path('run'))
    output = dict(passed=True, run_id=manifest['run_id'], reference=str(reference),
                  results=results, seconds=time.perf_counter()-started)
    Path('deployment_check/result.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
