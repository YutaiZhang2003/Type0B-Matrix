"""Directed comparison of both production executables at machine precision."""
import json
from pathlib import Path
import subprocess
import time

here = Path(__file__).resolve().parent
bin_dir = here.parents[1] / 'bin'
out_dir = here / 'cli_results'
out_dir.mkdir(exist_ok=True)
start = time.perf_counter()
results = []
for mode in ('ordinary', 'inserted'):
    for f in (0, 1):
        for eta in (-1, 1):
            data = []
            for exe in ('ramond', 'pbw'):
                name = f'{exe}_{mode}_f{f}_eta{eta}'
                output = out_dir / (name + '.json')
                args = [str(bin_dir / exe), '--mode', mode, '--level', '1',
                        '--dps', '0', '--p', '0', '--f', str(f), '--eta', str(eta),
                        '--json', str(output)]
                if exe == 'ramond':
                    args += ['--truncation', 'per-edge']
                with (out_dir / (name + '.log')).open('w') as log:
                    subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
                rows = json.loads(output.read_text())['coefficients']
                data.append({tuple(r['exponents']): r['values'] for r in rows})
            assert data[0].keys() == data[1].keys()
            maximum = 0.0
            count = 0
            for key, row in data[0].items():
                for x, y in zip(row, data[1][key]):
                    a = complex(float(x['real']), float(x['imag']))
                    b = complex(float(y['real']), float(y['imag']))
                    maximum = max(maximum, abs(a-b) / max(1.0, abs(a), abs(b)))
                    count += 1
            results.append(dict(mode=mode, f=f, eta=eta, components=count,
                                maximum_scaled_error=maximum, passed=maximum < 1e-8))
summary = dict(conventions='product_bpz_residue_2026-09-22', dps=0,
               per_edge_level=1, cases=results,
               seconds=time.perf_counter()-start,
               passed=all(r['passed'] for r in results))
(here / 'cli_results.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary['passed'] else 1)
