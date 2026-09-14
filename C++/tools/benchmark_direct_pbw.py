"""Fresh C++ PBW/DV per-edge benchmark; Python only launches and compares."""
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12'
OUT = ROOT / 'C++/results/direct_pbw_cpp_2026-09-12'
spec = importlib.util.spec_from_file_location('reference_measure', OLD / 'measure.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'running', 'dps': 40, 'precision_bits': 136,
        'q_level_cutoffs': [5, 5, 5], 'environment': platform.platform(),
        'protocol': 'sequential fresh C++ processes; no numerical caches loaded; '
                    'compilation excluded; Python only launches and compares',
        'source_sha256': {}, 'runs': [], 'comparisons': [],
    }
    for path in [ROOT / 'C++/include/ramond/direct_pbw.hpp',
                 ROOT / 'C++/drivers/pbw_main.cpp', ROOT / 'C++/bin/pbw',
                 OLD / 'ramond', OLD / 'include/ramond/ccy.hpp']:
        report['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['compiler'] = subprocess.check_output(['clang++', '--version'], text=True).strip()
    def save():
        (OUT / 'timings.json').write_text(json.dumps(report, indent=2) + '\n')
    for algorithm in ('pbw_cpp', 'double_virasoro_cpp'):
        for mode in ('ordinary', 'inserted'):
            stem = algorithm + '_' + mode
            command = ([str(ROOT / 'C++/bin/pbw')] if algorithm == 'pbw_cpp' else
                       [str(OLD / 'ramond'), '--truncation', 'per-edge', '--sector-policy', 'record'])
            command += ['--mode', mode, '--level', '5', '--dps', '40', '--json', str(OUT / (stem + '.json'))]
            row = {'algorithm': algorithm, 'mode': mode, 'command': command, 'status': 'running'}
            report['runs'].append(row)
            save()
            print('Starting', stem, flush=True)
            start = time.perf_counter()
            with (OUT / (stem + '.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                try:
                    status = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            row.update(wall_seconds=time.perf_counter()-start, exit_code=status,
                       status='completed' if status == 0 else 'failed')
            if status:
                report['status'] = 'failed'; save()
                raise RuntimeError((OUT / (stem + '.log')).read_text()[-4000:])
            data = json.loads((OUT / (stem + '.json')).read_text())
            row.update(timing_seconds=data['timing_seconds'], counts=data['counts'])
            if 'memory' in data:
                row['memory'] = data['memory']
            print(stem, row['wall_seconds'], 's wall;', row['timing_seconds'], flush=True)
            save()
    for mode in ('ordinary', 'inserted'):
        pbw = json.loads((OUT / ('pbw_cpp_' + mode + '.json')).read_text())
        for label, other, criterion in [
            ('saved_python_pbw', OLD / 'results' / ('pbw_optimized_' + mode + '.json'), '1e-28'),
            ('fresh_cpp_double_virasoro', OUT / ('double_virasoro_cpp_' + mode + '.json'), '2e-20'),
        ]:
            result = {'mode': mode, 'reference': label, **reference.compare(pbw, json.loads(other.read_text()))}
            result['criterion'] = criterion
            result['passed'] = Decimal(result['maximum_scaled_difference']) < Decimal(criterion)
            report['comparisons'].append(result)
            print(json.dumps(result), flush=True)
    report['status'] = 'completed' if all(r['passed'] for r in report['comparisons']) else 'comparison_failed'
    save()
    if report['status'] != 'completed':
        raise SystemExit('Directed comparison failed; inspect retained results.')


if __name__ == '__main__':
    main()
