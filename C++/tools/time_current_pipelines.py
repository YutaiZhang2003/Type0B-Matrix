"""Fresh sequential production timings. Does not run PBW or load numerical caches."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

CPP = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dps', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = CPP / 'bin/ramond'
    source_files = sorted([*CPP.glob('include/ramond/*.hpp'), *CPP.glob('src/*.cpp')])
    report = {'status': 'running', 'dps': args.dps, 'environment': platform.platform(),
              'binary_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(CPP)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_files},
              'protocol': 'separate sequential processes; empty numerical caches; no PBW', 'runs': []}
    summary = directory / 'timings.json'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    for level in [10, 15]:
        for mode in ['ordinary', 'inserted']:
            stem = f'{mode}_level{level}_{args.dps}dps'
            output = directory / f'{stem}.json'
            command = [str(executable), '--mode', mode, '--level', str(level), '--dps',
                       str(args.dps), '--sector-policy', 'record', '--json', str(output)]
            print(f'Starting {stem}', flush=True)
            start = time.perf_counter()
            with (directory / f'{stem}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                record = {'level': level, 'mode': mode, 'command': command, 'pid': process.pid,
                          'status': 'running', 'result': str(output), 'started_unix': time.time()}
                report['runs'].append(record)
                summary.write_text(json.dumps(report, indent=2) + '\n')
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    record['status'] = 'interrupted'
                    summary.write_text(json.dumps(report, indent=2) + '\n')
                    raise
            record['wall_seconds'] = time.perf_counter() - start
            record['exit_code'] = code
            record['status'] = 'completed' if code == 0 else 'failed'
            if code == 0:
                data = json.loads(output.read_text())
                record['timing_seconds'] = data['timing_seconds']
                record['diagnostics'] = data['diagnostics']
                record['counts'] = data['counts']
                record['coefficient_vectors'] = len(data['coefficients'])
            summary.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{stem}: {record["status"]}, {record["wall_seconds"]:.3f} s wall', flush=True)
            if code:
                print((directory / f'{stem}.log').read_text()[-3000:], flush=True)
    report['status'] = 'completed' if all(r['exit_code'] == 0 for r in report['runs']) else 'failures'
    summary.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
