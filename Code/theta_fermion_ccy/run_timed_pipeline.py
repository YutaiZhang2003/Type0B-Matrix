"""Run the production pipeline and record its complete child-process wall time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing-json',type=Path,required=True)
    args, pipeline_args = parser.parse_known_args()
    command = [sys.executable,str(Path(__file__).with_name('pipeline.py')),*pipeline_args]
    started = time.perf_counter()
    result = subprocess.run(command,check=False)
    elapsed = time.perf_counter()-started
    report = {'status':'completed' if result.returncode == 0 else 'failed',
              'returncode':result.returncode,'command':command,
              'process_wall_seconds':elapsed,
              'scope':'From child-process launch through exit, including imports and all result writes'}
    args.timing_json.parent.mkdir(parents=True,exist_ok=True)
    args.timing_json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
