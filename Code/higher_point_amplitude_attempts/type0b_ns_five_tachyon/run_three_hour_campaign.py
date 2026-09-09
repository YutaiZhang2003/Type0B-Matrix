"""Run four-worker phases with persistent results inside one Slurm allocation."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest['schema']!='type0b-fivepoint-three-hour-campaign-v1':
        raise ValueError('unexpected campaign schema')
    source=Path.cwd()
    root=args.run_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0')) < manifest['workers']:
        raise ValueError('not enough allocated CPUs')
    os.environ['TYPE0B_5PT_LOCAL_CACHE_ROOT']=tempfile.mkdtemp(
        prefix='type0b-cache-',dir=os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or '/tmp')
    driver=source/'Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_type0b_ns_five_tachyon_cluster.py'
    comparison=source/'Code/higher_point_amplitude_attempts/complex_frequency_comparison/compare.py'
    refinements=comparison.with_name('compare_refinements.py')
    started=time.monotonic()
    deadline=started+manifest['wall_seconds']-manifest['cleanup_seconds']
    status={'schema':manifest['schema'],'job_id':os.environ.get('SLURM_JOB_ID'),
            'state':'running','phases':[],'relative_accuracy_target':.1,'accuracy_target_established':False}
    stop=[]
    def request_stop(signum,_frame):
        stop.append(signum)
    signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    def save():
        status['elapsed_seconds']=time.monotonic()-started
        path=root/'campaign_status.json'
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(path)
    save()
    for phase in manifest['phases']:
        record={'name':phase['name'],'state':'pending'}
        status['phases'].append(record)
        if stop or deadline-time.monotonic() < phase['minimum_remaining_seconds']:
            record['state']='not_started_time_budget'
            save()
            continue
        config=args.manifest.parent/phase['config']
        phase_root=root if phase['name']=='primary' else root/'checks'/phase['name']
        shards=phase_root/'shards'
        shards.mkdir(parents=True,exist_ok=True)
        children=[]
        record.update(state='running',config=str(config),output=str(phase_root))
        save()
        print(json.dumps({'phase':phase['name'],'event':'start'}),flush=True)
        for i in range(manifest['workers']):
            handle=(shards/f'worker_{i}.log').open('w')
            command=[sys.executable,'-B',str(driver),'--config',str(config),'worker',
                     '--output-dir',str(shards),'--task-index',str(i)]
            children.append((subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT),handle))
        while any(p.poll() is None for p,_h in children):
            if stop or time.monotonic()>=deadline or any(p.poll() not in (None,0) for p,_h in children):
                for p,_h in children:
                    if p.poll() is None:p.terminate()
                grace=time.monotonic()+90
                for p,_h in children:
                    try:p.wait(timeout=max(.1,grace-time.monotonic()))
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                break
            time.sleep(1)
        codes=[p.wait() for p,_h in children]
        for _p,h in children:h.close()
        record['worker_exit_codes']=codes
        if any(codes):
            record['state']='interrupted' if stop or time.monotonic()>=deadline else 'failed'
            status['state']=record['state']
            save()
            return 124 if record['state']=='interrupted' else 1
        summary=phase_root/'summary.json'
        try:
            subprocess.run([sys.executable,'-B',str(driver),'--config',str(config),'reduce',
                            '--output-dir',str(shards),'--summary',str(summary)],check=True)
            subprocess.run([sys.executable,'-B',str(comparison),'--config',str(config),'--summary',str(summary),
                            '--output',str(phase_root/'comparison.json')],check=True)
            subprocess.run([sys.executable,'-B',str(refinements),'--run-root',str(root)],check=True)
        except subprocess.CalledProcessError:
            record['state']='postprocessing_failed'
            status['state']='failed'
            save()
            return 1
        record['state']='completed'
        save()
        print(json.dumps({'phase':phase['name'],'event':'completed'}),flush=True)
    status['state']='completed' if all(p['state']=='completed' for p in status['phases']) else 'primary_completed_controls_incomplete'
    save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
