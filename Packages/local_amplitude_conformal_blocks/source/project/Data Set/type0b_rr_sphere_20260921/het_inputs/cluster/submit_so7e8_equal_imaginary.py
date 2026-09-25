#!/usr/bin/env python3
"""Submit the equal-imaginary bank, selection, and amplitude stages once."""
import argparse
import json
from pathlib import Path
import subprocess

from so7e8_literature_campaign import read_manifest,atomic_json,_matching_record


def ranges(indices):
    out=[];first=last=indices[0]
    for i in indices[1:]:
        if i!=last+1:
            out.append(str(first) if first==last else f'{first}-{last}');first=i
        last=i
    out.append(str(first) if first==last else f'{first}-{last}')
    return ','.join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    m=read_manifest('native');missing=[]
    for i,task in enumerate(m['bank_tasks']):
        path=Path('native/banks')/f'{i:04d}.json'
        if path.exists():_matching_record(path,m,task)
        else:missing.append(i)
    jobs={};previous=None
    for stage in ('bank','expand','integrate','finish'):
        receipt=Path('submission')/(stage+'.json');intent=receipt.with_suffix('.intent.json')
        if receipt.exists():
            row=json.loads(receipt.read_text());assert row['manifest_sha256']==m['manifest_sha256']
            jobs[stage]=previous=row['job_id'];continue
        if intent.exists():raise RuntimeError('uncertain dispatch; inspect scheduler before retrying '+stage)
        if stage=='bank' and not missing:jobs[stage]=None;continue
        cmd=['sbatch','--parsable','--job-name=so7eq-'+stage]
        if previous:cmd+=['--dependency=afterok:'+previous]
        if stage=='bank':cmd+=['--array='+ranges(missing)+'%128']
        elif stage=='integrate':cmd+=['--array=0-8%9','--time=02:00:00']
        else:cmd+=['--time=01:00:00']
        cmd+=['cluster/submit_so7e8_equal_imaginary.slurm',stage]
        if a.dry_run:
            print(json.dumps(cmd));jobs[stage]=previous=stage.upper()+'_JOB';continue
        Path('logs').mkdir(exist_ok=True);atomic_json(intent,dict(command=cmd,manifest_sha256=m['manifest_sha256']))
        result=subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:
            atomic_json(receipt.with_suffix('.failure.json'),dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            raise RuntimeError(result.stderr)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('ambiguous submission; inspect scheduler')
        atomic_json(receipt,dict(job_id=job,command=cmd,stdout=result.stdout,manifest_sha256=m['manifest_sha256']))
        jobs[stage]=previous=job;print(stage,job,flush=True)
    if not a.dry_run:atomic_json('submission/jobs.json',jobs)


if __name__=='__main__':main()
