#!/usr/bin/env python3
"""Submit the prepared block bank and crossing gate, once, from its release root."""
import argparse
import json
from pathlib import Path
import subprocess

from so7e8_literature_campaign import atomic_json, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_manifest("run")
    cluster = manifest["config"]["cluster"]
    receipts = Path("submission")
    jobs = {}
    for stage in ("bank", "crossing"):
        receipt, intent = receipts/(stage+".json"), receipts/(stage+".intent.json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("submission belongs to another manifest")
            jobs[stage] = saved["job_id"]
            continue
        if intent.exists():
            raise RuntimeError(f"uncertain {stage} dispatch; inspect squeue/sacct before retrying")
        command = ["sbatch", "--parsable", "--job-name=so7e8-literature-"+stage]
        if stage == "bank":
            command += [f"--array=0-{len(manifest['bank_tasks'])-1}%{cluster['maximum_concurrency']}"]
        else:
            command += ["--dependency=afterok:"+jobs["bank"], "--time=02:00:00"]
        command += ["cluster/submit_so7e8_literature_stage.slurm", stage]
        if args.dry_run:
            print(json.dumps(command))
            jobs[stage] = "BANK_JOB_ID" if stage == "bank" else "CROSSING_JOB_ID"
            continue
        receipts.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        atomic_json(intent, dict(command=command, manifest_sha256=manifest["manifest_sha256"]))
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            atomic_json(receipts/(stage+".failure.json"), dict(returncode=result.returncode,
                        stdout=result.stdout, stderr=result.stderr))
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("unrecognized sbatch response; inspect submission before retrying")
        atomic_json(receipt, dict(job_id=job, command=command,
                    manifest_sha256=manifest["manifest_sha256"], stdout=result.stdout))
        jobs[stage] = job
        print(stage, job, flush=True)


if __name__ == "__main__":
    main()
