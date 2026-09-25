#!/usr/bin/env python3
"""Submit the regional-atlas completion once, preserving dispatch receipts."""
import argparse
import json
from pathlib import Path
import subprocess

from so7e8_literature_campaign import atomic_json, read_manifest, _matching_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_manifest("run")
    missing = []
    for index, task in enumerate(manifest["bank_tasks"]):
        bank = Path("run/banks")/f"{index:04d}.json"
        if bank.exists():
            _matching_record(bank, manifest, task)
        else:
            missing.append(index)
    jobs = {}
    receipts = Path("submission")
    for stage in ("bank", "reduce"):
        receipt, intent = receipts/(stage+".json"), receipts/(stage+".intent.json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("submission receipt belongs to another release")
            jobs[stage] = saved["job_id"]
            continue
        if intent.exists():
            raise RuntimeError("uncertain dispatch; inspect Slurm before resubmitting")
        command = ["sbatch", "--parsable", "--job-name=so7e8-atlas-"+stage]
        if stage == "bank":
            if not missing:
                jobs["bank"] = None
                continue
            # Compact consecutive indices; omit already completed valid work.
            ranges, start, last = [], missing[0], missing[0]
            for index in missing[1:]:
                if index != last+1:
                    ranges.append(str(start) if start == last else f"{start}-{last}")
                    start = index
                last = index
            ranges.append(str(start) if start == last else f"{start}-{last}")
            command += ["--array="+",".join(ranges)+f"%{manifest['config']['cluster']['maximum_concurrency']}"]
        else:
            if jobs["bank"]:
                command += ["--dependency=afterok:"+jobs["bank"]]
            command += ["--time=02:00:00"]
        command += ["cluster/submit_so7e8_literature_atlas.slurm", stage]
        if args.dry_run:
            print(json.dumps(command))
            jobs[stage] = stage.upper()+"_JOB_ID"
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
            raise RuntimeError("unrecognized sbatch response; inspect Slurm before retrying")
        atomic_json(receipt, dict(job_id=job, command=command, manifest_sha256=manifest["manifest_sha256"],
                    stdout=result.stdout))
        jobs[stage] = job
        print(stage, job, flush=True)


if __name__ == "__main__":
    main()
