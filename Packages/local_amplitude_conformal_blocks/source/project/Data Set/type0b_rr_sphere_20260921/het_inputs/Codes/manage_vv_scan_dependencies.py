"""Overlap VV auxiliary preparation and integrate each completed energy.

Scheduling only: the frozen mathematical implementation is left intact.
Each auxiliary/finish pair occupies the slot released by a distinct
preparation worker, so the complete pipeline remains within 200 cores.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def producers(energy_index, *, energies=30, nodes=272, workers=200):
    if not 0 <= energy_index < energies:
        raise ValueError('invalid energy index')
    return sorted({(node*energies+energy_index) % workers for node in range(nodes)})


def dependencies(energy_index, preparation_job, auxiliary_job, completed=()):
    tasks = [f'{preparation_job}_{i}' for i in producers(energy_index)]
    remaining = [task for task in tasks+[f'{auxiliary_job}_{energy_index}']
                 if task not in completed]
    return 'afterok:'+':'.join(remaining) if remaining else ''


def successful_tasks(job_ids):
    """Completed tasks may be gone from Slurm's live dependency lookup."""
    rows = call(['sacct', '-X', '-n', '-P', '-j', ','.join(job_ids),
                 '--format=JobID%40,State,ExitCode']).splitlines()
    result = set()
    for row in rows:
        task, state, exit_code = row.split('|')[:3]
        if state == 'COMPLETED' and exit_code == '0:0':
            result.add(task.strip())
    return result


def call(args):
    return subprocess.check_output(args, text=True).strip()


def write_json(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare_auxiliary(root, energy_index):
    from spin23_genus1_vv_energy_scan import load_run
    root = Path(root).resolve()
    manifest = load_run(root/'run')
    kappa = manifest['config']['kappas'][energy_index]
    directory = root/'run'/f'e{energy_index:02d}'
    for kind in ('ope', 'tail', 'cusp'):
        subprocess.run([sys.executable, '-u', str(root/'Codes/prepare_spin23_genus1_vv_completion.py'),
                        kind, '--omega', str(1j*kappa), '--workers', '1',
                        '--output', str(directory/f'{kind}.npz')], check=True)
    load_run(root/'run')
    print('Auxiliary banks complete for energy', energy_index, kappa, flush=True)


def apply(root):
    from spin23_genus1_vv_energy_scan import load_run
    root = Path(root).resolve(); os.chdir(root)
    manifest = load_run(root/'run'); config = manifest['config']
    if (len(config['kappas']), config['spectral_order'], config['workers']) != (30, 16, 200):
        raise ValueError('this dependency map is for the existing 30-energy, 200-worker scan')
    jobs = json.loads((root/'submitted_jobs.json').read_text())
    preparation = jobs['prepare']; integration = jobs['integrate']
    path = root/'energy_dependencies.json'
    if path.exists():
        ledger = json.loads(path.read_text())
        if ledger['run_id'] != manifest['run_id']:
            raise ValueError('dependency ledger belongs to another run')
        if ledger.get('status') == 'applied':
            print(json.dumps(ledger)); return
    else:
        states = call(['squeue', '-h', '-r', '-j', integration, '-o', '%T']).splitlines()
        if len(states) != 30 or set(states) != {'PENDING'}:
            raise RuntimeError('initial conversion requires all 30 integration tasks still pending')
        ledger = dict(run_id=manifest['run_id'], preparation_job=preparation,
            integration_job=integration, reduction_job=jobs['reduce'],
            started_at_utc=datetime.now(timezone.utc).isoformat(), status='planned',
            source_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                           for name in ('Codes/manage_vv_scan_dependencies.py',
                                        'cluster/submit_spin23_genus1_vv_auxiliary.slurm')},
            integration_dependencies={})
        write_json(path, ledger)
    # Prevent an integration from starting while its new auxiliary worker
    # writes the same banks. Holding affects only the still-pending array.
    if ledger.get('status') != 'integration_held':
        call(['scontrol', 'hold', integration])
    ledger['status'] = 'integration_held'; write_json(path, ledger)
    completed = successful_tasks([preparation])
    all_prepared = {f'{preparation}_{i}' for i in range(200)} <= completed
    auxiliary_active = (call(['squeue', '-h', '-r', '-j', ledger['auxiliary_job'], '-o', '%i'])
                        if 'auxiliary_job' in ledger else '')
    if all_prepared and not auxiliary_active:
        # The accounting database outlives live Slurm task records. Do not
        # attach new dependencies to already successful, possibly expired IDs.
        call(['scontrol', 'update', 'JobId='+integration, 'Dependency='])
        call(['scontrol', 'release', integration])
        ledger.update(status='applied', mode='all_preparation_completed_before_release',
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            confirmed_successful_preparation_workers=200, actual_integration_core_cap=30,
            integration_dependencies={str(i): '' for i in range(30)})
        write_json(path, ledger)
        print('All coefficient producers succeeded; released 30 integrations.', flush=True)
        return
    if 'auxiliary_job' not in ledger:
        auxiliary = call(['sbatch', '--parsable', '--array=0-29%30', '--hold',
                          '--kill-on-invalid-dep=yes',
                          'cluster/submit_spin23_genus1_vv_auxiliary.slurm']).split(';')[0]
        ledger['auxiliary_job'] = auxiliary; write_json(path, ledger)
        print('Submitted auxiliary array', auxiliary, flush=True)
    auxiliary = ledger['auxiliary_job']
    for ei in range(30):
        owner = f'{preparation}_{ei}'
        dependency = '' if owner in completed else 'afterok:'+owner
        call(['scontrol', 'update', f'JobId={auxiliary}_{ei}', 'Dependency='+dependency])
    call(['scontrol', 'release', auxiliary])
    completed = successful_tasks([preparation, auxiliary])
    ledger['already_successful_tasks'] = sorted(completed)
    ledger['applied_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for ei in range(30):
        dependency = dependencies(ei, preparation, auxiliary, completed)
        call(['scontrol', 'update', 'JobId='+integration+'_'+str(ei),
              'Dependency='+dependency])
        ledger['integration_dependencies'][str(ei)] = dependency
        write_json(path, ledger)
    load_run(root/'run')
    call(['scontrol', 'release', integration])
    ledger.update(status='applied', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        core_cap=200,
        concurrency_argument='Auxiliary energy i waits for unique prep worker i; '
        'integration i waits for its auxiliary job and all its coefficient producers. '
        'Thus each energy uses at most the single slot released by worker i.')
    write_json(path, ledger)
    print(json.dumps(dict(auxiliary_job=auxiliary, integration_job=integration,
                          updated_energy_dependencies=30, core_cap=200)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('apply', 'auxiliary'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--energy-index', type=int)
    args = parser.parse_args()
    if args.stage == 'apply': apply(args.root)
    else:
        if args.energy_index is None: parser.error('auxiliary requires --energy-index')
        prepare_auxiliary(args.root, args.energy_index)


if __name__ == '__main__': main()
