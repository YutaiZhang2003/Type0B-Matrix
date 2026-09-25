"""Preserve raw RC modular ledgers and their exact ordering corrections."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from spin23_genus1_svv_operator_order import canonicalize_record
from spin23_genus1_threepoint_integral import complex_json


def collect(directory):
    root=Path(__file__).resolve().parents[1]
    records=[]
    for path in sorted(directory.glob('corrected_*.json')):
        raw=json.loads(path.read_text())
        canonical=canonicalize_record(raw)
        records.append(dict(file=path.name,raw=raw,
            canonical={key:complex_json(value) for key,value in canonical.items()},
            maximum_component_modular_residual=max(abs(v-1) for v in canonical['individual_ratios']),
            summed_modular_residual=abs(canonical['total_ratio']-1)))
    local=[]
    for path in sorted(directory.glob('ns_self_*.json')):
        local.append(dict(file=path.name,record=json.loads(path.read_text())))
    names=('Codes/spin23_genus1_ns_pairing_audit.py','Codes/spin23_genus1_ns_point_sewing.py',
        'Codes/spin23_genus1_ramond_point_sewing.py','Codes/spin23_ramond_point_precision.py',
        'Codes/spin23_genus1_svv_operator_order.py','Codes/spin23_genus1_corrected_integrand.py',
        'tests/test_spin23_genus1_pairing_correction.py',
        'Codes/collect_spin23_genus1_pairing_correction.py')
    return dict(status='corrected finite-level modular diagnostics; not an integrated string amplitude',
        collected_at=datetime.now(timezone.utc).isoformat(),
        remote_directory='/n/holylabs/yin_lab/Lab/schristian/spin23_pairing_fix_20260909_F8Fu9R',
        rc_array_job=45638178,expected_array_tasks=10,
        rc_validation_job=45638175,
        completed_task_indices=[int(x['file'].removeprefix('corrected_').removesuffix('.json')) for x in records],
        unfinished_task_indices=[i for i in range(10) if not (directory/f'corrected_{i}.json').exists()],
        source_manifest=json.loads((directory/'source_manifest.json').read_text()),
        local_source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        ordering_note='Raw ledgers use the original edge-zero cut and component grouping. Canonical values apply the separately derived exact grouping and cut factors; no numerical integrations are changed.',
        rc_records=records,local_ns_self_checks=local)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('choose a new output artifact')
    with args.output.open('x') as stream:
        json.dump(collect(args.directory),stream,indent=2,allow_nan=False)
