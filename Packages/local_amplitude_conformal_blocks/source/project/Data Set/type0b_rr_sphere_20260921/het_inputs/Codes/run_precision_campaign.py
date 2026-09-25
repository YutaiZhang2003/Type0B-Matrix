#!/usr/bin/env python3
"""Run the precision repair with unchanged physical settings and frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from so7e8_literature_campaign import ROOT, read_manifest, atomic_json, _save_record


def bank(index):
    manifest = read_manifest(ROOT / 'native')
    # Some archived helpers set mpmath's global default at import time.
    # Load those before installing and explicitly scoping production precision.
    from so7e8_literature_atlas_campaign import build_atlas_node
    import literature_precision_blocks as precision
    digits = 60
    with precision.mp.workdps(digits):
        result = build_atlas_node(ROOT / 'native', index, manifest=manifest)
    result['arithmetic'] = dict(decimal_digits=digits,
        method='high precision finite branching, vertices, CCY products and assembled branch sum',
        isolation_tolerance=1e-8, high_precision_residual_limit=1e-40,
        source_sha256=hashlib.sha256(Path(precision.__file__).read_bytes()).hexdigest())
    _save_record(ROOT / 'native/banks' / f'{index:04d}.json', manifest,
                 manifest['bank_tasks'][index], result)
    atomic_json(ROOT / 'native/branch_checks' / f'{index:04d}.json', dict(
        decimal_digits=digits, branches=precision.DIAGNOSTICS,
        status='checked', manifest_sha256=manifest['manifest_sha256']))
    return {k: result[k] for k in ('order', 'status', 'seconds', 'arithmetic')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('bank', 'expand', 'integrate', 'finish'))
    p.add_argument('--task', type=int, default=0)
    a = p.parse_args()
    if a.stage == 'bank':
        result = bank(a.task)
    else:
        from so7e8_equal_imaginary_campaign import expand, integrate, finish
        result = integrate(ROOT, a.task) if a.stage == 'integrate' else globals()[a.stage](ROOT)
    print(json.dumps(result, indent=2), flush=True)
