#!/usr/bin/env python3
"""Raw-plane BPZ tensor-transport audit; not a complete Human Note dictionary."""
import argparse
import hashlib
import json
from pathlib import Path

from literature_component_blocks import (
    ComponentBlocks, PaperConstants, ThreePoint, _ground, _ns,
    sewn_integrand as literature_integrand,
)
from literature_human_bpz import HumanBPZBlocks, bpz_three_point, sewn_integrand


def run(note_path):
    constants = PaperConstants(32)
    p = (.21, .39, .31, .43)
    cases = [
        ('rrrr', (1, 0, 1, 0)), ('rrrr', (0, 1, 0, 1)),
        ('mixed_ns', (0, 1, (1, 0), (0, 0))),
        ('mixed_ns', (1, 1, (1, 1), (1, 1))),
        ('mixed_r', ((1, 0), 1, 0, (0, 0))),
        ('mixed_r', ((1, 1), 1, 1, (1, 1))),
    ]
    encode = lambda x: [float(x.real), float(x.imag)]
    rows = []
    for family, external in cases:
        for P in (.71, -.71):
            old = ComponentBlocks(family, p, P, 6)
            new = HumanBPZBlocks(family, p, P, 6)
            for z in (.5+.1j, .5-.1j):
                lhs = literature_integrand(old, constants, P, z, external)
                rhs = sewn_integrand(new, constants, P, z, external)
                rows.append(dict(family=family, external=external, P=P, z=encode(z),
                                 literature=encode(lhs), transported=encode(rhs),
                                 relative_error=float(abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-300))))
    nr = ThreePoint('NS', 'R', 'R', .31, .37, .43)
    actual = bpz_three_point(nr.nr(_ns(1), 0, _ground(1)), 'NR', 0, 1)
    printed = (1+1j)/2**.5*(nr.m1.beta-nr.m2.beta)
    repo = Path(__file__).resolve().parents[1]
    baseline_path = repo/'data_exports/so7e8_literature_native_20260920/generic_external_parity.json'
    baseline = json.loads(baseline_path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {name: sha(repo/'Codes'/name) for name in (
        'literature_component_blocks.py', 'literature_human_bpz.py', 'audit_literature_human_bpz.py')}
    if hashes['literature_component_blocks.py'] != baseline['code_sha256']:
        raise ValueError('native source no longer matches the recorded crossing baseline')
    return dict(
        stage='raw-plane linear BPZ transport; complete Type0B fixed-parity dictionary not certified',
        human_note=dict(path=str(note_path.resolve()), sha256=sha(note_path)),
        definition='same-beta linear BPZ, G_n^T=G_-n, B(w+,w+)=1, B(w-,w-)=i; mirror antiholomorphic frame',
        branch='beta=-i*p/sqrt(2); P_H=i*p; same principal slit coordinate lift as stage 1',
        integration='none; pointwise basis and coefficient-tensor transport',
        maximum_twice_level=6, momenta=p, code_sha256=hashes,
        max_transport_error=max(row['relative_error'] for row in rows), rows=rows,
        raw_plane_vs_fixed_parity=dict(
            location='SCblock.tex lines 2032-2045, r=s=1, (chi,w+,w-), f=0, eta=+1',
            ward_value=encode(actual), printed_value=encode(printed),
            ratio=encode(actual/printed),
            type0b_fixed_parity_value=encode(-actual),
            type0b_fixed_parity_residual=encode(-actual-printed),
            fixed_identity_derivation_verified=False,
            explanation='A separate Type0B check assumes a Ward identity whose value is minus the raw plane value for this ground component and agrees with Appendix A. That agreement does not derive the rephasing from the Human Note. The previous correction claim is withdrawn.'),
        previous_crossing=dict(
            path=str(baseline_path), sha256=sha(baseline_path), rerun=False,
            max_error={family: max(row['relative_error'] for row in baseline['rows']
                       if row['family'] == family and not row['vanishes_by_parity'])
                       for family in ('mixed', 'rrrr')},
            interpretation='Retained by pointwise tensor transport; not an independent validation of Appendix A.'),
        production_heterotic_assembly_certified=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human-note', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.human_note)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Maximum pointwise transport error:', result['max_transport_error'])
    print('Raw-plane versus fixed-parity example:', result['raw_plane_vs_fixed_parity'])
