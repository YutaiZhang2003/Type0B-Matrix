#!/usr/bin/env python3
"""Audit completed quadrature grids, frozen inputs, and cutoff controls."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/genus_2'))
import numpy as np
import refine_nsrr_bilinear_quadrature as run


def main():
    out = run.OUT
    config = run.read(out / 'config.json')
    summary = run.read(out / 'summary.json')
    digest = run.object_digest(config)
    for path, expected in {**config['implementation_sha256'], **config['baseline_sha256']}.items():
        assert run.sha(ROOT / path) == expected, path
    baseline = run.read(run.BASE / 'source.json')
    results = []; input_hashes = {}
    for row in summary['results']:
        n = row['N']; maximum_ward = 0.; maximum_occupation = 0
        node_counts = {}
        for channel in ('source', 'target'):
            manifest = run.read(out / f'provenance-N{n}.json')
            count = 0
            for path, expected in manifest.items():
                if f'/{channel}/' not in path:
                    continue
                assert run.sha(ROOT / path) == expected, path
                node = run.read(ROOT / path)
                assert node['config_digest'] == digest and node['N'] == n
                count += 1; input_hashes[path] = expected
                if channel == 'source':
                    maximum_ward = max(maximum_ward, node['native_checks']['branching_ward_residual'])
                    Q = config['b'] + 1 / config['b']
                    weights = [Q * Q / 8 + p * p / 2 + (1 / 16 if i < 2 else 0)
                               for i, p in enumerate(node['momenta'])]
                else:
                    assert node['global_nonconverged_calls'] == 0
                    maximum_occupation = max(maximum_occupation, node['global_max_occupation_used'])
                    weights = node['weights']
                primary = np.exp(sum(h * run.decode(z) for h, z in zip(weights, node['log_q'])))
                assert abs(primary / run.decode(node['primary']) - 1) < 1e-12
            node_counts[channel] = count
            reused = channel == 'source' and n == 3 or channel == 'target' and n in (3, 4)
            assert count == (0 if reused else n ** 3)
        comparisons = []
        for old in baseline:
            if old['t'] == config['point']['t'] and old['N'] == n and old['L'] == 3:
                errors = [abs(v['source_L3_Z'] / run.scalar(run.matrix(old), run.VECTORS[label]) - 1)
                          for label, v in row['values'].items()]
                assert max(errors) < 1e-9
                comparisons.append(dict(dataset=old['dataset'], maximum_L3_integral_relative_error=max(errors)))
        results.append(dict(N=n, fresh_node_counts=node_counts,
            maximum_native_ward_residual=maximum_ward if node_counts['source'] else None,
            maximum_target_global_occupation=maximum_occupation if node_counts['target'] else None,
            saved_L3_checks=comparisons,
            maximum_L3_to_L5_change=max(abs(v['source_L3_to_L5_relative_change']) for v in row['values'].values())))
    last = summary['results'][-1]
    previous = summary['results'][-2]
    changes = {}
    for key in ('source_Z', 'target_Z', 'ratio'):
        changes[key] = max(abs(v[key] / previous['values'][label][key] - 1) for label, v in last['values'].items())
    final = dict(status=summary['status'], completed_orders=summary['completed_orders'],
        checks=results, last_step_maximum_relative_changes=changes,
        stable_steps=summary['steps'][-2:],
        baseline_target_reuse_check=run.read(out / 'target_reuse_check.json'),
        extra_target_reuse_checks=run.read(out / 'target_extra_reuse_checks.json'),
        interpretation='Empirical quadrature convergence at fixed block cutoffs; no rigorous remainder bound.',
        audit_script_sha256=run.sha(Path(__file__)))
    run.save(out / 'audit.json', final)
    for p in (out / 'config.json', out / 'summary.json', out / 'target_reuse_check.json',
              out / 'target_extra_reuse_checks.json'):
        input_hashes[str(p.relative_to(ROOT))] = run.sha(p)
    run.save(out / 'audit_provenance.json', input_hashes)
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
