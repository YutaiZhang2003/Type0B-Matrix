"""Verify every expected coefficient validation result before summarizing it."""
import argparse
import json
import math
from pathlib import Path


def collect(directory, points, manifest):
    results = []
    source = {name: manifest['files'][name] for name in
              ('Codes/audit_spin23_ramond_continuation.py', 'Codes/spin23_genus1_branch_recursion.py')}
    words = [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    expected_files = {f'node_{i}.json' for i in range(len(points))}
    present_files = {p.name for p in directory.glob('node_*.json')}
    if present_files != expected_files:
        raise ValueError(f'incomplete or unexpected result files: {present_files ^ expected_files}')
    jobs = set()
    for index, point in enumerate(points):
        report = json.loads((directory/f'node_{index}.json').read_text())
        config = report['configuration']
        if report['source_sha256'] != source:
            raise ValueError(f'source mismatch at node {index}')
        if config['index'] != index or config['cutoff'] != 8 or not config['confirm'] or config['contours'] or config['ggg_only']:
            raise ValueError(f'incomplete validation configuration at node {index}')
        if len(report['results']) != 1:
            raise ValueError(f'wrong result count at node {index}')
        row = report['results'][0]
        if row['file'] != point['file'] or row['energy'] != point['record']['energy'] or row['momentum'] != point['record']['momentum']:
            raise ValueError(f'point mismatch at node {index}')
        if [w['word'] for w in row['words']] != words:
            raise ValueError(f'incomplete vertex words at node {index}')
        for word in row['words']:
            if len(word['tests']) != 2:
                raise ValueError(f'missing resolution confirmation at node {index}')
            for test in word['tests']:
                if not test['passed']:
                    raise ValueError(f'rejected coefficient comparison at node {index}')
                diag = test['diagnostics']
                if (diag['radius'], diag['check_radius']) != (.6, .7):
                    raise ValueError(f'wrong contours at node {index}')
                errors = [test['maximum_scaled_error'], diag['scaled_finite_part_error']]
                if 'angular_refinement_error' in test:
                    errors.append(test['angular_refinement_error'])
                if any(not math.isfinite(e) or e > 2e-7 for e in errors):
                    raise ValueError(f'accuracy failure at node {index}')
            if word['tests'][1]['diagnostics']['samples'] <= word['tests'][0]['diagnostics']['samples']:
                raise ValueError(f'confirmation did not increase resolution at node {index}')
            if 'angular_refinement_error' not in word['tests'][1]:
                raise ValueError(f'missing comparison between resolutions at node {index}')
        jobs.add(report['job_id'])
        results.append(dict(row, execution_job_id=report['job_id']))
    tests = [test for row in results for word in row['words'] for test in word['tests']]
    return dict(scope='level-four coefficient validation only; no amplitude integration',
        all_passed=True, original_failed_nodes=sum(p['file'].endswith('.json') for p in points),
        control_nodes=sum(not p['file'].endswith('.json') for p in points),
        vertex_words_per_node=4, form_sign_channels_per_word=32,
        coefficients_per_channel=35, batch_comparisons=len(tests),
        scalar_coefficient_comparisons=len(tests)*32*35,
        maximum_scaled_direct_error=max(t['maximum_scaled_error'] for t in tests),
        maximum_absolute_direct_error=max(t['maximum_absolute_error'] for t in tests),
        maximum_scaled_radius_error=max(t['diagnostics']['scaled_finite_part_error'] for t in tests),
        maximum_angular_refinement_error=max(t.get('angular_refinement_error', 0.) for t in tests),
        maximum_gram_condition=max(t['maximum_gram_condition'] for t in tests),
        angular_sample_counts=sorted({t['diagnostics']['samples'] for t in tests}),
        validation_job_ids=sorted(job for job in jobs if job is not None),
        locally_validated_nodes=[row['file'] for row in results if row['execution_job_id'] is None],
        validation_manifest=manifest, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.directory, json.loads(args.points.read_text()), json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('results', 'validation_manifest')}, indent=2))
