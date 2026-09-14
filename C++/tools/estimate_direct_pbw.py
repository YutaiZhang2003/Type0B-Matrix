"""Count C++ PBW work and scale fresh C++ stages; no block evaluations."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'C++/results/direct_pbw_cpp_2026-09-12'


def character(cutoff, fermions, bosons):
    values = [1] + [0] * cutoff
    for k in bosons:
        for j in range(k, cutoff + 1):
            values[j] += values[j-k]
    for k in fermions:
        for j in range(cutoff, k-1, -1):
            values[j] += values[j-k]
    return values


def counts(level):
    ns = character(2*level, range(1, 2*level+1, 2), range(2, 2*level+1, 2))
    rr = character(level, range(1, level+1), range(1, level+1))
    volume = sum(ns)*sum(rr)**2
    return {
        'ns_dimensions': ns, 'ramond_dimensions_per_parity': rr,
        'monomials': len(ns)*len(rr)**2,
        'vertex_entries_positive': 2*volume, 'vertex_entries_negative': 4*volume,
        'gram_triangle_entries': sum(d*(d+1)//2 for d in ns)+4*sum(d*(d+1)//2 for d in rr),
        'gram_cube_proxy': sum(d**3 for d in ns)+4*sum(d**3 for d in rr),
        'contraction_products': sum(2*a*b*c*(a+b+c+1) for a in ns for b in rr for c in rr),
        'basis_metadata_entries': sum(ns)+4*sum(rr),
        'mean_total_level_per_vertex_entry': float(sum(Fraction(i+2*j+2*k, 2)*a*b*c
            for i, a in enumerate(ns) for j, b in enumerate(rr) for k, c in enumerate(rr))/volume),
    }


def main():
    measured = json.loads((OUT/'timings.json').read_text())
    assert measured['status'] == 'completed' and measured['precision_bits'] == 136
    assert measured['q_level_cutoffs'] == [5]*3
    for relative, digest in measured['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    work = {n: counts(n) for n in (5, 10)}
    features = ('monomials', 'vertex_entries_positive', 'vertex_entries_negative',
                'gram_triangle_entries', 'gram_cube_proxy', 'contraction_products',
                'basis_metadata_entries', 'mean_total_level_per_vertex_entry')
    ratio = {k: work[10][k]/work[5][k] for k in features}
    forecasts = []
    for row in measured['runs']:
        if row['algorithm'] != 'pbw_cpp':
            continue
        mode = row['mode']
        key = 'vertex_entries_' + ('positive' if mode == 'ordinary' else 'negative')
        for actual, counted in [('vertex_entries', key), ('gram_entries', 'gram_triangle_entries'),
                                ('contraction_products', 'contraction_products')]:
            assert row['counts'][actual] == work[5][counted]
        t = row['timing_seconds']
        phases = {stage: t[stage]*ratio[feature] for stage, feature in (
            ('vertices', key), ('gram_entries', 'gram_triangle_entries'),
            ('gram_inverse', 'gram_cube_proxy'), ('contractions', 'contraction_products'),
            ('metadata', 'basis_metadata_entries'))}
        phases['setup'] = t['setup']
        # Unassigned time includes tensor/cache destruction, so use entries rather
        # than the much smaller monomial ratio. This is a proxy, not an exact count.
        phases['other_internal'] = max(0, t['total']-sum(t[k] for k in phases))*ratio[key]
        phases['process_overhead'] = max(0, row['wall_seconds']-t['total'])*ratio[key]
        steady = sum(phases.values())
        extra = phases['vertices']*(ratio['mean_total_level_per_vertex_entry']-1)
        memory = row['memory']
        count = work[10][key]
        # This platform uses 64-bit GMP limbs; 136-bit MPFR mantissas need at
        # least three limbs each, excluding allocation headers and rounding slack.
        mantissa_payload = 2*((136+63)//64)*8
        forecasts.append({
            'mode': mode, 'measured_level5_wall_seconds': row['wall_seconds'],
            'constant_unit_cost_stage_seconds': phases,
            'constant_unit_cost_seconds': steady,
            'level_weighted_ward_sensitivity_seconds': steady+extra,
            'ward_mean_level_cost_multiplier': ratio['mean_total_level_per_vertex_entry'],
            'top_level_ward_entries_at_least': count,
            'memory_lower_bound': {
                'top_level_key_bytes': count*memory['ward_key_bytes'],
                'top_level_key_value_object_bytes': count*memory['ward_entry_pair_bytes'],
                'top_level_mantissa_payload_bytes': count*mantissa_payload,
                'top_level_pair_and_mantissa_bytes': count*(memory['ward_entry_pair_bytes']+mantissa_payload),
                'per_entry_pair_bytes': memory['ward_entry_pair_bytes'],
                'per_entry_mantissa_payload_bytes': mantissa_payload,
                'tensor_permutation_cache_bytes': 0,
            },
        })
    report = {
        'precision_bits': 136, 'dps': 40, 'q_level_cutoffs': [10]*3,
        'implementation': 'C++17/MPC direct physical SCA PBW',
        'new_physical_gram_ward_block_evaluations': 0,
        'counts': work, 'growth_ratios': ratio, 'forecasts': forecasts,
        'assumptions': [
            'Baseline keeps measured seconds per requested vertex entry and contraction product fixed.',
            'Gram inversions scale by sum(d^3), a work proxy, not an exact operation count.',
            'The longer-Ward scenario multiplies only vertex work by the entry-weighted mean total-level ratio. No level-10 internal Ward-operation count was computed.',
            'Unassigned internal time and process startup/serialization/cache-destruction overhead scale by the top-level vertex-entry ratio as a proxy.',
            'Both scenarios assume sufficient RAM without paging, unchanged persistent caches, and unchanged 136-bit arithmetic. They are not runtime bounds or confidence intervals.',
            'Memory counts cover requested top-level Ward entries only: measured 96-byte key/value objects plus at least 48 bytes of complex mantissa payload at 136 bits with 64-bit GMP limbs.',
            'Hash-node links, buckets, additional internal Ward entries, allocator overhead, words, action caches, and contraction tensors require further memory. No level-10 peak RSS was measured.',
            'The C++ implementation uses integer keys and on-the-fly tensor indexing; it stores no tensor-permutation tables.',
            'No conclusion about achieved level-10 accuracy follows from equal working precision.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT/'C++/include/ramond/direct_pbw.hpp', OUT/'timings.json')},
    }
    (OUT/'level10_estimate.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'growth_ratios': ratio, 'forecasts': forecasts}, indent=2))


if __name__ == '__main__':
    main()
