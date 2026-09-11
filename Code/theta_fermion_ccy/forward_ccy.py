"""Forward CCY: accumulate null-shift path weights once for all coefficients.

The residue and the child central charge depend on accumulated internal
weight shifts, not on the final descendant coefficient being requested.
Expand the same CCY DAG from its root once, keep amplitudes labelled by
(weight shifts, last pole), and attach the c-independent global seeds only
afterwards. No new CFT recurrence or PBW ingredient is introduced.
"""

from functools import lru_cache
from itertools import product

from punctured_ccy import PuncturedCCY as BackwardPuncturedCCY
from punctured_ccy import _residue_labels, _vector_id, _vectors, downward_indices


@lru_cache(None)
def seed_splits(level_id):
    levels = _vectors[level_id]
    return tuple((_vector_id(shift), _vector_id(tuple(n-s for n, s in zip(levels, shift))))
                 for shift in product(*(tuple([0]+list(range(2, n+1))) for n in levels)))


class ForwardPuncturedCCY(BackwardPuncturedCCY):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forward_statistics = {}

    def reduced_series(self, cutoff=None, *, indices=None, degree_weights=(1, 1, 1, 1)):
        if indices is None:
            if cutoff is None:
                raise ValueError('supply cutoff or indices')
            indices = downward_indices(cutoff, degree_weights)
        ordered = tuple(sorted(set(tuple(map(int, n)) for n in indices), key=lambda n: (sum(n), n)))
        if not ordered:
            return {}
        allowed = set(ordered)
        for n in ordered:
            if len(n) != 4 or min(n) < 0:
                raise ValueError('four nonnegative descendant levels are required')
            for edge, level in enumerate(n):
                if level:
                    lower = n[:edge]+(level-1,)+n[edge+1:]
                    if lower not in allowed:
                        raise ValueError('forward CCY requires a downward-closed index set')
        maximum = tuple(max(n[edge] for n in ordered) for edge in range(4))
        amplitudes = {0: {0: self.mp.mpf(1)}}
        totals = {}
        transitions = contributions = pole_states = 0
        # Each target shift is larger than its parent. Summation by total
        # shift is therefore topological, irrespective of the plumbing set.
        for shift in ordered:
            shift_id = _vector_id(shift)
            incoming = amplitudes.get(shift_id)
            if incoming is None:
                continue
            pole_states += len(incoming)
            totals[shift_id] = sum(incoming.values(), self.mp.mpf(0))
            for edge in range(4):
                for r, s, degree in _residue_labels(maximum[edge]-shift[edge]):
                    changed = shift[:edge]+(shift[edge]+degree,)+shift[edge+1:]
                    if changed not in allowed:
                        continue
                    template = self._template(shift_id, edge, r, s)
                    if template is None:
                        continue
                    pole_id, _, changed_id = self._templates[template]
                    value = self.mp.mpf(0)
                    for central_id, amplitude in incoming.items():
                        # Each (parent shift, parent pole, transition) is
                        # visited once. A factor-result cache is unnecessary.
                        value += amplitude*self._factor_uncached(central_id, template)
                        contributions += 1
                    row = amplitudes.setdefault(changed_id, {})
                    row[pole_id] = row.get(pole_id, self.mp.mpf(0))+value
                    transitions += 1
            del amplitudes[shift_id]
        result = {}
        global_terms = 0
        for n in ordered:
            value = self.mp.mpf(0)
            for shift_id, remainder_id in seed_splits(_vector_id(n)):
                weight = totals.get(shift_id)
                if weight is None:
                    continue
                # A vanishing summed seed weight does not justify dropping
                # its outgoing transitions; those were already propagated.
                value += weight*self._global_indexed(shift_id, remainder_id)
                global_terms += 1
            result[n] = value
        self._forward_statistics = dict(
            reachable_shift_vectors=len(totals), accumulated_pole_states=pole_states,
            shift_transitions=transitions, residue_weight_multiply_adds=contributions,
            global_seed_multiply_adds=global_terms,
            total_multiply_adds=contributions+global_terms,
            shared_seed_splits=seed_splits.cache_info()._asdict())
        return result

    def cache_info(self):
        result = super().cache_info()
        if self._forward_statistics:
            result['recursion_algorithm'] = 'forward null-shift weights, then global-seed sums, Python'
        result['forward'] = self._forward_statistics.copy()
        return result

    def clear_caches(self):
        super().clear_caches()
        self._forward_statistics.clear()
