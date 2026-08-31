import json
from pathlib import Path
import tempfile
import unittest

from compare_refinements import compare_refinements


def encoded(z):
    return {'real':z.real,'imag':z.imag}


class RefinementTests(unittest.TestCase):
    def test_common_corner_and_replicate_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for control in (False,True):
                folder=root/'checks'/'block_depth' if control else root
                (folder/'shards').mkdir(parents=True)
                (folder/'summary.json').write_text('{}')
                for shard in range(2):
                    values=[complex(10*shard+rep) for rep in range(2)]
                    row={'collar_radius':.01,'replicates':2,'corner_contribution_computed':shard==0,
                         'corner_contribution':encoded(complex(5 if control else 1)),
                         'bulk_samples_per_replicate':32,'face_samples_per_replicate':32,
                         'bulk_estimates':[encoded(z+(3+4j if control else 0)) for z in values],
                         'face_estimates':[encoded(0j)]*2,
                         'sampling_prefix_estimates':[{'replicate':rep,'bulk_samples':32,'face_samples':32,
                                                      'bulk_estimate':encoded(z),'face_estimate':encoded(0j)}
                                                     for rep,z in enumerate(values)]}
                    data={'cluster_task':{'shard_index':shard,'seed':100+10*shard},'results':[row]}
                    (folder/'shards'/f'task_{shard:05d}.json').write_text(json.dumps(data))
            result=compare_refinements(root)
            row=result['comparisons'][0]
            self.assertEqual(row['raw_paired_shift'],encoded(7+4j))
            self.assertEqual(row['raw_standard_error_real'],0)
            self.assertEqual(row['raw_standard_error_imag'],0)
            self.assertEqual(result['pending_controls'],['projection_radius','momentum_order','momentum_tail'])
            self.assertFalse(result['accuracy_target_established'])


if __name__=='__main__':
    unittest.main()
