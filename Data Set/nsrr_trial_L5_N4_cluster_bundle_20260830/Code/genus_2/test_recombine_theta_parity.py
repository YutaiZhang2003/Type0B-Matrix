from __future__ import annotations

import io
import json
import math
import tarfile
import tempfile
import unittest
from pathlib import Path

from genus_2.recombine_theta_parity import recombine_archive


class ThetaArchiveRecombinationTests(unittest.TestCase):
    @staticmethod
    def _add_json(
        archive: tarfile.TarFile, name: str, value: object
    ) -> None:
        payload = json.dumps(value).encode("utf-8")
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    def test_even_minus_odd_recombination(self) -> None:
        schema = "source-v1"
        digest = "config-digest"
        fingerprint = "implementation-fingerprint"
        radius = 0.035
        summary = {
            "schema": schema,
            "config_digest": digest,
            "implementation_fingerprint": fingerprint,
            "task_count": 2,
            "rows": [
                {
                    "point_id": "test-point",
                    "channel": "theta",
                    "recursion_order": 24,
                    "quadrature_order": 2,
                    "node_count": 2,
                    "finite_part_radius": radius,
                    "z_liouville": 10.0,
                    "z_free_superfield": 2.0,
                    "q_l": 10.0 / 2.0**9,
                }
            ],
        }

        def shard(index: int, even: float, odd: float) -> dict[str, object]:
            return {
                "schema": schema,
                "task_index": index,
                "point_id": "test-point",
                "channel": "theta",
                "recursion_order": 24,
                "quadrature_order": 2,
                "radius_results": [
                    {
                        "finite_part_radius": radius,
                        "contribution": even + odd,
                        "sectors": [
                            {"sector": 0, "contribution": even},
                            {"sector": 1, "contribution": odd},
                        ],
                    }
                ],
                "config_digest": digest,
                "implementation_fingerprint": fingerprint,
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "shards.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                self._add_json(archive, "run/summary.json", summary)
                # Deliberately reverse physical member order.
                self._add_json(
                    archive, "run/shards/task-000001.json", shard(1, 4.0, 3.0)
                )
                self._add_json(
                    archive, "run/shards/task-000000.json", shard(0, 2.0, 1.0)
                )

            result = recombine_archive(archive_path)

        row = result["corrected_rows"][0]
        self.assertEqual(row["z_even"], 6.0)
        self.assertEqual(row["z_odd_unsigned"], 4.0)
        self.assertEqual(row["z_liouville_corrected"], 2.0)
        self.assertTrue(math.isclose(row["q_l_corrected"], 2.0 / 2.0**9))
        self.assertEqual(result["source"]["theta_shard_count"], 2)


if __name__ == "__main__":
    unittest.main()
