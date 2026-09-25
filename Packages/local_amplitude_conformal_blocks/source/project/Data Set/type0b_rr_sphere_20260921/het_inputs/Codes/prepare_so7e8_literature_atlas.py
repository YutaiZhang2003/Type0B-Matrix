#!/usr/bin/env python3
"""Freeze a child release for only the missing/unresolved regional banks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from so7e8_literature_atlas import atlas_point_sets
from so7e8_literature_campaign import ROOT, atomic_json, digest, pairs, read_manifest


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(parent_run, audit_run, output):
    parent_run, audit_run, output = (Path(p).resolve() for p in (parent_run, audit_run, output))
    if output.exists():
        raise FileExistsError("choose a new immutable output directory")
    parent = read_manifest(parent_run, check_sources=False)
    audit_bytes = (audit_run/"atlas_audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    if audit["parent_manifest_sha256"] != parent["manifest_sha256"]:
        raise ValueError("atlas audit belongs to a different parent")
    for name, expected in audit["source_sha256"].items():
        if sha((ROOT/name).read_bytes()) != expected:
            raise ValueError("atlas audit source changed: "+name)
    # Other tasks can edit the live parent files after this audit. We use
    # their verified archive bytes below, so bind the audit to those bytes.
    for name, expected in audit["verified_runtime_parent_source_sha256"].items():
        if parent["source_sha256"].get(name) != expected:
            raise ValueError("atlas audit did not use the frozen parent dependency: "+name)
    source_files = {}
    # Reuse the exact frozen parent sources, not unrelated live-workspace edits.
    with tarfile.open(parent_run/"sources.tar.gz", "r:gz") as archive:
        for name, expected in parent["source_sha256"].items():
            data = archive.extractfile(name).read()
            if sha(data) != expected:
                raise ValueError("parent source archive integrity failure: "+name)
            source_files[name] = data
    new_names = ["Codes/"+n for n in (
        "so7e8_literature_amplitude.py", "so7e8_literature_atlas.py",
        "audit_so7e8_literature_atlas.py", "prepare_so7e8_literature_atlas.py",
        "so7e8_literature_atlas_campaign.py", "resolve_so7e8_literature_atlas.py")]
    new_names += ["cluster/submit_so7e8_literature_atlas.py", "cluster/submit_so7e8_literature_atlas.slurm",
                  "tests/test_so7e8_literature_atlas.py"]
    source_files.update({name: (ROOT/name).read_bytes() for name in new_names})
    points, geometry = atlas_point_sets(parent["config"]["moduli"]["settings"])
    serialized_points = {name: pairs(p) for name, p in points.items()}
    if {name: digest(p) for name, p in serialized_points.items()} != audit["point_set_sha256"]:
        raise ValueError("moduli points changed since the atlas audit")
    points_bytes = (json.dumps(serialized_points, separators=(",", ":"))+"\n").encode()
    data_files = {"run/point_sets.json": points_bytes, "run/parent_atlas_audit.json": audit_bytes,
                  "run/parent/manifest.json": (parent_run/"manifest.json").read_bytes()}
    parent_bank_hashes = {}
    from so7e8_literature_campaign import _matching_record
    for index, task in enumerate(parent["bank_tasks"]):
        path = parent_run/"banks"/f"{index:04d}.json"
        _matching_record(path, parent, task)
        data = path.read_bytes()
        parent_bank_hashes[str(index)] = sha(data)
        data_files[f"run/parent/banks/{index:04d}.json"] = data
    tasks = []
    for row in audit["rows"]:
        # Reuse every existing native table. The reducer tests alternatives
        # for its unresolved points before requesting any higher-order work.
        if row["task"]["parent_bank"] is not None:
            continue
        task = dict(row["task"], atlas_assignment=row["task_index"], seed_file=None)
        tasks.append(task)
    manifest = dict(schema="so7e8-literature-regional-atlas-cannon-v1",
        config=parent["config"], energies=parent["energies"], momentum_rules=parent["momentum_rules"],
        parent_manifest_sha256=parent["manifest_sha256"], parent_atlas_audit_sha256=sha(audit_bytes),
        parent_bank_sha256=parent_bank_hashes,
        point_sets_file="point_sets.json", point_sets_sha256=sha(points_bytes), geometry=geometry,
        bank_tasks=tasks, source_sha256={name: sha(data) for name, data in source_files.items()},
        release_scope="third-channel banks at the first adjacent comparison, followed by regional selection; unresolved points require refinement through the original cap; no moduli integration",
        first_pass_only=True, refinement_cap=parent["config"]["blocks"]["maximum_order"],
        production_enabled=False, physical_amplitude_certified=False)
    manifest["manifest_sha256"] = digest(manifest)
    manifest_bytes = (json.dumps(manifest, indent=2)+"\n").encode()
    data_files["run/manifest.json"] = manifest_bytes
    output.mkdir(parents=True)
    (output/"manifest.json").write_bytes(manifest_bytes)
    # Keep the executable run inputs locally too, for the monitor and audit.
    for name, data in data_files.items():
        path = output/Path(name).relative_to("run")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with tarfile.open(output/"sources.tar.gz", "w:gz") as archive:
        for name, data in {**source_files, **data_files}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    result = dict(status="prepared_not_submitted", manifest_sha256=manifest["manifest_sha256"],
        archive_sha256=sha((output/"sources.tar.gz").read_bytes()),
        executable_tasks=len(tasks), missing_tasks=sum(t["seed_file"] is None for t in tasks),
        refinement_tasks=sum(t["seed_file"] is not None for t in tasks),
        unchanged_reused_assignments=len(audit["rows"])-len(tasks), physical_amplitude_certified=False)
    atomic_json(output/"prepared.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_run, args.audit_run, args.output), indent=2))
