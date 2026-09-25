#!/usr/bin/env python3
"""Build the complete, immutable SO(7) Ramond four-point RC task bundle."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile

from benchmark_so7e8_four_ramond_convergence import MOMENTA as PILOT
from benchmark_so7e8_two_ramond_generic_scan import MOMENTA as EXISTING
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_two_ramond_liouville_integral import collision_convergence_margins

ROOT = Path(__file__).resolve().parents[1]


def build_tasks():
    points = {"pilot": PILOT,
              "independent": (.03+.12j,.04+.18j,.02+.26j,.09+.56j),
              **EXISTING}
    tasks = []
    for name,m in points.items():
        require_conservative_real_p_chamber(m[:3],m[3])
        if not collision_convergence_margins(m).convergent:
            raise ValueError(f"Nonconvergent energy point {name}")
        base = dict(energy=name,momenta=[[p.real,p.imag] for p in m])
        settings = [("base",15,48,"fine",1), ("block",19,48,"fine",1),
                    ("momentum",19,64,"fine",1), ("moduli",19,64,"ultra",1),
                    ("endpoint",19,64,"ultra_p5",1),
                    ("p_scale",19,64,"ultra_p5",.7)]
        for label,order,nodes,grid,scale in settings:
            tasks.append(dict(base,kind="fourr_integral",setting=label,
                              order=order,p_nodes=nodes,grid=grid,p_scale=scale))
    for name,m in points.items():
        base = dict(energy=name,momenta=[[p.real,p.imag] for p in m])
        for species in ("SS","SV","VS","VV"):
            low,high = (11,15) if species=="SS" else (15,23)
            settings = [("base",low,48,"fine"), ("block",high,48,"fine"),
                        ("momentum",high,64,"fine"), ("endpoint",high,64,"ultra_p5")]
            for label,order,nodes,grid in settings:
                tasks.append(dict(base,kind="mixed_integral",species=species,
                    setting=label,order=order,p_nodes=nodes,grid=grid))
    for name,m in points.items():
        base = dict(energy=name,momenta=[[p.real,p.imag] for p in m])
        tasks.append(dict(base,kind="fourr_crossing",order=19,p_nodes=64))
        tasks.append(dict(base,kind="mixed_ward",order=15))
    for name in ("pilot","independent"):
        base = dict(energy=name,momenta=[[p.real,p.imag] for p in points[name]])
        for species in ("SS","SV","VS","VV"):
            for picture in ("z","one"):
                tasks.append(dict(base,kind="mixed_picture_integral",species=species,
                    picture=picture,order=15 if species=="SS" else 23,
                    p_nodes=64,grid="ultra_p5"))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,
                        default=ROOT/"data_exports/so7e8_fourpoint_rc_complete_20260911")
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"submission.json").exists() and (folder/"submission.json").stat().st_size:
        raise RuntimeError("Do not replace a submitted source bundle")
    tasks = build_tasks()
    (folder/"tasks.json").write_text(json.dumps(tasks,indent=2)+"\n")
    files = set(ROOT.glob("*.py"))
    for name in ("Codes", "tests", "tools", "spin23_generated"):
        files.update((ROOT/name).rglob("*.py"))
    files.add(ROOT / "pytest.ini")
    for name in ('cluster/submit_so7e8_fourpoint.slurm','cluster/submit_so7e8_fourpoint_rc.sh',
                 'docs/so7e8/SO7E8_FOUR_RAMOND_REPAIR_20260911.md','docs/so7e8/SO7E8_MIXED_REPAIR_20260911.md'):
        files.add(ROOT/name)
    for p in files:
        if p.suffix==".py": ast.parse(p.read_text(),filename=str(p))
    manifest = dict(sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(files)},paper_tex_included=False,
                    scope="four-R and SS/SV/VS/VV reduced integrals with independent convergence tests")
    (folder/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with tarfile.open(folder/"sources.tar.gz","w:gz") as tar:
        for p in sorted(files): tar.add(p,arcname=str(p.relative_to(ROOT)))
        for name in ("source_manifest.json","tasks.json"): tar.add(folder/name,arcname=name)
    summary = {kind:sum(t["kind"]==kind for t in tasks) for kind in sorted({t["kind"] for t in tasks})}
    metadata = dict(task_count=len(tasks),counts=summary,maximum_concurrent_cores=128,
        source_archive_sha256=hashlib.sha256((folder/"sources.tar.gz").read_bytes()).hexdigest(),
        status="prepared_not_submitted")
    (folder/"prepared.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__": main()
