"""Regenerate the release manifest after deliberate changes to the package."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
files = {}
for path in sorted(ROOT.rglob('*')):
    relative = path.relative_to(ROOT)
    if (not path.is_file() or path.name == 'MANIFEST.json'
            or any(x in ('.venv', '__pycache__', '.git', '.pytest_cache') for x in relative.parts)
            or relative.parts[0] in ('results', 'build')
            or path.name == '.DS_Store' or path.suffix in ('.pyc', '.pyo')
            or (relative.parts[0] == 'notes' and path.suffix in
                ('.aux', '.log', '.out', '.toc', '.fdb_latexmk', '.fls'))):
        continue
    if path.is_symlink():
        raise ValueError('Release must contain real files, not symlinks: '+str(relative))
    data = path.read_bytes()
    files[relative.as_posix()] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
manifest = dict(schema='local-amplitude-conformal-blocks-manifest-v1',
    convention='local_amplitude_production_2026-09-24',
    base_commit='4887104301ec1f3666cfad058518cd01f53b761e', files=files)
(ROOT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Manifest contains', len(files), 'files.')
