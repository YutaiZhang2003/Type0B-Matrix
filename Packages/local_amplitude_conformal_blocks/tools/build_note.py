"""Compile the TeX note in a temporary directory and retain only its PDF."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
compiler = shutil.which('latexmk')
if compiler is None:
    raise SystemExit('Install latexmk/pdflatex and put them on PATH to rebuild the note.')
with tempfile.TemporaryDirectory(prefix='local-amplitude-tex-') as build:
    subprocess.run([compiler, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                    '-outdir='+build, 'local_amplitude_blocks.tex'],
                   cwd=ROOT/'notes', check=True)
    log = (Path(build)/'local_amplitude_blocks.log').read_text()
    if 'Overfull' in log or 'undefined references' in log:
        raise RuntimeError('Inspect the TeX layout or references before releasing the note.')
    output = ROOT/'notes/local_amplitude_blocks.pdf'
    shutil.copyfile(Path(build)/output.name, output)
    print('Wrote', output)
