"""Build the current pipeline with the two measured CCY optimizations, in isolation."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
CPP = HERE.parents[1]
source = CPP/'experiments/ccy_assembly_2026-09-12/ccy_prototype.hpp'
code = source.read_text()
old = '#ifdef CCY_GROUPED_ASSEMBLY\n        const size_t box_size'
assert code.count(old) == 1
code = code.replace(old, '#ifdef CCY_GROUPED_ASSEMBLY\n        if (dim_ == 4) {\n        const size_t box_size')
old = '#else\n        S scratch;\n        for (auto n : indices)'
assert code.count(old) == 1
code = code.replace(old, '} else\n#endif\n        {\n        S scratch;\n        for (auto n : indices)')
old = '#endif\n        assembly_seconds = seconds() - timer;'
assert code.count(old) == 1
code = code.replace(old, '}\n        assembly_seconds = seconds() - timer;')
include = HERE/'include/ramond'
include.mkdir(parents=True,exist_ok=True)
(include/'ccy.hpp').write_text(code)
(include/'pipeline.hpp').write_text((CPP/'include/ramond/pipeline.hpp').read_text())
command = ['clang++','-O3','-std=c++17','-Wall','-Wextra',
           '-DCCY_GROUPED_ASSEMBLY','-DCCY_DENSE_REUSE',
           '-I'+str(HERE/'include'),'-I'+str(CPP/'include'),'-I'+str(CPP/'include/ramond'),
           '-I/opt/homebrew/include',str(CPP/'src/main.cpp'),'-L/opt/homebrew/lib',
           '-lmpc','-lmpfr','-lgmpxx','-lgmp','-framework','Accelerate','-o',str(HERE/'ramond')]
result = subprocess.run(command,capture_output=True,text=True)
(HERE/'build.log').write_text(result.stdout+result.stderr)
if result.returncode:
    raise SystemExit(result.stderr)
paths = [source,*sorted(include.glob('*.hpp')),*sorted((CPP/'include/ramond').glob('*.hpp')),CPP/'src/main.cpp']
(HERE/'build.json').write_text(json.dumps({'command':command,'source_sha256':{
    str(p.relative_to(CPP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'binary_sha256':hashlib.sha256((HERE/'ramond').read_bytes()).hexdigest()},indent=2)+'\n')
print('Built isolated full CCY pipeline with grouped assembly and dense caches.')
