"""Load the same physical PBW implementation with native binary64 arithmetic.

The numerical substitution follows check_level10_complex.load_runner, but
does not import its FLINT scalar or matrix backend. The caller invokes only
Check.physical_coefficient; no enlarged/auxiliary calculation is performed.
"""

from pathlib import Path
import sys
import types


HERE = Path(__file__).resolve().parent


def load_backend():
    name = "zero_mode_machine_backend"
    module = types.ModuleType(name)
    module.__file__ = str(HERE/"modular_backend.py")
    sys.modules[name] = module
    source = (HERE/"modular_backend.py").read_text()
    source = source.replace("from modular_arithmetic import", "from machine_arithmetic import")
    first = source.index("def sqrt(value):")
    last = source.index("\nnamespace = dict(", first)
    source = source[:first]+'''def sqrt(value):
    import cmath
    return cmath.sqrt(F(value))

EIGHTH_ROOT_TWO = F(2)**Fraction(1,8)

'''+source[last:]
    source = source.replace("F(int(coefficient))", "F(coefficient)").replace("F(int(c))", "F(c)")
    exec(compile(source,module.__file__,"exec"),module.__dict__)
    module.Weights.triple = module.lru_cache(None)(module.Weights.triple)
    return module


def load_runner():
    load_backend()
    name = "zero_mode_machine_runner"
    module = types.ModuleType(name)
    sys.modules[name] = module
    source = (HERE/"check_level10_modular.py").read_text()
    source = source.replace("import modular_backend as mb", "import zero_mode_machine_backend as mb")
    source = source.replace("from modular_arithmetic import", "from machine_arithmetic import")
    source = source.replace("import numpy as np", "from machine_arithmetic import NumpyProxy\nnp=NumpyProxy()")
    for expression in ("norm","1/norm","c","F(Fraction(1,2))","mm(left.reshape(-1),value.reshape(-1))"):
        source = source.replace("int("+expression+")",expression)
    exec(compile(source,str(HERE/"check_level10_modular.py"),"exec"),module.__dict__)
    return module
