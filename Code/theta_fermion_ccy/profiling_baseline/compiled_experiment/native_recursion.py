"""Optional compiled CCY loop, using the existing Python numeric operations.

No package installation is required. Build once with the local C++ compiler;
the source/ABI-keyed extension is cached in __pycache__. A pipeline can include
the build in its timing by setting CCY_REBUILD_NATIVE=1 before process launch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

HERE = Path(__file__).resolve().parent
_module = None
_attempted = False
_info = {}


def load_native():
    global _module, _attempted, _info
    if _attempted:
        return _module
    _attempted = True
    source = HERE / "ccy_native.cpp"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"]
    if sys.platform == "darwin":
        flags += ["-undefined", "dynamic_lookup"]
    signature = hashlib.sha256((source_hash + repr(flags) + sys.version +
                                str(sysconfig.get_config_var("SOABI"))).encode()).hexdigest()[:16]
    cache = HERE / "__pycache__"
    binary = cache / ("_ccy_native_" + signature + suffix)
    _info = {"source_sha256":source_hash, "binary":str(binary),
             "build_seconds":0.0, "built_this_process":False}
    if not binary.exists() or os.environ.get("CCY_REBUILD_NATIVE") == "1":
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            _info["fallback_reason"] = "No C++ compiler and no cached extension"
            return None
        cache.mkdir(exist_ok=True)
        temporary = binary.with_name(binary.name + f".{os.getpid()}.tmp")
        command = [compiler, *flags, "-I" + sysconfig.get_path("include"),
                   str(source), "-o", str(temporary)]
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        _info.update(build_seconds=time.perf_counter()-started,
                     built_this_process=True, build_command=command)
        if result.returncode:
            temporary.unlink(missing_ok=True)
            _info["fallback_reason"] = result.stderr[-4000:]
            return None
        temporary.replace(binary)
    spec = importlib.util.spec_from_file_location("_ccy_native", binary)
    try:
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
    except (ImportError, OSError) as error:
        _module = None
        _info["fallback_reason"] = str(error)
    return _module


def native_info():
    return dict(_info)
