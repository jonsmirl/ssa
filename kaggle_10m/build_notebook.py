"""Build the self-contained Kaggle notebook for the >10M SSA experiment."""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCES = (
    "ssa/__init__.py",
    "ssa/ssa_kernel.py",
    "ssa/ivf_kernel.py",
    "ssa/cascade_router.py",
    "ssa/streaming_qwen.py",
    "ssa/kaggle_10m_runner.py",
)


def code(source, cell_id):
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def markdown(source, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(True)}


payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii") for name in SOURCES}
install = r'''# ARC3's RTX Pro 6000 tier requires internet OFF. Install the two pinned public wheels
# from our private attachment; --no-index proves the experiment has no runtime dependency download.
import glob, os, subprocess, sys
wheels = sorted(glob.glob("/kaggle/input/**/transformers-5.12.1*.whl", recursive=True))
faiss = sorted(glob.glob("/kaggle/input/**/faiss_gpu_cu12-1.14.1*.whl", recursive=True))
if not wheels or not faiss:
    raise FileNotFoundError("offline Transformers/FAISS wheel attachment was not mounted")
cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", "--no-index", "--no-deps",
       wheels[0], faiss[0]]
r = subprocess.run(cmd, text=True, capture_output=True)
print("pip rc", r.returncode, (r.stderr or r.stdout)[-2000:])
if r.returncode:
    raise RuntimeError("dependency installation failed")
'''
write_sources = f'''import base64, os
files = {json.dumps(payload, sort_keys=True)}
for name, encoded in files.items():
    path = "/kaggle/working/" + name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(encoded))
print("staged", len(files), "SSA source files")
'''
run = '''import runpy, sys
sys.path.insert(0, "/kaggle/working")
runpy.run_path("/kaggle/working/ssa/kaggle_10m_runner.py", run_name="__main__")
'''

nb = {
    "cells": [
        markdown("# SSA complete transformer at >10M tokens\n\nPrivate RTX Pro 6000 capacity run; ARC3 attached for accelerator access.", "title"),
        code(install, "offline-deps"), code(write_sources, "stage-source"), code(run, "execute"),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
(HERE / "ssa_10m.ipynb").write_text(json.dumps(nb))
print(HERE / "ssa_10m.ipynb")
