"""Package the frozen-gain full-scale protocol and cached public test tokens."""
import base64
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--tail-mode", choices=("prototype", "jensen"), default="prototype")
parser.add_argument("--gain-mode", choices=("saved", "zero"), default="saved")
parser.add_argument("--max-tail-share", type=float)
parser.add_argument("--selection", type=Path, help="optional frozen validation-selection JSON to embed")
args = parser.parse_args()
if args.max_tail_share is not None and not (math.isfinite(args.max_tail_share) and 0 <= args.max_tail_share <= 1):
    parser.error("max-tail-share must be finite and in [0,1]")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
arrow = next((Path.home() / ".cache/huggingface/datasets/wikitext").glob("wikitext-2-raw-v1/*/*/wikitext-test.arrow"))
ds = Dataset.from_file(str(arrow))
text = "\n".join(s for s in ds["text"] if len(s.strip()) > 0)
tokens = np.asarray(tokenizer(text, truncation=False)["input_ids"], dtype=np.int32)
memory = io.BytesIO(); np.savez_compressed(memory, test=tokens)
sources = ["ssa/__init__.py", "ssa/ssa_kernel.py", "ssa/ivf_kernel.py", "ssa/cascade_router.py",
           "ssa/streaming_qwen.py", "ssa/kaggle_10m_runner.py", "ssa/hybrid_tail_attention.py",
           "ssa/tail_tree_router.py", "ssa/qwen_tail_demo.py", "ssa/fullscale_tail_runner.py", "ssa/tail_correction.py"]
files = {name: (ROOT / name).read_bytes() for name in sources}
files["tail_bundle/tokens.npz"] = memory.getvalue()
files["tail_bundle/gains.json"] = (ROOT / "runs/qwen_tail_final/results.json").read_bytes()
if args.selection is not None:
    selection_data = args.selection.read_bytes()
    selected = json.loads(selection_data)["selected_config"]
    requested = {"tail_mode": args.tail_mode, "gain_mode": args.gain_mode,
                 "max_tail_share": args.max_tail_share}
    if selected != requested:
        raise ValueError("packaged configuration differs from frozen selection")
    files["tail_bundle/selection.json"] = selection_data
manifest = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            "test_tokens": len(tokens), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "protocol": "all official WikiText-2 test windows at 512,4096,8192,32768; 9 fixed NIAH probes through 131072; no test fitting",
            "revision": {**vars(args), "selection": str(args.selection) if args.selection else None}}
files["tail_bundle/manifest.json"] = json.dumps(manifest, indent=2).encode()
# Also stage an identical local bundle for the GPU preflight.
bundle = ROOT / "runs/kaggle_tail_bundle"
bundle.mkdir(parents=True, exist_ok=True)
for name in ("tokens.npz", "gains.json", "manifest.json", *(["selection.json"] if args.selection else [])):
    (bundle / name).write_bytes(files["tail_bundle/" + name])
payload = {name: base64.b64encode(data).decode() for name, data in files.items()}
install = '''import glob, os, subprocess, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
wheels = [sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))
          for pattern in ('transformers-5.12.1*.whl', 'faiss_gpu_cu12-1.14.1*.whl')]
if not all(wheels):
    raise FileNotFoundError('offline pinned wheels not mounted')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', '--no-index', '--no-deps',
                *[items[0] for items in wheels]], check=True)
'''
stage = f'''import base64
from pathlib import Path
files = {json.dumps(payload)}
for name, data in files.items():
    path = Path('/kaggle/working') / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
print('staged', len(files), 'files')
'''
run_args = ["ssa.fullscale_tail_runner", "--tail-mode", args.tail_mode, "--gain-mode", args.gain_mode]
if args.max_tail_share is not None:
    run_args += ["--max-tail-share", str(args.max_tail_share)]
run = f'''import runpy, sys
sys.path.insert(0, '/kaggle/working')
sys.argv = {run_args!r}
runpy.run_module('ssa.fullscale_tail_runner', run_name='__main__')
'''
cells = [{"cell_type": "code", "id": str(i), "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(True)}
         for i, code in enumerate((install, stage, run))]
notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
(HERE / "ssa_tail_fullscale.ipynb").write_text(json.dumps(notebook))
print(json.dumps(manifest, indent=2))
