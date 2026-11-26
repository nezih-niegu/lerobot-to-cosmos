"""``cosmos-check``: test a converted dataset with NVIDIA's Cosmos pipeline.

* ``cosmos-check loader`` -- macOS or Linux, CPU only. Runs cosmos-framework's own SFT data
  loader (``get_sft_dataset`` + ``SFTDataset``) over every clip, with the settings read from
  the ``vision_sft_nano`` recipe.
* ``cosmos-check train``  -- Linux + NVIDIA GPUs. Follows NVIDIA's documented setup
  (docs/setup.md, docs/training.md), then runs a short SFT smoke run on the dataset, the
  same way NVIDIA's own tests/nano_training_smoke_test.py does with Bridge.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

COSMOS_GIT = "https://github.com/nvidia/cosmos-framework.git"
COSMOS_TARBALL = "https://codeload.github.com/nvidia/cosmos-framework/tar.gz/{ref}"
# Commit the checks were written and tested against (2026-11-20). --cosmos-ref main to track upstream.
PINNED_REF = "a3fb34690e157972a9d04244c11b16022c9d6479"
QWEN_TOKENIZER = "Qwen/Qwen3-VL-8B-Instruct"
# Smallest set that imports cosmos_framework...sft_dataset and runs it on CPU.
LOADER_PYTHON = "3.13"  # same as cosmos-framework's training environment
LOADER_DEPS = ["torch", "numpy", "boto3", "loguru", "omegaconf", "transformers>=4.57.1,<5", "termcolor",
               "attrs", "pillow", "pydantic", "iopath", "static-ffmpeg"]
CHECK_SCRIPT = Path(__file__).with_name("cosmos_loader_check.py")
LOSS_RE = re.compile(
    r"(?:\[RANK\s+(?P<rank>\d+)\]\s+)?Iteration\s+\d+:\s+Hit counter:[^|]+\|\s+Loss:\s+(?P<loss>[-+0-9.eE]+|nan|inf)",
    re.IGNORECASE,
)
MODELS = {"nano": "Cosmos3-Nano", "edge": "Cosmos3-Edge", "super": "Cosmos3-Super"}

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Test a converted dataset with NVIDIA's Cosmos pipeline (cosmos-framework).")


def echo(msg: str = "") -> None:
    typer.echo(msg)


class CheckError(RuntimeError):
    pass


# --------------------------------------------------------------------------- shared helpers
def cache_dir() -> Path:
    base = os.environ.get("LEROBOT_TO_COSMOS_CACHE") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "lerobot-to-cosmos"


def uv_exe() -> str:
    exe = os.environ.get("UV") or shutil.which("uv")
    if not exe:
        raise CheckError("uv not found on PATH (install: curl -LsSf https://astral.sh/uv/install.sh | sh)")
    return exe


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for child processes, without this project's own virtualenv leaking in."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME")}
    env.update(extra or {})
    return env


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        log: Path | None = None) -> tuple[int, str]:
    """Run a command, streaming output to the terminal (and a log file); return (rc, output)."""
    echo(f"$ {shlex.join(cmd)}" + (f"   (in {cwd})" if cwd else ""))
    out: list[str] = []
    fh = None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out.append(line)
            if fh:
                fh.write(line)
        rc = proc.wait()
    finally:
        if fh:
            fh.close()
    return rc, "".join(out)


def must(cmd: list[str], what: str, **kw) -> str:
    rc, out = run(cmd, **kw)
    if rc != 0:
        raise CheckError(f"{what} failed (exit {rc}). Last output:\n{out[-3000:]}")
    return out


def fetch_cosmos(dest: Path, ref: str, *, full: bool = False) -> Path:
    """Check out cosmos-framework at ``ref`` into ``dest`` (git; tarball fallback for the loader check)."""
    marker = dest / ".lerobot_to_cosmos_ref"
    if dest.exists() and marker.exists() and marker.read_text().strip() == ref and ref != "main":
        echo(f"cosmos-framework {ref[:12]} already at {dest}")
        return dest
    git = shutil.which("git")
    if git:
        if not (dest / ".git").exists():
            dest.mkdir(parents=True, exist_ok=True)
            must([git, "init", "-q"], "git init", cwd=dest)
            must([git, "remote", "add", "origin", COSMOS_GIT], "git remote add", cwd=dest)
        must([git, "fetch", "-q", "--depth", "1", "origin", ref], "git fetch cosmos-framework", cwd=dest)
        must([git, "checkout", "-q", "--force", "FETCH_HEAD"], "git checkout", cwd=dest)
        sha = subprocess.run([git, "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=False).stdout.strip()
    elif not full:
        echo(f"git not found; downloading the cosmos-framework {ref[:12]} source tarball")
        tmp = Path(tempfile.mkdtemp())
        tgz = tmp / "src.tar.gz"
        urllib.request.urlretrieve(COSMOS_TARBALL.format(ref=ref), tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp, filter="data")
        top = next(p for p in tmp.iterdir() if p.is_dir())
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(top), dest)
        shutil.rmtree(tmp, ignore_errors=True)
        sha = ref
    else:
        raise CheckError("git is required for the training check (sudo apt-get install -y git)")
    marker.write_text(sha if ref != "main" else "main")
    echo(f"cosmos-framework at {sha[:12]} in {dest}")
    return dest


def loader_python(torch_backend: str) -> Path:
    """A cached venv with LOADER_DEPS (CPU torch on Linux unless told otherwise)."""
    uv = uv_exe()
    key = hashlib.sha256(json.dumps([LOADER_PYTHON, LOADER_DEPS, torch_backend, sys.platform, platform.machine()]).encode()).hexdigest()[:12]
    venv = cache_dir() / f"loader-env-{key}"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ok = venv / ".ready"
    if ok.exists() and py.exists():
        return py
    echo(f"Creating the loader-check environment in {venv} (first run only; downloads PyTorch)")
    must([uv, "venv", "--quiet", "--python", LOADER_PYTHON, str(venv)], "uv venv", env=clean_env())
    cmd = [uv, "pip", "install", "--quiet", "--python", str(py)]
    if torch_backend != "pypi":
        cmd += ["--torch-backend", torch_backend]
    must(cmd + LOADER_DEPS, "installing loader dependencies", env=clean_env())
    ok.write_text("ok")
    return py


def default_torch_backend() -> str:
    # macOS wheels on PyPI are already CPU/MPS; on Linux skip the multi-GB CUDA build.
    return "pypi" if sys.platform == "darwin" else "cpu"


def dataset_dir(path: Path) -> Path:
    p = path.expanduser().resolve()
    if not p.exists():
        hint = ""
        near = [c for c in (Path.cwd() / path.name, Path.cwd().parent / path.name, Path.home() / path.name)
                if c != p and (c / "train" / "video_dataset_file.jsonl").is_file()]
        if near:
            rel = os.path.relpath(near[0], Path.cwd())
            hint = f"; did you mean -d {rel} ?"
        raise CheckError(f"{p} doesn't exist{hint}")
    if not (p / "train" / "video_dataset_file.jsonl").is_file():
        raise CheckError(f"{p} is not a converted dataset (missing train/video_dataset_file.jsonl)")
    return p


def run_loader_check(py: Path, repo: Path, dataset: Path, tokenizer: str, max_samples: int,
                     log: Path | None = None) -> int:
    cmd = [str(py), str(CHECK_SCRIPT), "--repo", str(repo), "--dataset", str(dataset),
           "--tokenizer", tokenizer, "--max-samples", str(max_samples),
           "--report", str(dataset / "cosmos_loader_check.json"),
           "--preview", str(dataset / "cosmos_loader_preview.png")]
    env = clean_env({"PYTHONPATH": str(repo), "PYTHONWARNINGS": "ignore", "TOKENIZERS_PARALLELISM": "false"})
    rc, _ = run(cmd, env=env, log=log)
    return rc


# --------------------------------------------------------------------------- loader
@app.command()
def loader(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Converted dataset folder (contains train/).")],
    cosmos_repo: Annotated[Path | None, typer.Option("--cosmos-repo", help="Use an existing cosmos-framework checkout.")] = None,
    cosmos_ref: Annotated[str, typer.Option("--cosmos-ref", help="cosmos-framework commit to test against ('main' = latest).")] = PINNED_REF,
    tokenizer: Annotated[str, typer.Option("--tokenizer", help="Tokenizer the recipe uses (HF id or local folder).")] = QWEN_TOKENIZER,
    python: Annotated[Path | None, typer.Option("--python", help="Python that already has the loader dependencies (skips the managed env).")] = None,
    torch_backend: Annotated[str | None, typer.Option("--torch-backend", help="uv --torch-backend for the managed env: cpu, auto, cu128, ... or 'pypi'. Default: cpu on Linux, pypi on macOS.")] = None,
    max_samples: Annotated[int, typer.Option("--max-samples", help="Check only the first N clips (0 = all).")] = 0,
) -> None:
    """Run NVIDIA's own SFT data loader over every clip (macOS or Linux, no GPU needed)."""
    try:
        ds = dataset_dir(dataset)
        repo = cosmos_repo.expanduser().resolve() if cosmos_repo else fetch_cosmos(cache_dir() / f"cosmos-framework-{cosmos_ref[:12]}", cosmos_ref)
        py = python.expanduser().absolute() if python else loader_python(torch_backend or default_torch_backend())
        rc = run_loader_check(py, repo, ds, tokenizer, max_samples)
    except CheckError as e:
        echo(f"Error: {e}")
        raise typer.Exit(2) from None
    raise typer.Exit(rc)


# --------------------------------------------------------------------------- train
def edit_toml(base_toml: str, edits: dict[tuple[str, str], str]) -> str:
    """Set ``[section] key = value`` lines in a recipe TOML, keeping layout and comments."""
    done: set[tuple[str, str]] = set()
    section, out = "", []
    for line in base_toml.splitlines():
        m = re.match(r"\s*\[([^\]]+)\]\s*(#.*)?$", line)
        if m:
            section = m.group(1).strip()
        else:
            km = re.match(r"(\s*)([A-Za-z0-9_]+)(\s*=\s*)([^#]*?)(\s*#.*)?$", line)
            if km and (section, km.group(2)) in edits:
                key = (section, km.group(2))
                line = f"{km.group(1)}{km.group(2)}{km.group(3)}{edits[key]}{km.group(5) or ''}"
                done.add(key)
        out.append(line)
    missing = set(edits) - done
    if missing:
        raise CheckError(f"Recipe TOML lacks expected keys {sorted(missing)}; cosmos-framework changed?")
    text = "\n".join(out) + "\n"
    tomllib.loads(text)
    return text


def smoke_toml(base_toml: str, *, name: str, iters: int) -> str:
    """Recipe TOML -> short smoke run (what NVIDIA's tests/vision_sft_nano_5iter.toml changes)."""
    text = edit_toml(base_toml, {("job", "name"): json.dumps(name), ("trainer", "max_iter"): str(iters),
                                 ("checkpoint", "save_iter"): str(iters), ("scheduler", "warm_up_steps"): "[1]"})
    parsed = tomllib.loads(text)
    assert parsed["trainer"]["max_iter"] == iters and parsed["checkpoint"]["save_iter"] == iters
    return text


def full_toml(base_toml: str, *, name: str, iters: int | None, save_every: int | None) -> str:
    """Recipe TOML -> full run: the recipe as NVIDIA ships it, only the run name (and optionally
    the length / checkpoint interval) changed. If a run is longer than the LR cycle, the cycle is
    stretched to match so the learning rate still decays to its minimum at the end."""
    base = tomllib.loads(base_toml)
    edits = {("job", "name"): json.dumps(name)}
    if iters:
        edits[("trainer", "max_iter")] = str(iters)
        cycle = (base.get("scheduler", {}).get("cycle_lengths") or [0])[0]
        if cycle and iters > cycle:
            edits[("scheduler", "cycle_lengths")] = f"[{iters}]"
    if save_every:
        edits[("checkpoint", "save_iter")] = str(save_every)
    return edit_toml(base_toml, edits)


LAUNCHER = """#!/usr/bin/env bash
# Written by lerobot-to-cosmos `cosmos-check train`: {model} vision SFT ({mode}) on a converted dataset.
TOML_FILE="examples/toml/sft_config/{toml_name}"
: "${{BASE_CHECKPOINT_PATH:=examples/checkpoints/{ckpt}}}"
EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || {{ echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }}'
source "$(dirname "${{BASH_SOURCE[0]}}")/_sft_launcher_common.sh"
"""


def parse_losses(text: str) -> list[float]:
    vals = []
    for m in LOSS_RE.finditer(text):
        if (m.group("rank") or "0") == "0":
            try:
                vals.append(float(m.group("loss")))
            except ValueError:
                vals.append(float("nan"))
    return vals


def iteration_seconds(text: str) -> float | None:
    """Median seconds per iteration from the trainer's per-iteration log lines (skips compile warm-up)."""
    times = [float(t) for t in re.findall(r"Hit counter:[^|]+\|\s+Loss:[^|]+\|\s+Time:\s*([0-9.]+)s", text)]
    steady = times[2:] if len(times) > 3 else times[1:]
    if not steady:
        return None
    steady.sort()
    return steady[len(steady) // 2]


def judge_training(rc: int, log_text: str, iters: int, *, full: bool = False) -> tuple[bool, list[str]]:
    losses = parse_losses(log_text)
    lines, ok = [], True
    if rc != 0:
        ok = False
        lines.append(f"training exited with code {rc}")
    if "Done with training" not in log_text:
        ok = False
        lines.append("no 'Done with training' line in the log")
    # Smoke runs log every iteration. Full runs only log the first iterations line by line
    # (the trainer's IterSpeed callback), so the count isn't checked there.
    if not full and len(losses) != iters:
        ok = False
        lines.append(f"expected {iters} logged losses, found {len(losses)}")
    if any(not math.isfinite(v) for v in losses):
        ok = False
        lines.append("non-finite loss (NaN/inf)")
    saved = re.findall(r"Saved checkpoint to (\S+)", log_text)
    if not saved:
        ok = False
        lines.append("no checkpoint saved")
    if losses:
        if full and len(losses) >= 10:
            k = max(1, len(losses) // 5)
            first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
            lines.append(f"loss: mean of first {k} logged {first:.4f} -> last {k} logged {last:.4f}")
        else:
            trend = "down" if min(losses[1:] or losses) < losses[0] else "not down (fine for a smoke run on new data)"
            lines.append(f"losses: {', '.join(f'{v:.4f}' for v in losses)}  (trend {trend})")
    if saved:
        lines.append(f"checkpoint: {saved[-1]}")
    return ok, lines


@dataclass
class Step:
    name: str
    cmd: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    skip_reason: str = ""
    slug: str = ""
    action: object = None          # optional callable run instead of cmd (for composite steps)


def gpu_count() -> int:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0
    res = subprocess.run([smi, "-L"], capture_output=True, text=True, check=False)
    return sum(1 for line in res.stdout.splitlines() if line.startswith("GPU "))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def required_uv(repo: Path) -> str | None:
    pp = repo / "pyproject.toml"
    if not pp.exists():
        return None
    req = tomllib.loads(pp.read_text(encoding="utf-8")).get("tool", {}).get("uv", {}).get("required-version", "")
    m = re.fullmatch(r"\s*==\s*([0-9][0-9A-Za-z.+-]*)\s*", req or "")
    return m.group(1) if m else None


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def latest_checkpoint(run_dir: Path) -> Path | None:
    f = run_dir / "checkpoints" / "latest_checkpoint.txt"
    if not f.is_file():
        return None
    ck = run_dir / "checkpoints" / f.read_text().strip()
    return ck if ck.is_dir() else None


# --------------------------------------------------------------------------- layout of a run
@dataclass
class Layout:
    """Where everything for one model lives under --workdir."""
    work: Path
    model: str
    run_name: str

    @property
    def repo(self) -> Path:
        return self.work / "cosmos-framework"

    @property
    def venv_bin(self) -> Path:
        return self.repo / ".venv" / "bin"

    @property
    def ckpt_name(self) -> str:
        return MODELS[self.model]

    @property
    def out_root(self) -> Path:
        return self.work / "train_out" / self.model

    @property
    def run_dir(self) -> Path:
        # $IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>; the vision_sft_* recipes use cosmos3/sft.
        return self.out_root / "cosmos3" / "sft" / self.run_name

    @property
    def logs(self) -> Path:
        return self.work / "logs"

    def venv_env(self) -> dict[str, str]:
        return {"PATH": f"{self.venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "VIRTUAL_ENV": str(self.repo / ".venv"), "PYTHONPATH": "."}


# --------------------------------------------------------------------------- validation videos
def pick_val_prompts(ds: Path, n: int, camera: str | None) -> list[Path]:
    """Image-to-video prompts of held-out clips: the requested camera, else RGB views first."""
    files = sorted((ds / "val" / "inference_prompt_i2v").glob("*.json"))
    if not files:
        raise CheckError(f"No image-to-video prompts in {ds / 'val' / 'inference_prompt_i2v'}")
    if camera:
        chosen = [f for f in files if f.stem.endswith(f"_{camera}")]
        if not chosen:
            cams = sorted({f.stem.split("_clip")[-1].split("_", 1)[-1] for f in files})
            raise CheckError(f"No validation clips for camera {camera!r}; available: {cams}")
    else:
        rgb = [f for f in files if not f.stem.endswith("_depth")]
        pref = [f for f in rgb if f.stem.endswith("_realsense")]
        chosen = pref or rgb or files
    return chosen[:n]


def prepare_prompts(prompt_files: list[Path], ds: Path, out_dir: Path, seconds: float) -> list[dict]:
    """Copy val prompts for inference: absolute condition paths, ~`seconds` long (4k+1 frames)."""
    from .media import probe

    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for f in prompt_files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        uuid = f.stem
        fps = float(rec.get("fps") or 5)
        gt = ds / "val" / "videos" / f"{uuid}.mp4"
        gt_frames = probe(gt)["total_frames"] if gt.exists() else 10**6
        nf = max(5, min(round(seconds * fps), gt_frames, int(rec.get("num_frames") or 10**6)))
        nf = (nf - 1) // 4 * 4 + 1
        rec["name"] = uuid
        rec["num_frames"] = nf
        if rec.get("vision_path"):
            rec["vision_path"] = str((f.parent / rec["vision_path"]).resolve())
        (out_dir / f"{uuid}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        items.append({"uuid": uuid, "fps": fps, "num_frames": nf, "gt": gt})
    return items


def _font() -> str | None:
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(cand).is_file():
            return cand
    return None


def compose_comparison(panels: list[tuple[str, Path]], fps: float, frames: int, out: Path) -> bool:
    """Side-by-side video of the given clips (same height, same fps, same length); labels if a font
    is available. Returns True if labels were drawn."""
    from .media import run_ffmpeg

    out.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    for _, path in panels:
        args += ["-i", str(path)]

    def graph(labels: bool) -> str:
        parts = []
        font = _font()
        for i, (label, _) in enumerate(panels):
            chain = f"[{i}:v]fps={fps:g},scale=-2:256,setsar=1,trim=end_frame={frames},setpts=PTS-STARTPTS"
            if labels and font:
                safe = label.replace(":", r"\:").replace("'", "")
                chain += (f",drawtext=fontfile='{font}':text='{safe}':x=8:y=8:fontsize=16:fontcolor=white"
                          f":box=1:boxcolor=black@0.55:boxborderw=4")
            parts.append(chain + f"[v{i}]")
        stack = "".join(f"[v{i}]" for i in range(len(panels)))
        parts.append(f"{stack}hstack=inputs={len(panels)}[out]" if len(panels) > 1 else f"{stack}null[out]")
        return ";".join(parts)

    enc = ["-map", "[out]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"]
    try:
        run_ffmpeg([*args, "-filter_complex", graph(True), *enc, str(out)])
        return _font() is not None
    except RuntimeError:
        run_ffmpeg([*args, "-filter_complex", graph(False), *enc, str(out)])
        return False


def sample_steps(L: Layout, ds: Path, *, seconds: float, clips: int, camera: str | None, compare_base: bool,
                 checkpoint: Path | None = None) -> list[Step]:
    """Export the trained checkpoint, generate image-to-video clips for held-out episodes with the
    fine-tuned (and base) model, and put them side by side with the real episode."""
    venv = L.venv_env()
    py = str(L.venv_bin / "python")
    samples = L.run_dir / "val_samples"
    export_dir = L.run_dir / "model"
    marker = export_dir / ".lerobot_to_cosmos_exported_from"

    def resolve_ckpt() -> Path:
        ck = checkpoint or latest_checkpoint(L.run_dir)
        if not ck:
            raise CheckError(f"No checkpoint found in {L.run_dir / 'checkpoints'}; train first.")
        return ck

    def do_export() -> None:
        ck = resolve_ckpt()
        if marker.is_file() and marker.read_text().strip() == str(ck):
            echo(f"  already exported from {ck.name}")
            return
        if export_dir.exists():
            shutil.rmtree(export_dir)
        cmd = [py, "-m", "cosmos_framework.scripts.export_model", "--checkpoint-path", str(ck),
               "--config-file", str(L.run_dir / "config.yaml"), "--no-vit", "-o", str(export_dir)]
        rc, _ = run(cmd, cwd=L.repo, env=clean_env(venv), log=L.logs / f"export_{L.run_name}.log")
        if rc != 0:
            raise CheckError(f"export_model failed (exit {rc}); log: {L.logs / f'export_{L.run_name}.log'}")
        marker.write_text(str(ck))

    def do_generate() -> None:
        items = prepare_prompts(pick_val_prompts(ds, clips, camera), ds, samples / "prompts", seconds)
        inputs = [str(samples / "prompts" / f"{it['uuid']}.json") for it in items]
        runs = [("fine-tuned", str(export_dir), samples / "finetuned")]
        if compare_base:
            runs.insert(0, ("base", L.ckpt_name, samples / "base"))
        for label, ckpt, out in runs:
            cmd = [py, "-m", "cosmos_framework.scripts.inference", "--parallelism-preset=latency",
                   "-i", *inputs, "-o", str(out), "--checkpoint-path", ckpt,
                   "--model-mode", "image2video", "--seed", "0"]
            rc, _ = run(cmd, cwd=L.repo, env=clean_env(venv), log=L.logs / f"sample_{label}_{L.run_name}.log")
            if rc != 0:
                raise CheckError(f"inference with the {label} model failed (exit {rc}); "
                                 f"log: {L.logs / f'sample_{label}_{L.run_name}.log'}")
        made = []
        for it in items:
            panels = [("real episode", it["gt"])] if it["gt"].exists() else []
            if compare_base:
                panels.append((f"base {L.ckpt_name}", samples / "base" / it["uuid"] / "vision.mp4"))
            panels.append(("fine-tuned", samples / "finetuned" / it["uuid"] / "vision.mp4"))
            missing = [str(p) for _, p in panels if not p.exists()]
            if missing:
                raise CheckError(f"expected generated video(s) not found: {missing}")
            out = samples / f"{it['uuid']}_comparison.mp4"
            labelled = compose_comparison(panels, it["fps"], it["num_frames"], out)
            made.append(out)
            dur = (it["num_frames"] - 1) / it["fps"]
            echo(f"  {out}  ({dur:.1f} s, {it['num_frames']} frames @ {it['fps']:g} fps; "
                 f"left to right: {', '.join(lbl for lbl, _ in panels)}{'' if labelled else ', no labels'})")
        echo(f"\nValidation videos in {samples}:")
        for m in made:
            echo(f"  {m}")

    return [
        Step(f"Export {L.run_name} checkpoint to Hugging Face format", [py, "-m", "cosmos_framework.scripts.export_model",
             "--checkpoint-path", "<latest checkpoint>", "--config-file", str(L.run_dir / "config.yaml"),
             "--no-vit", "-o", str(export_dir)], L.repo, venv, slug="export", action=do_export),
        Step(f"Validation video{'s' if clips > 1 else ''}: {seconds:g} s image-to-video from held-out episode"
             f"{'s' if clips > 1 else ''}" + (" (base vs fine-tuned)" if compare_base else ""),
             [py, "-m", "cosmos_framework.scripts.inference", "--parallelism-preset=latency",
              "-i", str(samples / "prompts" / "*.json"), "-o", str(samples / "finetuned"),
              "--checkpoint-path", str(export_dir), "--model-mode", "image2video", "--seed", "0"],
             L.repo, venv, slug="sample", action=do_generate),
    ]


# --------------------------------------------------------------------------- train
@app.command()
def train(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Converted dataset folder (contains train/).")],
    model: Annotated[str, typer.Option("--model", help="nano (Cosmos3-Nano, 8 GPUs recommended), edge (Cosmos3-Edge, fits 1-4) or super.")] = "nano",
    full: Annotated[bool, typer.Option("--full", help="Full fine-tune with NVIDIA's recipe (500 iterations, checkpoint every 100), then export it and generate a validation video. Default is a 5-iteration smoke run.")] = False,
    iters: Annotated[int | None, typer.Option("--iters", help="Iterations (smoke default 5; full default: the recipe's 500).")] = None,
    save_every: Annotated[int | None, typer.Option("--save-every", help="Full run: checkpoint every N iterations (recipe: 100). Each checkpoint holds optimizer state, so fewer saves = less disk.")] = None,
    run_name: Annotated[str | None, typer.Option("--run-name", help="Name of the training run (default so101_full_<model>). Rerunning the same name resumes it; pick a new name to start over.")] = None,
    sample: Annotated[bool | None, typer.Option("--sample/--no-sample", help="After training, generate validation videos (default: on for --full).")] = None,
    sample_seconds: Annotated[float, typer.Option("--sample-seconds", help="Length of each validation video.")] = 5.0,
    sample_clips: Annotated[int, typer.Option("--sample-clips", help="How many held-out episodes to generate.")] = 1,
    sample_camera: Annotated[str | None, typer.Option("--sample-camera", help="Camera of the validation clips (e.g. realsense, logitech). Default: realsense, else any RGB view.")] = None,
    compare_base: Annotated[bool, typer.Option("--compare-base/--no-compare-base", help="Also generate with the base model for a side-by-side comparison.")] = True,
    gpus: Annotated[int, typer.Option("--gpus", help="GPUs to use (default: all that nvidia-smi lists).")] = 0,
    cuda: Annotated[str, typer.Option("--cuda", help="CUDA build of the training env: 13.0 (driver >= 580) or 12.8.")] = "13.0",
    workdir: Annotated[Path, typer.Option("--workdir", help="Where cosmos-framework, checkpoints and outputs go.")] = Path("cosmos_work"),
    cosmos_ref: Annotated[str, typer.Option("--cosmos-ref", help="cosmos-framework commit ('main' = latest).")] = PINNED_REF,
    tokenizer: Annotated[str, typer.Option("--tokenizer", help="Tokenizer for the loader pre-check.")] = QWEN_TOKENIZER,
    skip_loader_check: Annotated[bool, typer.Option("--skip-loader-check", help="Don't run the data-loader check before training.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print every step without running anything.")] = False,
) -> None:
    """Cosmos SFT on the dataset (Linux + NVIDIA GPUs): a 5-iteration smoke run, or --full."""
    try:
        code = _train(dataset, model, full, iters, save_every, run_name, sample, sample_seconds, sample_clips,
                      sample_camera, compare_base, gpus, cuda, workdir, cosmos_ref, tokenizer, skip_loader_check,
                      dry_run)
    except CheckError as e:
        echo(f"Error: {e}")
        raise typer.Exit(2) from None
    raise typer.Exit(code)


def _preflight(model: str, gpus: int, dry_run: bool, work: Path) -> int:
    problems = []
    if sys.platform != "linux":
        problems.append(f"training needs Linux with NVIDIA GPUs (this is {platform.system()}); "
                        f"run `cosmos-check loader` here instead")
    n_gpu = gpu_count()
    if sys.platform == "linux" and n_gpu == 0:
        problems.append("no NVIDIA GPU found (nvidia-smi missing or lists none)")
    for tool, hint in (("git", "sudo apt-get install -y git"), ("ffmpeg", "sudo apt-get install -y ffmpeg"),
                       ("ffprobe", "sudo apt-get install -y ffmpeg")):
        if not shutil.which(tool):
            problems.append(f"{tool} not on PATH ({hint}); NVIDIA's data loader shells out to ffmpeg/ffprobe")
    nproc = gpus or n_gpu or 8
    if model == "nano" and nproc < 8:
        echo(f"note: Cosmos3-Nano is set up for 8 GPUs; with {nproc} it may run out of memory "
             f"(Cosmos3-Edge fits fewer: --model edge)")
    if problems and not dry_run:
        raise CheckError("cannot run training here:\n  - " + "\n  - ".join(problems))
    for p in problems:
        echo(f"[dry-run] would stop here: {p}")
    return nproc


def _train(dataset: Path, model: str, full: bool, iters: int | None, save_every: int | None, run_name: str | None,
           sample: bool | None, sample_seconds: float, sample_clips: int, sample_camera: str | None,
           compare_base: bool, gpus: int, cuda: str, workdir: Path, cosmos_ref: str, tokenizer: str,
           skip_loader_check: bool, dry_run: bool) -> int:
    if model not in MODELS:
        raise CheckError(f"--model must be one of {list(MODELS)}")
    group = {"13.0": "cu130-train", "12.8": "cu128-train"}.get(cuda)
    if not group:
        raise CheckError("--cuda must be 13.0 or 12.8")
    if save_every and not full:
        raise CheckError("--save-every only applies to --full runs")
    ds = dataset_dir(dataset)
    mode = "full" if full else "smoke"
    iters_eff = iters or (None if full else 5)
    L = Layout(workdir.expanduser().resolve(), model, run_name or f"so101_{mode}_{model}")
    do_sample = full if sample is None else sample
    nproc = _preflight(model, gpus, dry_run, L.work)

    free_gb = shutil.disk_usage(L.work if L.work.exists() else L.work.parent).free / 1e9
    smoke_ck = latest_checkpoint(L.out_root / "cosmos3" / "sft" / f"so101_smoke_{model}")
    if full and smoke_ck:
        per_ck = dir_size(smoke_ck) / 1e9
        n_saves = math.ceil((iters_eff or 500) / (save_every or 100))
        need = per_ck * n_saves
        echo(f"note: each checkpoint is ~{per_ck:.0f} GB (measured from your smoke run); {n_saves} saves = "
             f"~{need:.0f} GB, {free_gb:.0f} GB free")
        if need > free_gb * 0.9:
            echo(f"      that won't fit: use fewer saves, e.g. --save-every {iters_eff or 500}")
        speed = None
        for lg in sorted(L.logs.glob("*.log")):
            speed = speed or iteration_seconds(lg.read_text(encoding="utf-8", errors="replace"))
        if speed:
            echo(f"note: ~{speed:.0f} s per iteration in your smoke run -> {iters_eff or 500} iterations "
                 f"take roughly {(iters_eff or 500) * speed / 3600:.1f} h")
    elif free_gb < 150:
        echo(f"note: {free_gb:.0f} GB free in {L.work.parent}; base checkpoints, the DCP conversion and "
             f"training checkpoints can take 100+ GB")

    if dry_run and not L.repo.exists():
        uv_pin = "0.11.29"  # what the pinned commit requires; read from pyproject.toml for real runs
    else:
        if not dry_run:
            fetch_cosmos(L.repo, cosmos_ref, full=True)
        uv_pin = required_uv(L.repo) or ""
    uvx = shutil.which("uvx") or "uvx"
    uv_cmd = [uvx, "--no-config", f"uv@{uv_pin}"] if uv_pin else [uv_exe()]
    vae = L.repo / "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
    dcp = L.repo / "examples/checkpoints" / L.ckpt_name
    toml_name = f"{L.run_name}.toml"
    launcher = L.repo / "examples" / f"{L.run_name}.sh"
    venv = L.venv_env()

    steps = [
        Step(f"Fetch cosmos-framework @ {cosmos_ref[:12]}", ["git", "fetch", "--depth", "1", COSMOS_GIT, cosmos_ref],
             L.repo, skip_reason="done above" if not dry_run else "", slug="fetch"),
        Step(f"Training environment (uv sync, {group})", [*uv_cmd, "sync", "--all-extras", f"--group={group}"],
             L.repo, skip_reason="already synced" if (L.venv_bin / "torchrun").exists() else "", slug="sync"),
        Step("Wan2.2 VAE", [uvx, "--no-config", "hf@latest", "download", "Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth",
                            "--local-dir", str(vae.parent), "--quiet"], L.work,
             skip_reason="already downloaded" if vae.is_file() else "", slug="vae"),
        Step(f"{L.ckpt_name} -> DCP checkpoint", ["python", "-m", "cosmos_framework.scripts.convert_model_to_dcp",
                                                 "--checkpoint-path", L.ckpt_name, "-o", str(dcp)], L.repo, venv,
             skip_reason="already converted" if dcp.is_dir() and any(dcp.iterdir()) else "", slug="dcp"),
    ]
    if not skip_loader_check:
        steps.append(Step("Data-loader check (NVIDIA's SFT loader, every clip)",
                          [str(L.venv_bin / "python"), str(CHECK_SCRIPT), "--repo", str(L.repo), "--dataset", str(ds),
                           "--tokenizer", tokenizer, "--report", str(ds / "cosmos_loader_check.json")], None,
                          {**venv, "PYTHONPATH": str(L.repo), "PYTHONWARNINGS": "ignore"}, slug="loader"))
    resume_from = latest_checkpoint(L.run_dir) if full else None
    recipe_toml = L.repo / "examples/toml/sft_config" / f"vision_sft_{model}.toml"
    final_iter = iters_eff or (tomllib.loads(recipe_toml.read_text(encoding="utf-8"))["trainer"]["max_iter"]
                               if recipe_toml.exists() else 500)
    done_iter = int(resume_from.name.rsplit("_", 1)[-1]) if resume_from else 0
    desc = (f"SFT {mode} run: {L.ckpt_name}, {final_iter} iterations on {nproc} GPU(s), "
            f"run '{L.run_name}'" + (f", resuming from {resume_from.name}" if resume_from else ""))
    train_env = {**venv, "DATASET_PATH": str(ds), "NPROC_PER_NODE": str(nproc), "MASTER_PORT": str(free_port()),
                 "OUTPUT_ROOT": str(L.out_root)}
    train_step = Step(desc, ["bash", str(launcher.relative_to(L.repo))], L.repo, train_env, slug=f"train_{L.run_name}",
                      skip_reason=f"already finished ({resume_from.name})" if full and done_iter >= final_iter else "")
    steps.append(train_step)
    if do_sample:
        steps += sample_steps(L, ds, seconds=sample_seconds, clips=sample_clips, camera=sample_camera,
                              compare_base=compare_base)

    echo(f"\nDataset: {ds}\nWork dir: {L.work}\nRun dir: {L.run_dir}\n")
    train_ok = True
    for i, st in enumerate(steps, 1):
        tag = f"[{i}/{len(steps)}] {st.name}"
        if st.skip_reason:
            echo(f"{tag}: skipped ({st.skip_reason})")
            continue
        if dry_run:
            env_s = " ".join(f"{k}={shlex.quote(v)}" for k, v in st.env.items() if k not in ("PATH", "VIRTUAL_ENV"))
            echo(f"{tag}\n    {('cd ' + str(st.cwd) + ' && ') if st.cwd else ''}{env_s + ' ' if env_s else ''}{shlex.join(st.cmd)}")
            continue
        echo(f"\n{tag}")
        if st.action:
            st.action()
            continue
        if st is train_step:
            base = (L.repo / "examples/toml/sft_config" / f"vision_sft_{model}.toml").read_text(encoding="utf-8")
            text = (full_toml(base, name=L.run_name, iters=iters_eff, save_every=save_every) if full
                    else smoke_toml(base, name=L.run_name, iters=iters_eff))
            (L.repo / "examples/toml/sft_config" / toml_name).write_text(text, encoding="utf-8")
            launcher.write_text(LAUNCHER.format(model=L.ckpt_name, mode=mode, toml_name=toml_name, ckpt=L.ckpt_name),
                                encoding="utf-8")
            launcher.chmod(0o755)
            if not full and L.run_dir.exists():
                shutil.rmtree(L.run_dir)   # a smoke run always starts fresh (it would otherwise resume at its end)
        cmd = st.cmd
        if cmd[0] == "python":
            cmd = [str(L.venv_bin / "python"), *cmd[1:]]
        log = L.logs / f"{st.slug or f'step{i}'}.log"
        rc, out = run(cmd, cwd=st.cwd, env=clean_env(st.env), log=log)
        if st is train_step:
            train_ok, lines = judge_training(rc, out, iters_eff or 0, full=full)
            echo("\n" + ("PASS" if train_ok else "FAIL") + f": Cosmos SFT {mode} run on {ds.name}")
            for ln in lines:
                echo(f"  {ln}")
            echo(f"  run dir: {L.run_dir}\n  log: {log}")
            if not train_ok:
                if full:
                    echo("  Rerun the same command to resume from the last saved checkpoint.")
                return 1
            continue
        if rc != 0:
            raise CheckError(f"step '{st.name}' failed (exit {rc}); log: {log}")
    if dry_run:
        echo("\nDry run only: nothing was downloaded or executed.")
    return 0 if train_ok else 1


# --------------------------------------------------------------------------- sample
@app.command("sample")
def sample_cmd(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Converted dataset folder (contains val/).")],
    model: Annotated[str, typer.Option("--model", help="Model of the training run: nano, edge or super.")] = "edge",
    run_name: Annotated[str | None, typer.Option("--run-name", help="Training run to sample from (default so101_full_<model>).")] = None,
    checkpoint_iter: Annotated[int | None, typer.Option("--checkpoint-iter", help="Use checkpoints/iter_<N> instead of the latest.")] = None,
    seconds: Annotated[float, typer.Option("--seconds", help="Length of each validation video.")] = 5.0,
    clips: Annotated[int, typer.Option("--clips", help="How many held-out episodes to generate.")] = 1,
    camera: Annotated[str | None, typer.Option("--camera", help="Camera of the validation clips (default: realsense, else any RGB view).")] = None,
    compare_base: Annotated[bool, typer.Option("--compare-base/--no-compare-base", help="Also generate with the base model for a side-by-side comparison.")] = True,
    workdir: Annotated[Path, typer.Option("--workdir", help="The --workdir used for training.")] = Path("cosmos_work"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the steps without running them.")] = False,
) -> None:
    """Generate validation videos from a trained checkpoint (export + image-to-video + side-by-side)."""
    try:
        if model not in MODELS:
            raise CheckError(f"--model must be one of {list(MODELS)}")
        ds = dataset_dir(dataset)
        L = Layout(workdir.expanduser().resolve(), model, run_name or f"so101_full_{model}")
        ck = None
        if checkpoint_iter is not None:
            ck = L.run_dir / "checkpoints" / f"iter_{checkpoint_iter:09d}"
            if not ck.is_dir() and not dry_run:
                raise CheckError(f"{ck} not found")
        if not dry_run and not (L.venv_bin / "python").exists():
            raise CheckError(f"No training environment in {L.repo}; run `cosmos-check train` first.")
        steps = sample_steps(L, ds, seconds=seconds, clips=clips, camera=camera, compare_base=compare_base,
                             checkpoint=ck)
        for i, st in enumerate(steps, 1):
            echo(f"\n[{i}/{len(steps)}] {st.name}")
            if dry_run:
                echo(f"    cd {st.cwd} && {shlex.join(st.cmd)}")
            else:
                st.action()
        if dry_run:
            echo("\nDry run only: nothing was executed.")
    except CheckError as e:
        echo(f"Error: {e}")
        raise typer.Exit(2) from None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
