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
def smoke_toml(base_toml: str, *, name: str, iters: int) -> str:
    """Recipe TOML -> short smoke run (what NVIDIA's tests/vision_sft_nano_5iter.toml changes)."""
    edits = {("job", "name"): json.dumps(name), ("trainer", "max_iter"): str(iters),
             ("checkpoint", "save_iter"): str(iters), ("scheduler", "warm_up_steps"): "[1]"}
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
    parsed = tomllib.loads(text)
    assert parsed["trainer"]["max_iter"] == iters and parsed["checkpoint"]["save_iter"] == iters
    return text


LAUNCHER = """#!/usr/bin/env bash
# Written by lerobot-to-cosmos `cosmos-check train`: {model} vision SFT smoke run on a converted dataset.
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


def judge_training(rc: int, log_text: str, iters: int) -> tuple[bool, list[str]]:
    losses = parse_losses(log_text)
    lines, ok = [], True
    if rc != 0:
        ok = False
        lines.append(f"training exited with code {rc}")
    if "Done with training" not in log_text:
        ok = False
        lines.append("no 'Done with training' line in the log")
    if len(losses) != iters:
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


@app.command()
def train(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Converted dataset folder (contains train/).")],
    model: Annotated[str, typer.Option("--model", help="nano (Cosmos3-Nano, 8 GPUs recommended), edge (Cosmos3-Edge, fits 4) or super.")] = "nano",
    gpus: Annotated[int, typer.Option("--gpus", help="GPUs to use (default: all that nvidia-smi lists).")] = 0,
    iters: Annotated[int, typer.Option("--iters", help="Training iterations for the smoke run.")] = 5,
    cuda: Annotated[str, typer.Option("--cuda", help="CUDA build of the training env: 13.0 (driver >= 580) or 12.8.")] = "13.0",
    workdir: Annotated[Path, typer.Option("--workdir", help="Where cosmos-framework, checkpoints and outputs go.")] = Path("cosmos_work"),
    cosmos_ref: Annotated[str, typer.Option("--cosmos-ref", help="cosmos-framework commit ('main' = latest).")] = PINNED_REF,
    tokenizer: Annotated[str, typer.Option("--tokenizer", help="Tokenizer for the loader pre-check.")] = QWEN_TOKENIZER,
    skip_loader_check: Annotated[bool, typer.Option("--skip-loader-check", help="Don't run the data-loader check before training.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print every step without running anything.")] = False,
) -> None:
    """Short Cosmos SFT run on the dataset (Linux + NVIDIA GPUs), following NVIDIA's documented recipe."""
    try:
        code = _train(dataset, model, gpus, iters, cuda, workdir, cosmos_ref, tokenizer, skip_loader_check, dry_run)
    except CheckError as e:
        echo(f"Error: {e}")
        raise typer.Exit(2) from None
    raise typer.Exit(code)


def _train(dataset: Path, model: str, gpus: int, iters: int, cuda: str, workdir: Path, cosmos_ref: str,
           tokenizer: str, skip_loader_check: bool, dry_run: bool) -> int:
    if model not in MODELS:
        raise CheckError(f"--model must be one of {list(MODELS)}")
    group = {"13.0": "cu130-train", "12.8": "cu128-train"}.get(cuda)
    if not group:
        raise CheckError("--cuda must be 13.0 or 12.8")
    ds = dataset_dir(dataset)
    work = workdir.expanduser().resolve()
    repo = work / "cosmos-framework"
    ckpt_name = MODELS[model]

    # Preflight: what NVIDIA's setup guide requires.
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
             f"(Cosmos3-Edge fits 4: --model edge)")
    free_gb = shutil.disk_usage(work if work.exists() else work.parent).free / 1e9
    if free_gb < 150:
        echo(f"note: {free_gb:.0f} GB free in {work.parent}; base checkpoints, the DCP conversion and "
             f"training checkpoints can take 100+ GB")
    if problems and not dry_run:
        raise CheckError("cannot run training here:\n  - " + "\n  - ".join(problems))
    for p in problems:
        echo(f"[dry-run] would stop here: {p}")

    if dry_run and not repo.exists():
        uv_pin = "0.11.29"  # what the pinned commit requires; read from pyproject.toml for real runs
    else:
        if not dry_run:
            fetch_cosmos(repo, cosmos_ref, full=True)
        uv_pin = required_uv(repo) or ""
    uvx = shutil.which("uvx") or "uvx"
    uv_cmd = [uvx, "--no-config", f"uv@{uv_pin}"] if uv_pin else [uv_exe()]
    venv_bin = repo / ".venv" / "bin"
    vae = repo / "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
    dcp = repo / "examples/checkpoints" / ckpt_name
    toml_name = f"so101_smoke_{model}.toml"
    launcher = repo / "examples" / f"so101_smoke_{model}.sh"
    out_root = work / "train_out" / model
    logs = work / "logs"
    venv_env = {"PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}", "VIRTUAL_ENV": str(repo / ".venv"),
                "PYTHONPATH": "."}

    steps = [
        Step(f"Fetch cosmos-framework @ {cosmos_ref[:12]}", ["git", "fetch", "--depth", "1", COSMOS_GIT, cosmos_ref], repo,
             skip_reason="done above" if not dry_run else ""),
        Step(f"Training environment (uv sync, {group})", [*uv_cmd, "sync", "--all-extras", f"--group={group}"], repo,
             skip_reason="already synced" if (venv_bin / "torchrun").exists() else ""),
        Step("Wan2.2 VAE", [uvx, "--no-config", "hf@latest", "download", "Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth",
                            "--local-dir", str(vae.parent), "--quiet"], work,
             skip_reason="already downloaded" if vae.is_file() else ""),
        Step(f"{ckpt_name} -> DCP checkpoint", ["python", "-m", "cosmos_framework.scripts.convert_model_to_dcp",
                                               "--checkpoint-path", ckpt_name, "-o", str(dcp)], repo, venv_env,
             skip_reason="already converted" if dcp.is_dir() and any(dcp.iterdir()) else ""),
    ]
    if not skip_loader_check:
        steps.append(Step("Data-loader check (NVIDIA's SFT loader, every clip)",
                          [str(venv_bin / "python"), str(CHECK_SCRIPT), "--repo", str(repo), "--dataset", str(ds),
                           "--tokenizer", tokenizer, "--report", str(ds / "cosmos_loader_check.json")], None,
                          {**venv_env, "PYTHONPATH": str(repo), "PYTHONWARNINGS": "ignore"}))
    train_env = {**venv_env, "DATASET_PATH": str(ds), "NPROC_PER_NODE": str(nproc), "MASTER_PORT": str(free_port()),
                 "OUTPUT_ROOT": str(out_root)}
    steps.append(Step(f"SFT smoke run: {ckpt_name}, {iters} iterations on {nproc} GPU(s)",
                      ["bash", str(launcher.relative_to(repo))], repo, train_env))

    echo(f"\nDataset: {ds}\nWork dir: {work}\n")
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
        if st is steps[-1]:
            base = (repo / "examples/toml/sft_config" / f"vision_sft_{model}.toml").read_text(encoding="utf-8")
            (repo / "examples/toml/sft_config" / toml_name).write_text(
                smoke_toml(base, name=f"so101_smoke_{model}", iters=iters), encoding="utf-8")
            launcher.write_text(LAUNCHER.format(model=ckpt_name, toml_name=toml_name, ckpt=ckpt_name), encoding="utf-8")
            launcher.chmod(0o755)
        cmd = st.cmd
        if cmd[0] == "python":
            cmd = [str(venv_bin / "python"), *cmd[1:]]
        rc, out = run(cmd, cwd=st.cwd, env=clean_env(st.env), log=logs / f"step{i}.log")
        if st is steps[-1]:
            ok, lines = judge_training(rc, out, iters)
            echo("\n" + ("PASS" if ok else "FAIL") + f": Cosmos SFT smoke run on {ds.name}")
            for ln in lines:
                echo(f"  {ln}")
            echo(f"  log: {logs / f'step{i}.log'}")
            return 0 if ok else 1
        if rc != 0:
            raise CheckError(f"step '{st.name}' failed (exit {rc}); log: {logs / f'step{i}.log'}")
    if dry_run:
        echo("\nDry run only: nothing was downloaded or executed.")
    return 0


def main() -> None:
    app()


if __name__ == "__main__":
    main()
