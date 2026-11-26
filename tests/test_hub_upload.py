"""--push-to-hub / --upload-only, with a stand-in `hf` CLI and a stubbed whoami (no network)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lerobot_to_cosmos.cli import app
from lerobot_to_cosmos.hub_upload import CARD_MARKER

runner = CliRunner()
TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7"
FAKE_HF = """#!{python}
# Stand-in for the `hf` CLI: a tiny Hub kept in $FAKE_HF_REMOTE (one folder = one repo's files).
import fnmatch, json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_HF_LOG"], "a") as f:
    f.write(json.dumps({{"argv": args, "token": os.environ.get("HF_TOKEN")}}) + "\\n")
remote = Path(os.environ["FAKE_HF_REMOTE"])
def positional(rest):
    out = []
    for a in rest:
        if a.startswith("--"):
            break
        out.append(a)
    return out
def opt(name):
    return args[args.index(name) + 1] if name in args else None
if args[0] == "upload":
    if os.environ.get("FAKE_HF_FAIL"):
        print("401 Unauthorized for token " + os.environ.get("HF_TOKEN", "")); sys.exit(1)
    repo, local, dest = (positional(args[1:]) + ["."])[:3]
    local, root = Path(local), remote / repo
    if local.is_file():
        (root / dest).parent.mkdir(parents=True, exist_ok=True); shutil.copy(local, root / dest)
    else:
        for p in local.rglob("*"):
            if p.is_file() and p.name != ".DS_Store":
                t = root / dest / p.relative_to(local); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, t)
elif args[0] == "download":
    repo, name = positional(args[1:])[:2]
    src = remote / repo / name
    if not src.is_file():
        print("404 Entry Not Found: " + name); sys.exit(1)
    dst = Path(opt("--local-dir")) / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
elif args[:2] == ["repos", "delete-files"]:
    repo, *patterns = positional(args[2:])
    root = remote / repo
    for p in list(root.rglob("*")):
        if p.is_file() and any(fnmatch.fnmatch(p.relative_to(root).as_posix(), pat) for pat in patterns):
            p.unlink()
elif args[:2] == ["repos", "create"]:
    (remote / positional(args[2:])[0]).mkdir(parents=True, exist_ok=True)
print("ok")
"""


def whoami_info(role="fineGrained", perms=("repo.content.read", "repo.write"), name="me", orgs=()):
    access = {"displayName": "so101-upload", "role": role}
    if role == "fineGrained":
        access["fineGrained"] = {"scoped": [{"entity": {"type": "user", "name": name}, "permissions": list(perms)}],
                                 "global": []}
    return {"name": name, "orgs": [{"name": o} for o in orgs], "auth": {"accessToken": access}}


@pytest.fixture
def fake_hf(tmp_path, monkeypatch):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    exe = bindir / "hf"
    exe.write_text(FAKE_HF.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "hf_calls.jsonl"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_HF_LOG", str(log))
    remote = tmp_path / "fake_hub"
    remote.mkdir()
    monkeypatch.setenv("FAKE_HF_REMOTE", str(remote))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    import huggingface_hub

    state = {"info": whoami_info(), "license": "apache-2.0"}

    class Card(dict):
        pass

    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", lambda self, token=None: state["info"])

    def list_repo_files(self, repo_id, repo_type=None, token=None, **kw):
        root = remote / repo_id
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", list_repo_files)
    state["remote"] = remote
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info",
                        lambda self, repo_id, token=None: type("I", (), {"card_data": Card(license=state["license"])})())

    def calls():
        return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []

    return state, calls


@pytest.fixture(scope="module")
def converted(tmp_path_factory, lerobot_src, bridge_ref) -> Path:
    out = tmp_path_factory.mktemp("conv") / "sft_dataset_so101"
    r = runner.invoke(app, ["--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                            "-c", "observation.images.realsense", "-c", "observation.images.realsense_depth",
                            "--max-episodes", "2", "--val-episodes", "1"])
    assert r.exit_code == 0, r.output
    return out


def copy_of(converted: Path, tmp_path: Path) -> Path:
    dst = tmp_path / converted.name
    shutil.copytree(converted, dst)
    return dst


def test_push_after_conversion_prompts_for_token(tmp_path, fake_hf, lerobot_src, bridge_ref):
    _state, calls = fake_hf
    out = tmp_path / "sft_dataset_so101"
    r = runner.invoke(app, ["--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                            "-c", "observation.images.realsense", "--max-episodes", "2", "--val-episodes", "1",
                            "--push-to-hub", "me/so101-cups-cosmos"],
                      input=f"not-a-token\n{TOKEN}\n")
    print(r.output)
    assert r.exit_code == 0
    assert "fine-grained" in r.output and "doesn't look like a Hugging Face token" in r.output
    assert TOKEN not in r.output                                           # never echoed
    c = calls()
    assert [x["argv"][:2] for x in c] == [["repos", "create"], ["upload", "me/so101-cups-cosmos"]]
    assert all(x["token"] == TOKEN for x in c)                             # passed via env ...
    assert all(TOKEN not in " ".join(x["argv"]) and "--token" not in x["argv"] for x in c)   # ... never argv
    assert "--private" in c[0]["argv"] and "dataset" in c[0]["argv"]
    up = c[1]["argv"]
    assert up[2:4] == [str(out), "."] and "--repo-type" in up and ".DS_Store" in up
    card = (out / "README.md").read_text()
    assert card.startswith("---\ntags:") and CARD_MARKER in card
    assert "license:" not in card                                          # local source: no card to inherit from
    assert "| Training clips | 2 " in card and "hf download me/so101-cups-cosmos" in card
    assert "https://huggingface.co/datasets/me/so101-cups-cosmos" in r.output


def test_upload_only_with_env_token_public(tmp_path, fake_hf, converted, monkeypatch):
    state, calls = fake_hf
    out = copy_of(converted, tmp_path)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    state["info"] = whoami_info(name="me", orgs=["my-lab"])
    r = runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", "my-lab/so101-cosmos", "--public"])
    print(r.output)
    assert r.exit_code == 0 and "HF_TOKEN environment variable" in r.output
    c = calls()
    assert "--public" in c[0]["argv"] and c[1]["argv"][1] == "my-lab/so101-cosmos"
    assert "Converting" not in r.output and "[reference]" not in r.output   # nothing reconverted
    card = (out / "README.md").read_text()
    assert "`realsense_depth` clips are rendered depth" in card


@pytest.mark.parametrize("info, repo, message", [
    (whoami_info(name="me"), "someone-else/x", "neither your account"),
    (whoami_info(role="read"), "me/x", "read-only"),
    (whoami_info(perms=("repo.content.read",)), "me/x", "no write permission"),
])
def test_token_problems_stop_before_uploading(tmp_path, fake_hf, converted, monkeypatch, info, repo, message):
    state, calls = fake_hf
    state["info"] = info
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    r = runner.invoke(app, ["--out", str(copy_of(converted, tmp_path)), "--upload-only", "--push-to-hub", repo])
    assert r.exit_code == 2 and message in r.output
    assert calls() == []


def test_classic_write_token_warns_but_works(tmp_path, fake_hf, converted, monkeypatch):
    state, calls = fake_hf
    state["info"] = whoami_info(role="write")
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    r = runner.invoke(app, ["--out", str(copy_of(converted, tmp_path)), "--upload-only", "--push-to-hub", "me/x"])
    assert r.exit_code == 0 and "not a fine-grained one" in r.output and len(calls()) == 2


def test_hf_failure_is_reported_without_leaking_token(tmp_path, fake_hf, converted, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("FAKE_HF_FAIL", "1")
    r = runner.invoke(app, ["--out", str(copy_of(converted, tmp_path)), "--upload-only", "--push-to-hub", "me/x"])
    assert r.exit_code == 2 and "'hf upload' failed" in r.output
    assert TOKEN not in r.output and "hf_***" in r.output


def test_refuses_failed_conversion_and_keeps_user_readme(tmp_path, fake_hf, converted, monkeypatch):
    _state, calls = fake_hf
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    out = copy_of(converted, tmp_path)
    (out / "README.md").write_text("# my own card\n")
    r = runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", "me/x"])
    assert r.exit_code == 0 and "Keeping your own README.md" in r.output
    assert (out / "README.md").read_text() == "# my own card\n"

    s = json.loads((out / "conversion_summary.json").read_text())
    s["validation"]["errors"] = ["boom"]
    (out / "conversion_summary.json").write_text(json.dumps(s))
    n = len(calls())
    r = runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", "me/x"])
    assert r.exit_code == 2 and "failed validation" in r.output and len(calls()) == n


@pytest.mark.parametrize("args, message", [
    (["--upload-only"], "--upload-only needs --push-to-hub"),
    (["--push-to-hub", "not a repo"], "dataset id like"),
    (["--push-to-hub", "me/x", "--check-only"], "can't be combined with --check-only"),
])
def test_argument_errors(tmp_path, fake_hf, args, message):
    r = runner.invoke(app, ["--out", str(tmp_path / "o"), *args])
    assert r.exit_code == 2 and message in r.output


def test_three_bad_tokens_give_up(tmp_path, fake_hf, converted):
    r = runner.invoke(app, ["--out", str(copy_of(converted, tmp_path)), "--upload-only", "--push-to-hub", "me/x"],
                      input="a\nb\nc\n")
    assert r.exit_code == 2 and "No valid Hugging Face token" in r.output


def test_card_inherits_license_from_hub_source(tmp_path, fake_hf, converted, monkeypatch):
    state, _calls = fake_hf
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    out = copy_of(converted, tmp_path)
    s = json.loads((out / "conversion_summary.json").read_text())
    s["source"] = "https://huggingface.co/datasets/omarcavazosgzz/SO101cupsRGBD"
    (out / "conversion_summary.json").write_text(json.dumps(s))
    r = runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", "me/x"])
    assert r.exit_code == 0
    card = (out / "README.md").read_text()
    assert "license: apache-2.0" in card and "[omarcavazosgzz/SO101cupsRGBD]" in card
    state["license"] = None                                                # source card without a license
    assert runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", "me/x"]).exit_code == 0
    assert "license:" not in (out / "README.md").read_text()               # regenerated card, still ours


# --------------------------------------------------------------------------- several versions in one repo
@pytest.fixture(scope="module")
def converted_30fps(tmp_path_factory, lerobot_src, bridge_ref) -> Path:
    out = tmp_path_factory.mktemp("conv30") / "sft_dataset_so101_30fps"
    r = runner.invoke(app, ["--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                            "-c", "observation.images.realsense", "-c", "observation.images.realsense_depth",
                            "--max-episodes", "2", "--val-episodes", "1", "--fps", "0"])
    assert r.exit_code == 0, r.output
    return out


def remote_files(state, repo):
    root = state["remote"] / repo
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_two_versions_share_one_repo(tmp_path, fake_hf, converted, converted_30fps, monkeypatch):
    state, _calls = fake_hf
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    repo = "me/so101-cups-cosmos"
    for out, sub in ((copy_of(converted, tmp_path / "a"), "5fps"), (copy_of(converted_30fps, tmp_path / "b"), "30fps")):
        r = runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", repo, "--path-in-repo", sub])
        print(r.output)
        assert r.exit_code == 0 and f"/tree/main/{sub}" in r.output

    files = remote_files(state, repo)
    for sub in ("5fps", "30fps"):
        assert f"{sub}/train/video_dataset_file.jsonl" in files and f"{sub}/val/inference_prompt" in " ".join(files)
    assert not any(f.startswith(("train/", "val/")) for f in files)          # nothing at the root but the card
    a = {f.split("/", 1)[1] for f in files if f.startswith("5fps/")}
    b = {f.split("/", 1)[1] for f in files if f.startswith("30fps/")}
    assert a == b                                                            # same layout, separate copies

    index = (state["remote"] / repo / "README.md").read_text()
    assert index.startswith("---\ntags:") and CARD_MARKER in index
    assert "[`30fps/`]" in index and "[`5fps/`]" in index
    assert "256x256, 5 fps" in index and "256x256, 30 fps" in index
    assert '--include "30fps/*"' in index                                    # versions sorted: 30fps first
    sub_card = (state["remote"] / repo / "30fps" / "README.md").read_text()
    assert not sub_card.startswith("---") and "# so101-cups-cosmos / 30fps" in sub_card
    assert "DATASET_PATH=$PWD/so101-cups-cosmos/30fps" in sub_card


def test_root_upload_then_folders_needs_clean_root(tmp_path, fake_hf, converted, converted_30fps, monkeypatch):
    state, _calls = fake_hf
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    repo = "me/x"
    five, thirty = copy_of(converted, tmp_path / "a"), copy_of(converted_30fps, tmp_path / "b")
    assert runner.invoke(app, ["--out", str(five), "--upload-only", "--push-to-hub", repo]).exit_code == 0
    assert "train/video_dataset_file.jsonl" in remote_files(state, repo)     # earlier, flat upload

    r = runner.invoke(app, ["--out", str(thirty), "--upload-only", "--push-to-hub", repo, "--path-in-repo", "30fps"])
    assert r.exit_code == 2 and "--clean-root" in r.output
    assert not any(f.startswith("30fps/") for f in remote_files(state, repo))   # stopped before uploading

    for out, sub in ((five, "5fps"), (thirty, "30fps")):
        r = runner.invoke(app, ["--out", str(out), "--upload-only", "--push-to-hub", repo, "--path-in-repo", sub,
                                "--clean-root"])
        print(r.output)
        assert r.exit_code == 0
    files = remote_files(state, repo)
    assert not any(f.startswith(("train/", "val/")) or f == "conversion_summary.json" for f in files)
    assert "5fps/train/video_dataset_file.jsonl" in files and "30fps/train/video_dataset_file.jsonl" in files
    assert "[`5fps/`]" in (state["remote"] / repo / "README.md").read_text()


def test_own_root_readme_is_kept(tmp_path, fake_hf, converted, monkeypatch):
    state, _calls = fake_hf
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    (state["remote"] / "me/y").mkdir(parents=True)
    (state["remote"] / "me/y/README.md").write_text("# hand-written card\n")
    r = runner.invoke(app, ["--out", str(copy_of(converted, tmp_path)), "--upload-only", "--push-to-hub", "me/y",
                            "--path-in-repo", "5fps"])
    assert r.exit_code == 0 and "Keeping the repo's own README.md" in r.output
    assert (state["remote"] / "me/y/README.md").read_text() == "# hand-written card\n"


@pytest.mark.parametrize("args, message", [
    (["--path-in-repo", "30fps"], "only apply with --push-to-hub"),
    (["--push-to-hub", "me/x", "--path-in-repo", "../up"], "--path-in-repo must be a folder name"),
])
def test_path_in_repo_argument_errors(tmp_path, fake_hf, args, message):
    r = runner.invoke(app, ["--out", str(tmp_path / "o"), *args])
    assert r.exit_code == 2 and message in r.output
