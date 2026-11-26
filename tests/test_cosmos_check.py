"""Tests for `cosmos-check`. The pure-Python parts always run; the ones that execute NVIDIA's
loader need a Python with its dependencies and a cosmos-framework checkout:

    COSMOS_CHECK_PYTHON=/path/to/python COSMOS_REPO=/path/to/cosmos-framework \
    COSMOS_TEST_TOKENIZER=Qwen/Qwen3-VL-8B-Instruct uv run pytest tests/test_cosmos_check.py
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lerobot_to_cosmos import cosmos_check as cc
from lerobot_to_cosmos import cosmos_loader_check as lc
from lerobot_to_cosmos.cli import app as convert_app

runner = CliRunner()

RECIPE_PY = '''
from x import L, PackingDataLoader, get_sft_dataset, create_tok
cfg = dict(
    dataloader_train=L(PackingDataLoader)(
        max_sequence_length=45056, patch_spatial=2,
        tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4,
        dataloader=dict(datasets=dict(video=dict(dataset=L(get_sft_dataset)(
            resolution="256", num_video_frames=-1, temporal_interval_mode="max_30fps",
            frame_selection_mode="first", conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
            conditioning_fps=-1, cfg_dropout_rate=0.1, max_caption_tokens=1024,
            jsonl_paths=["${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl"],
            tokenizer_config="${model.config.vlm_config.tokenizer}",
        )))),
    ),
)
other = L(get_sft_dataset)(resolution="480")   # later variant: must be ignored
'''

RECIPE_TOML = '''
[job]
task         = "vfm"
name         = "vision_sft_nano"
wandb_mode   = "disabled"

[scheduler]
cycle_lengths      = [1000]
warm_up_steps      = [50]

[trainer]
grad_accum_iter         = 2
max_iter                = 500

[checkpoint]
load_path            = "${oc.env:BASE_CHECKPOINT_PATH}"
save_iter            = 100

[dataloader_train]
max_sequence_length = 45056
max_caption_tokens = 2048   # comment
'''


@pytest.fixture
def fake_repo(tmp_path) -> Path:
    (tmp_path / lc.RECIPE_DIR).mkdir(parents=True)
    (tmp_path / lc.RECIPE_DIR / "vision_sft_nano.py").write_text(RECIPE_PY)
    (tmp_path / lc.TOML_DIR).mkdir(parents=True)
    (tmp_path / lc.TOML_DIR / "vision_sft_nano.toml").write_text(RECIPE_TOML)
    return tmp_path


def test_recipe_settings_read_from_source(fake_repo):
    s = lc.recipe_settings(fake_repo, "vision_sft_nano")
    ds, pack = s["dataset"], s["packing"]
    assert ds["resolution"] == "256" and ds["num_video_frames"] == -1           # first call in source order
    assert ds["conditioning_config"] == {0: 0.7, 1: 0.2, 2: 0.1}
    assert ds["max_caption_tokens"] == 2048                                     # TOML override wins
    assert pack["max_sequence_length"] == 45056 and pack["patch_spatial"] == 2


def test_smoke_toml_matches_nvidia_smoke_edits():
    text = cc.smoke_toml(RECIPE_TOML, name="so101_smoke_nano", iters=5)
    t = tomllib.loads(text)
    assert t["job"]["name"] == "so101_smoke_nano"
    assert t["trainer"]["max_iter"] == 5 and t["checkpoint"]["save_iter"] == 5
    assert t["scheduler"]["warm_up_steps"] == [1]
    assert t["trainer"]["grad_accum_iter"] == 2 and t["dataloader_train"]["max_caption_tokens"] == 2048
    assert "# comment" in text


def test_smoke_toml_detects_upstream_changes():
    with pytest.raises(cc.CheckError):
        cc.smoke_toml("[job]\nname = 'x'\n", name="n", iters=5)


def test_judge_training_log():
    log = "[RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 0.2302 | Time: 1s\n[RANK 1] Iteration 1: Hit counter: 1/50 | Loss: 0.9999 | Time: 1s\nIteration 2: Hit counter: 2/50 | Loss: 0.2100 | Time: 1s\nSaved checkpoint to /out/checkpoints/iter_000000002\nDone with training."
    ok, lines = cc.judge_training(0, log, iters=2)
    assert ok and "0.2302, 0.2100" in lines[0] and "trend down" in lines[0]
    ok, lines = cc.judge_training(0, log.replace("0.2100", "nan"), iters=2)
    assert not ok and any("non-finite" in ln for ln in lines)
    ok, lines = cc.judge_training(1, "", iters=5)
    assert not ok and any("exit" in ln for ln in lines)


@pytest.mark.skipif(sys.platform != "linux", reason="exercises the Linux preflight")
def test_train_dry_run_and_preflight(tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    (ds / "train").mkdir(parents=True)
    (ds / "train" / "video_dataset_file.jsonl").write_text("{}\n")
    monkeypatch.setattr(cc, "gpu_count", lambda: 0)
    work = tmp_path / "work"
    r = runner.invoke(cc.app, ["train", "-d", str(ds), "--workdir", str(work), "--dry-run"])
    assert r.exit_code == 0, r.output
    for needle in ("uv@0.11.29", "sync --all-extras --group=cu130-train", "Wan2.2_VAE.pth",
                   "convert_model_to_dcp --checkpoint-path Cosmos3-Nano", "cosmos_loader_check.py",
                   f"DATASET_PATH={ds}", "NPROC_PER_NODE=8", "bash examples/so101_smoke_nano.sh",
                   "would stop here: no NVIDIA GPU", "Dry run only"):
        assert needle in r.output, needle
    assert not work.exists()                                   # dry run touches nothing

    r = runner.invoke(cc.app, ["train", "-d", str(ds), "--workdir", str(work)])
    assert r.exit_code == 2 and "no NVIDIA GPU" in r.output and not (work / "cosmos-framework").exists()

    r = runner.invoke(cc.app, ["train", "-d", str(ds), "--model", "edge", "--cuda", "12.8", "--gpus", "4",
                               "--workdir", str(work), "--dry-run"])
    assert "Cosmos3-Edge" in r.output and "cu128-train" in r.output and "NPROC_PER_NODE=4" in r.output


def test_rejects_non_dataset(tmp_path):
    r = runner.invoke(cc.app, ["loader", "-d", str(tmp_path), "--python", sys.executable, "--cosmos-repo", str(tmp_path)])
    assert r.exit_code == 2 and "not a converted dataset" in r.output


# --------------------------------------------------------------------------- NVIDIA's real loader
COSMOS_PY = os.environ.get("COSMOS_CHECK_PYTHON")
COSMOS_REPO = os.environ.get("COSMOS_REPO")
needs_cosmos = pytest.mark.skipif(not (COSMOS_PY and COSMOS_REPO),
                                  reason="set COSMOS_CHECK_PYTHON and COSMOS_REPO to run NVIDIA's loader")


@needs_cosmos
def test_nvidia_loader_accepts_converted_dataset(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    r = runner.invoke(convert_app, ["--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                                    "--val-episodes", "1"])
    assert r.exit_code == 0, r.output
    tok = os.environ.get("COSMOS_TEST_TOKENIZER", cc.QWEN_TOKENIZER)
    r = runner.invoke(cc.app, ["loader", "-d", str(out), "--python", COSMOS_PY, "--cosmos-repo", COSMOS_REPO,
                               "--tokenizer", tok])
    print(r.output)
    assert r.exit_code == 0
    rep = json.loads((out / "cosmos_loader_check.json").read_text())
    assert rep["jsonl_records"] == rep["kept_by_loader"] == rep["checked"] == rep["passed"] == 28
    assert rep["iter_samples"] == 3 and not rep["errors"]
    fps = {s["uuid"][:14]: s["fps"] for s in rep["samples"]}
    assert fps["episode_000002"] == 10.0 and fps["episode_000000"] == 5.0   # raised-fps clip read correctly
    assert (out / "cosmos_loader_preview.png").stat().st_size > 1000

    # After structured captions are added (+ --rebuild-only), the loader must train on caption_json.
    cj = {"subjects": [{"description": "A white SO-101 arm"}], "temporal_caption": "The arm stacks red cups.",
          "resolution": {"H": 256, "W": 256}, "aspect_ratio": "1,1", "fps": 5}
    for d in (out / "train" / "captions").iterdir():
        (d / "caption.json").write_text(json.dumps(cj))
    assert runner.invoke(convert_app, ["--out", str(out), "--reference", str(bridge_ref), "--rebuild-only"]).exit_code == 0
    r = runner.invoke(cc.app, ["loader", "-d", str(out), "--python", COSMOS_PY, "--cosmos-repo", COSMOS_REPO,
                               "--tokenizer", tok, "--max-samples", "4"])
    assert r.exit_code == 0, r.output
    rep = json.loads((out / "cosmos_loader_check.json").read_text())
    assert rep["caption_styles"] == {"caption_json": 4}


@needs_cosmos
def test_nvidia_loader_catches_broken_dataset(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    assert runner.invoke(convert_app, ["--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                                       "-c", "observation.images.realsense", "--max-episodes", "2",
                                       "--val-episodes", "1"]).exit_code == 0
    victim = min((out / "train" / "videos").glob("*.mp4"))
    victim.write_bytes(victim.read_bytes()[:2000])          # truncated / corrupt video
    tok = os.environ.get("COSMOS_TEST_TOKENIZER", cc.QWEN_TOKENIZER)
    r = runner.invoke(cc.app, ["loader", "-d", str(out), "--python", COSMOS_PY, "--cosmos-repo", COSMOS_REPO,
                               "--tokenizer", tok])
    print(r.output)
    assert r.exit_code == 1 and "FAIL" in r.output and victim.stem in r.output


def test_missing_dataset_suggests_the_right_path(tmp_path, monkeypatch):
    proj = tmp_path / "Projects" / "lerobot-to-cosmos"
    (proj / "sft_dataset_so101" / "train").mkdir(parents=True)
    (proj / "sft_dataset_so101" / "train" / "video_dataset_file.jsonl").write_text("{}\n")
    monkeypatch.chdir(proj)
    r = runner.invoke(cc.app, ["train", "-d", "../sft_dataset_so101", "--dry-run"])
    assert r.exit_code == 2 and "doesn't exist" in r.output and "did you mean -d sft_dataset_so101" in r.output


# --------------------------------------------------------------------------- full runs + validation videos
def test_full_toml_keeps_recipe_and_stretches_lr_cycle():
    t = tomllib.loads(cc.full_toml(RECIPE_TOML, name="so101_full_edge", iters=None, save_every=None))
    assert t["job"]["name"] == "so101_full_edge"
    assert t["trainer"]["max_iter"] == 500 and t["checkpoint"]["save_iter"] == 100
    assert t["scheduler"]["warm_up_steps"] == [50] and t["scheduler"]["cycle_lengths"] == [1000]
    t = tomllib.loads(cc.full_toml(RECIPE_TOML, name="n", iters=2000, save_every=500))
    assert t["trainer"]["max_iter"] == 2000 and t["checkpoint"]["save_iter"] == 500
    assert t["scheduler"]["cycle_lengths"] == [2000]                  # LR still decays to the end


def test_iteration_seconds_and_full_judge():
    log = "\n".join(f"[RANK 0] Iteration {i}: Hit counter: {i}/50 | Loss: {0.3 - i/100:.4f} | Time: {t}s"
                    for i, t in enumerate([90.0, 40.0, 12.5, 12.0, 13.0, 12.2], 1))
    assert cc.iteration_seconds(log) == pytest.approx(12.5)
    full_log = log + "\nSaved checkpoint to /o/iter_000000500\nDone with training."
    ok, lines = cc.judge_training(0, full_log, 500, full=True)     # only 6 logged lines for 500 iters: fine
    assert ok and "checkpoint: /o/iter_000000500" in lines


def _ds(tmp_path):
    ds = tmp_path / "ds"
    (ds / "train").mkdir(parents=True)
    (ds / "train" / "video_dataset_file.jsonl").write_text("{}\n")
    (ds / "val" / "inference_prompt_i2v").mkdir(parents=True)
    return ds


@pytest.mark.skipif(sys.platform != "linux", reason="exercises the Linux preflight")
def test_full_dry_run_plans_training_export_and_videos(tmp_path, monkeypatch):
    ds = _ds(tmp_path)
    monkeypatch.setattr(cc, "gpu_count", lambda: 1)
    work = tmp_path / "work"
    r = runner.invoke(cc.app, ["train", "-d", str(ds), "--model", "edge", "--full", "--workdir", str(work), "--dry-run"])
    print(r.output)
    assert r.exit_code == 0
    for needle in ("SFT full run: Cosmos3-Edge, 500 iterations on 1 GPU(s), run 'so101_full_edge'",
                   "bash examples/so101_full_edge.sh", "export_model", "--no-vit", "Validation video",
                   "(base vs fine-tuned)", "--model-mode image2video", "Run dir: " + str(work)):
        assert needle in r.output, needle
    assert not work.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="exercises the Linux preflight")
def test_full_run_resumes_and_skips_when_finished(tmp_path, monkeypatch):
    ds = _ds(tmp_path)
    monkeypatch.setattr(cc, "gpu_count", lambda: 1)
    L = cc.Layout(tmp_path / "work", "edge", "so101_full_edge")
    ck = L.run_dir / "checkpoints"
    (ck / "iter_000000200").mkdir(parents=True)
    (ck / "latest_checkpoint.txt").write_text("iter_000000200")
    args = ["train", "-d", str(ds), "--model", "edge", "--full", "--workdir", str(L.work), "--dry-run"]
    r = runner.invoke(cc.app, args)
    assert "resuming from iter_000000200" in r.output and "skipped (already finished" not in r.output
    (ck / "iter_000000500").mkdir()
    (ck / "latest_checkpoint.txt").write_text("iter_000000500")
    r = runner.invoke(cc.app, args)
    assert "skipped (already finished (iter_000000500))" in r.output and "Validation video" in r.output


def test_sample_dry_run_and_errors(tmp_path):
    ds = _ds(tmp_path)
    r = runner.invoke(cc.app, ["sample", "-d", str(ds), "--workdir", str(tmp_path / "w"), "--dry-run", "--clips", "2"])
    assert r.exit_code == 0 and "export_model" in r.output and "Validation videos" in r.output
    r = runner.invoke(cc.app, ["sample", "-d", str(ds), "--workdir", str(tmp_path / "w")])
    assert r.exit_code == 2 and "run `cosmos-check train` first" in r.output


def test_validation_prompts_and_side_by_side(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    assert runner.invoke(convert_app, ["--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                                       "--val-episodes", "1"]).exit_code == 0
    files = cc.pick_val_prompts(out, 5, None)
    assert files and all(f.stem.endswith("_realsense") for f in files)           # RGB realsense preferred
    assert all(f.stem.endswith("_logitech") for f in cc.pick_val_prompts(out, 5, "logitech"))
    with pytest.raises(cc.CheckError):
        cc.pick_val_prompts(out, 1, "thermal")

    items = cc.prepare_prompts(files[:1], out, tmp_path / "prompts", seconds=5)
    it = items[0]
    rec = json.loads((tmp_path / "prompts" / f"{it['uuid']}.json").read_text())
    assert rec["num_frames"] == it["num_frames"] == 25 and (rec["num_frames"] - 1) % 4 == 0   # 5 s at 5 fps
    assert Path(rec["vision_path"]).is_absolute() and Path(rec["vision_path"]).is_file()
    assert rec["name"] == it["uuid"] and rec["fps"] == 5

    # stand-ins for the two generated videos: any clips of the same rate
    gen = sorted((out / "train" / "videos").glob("*_logitech.mp4"))[:2]
    combo = tmp_path / "cmp.mp4"
    cc.compose_comparison([("real episode", it["gt"]), ("base", gen[0]), ("fine-tuned", gen[1])],
                          it["fps"], it["num_frames"], combo)
    from lerobot_to_cosmos.media import probe
    m = probe(combo)
    assert (m["width"], m["height"], m["total_frames"]) == (768, 256, 25)
