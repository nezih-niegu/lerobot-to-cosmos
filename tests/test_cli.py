from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest
from typer.testing import CliRunner

from lerobot_to_cosmos.cli import app
from lerobot_to_cosmos.depth import DepthQuant, render
from lerobot_to_cosmos.hub import DatasetLinkError, Repo
from lerobot_to_cosmos.media import probe, run_ffmpeg

from .conftest import CAMERAS, DEPTH_KEY, patch_depth_m

runner = CliRunner()
CAM_NAMES = [k.split(".")[-1] for k in CAMERAS]


def invoke(*args: str):
    result = runner.invoke(app, list(args))
    print(result.output)
    return result


def jsonl(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "train" / "video_dataset_file.jsonl").read_text().splitlines()]


def summary(out: Path) -> dict:
    return json.loads((out / "conversion_summary.json").read_text())


def frames(path: Path, n: int, fmt: str = "rgb24") -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format=fmt).astype(float) for _, f in zip(range(n), c.decode(video=0))]


# --------------------------------------------------------------------------- links
@pytest.mark.parametrize("link, repo_id, rev, sub", [
    ("https://huggingface.co/datasets/omarcavazosgzz/SO101cupsRGBD", "omarcavazosgzz/SO101cupsRGBD", None, ""),
    (" https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions/", "nvidia/BridgeData2-Subset-Synthetic-Captions", None, ""),
    ("https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions/tree/main/sft_dataset_bridge",
     "nvidia/BridgeData2-Subset-Synthetic-Captions", "main", "sft_dataset_bridge"),
    ("omarcavazosgzz/SO101cupsRGBD", "omarcavazosgzz/SO101cupsRGBD", None, ""),
])
def test_parse_links(tmp_path, link, repo_id, rev, sub):
    r = Repo(link, tmp_path)
    assert (r.repo_id, r.revision, r.subdir) == (repo_id, rev, sub)


def test_bad_link(tmp_path):
    with pytest.raises(DatasetLinkError):
        Repo("https://example.com/not-a-dataset", tmp_path)


# --------------------------------------------------------------------------- default run: links, all episodes, all cameras
def test_default_links_whole_dataset_all_cameras(tmp_path, fake_hub):
    out = tmp_path / "out"
    r = invoke("--out", str(out), "--download-dir", str(tmp_path / "dl"), "--val-episodes", "1")
    assert r.exit_code == 0
    assert "NOT COMPATIBLE" in r.output and "Validation against nvidia/BridgeData2-Subset-Synthetic-Captions: PASS" in r.output

    src_dl = fake_hub["omarcavazosgzz/SO101cupsRGBD"]
    assert not any(f.startswith("data/") for f in src_dl)                       # robot tables not needed
    for key in CAMERAS:                                                          # all 4 camera streams fetched
        assert any(f.startswith(f"videos/{key}/") for f in src_dl)
    assert len(fake_hub["nvidia/BridgeData2-Subset-Synthetic-Captions"]) < 15    # only a reference sample

    rows = jsonl(out)
    # 7 episodes x 4 cameras: the 1.5 s episode is dropped, the 70 s one becomes 2 clips -> 7 clips per camera
    assert len(rows) == 28
    for cam in CAM_NAMES:
        assert sum(r["uuid"].endswith(f"_{cam}") for r in rows) == 7
    assert {int(r["uuid"][8:14]) for r in rows} == {0, 1, 2, 4, 5, 6}
    for row in rows:
        assert list(row) == ["uuid", "duration", "width", "height", "vision_path", "t2w_windows"]
        m = probe(out / "train" / row["vision_path"])
        assert (m["width"], m["height"]) == (256, 256)
        assert m["total_frames"] == row["t2w_windows"][0]["end_frame"] + 1 >= 61
        expected_fps = 10 if row["uuid"].startswith("episode_000002_") else 5   # 8 s episode keeps 61+ frames
        assert round(m["fps"]) == expected_fps

    s = summary(out)
    assert s["episodes_exported"] == 6 and len(s["dropped_before_export"]) == 4
    val_ids = {p.stem for p in (out / "val" / "videos").glob("*.mp4")}
    assert val_ids and val_ids <= {r["uuid"] for r in rows}                     # val copies; every episode trains
    assert len(val_ids) % 4 == 0                                                 # all cameras of the val episode
    assert sorted(p.name for p in (out / "val").iterdir()) == sorted(
        ["captions", "images", "inference_prompt", "inference_prompt_i2v", "inference_prompt_v2v", "videos", "videos_5frames"])
    v2v = json.loads(next((out / "val" / "inference_prompt_v2v").glob("*.json")).read_text())
    assert v2v["name"].startswith("v2v/") and v2v["vision_path"].startswith("../videos_5frames/")
    depth_caption = next((out / "train" / "captions").glob("*_realsense_depth")) / "caption.txt"
    assert "depth map" in depth_caption.read_text()


def test_val_holdout(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    assert invoke("--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                  "--val-episodes", "1", "--val-holdout").exit_code == 0
    train = {r["uuid"] for r in jsonl(out)}
    val = {p.stem for p in (out / "val" / "videos").glob("*.mp4")}
    assert val and not (train & val) and len(train | val) == 28


def test_check_only(tmp_path, fake_hub):
    out = tmp_path / "out"
    r = invoke("--out", str(out), "--download-dir", str(tmp_path / "dl"), "--check-only")
    assert r.exit_code == 0 and "NOT COMPATIBLE" in r.output and "Depth:" in r.output
    assert not out.exists()
    assert all(f.startswith("meta/") for f in fake_hub["omarcavazosgzz/SO101cupsRGBD"])


def test_already_cosmos(tmp_path, bridge_ref):
    r = invoke("--out", str(tmp_path / "o"), "--source", str(bridge_ref), "--reference", str(bridge_ref))
    assert r.exit_code == 0 and "COMPATIBLE: the source already" in r.output


# --------------------------------------------------------------------------- frame accuracy
def test_rgb_frames_are_exact(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    assert invoke("--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                  "-c", "observation.images.realsense", "--val-episodes", "1").exit_code == 0
    # episode 4, clip 1 starts 35 s into file-001 -> source frame 1050; then every 6th frame.
    src = lerobot_src / "videos/observation.images.realsense/chunk-000/file-001.mp4"
    vf = "scale=256:256:force_original_aspect_ratio=increase,crop=256:256"

    def src_frame(n: int) -> np.ndarray:
        png = tmp_path / f"s{n}.png"
        run_ffmpeg(["-i", str(src), "-vf", f"select=eq(n\\,{n}),{vf}", "-frames:v", "1", str(png)])
        return frames(png, 1)[0]

    for i, frame in enumerate(frames(out / "train/videos/episode_000004_clip001.mp4", 2)):
        exact = np.abs(frame - src_frame(1050 + 6 * i)).mean()
        neighbour = np.abs(frame - src_frame(1050 + 6 * i + 2)).mean()
        assert exact < neighbour * 0.75


def test_depth_frames_and_values(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    r = invoke("--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref), "-c", DEPTH_KEY,
               "--resize-mode", "stretch", "--depth-near", "0.25", "--depth-far", "1.5", "--val-episodes", "1")
    assert r.exit_code == 0
    quant = DepthQuant()

    def expected_gray(depth_m: float) -> float:
        return float(render(quant.to_code(np.array([[depth_m]])), quant, 0.25, 1.5)[0, 0])

    clip = frames(out / "train/videos/episode_000004_clip001.mp4", 3, fmt="gray")
    for i, f in enumerate(clip):
        got = f[100:160, 110:150].mean()                      # inside the per-frame marker patch
        want = expected_gray(patch_depth_m(1050 + 6 * i))
        prev = expected_gray(patch_depth_m(1050 + 6 * i - 6))
        assert abs(got - want) < 1.5 and abs(got - want) < abs(got - prev)
    assert clip[0][245:254, 245:254].mean() < 8                # no-reading corner renders black
    assert abs(clip[0][100:160, 200:240].mean() - expected_gray(0.9)) < 1.5   # table plane at 0.9 m


def test_depth_auto_range(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    assert invoke("--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref), "-c", DEPTH_KEY,
                  "--max-episodes", "2", "--val-episodes", "1").exit_code == 0
    d = summary(out)["settings"]["depth"][DEPTH_KEY]
    assert 0.2 < d["near_m"] < 0.9 and 1.3 < d["far_m"] < 1.8
    assert d["quantization"] == {"depth_min": 0.01, "depth_max": 10.0, "shift": 3.5, "use_log": True}


# --------------------------------------------------------------------------- options
def test_rgb_cameras_keep_native_fps(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    r = invoke("--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
               "-c", "all-rgb", "--resize-mode", "keep", "--fps", "0", "--max-episodes", "2", "--val-episodes", "1")
    assert r.exit_code == 0
    names = sorted(p.stem for p in (out / "train" / "videos").glob("*.mp4"))
    assert names == sorted(f"episode_00000{e}_clip000_{c}" for e in (0, 1) for c in ("realsense", "logitech", "usb"))
    m = probe(out / "train" / "videos" / "episode_000001_clip000_logitech.mp4")
    assert (m["width"], m["height"], round(m["fps"]), m["total_frames"]) == (456, 256, 30, 450)


def test_rebuild_only_with_structured_captions(tmp_path, lerobot_src, bridge_ref):
    out = tmp_path / "out"
    assert invoke("--out", str(out), "--source", str(lerobot_src), "--reference", str(bridge_ref),
                  "--max-episodes", "2", "--val-episodes", "1").exit_code == 0
    cj = {"subjects": [{"description": "A white SO-101 arm"}], "temporal_caption": "The arm lifts a cup.",
          "resolution": {"H": 256, "W": 256}, "aspect_ratio": "1,1", "duration": "20s", "fps": 5}
    for split in ("train", "val"):
        for d in (out / split / "captions").iterdir():
            (d / "caption.json").write_text(json.dumps(cj))

    r = invoke("--out", str(out), "--reference", str(bridge_ref), "--rebuild-only")
    assert r.exit_code == 0 and "PASS" in r.output and "caption_json" not in r.output.split("PASS")[1]
    assert all(list(row["t2w_windows"][0]) == ["start_frame", "end_frame", "temporal_interval", "caption_json", "caption"]
               for row in jsonl(out))
    t2v = json.loads(next((out / "val" / "inference_prompt").glob("*.json")).read_text())
    assert t2v["prompt"] == json.dumps(cj, ensure_ascii=False)
