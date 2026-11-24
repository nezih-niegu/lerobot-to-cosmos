"""Synthetic fixtures: a LeRobot v3.0 dataset shaped like SO101cupsRGBD and a tiny
Cosmos-format reference shaped like nvidia/BridgeData2-Subset-Synthetic-Captions."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

# Use the ffmpeg bundled with imageio-ffmpeg (what users get), not a system override.
os.environ.pop("IMAGEIO_FFMPEG_EXE", None)

from lerobot_to_cosmos.media import run_ffmpeg

EPISODE_SECONDS = [20, 15, 8, 1.5, 70, 25, 18]  # 8 s -> fps raised to 10; 1.5 s -> dropped; 70 s -> 2 clips
EPISODE_FILE = [0, 0, 0, 0, 1, 1, 1]
TASK = "Pick up the cup and place it on the plate"
DEPTH_KEY = "observation.images.realsense_depth"
CAMERAS = {  # key: (w, h, channels) -- same 4 streams as SO101cupsRGBD, at a small size for speed
    "observation.images.realsense": (160, 120, 3),
    DEPTH_KEY: (160, 120, 1),
    "observation.images.logitech": (160, 90, 3),
    "observation.images.usb": (160, 90, 3),
}
DEPTH_INFO = {"is_depth_map": True, "video.codec": "hevc", "video.pix_fmt": "gray12le", "video.depth_min": 0.01,
              "video.depth_max": 10.0, "video.shift": 3.5, "video.use_log": True}
PATCH = (slice(40, 80), slice(60, 100))  # rows, cols of the per-frame depth marker


def patch_depth_m(global_frame: int) -> float:
    """Depth of the marker patch in source frame n of a depth file (unique per frame within 200)."""
    return 0.3 + (global_frame % 200) * 0.005


def _testsrc(path: Path, w: int, h: int, fps: float, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps:g}:duration={seconds}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)])


def _depth_video(path: Path, w: int, h: int, fps: int, seconds: float) -> None:
    """Lossless 12-bit HEVC, quantized exactly like LeRobot (log space, metres)."""
    import av
    import numpy as np

    from lerobot_to_cosmos.depth import DepthQuant

    quant = DepthQuant()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.full((h, w), 0.9)          # table plane at 0.9 m
    base[:, : w // 8] = 1.4              # far wall strip
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx265", rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "gray12le"
        st.options = {"x265-params": "lossless=1:log-level=error", "preset": "ultrafast"}
        for n in range(round(seconds * fps)):
            d = base.copy()
            d[PATCH] = patch_depth_m(n)
            q = quant.to_code(d)
            q[-10:, -10:] = 0                # no-reading corner
            frame = av.VideoFrame(w, h, "gray12le")
            plane = frame.planes[0]
            buf = np.zeros((h, plane.line_size // 2), np.uint16)
            buf[:, :w] = q
            plane.update(buf.tobytes())
            for pkt in st.encode(frame):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


@pytest.fixture(scope="session")
def lerobot_src(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("so101_fake")
    feats = {k: {"dtype": "video", "shape": [h, w, c]} for k, (w, h, c) in CAMERAS.items()}
    feats[DEPTH_KEY]["info"] = DEPTH_INFO
    feats["action"] = {"dtype": "float32", "shape": [6]}
    feats["observation.state"] = {"dtype": "float32", "shape": [6]}
    info = {"codebase_version": "v3.0", "robot_type": "so_follower", "fps": 30, "features": feats,
            "total_episodes": len(EPISODE_SECONDS), "total_frames": round(sum(EPISODE_SECONDS) * 30),
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows, offset = [], {0: 0.0, 1: 0.0}
    for ep, (d, f) in enumerate(zip(EPISODE_SECONDS, EPISODE_FILE)):
        r = {"episode_index": ep, "tasks": [TASK], "length": round(d * 30)}
        for k in CAMERAS:
            r |= {f"videos/{k}/chunk_index": 0, f"videos/{k}/file_index": f,
                  f"videos/{k}/from_timestamp": offset[f], f"videos/{k}/to_timestamp": offset[f] + d}
        offset[f] += d
        rows.append(r)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame({"action": [[0.0] * 6]}).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    for k, (w, h, c) in CAMERAS.items():
        for f, total in offset.items():
            path = root / f"videos/{k}/chunk-000/file-{f:03d}.mp4"
            if c == 1:
                _depth_video(path, w, h, 30, total)
            else:
                _testsrc(path, w, h, 30, total)
    return root


@pytest.fixture(scope="session")
def bridge_ref(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("bridge_fake")
    base = root / "sft_dataset_bridge"
    uid = "episode_000015_clip000"
    _testsrc(base / "train" / "videos" / f"{uid}.mp4", 256, 256, 5, 17.4)
    cap = "A black robotic arm moves beads across a white tray."
    cj = {"subjects": [{"description": "A black robotic arm"}], "temporal_caption": cap,
          "resolution": {"H": 256, "W": 256}, "aspect_ratio": "1,1", "duration": "17s", "fps": 5}
    for split in ("train", "val"):
        d = base / split / "captions" / uid
        d.mkdir(parents=True)
        (d / "caption.txt").write_text(cap, encoding="utf-8")
        (d / "caption.json").write_text(json.dumps(cj), encoding="utf-8")
    rec = {"uuid": uid, "duration": 17.4, "width": 256, "height": 256, "vision_path": f"videos/{uid}.mp4",
           "t2w_windows": [{"start_frame": 0, "end_frame": 86, "temporal_interval": 1,
                            "caption_json": cj, "caption": cap}]}
    (base / "train" / "video_dataset_file.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    for d in ("videos", "videos_5frames"):
        (base / "val" / d).mkdir(parents=True)
        shutil.copy(base / "train" / "videos" / f"{uid}.mp4", base / "val" / d / f"{uid}.mp4")
    (base / "val" / "images").mkdir()
    run_ffmpeg(["-i", str(base / "val" / "videos" / f"{uid}.mp4"), "-frames:v", "1", str(base / "val" / "images" / f"{uid}.jpg")])
    common = {"prompt": json.dumps(cj), "resolution": "256", "aspect_ratio": "1,1", "num_frames": 93, "fps": 5}
    for d, kind, vp in (("inference_prompt", "t2v", None), ("inference_prompt_i2v", "i2v", f"../images/{uid}.jpg"),
                        ("inference_prompt_v2v", "v2v", f"../videos_5frames/{uid}.mp4")):
        (base / "val" / d).mkdir()
        rec = {"name": f"{kind}/{uid}", **common} | ({"vision_path": vp} if vp else {})
        (base / "val" / d / f"{uid}.json").write_text(json.dumps(rec, indent=4), encoding="utf-8")
    (root / "README.md").write_text("fake bridge\n", encoding="utf-8")
    return root


@pytest.fixture
def fake_hub(monkeypatch, lerobot_src, bridge_ref):
    """Serve the two default Hugging Face links from the local fixtures; record what is downloaded."""
    import huggingface_hub
    from huggingface_hub.utils import filter_repo_objects

    repos = {"omarcavazosgzz/SO101cupsRGBD": lerobot_src,
             "nvidia/BridgeData2-Subset-Synthetic-Captions": bridge_ref}
    downloaded: dict[str, set[str]] = {k: set() for k in repos}

    def listing(repo_id):
        root = repos[repo_id]
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    def copy(repo_id, rel, local_dir):
        dst = Path(local_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repos[repo_id] / rel, dst)
        downloaded[repo_id].add(rel)
        return str(dst)

    def list_repo_files(repo_id, revision=None, repo_type=None, token=None):
        assert repo_type == "dataset"
        return listing(repo_id)

    def hf_hub_download(repo_id, filename, repo_type=None, revision=None, local_dir=None, **kw):
        return copy(repo_id, filename, local_dir)

    def snapshot_download(repo_id, repo_type=None, revision=None, local_dir=None, allow_patterns=None, **kw):
        for rel in filter_repo_objects(listing(repo_id), allow_patterns=allow_patterns):
            copy(repo_id, rel, local_dir)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", list_repo_files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return downloaded
