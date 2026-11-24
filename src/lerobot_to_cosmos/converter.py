"""LeRobot v3.0 -> Cosmos3 video-SFT conversion, driven by a reference dataset.

Target format: the one consumed by NVIDIA cosmos-framework's SFT loader
(cosmos_framework/data/generator/local_datasets/sft_dataset.py) and shipped in
nvidia/BridgeData2-Subset-Synthetic-Captions:

    <out>/train/{videos/, captions/<uuid>/caption.txt[, caption.json], video_dataset_file.jsonl}
    <out>/val/{videos/, videos_5frames/, images/, captions/,
               inference_prompt/, inference_prompt_i2v/, inference_prompt_v2v/}
"""

from __future__ import annotations

import json
import math
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .depth import DepthQuant, encode_depth_clip, estimate_range, is_depth_feature
from .hub import Repo
from .media import probe, run_ffmpeg

DEFAULT_SOURCE = "https://huggingface.co/datasets/omarcavazosgzz/SO101cupsRGBD"
DEFAULT_REFERENCE = "https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions"
DEFAULT_CAMERAS = ["all"]
DEFAULT_CAPTION_TEMPLATE = (
    "A single SO-101 robotic arm on a tabletop workspace performs the task: {task}. "
    "The arm moves with deliberate, controlled motions, grasping and repositioning objects "
    "with its gripper. The camera remains static throughout, recording the scene from a "
    "fixed {camera} viewpoint under even indoor lighting."
)
DEFAULT_DEPTH_CAPTION_TEMPLATE = (
    "A depth map video of a single SO-101 robotic arm on a tabletop workspace performing the task: {task}. "
    "The scene is rendered in grayscale, where brighter pixels are closer to the fixed {camera} camera, "
    "darker pixels are farther away, and black marks missing depth. The camera remains static throughout."
)

# Loader constants (sft_dataset.py)
MAX_DURATION_S = 61.0
MIN_WINDOW_FRAMES = 61
JSONL_NAME = "video_dataset_file.jsonl"
LOADER_RECORD_KEYS = ["uuid", "duration", "width", "height", "vision_path", "t2w_windows"]
LOADER_WINDOW_KEYS = ["start_frame", "end_frame", "temporal_interval"]
ASPECT_BUCKETS = [(0.65, "9,16"), (0.88, "3,4"), (1.16, "1,1"), (1.55, "4,3"), (math.inf, "16,9")]
PROMPT_DIRS_DEFAULT = {"inference_prompt": ("t2v", None), "inference_prompt_i2v": ("i2v", "images"),
                       "inference_prompt_v2v": ("v2v", "videos_5frames")}


class ConversionError(RuntimeError):
    pass


def log(msg: str = "") -> None:
    print(msg, flush=True)


@dataclass
class Options:
    out: Path
    source: str = DEFAULT_SOURCE
    reference: str = DEFAULT_REFERENCE
    download_dir: Path = Path("hf_downloads")
    check_only: bool = False
    rebuild_only: bool = False
    cameras: list[str] = field(default_factory=lambda: list(DEFAULT_CAMERAS))
    fps: float | None = None          # None = reference fps, 0 = keep source fps
    width: int | None = None
    height: int | None = None
    resize_mode: str = "crop"
    max_clip_seconds: float = 60.0
    val_ratio: float = 0.05
    val_episodes: int | None = None
    val_holdout: bool = False         # False: val episodes are also kept in train (whole dataset trains)
    max_episodes: int = 0
    caption_template: str = DEFAULT_CAPTION_TEMPLATE
    depth_caption_template: str = DEFAULT_DEPTH_CAPTION_TEMPLATE
    depth_near: float | None = None   # metres; None = estimated from the data
    depth_far: float | None = None
    crf: int = 18
    workers: int = 4
    seed: int = 0
    overwrite: bool = False
    overwrite_captions: bool = False


# --------------------------------------------------------------------------- small helpers
def aspect_bucket(w: int, h: int) -> str:
    r = w / h
    for limit, name in ASPECT_BUCKETS:
        if r < limit:
            return name
    return "16,9"


def scale_filter(w: int, h: int, mode: str) -> str:
    if mode == "crop":
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    if mode == "pad":
        return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
    if mode == "stretch":
        return f"scale={w}:{h}"
    if mode == "keep":
        s = min(w, h)
        return f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)'"
    raise ValueError(f"Unknown resize mode {mode!r}")


def camera_short(key: str) -> str:
    return key.split(".")[-1]


def caption_json_to_prompt(caption_json: dict) -> str:
    # Byte-identical to cosmos_framework.inference.structured_caption.caption_json_to_prompt
    return json.dumps(caption_json, ensure_ascii=False)


def g(x: float) -> str:
    return f"{x:g}"


# --------------------------------------------------------------------------- reference spec
@dataclass
class RefSpec:
    name: str
    width: int
    height: int
    fps: float
    val_num_frames: int
    record_keys: list[str]
    window_keys: list[str]
    val_dirs: list[str]
    prompt_keys: dict[str, list[str]]
    prompt_names: dict[str, str]
    prompt_vision: dict[str, str]
    caption_files: list[str]
    n_train: int


def load_reference(ref: Repo) -> RefSpec:
    log(f"\n[reference] {ref.name}")
    files = ref.files()
    jsonls = [f for f in files if f.endswith(f"train/{JSONL_NAME}")]
    if not jsonls:
        raise ConversionError(f"Reference has no train/{JSONL_NAME}; it is not a Cosmos video-SFT dataset.")
    jl = jsonls[0]
    pre = jl[: -len(f"train/{JSONL_NAME}")]

    def first(prefix: str, suffix: str = "") -> str | None:
        return next((f for f in files if f.startswith(prefix) and f.endswith(suffix)), None)

    sample_video = first(f"{pre}train/videos/", ".mp4")
    cap_any = first(f"{pre}train/captions/")
    sample_cap_dir = cap_any.rsplit("/", 1)[0] if cap_any else None
    val_dirs = sorted({f[len(f"{pre}val/"):].split("/")[0] for f in files
                       if f.startswith(f"{pre}val/") and "/" in f[len(f"{pre}val/"):]})
    prompt_samples = {d: first(f"{pre}val/{d}/", ".json") for d in val_dirs if d.startswith("inference_prompt")}

    wanted = [jl, *filter(None, [sample_video, *prompt_samples.values()])]
    if sample_cap_dir:
        wanted += [f for f in files if f.startswith(sample_cap_dir + "/")]
    log(f"  downloading {len(wanted)} sample files (not the whole dataset)")
    ref.fetch(wanted)
    root = ref.root

    lines = [ln for ln in (root / jl).read_text(encoding="utf-8").splitlines() if ln.strip()]
    rec = json.loads(lines[0])
    win = rec["t2w_windows"][0]
    width, height, fps = int(rec["width"]), int(rec["height"]), None
    if sample_video:
        m = probe(root / sample_video)
        fps, width, height = m["fps"], m["width"], m["height"]

    prompt_keys, prompt_names, prompt_vision, val_nf = {}, {}, {}, 93
    for d, f in prompt_samples.items():
        if not f:
            continue
        pj = json.loads((root / f).read_text(encoding="utf-8"))
        prompt_keys[d] = list(pj.keys())
        prompt_names[d] = str(pj.get("name", "t2v/")).split("/")[0]
        if pj.get("vision_path"):
            prompt_vision[d] = Path(pj["vision_path"]).parent.name
        val_nf = int(pj.get("num_frames", val_nf))
        if fps is None and "fps" in pj:
            fps = float(pj["fps"])
    if fps is None:
        raise ConversionError("Could not determine the reference frame rate.")
    caption_files = sorted(Path(f).name for f in files if sample_cap_dir and f.startswith(sample_cap_dir + "/"))

    spec = RefSpec(ref.name, width, height, round(fps, 3), val_nf, list(rec.keys()), list(win.keys()),
                   val_dirs, prompt_keys, prompt_names, prompt_vision, caption_files, len(lines))
    log(f"  layout: {pre or './'}train/ + val/{{{', '.join(val_dirs)}}}")
    log(f"  clips: {width}x{height} @ {g(spec.fps)} fps; {spec.n_train} training records")
    log(f"  JSONL record keys: {spec.record_keys}")
    log(f"  t2w_window keys:   {spec.window_keys}")
    log(f"  caption files per clip: {caption_files}")
    return spec


# --------------------------------------------------------------------------- source inspection
def inspect_source(src: Repo) -> dict:
    log(f"\n[source] {src.name}")
    files = src.files()
    if any(f.endswith(f"train/{JSONL_NAME}") for f in files):
        return {"format": "cosmos-sft"}
    info_f = next((f for f in files if f.endswith("meta/info.json")), None)
    if not info_f:
        raise ConversionError("Source has neither meta/info.json (LeRobot) nor train/video_dataset_file.jsonl (Cosmos).")
    pre = info_f[: -len("meta/info.json")]
    log("  downloading meta/ only")
    src.fetch_patterns([f"{pre}meta/*"])
    root = src.root / pre if pre else src.root
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    feats = info["features"]
    vids = {k: v for k, v in feats.items() if v.get("dtype") == "video"}
    depth = [k for k, v in vids.items() if is_depth_feature(v)]
    return {"format": f"lerobot-{info.get('codebase_version')}", "root": root, "prefix": pre, "info": info,
            "video_keys": list(vids), "depth_keys": depth,
            "rgb_keys": [k for k, v in vids.items() if v["shape"][-1] == 3 and k not in depth],
            "other_keys": [k for k, v in feats.items() if v.get("dtype") != "video"]}


def compatibility_report(s: dict, ref: RefSpec) -> list[str]:
    info, feats = s["info"], s["info"]["features"]
    cams = ", ".join(f"{k} {feats[k]['shape'][1]}x{feats[k]['shape'][0]}x{feats[k]['shape'][2]}"
                     for k in s["video_keys"])
    lines = [
        (f"Format: source is LeRobot {info.get('codebase_version')} (parquet tables + concatenated per-camera "
         f"videos); reference is a Cosmos video-SFT folder (one mp4 per clip + {JSONL_NAME})."),
        (f"Episodes vs clips: source has {info.get('total_episodes')} episodes / {info.get('total_frames')} frames "
         f"stored inside shared video files; reference has one file per clip."),
        f"Cameras: source has {len(s['video_keys'])} streams ({cams}); reference has 1 RGB view per clip.",
        f"Frame rate / size: source {info.get('fps')} fps; reference {ref.width}x{ref.height} @ {g(ref.fps)} fps.",
        (f"Text: source only has task strings (meta/tasks.parquet); reference needs per-clip captions "
         f"({', '.join(ref.caption_files) or 'caption.txt'}) and t2w_windows {ref.window_keys}."),
        f"Robot data: source columns {s['other_keys']} have no place in the reference format (dropped).",
    ]
    if s["depth_keys"]:
        lines.append(f"Depth: {s['depth_keys']} store 12-bit log-quantized depth, which the RGB reference format "
                     f"can't hold directly; they are converted to 8-bit grayscale depth videos (brighter = closer).")
    lines.append(f"Validation split: reference ships val/{{{', '.join(ref.val_dirs)}}}; source has none.")
    return lines


# --------------------------------------------------------------------------- episodes -> clip jobs
@dataclass
class ClipJob:
    uuid: str
    splits: tuple[str, ...]   # ("train",), ("val",) or ("train", "val")
    episode_index: int
    camera: str
    is_depth: bool
    src_video: Path
    start_s: float
    duration_s: float
    task: str
    src_fps: float
    fps: float                # output fps of this clip (0 = keep source fps)
    fps_raised: bool = False


def load_episodes(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not ep_files:
        raise ConversionError(f"No episode metadata under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    tp = root / "meta" / "tasks.parquet"
    return episodes, (pd.read_parquet(tp) if tp.exists() else pd.DataFrame())


def episode_task(row: pd.Series, tasks: pd.DataFrame) -> str:
    t = row.get("tasks")
    if t is not None and len(t) > 0:
        return str(t[0])
    if len(tasks):
        return str(tasks.index[0])
    return "manipulate the objects on the table"


def select_cameras(s: dict, requested: list[str]) -> list[str]:
    if requested == ["all"]:
        return list(s["video_keys"])
    if requested == ["all-rgb"]:
        return list(s["rgb_keys"])
    for c in requested:
        if c not in s["video_keys"]:
            raise ConversionError(f"Camera {c!r} not in source. Video keys: {s['video_keys']}")
        if c not in s["rgb_keys"] and c not in s["depth_keys"]:
            raise ConversionError(f"Camera {c!r} is neither RGB nor a depth map; can't convert it.")
    return list(requested)


def clip_fps(clip_s: float, src_fps: float, target_fps: float) -> tuple[float, bool] | None:
    """Output fps for a clip: the target, or the lowest exact decimation of the source that still
    yields the loader's 61-frame minimum. None if even the source fps is too short."""
    if target_fps <= 0:
        return (src_fps, False) if clip_s * src_fps >= MIN_WINDOW_FRAMES - 1e-6 else None
    if clip_s * target_fps >= MIN_WINDOW_FRAMES - 1e-6:
        return target_fps, False
    k = math.floor(clip_s * src_fps / MIN_WINDOW_FRAMES + 1e-9)
    return (src_fps / k, True) if k >= 1 else None


def build_jobs(root: Path, info: dict, episodes: pd.DataFrame, tasks: pd.DataFrame, cams: list[str],
               depth_keys: list[str], target_fps: float, o: Options) -> tuple[list[ClipJob], list[dict]]:
    src_fps = float(info["fps"])
    tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    ep_ids = episodes["episode_index"].tolist()
    if o.max_episodes:
        ep_ids = ep_ids[: o.max_episodes]
    multi = len(cams) > 1
    jobs, dropped = [], []
    for _, row in episodes[episodes["episode_index"].isin(ep_ids)].iterrows():
        ep = int(row["episode_index"])
        task = episode_task(row, tasks)
        splits = ("train",)
        for cam in cams:
            src = root / tpl.format(video_key=cam, chunk_index=int(row[f"videos/{cam}/chunk_index"]),
                                    file_index=int(row[f"videos/{cam}/file_index"]))
            t0, t1 = float(row[f"videos/{cam}/from_timestamp"]), float(row[f"videos/{cam}/to_timestamp"])
            dur = t1 - t0
            n = max(1, math.ceil(dur / o.max_clip_seconds))
            clip_len = dur / n
            for c in range(n):
                uuid = f"episode_{ep:06d}_clip{c:03d}" + (f"_{camera_short(cam)}" if multi else "")
                chosen = clip_fps(clip_len, src_fps, target_fps)
                if chosen is None:
                    dropped.append({"uuid": uuid, "reason": f"too_short ({clip_len:.2f}s has fewer than "
                                    f"{MIN_WINDOW_FRAMES} frames even at the source {g(src_fps)} fps)"})
                    continue
                fps, raised = chosen
                jobs.append(ClipJob(uuid, splits, ep, cam, cam in depth_keys, src, t0 + c * clip_len, clip_len,
                                    task, src_fps, fps, raised))

    # Val episodes are drawn from episodes that produced clips, so val/ is never empty.
    kept = sorted({j.episode_index for j in jobs})
    random.Random(o.seed).shuffle(kept)
    n_val = o.val_episodes if o.val_episodes is not None else max(1, round(len(kept) * o.val_ratio))
    val_set = set(kept[:n_val])
    for j in jobs:
        if j.episode_index in val_set:
            j.splits = ("val",) if o.val_holdout else ("train", "val")
    return jobs, dropped


def process_clip(job: ClipJob, out: Path, w: int, h: int, o: Options, depth_cfg: dict) -> dict:
    first = out / job.splits[0]
    vid = first / "videos" / f"{job.uuid}.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(o.crf), "-pix_fmt", "yuv420p"]
    step = max(1, round(job.src_fps / job.fps))
    exact = abs(job.src_fps / job.fps - step) < 1e-6
    if o.overwrite or not vid.exists():
        if job.is_depth:
            quant, near, far = depth_cfg[job.camera]
            if not exact:
                raise ConversionError(f"Depth clips need an integer fps decimation ({g(job.src_fps)} -> {g(job.fps)}).")
            vf = scale_filter(w, h, o.resize_mode) + ",setsar=1"
            encode_depth_clip(job.src_video, job.start_s, job.duration_s, job.src_fps, step, job.fps,
                              quant, near, far, vf, enc, vid)
        else:
            vf = scale_filter(w, h, o.resize_mode) + ",setsar=1"
            if exact:
                # exact decimation from the episode's first frame (ffmpeg's fps filter lands ~half a step late)
                pre = f"select=not(mod(n\\,{step})),setpts=N/({g(job.fps)}*TB)," if step > 1 else ""
            else:
                pre = f"fps={g(job.fps)},"
            run_ffmpeg(["-ss", f"{job.start_s:.6f}", "-i", str(job.src_video), "-t", f"{job.duration_s:.6f}",
                        "-vf", pre + vf, "-r", g(job.fps), "-an", *enc, "-movflags", "+faststart", str(vid)])

    template = o.depth_caption_template if job.is_depth else o.caption_template
    caption = template.format(task=job.task.strip().rstrip("."), camera=camera_short(job.camera)) + "\n"
    for split_name in job.splits:
        split = out / split_name
        v = split / "videos" / f"{job.uuid}.mp4"
        if v != vid and (o.overwrite or not v.exists()):
            v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vid, v)
        if split_name == "val":
            img, v5 = split / "images" / f"{job.uuid}.jpg", split / "videos_5frames" / f"{job.uuid}.mp4"
            img.parent.mkdir(parents=True, exist_ok=True)
            v5.parent.mkdir(parents=True, exist_ok=True)
            if o.overwrite or not img.exists():
                run_ffmpeg(["-i", str(v), "-frames:v", "1", "-q:v", "2", str(img)])
            if o.overwrite or not v5.exists():
                run_ffmpeg(["-i", str(v), "-frames:v", "5", "-an", *enc, str(v5)])
        cap_dir = split / "captions" / job.uuid
        cap_dir.mkdir(parents=True, exist_ok=True)
        if o.overwrite_captions or not (cap_dir / "caption.txt").exists():
            (cap_dir / "caption.txt").write_text(caption, encoding="utf-8")
    return {"uuid": job.uuid, "splits": list(job.splits), "episode_index": job.episode_index, "camera": job.camera,
            "kind": "depth" if job.is_depth else "rgb", "fps": round(job.fps, 4), "fps_raised": job.fps_raised,
            "source_video": str(job.src_video), "source_start_s": round(job.start_s, 4),
            "source_duration_s": round(job.duration_s, 4), "task": job.task}


# --------------------------------------------------------------------------- JSONL + prompts
def read_captions(cap_dir: Path) -> tuple[str, dict | None]:
    txt, js = cap_dir / "caption.txt", cap_dir / "caption.json"
    dense = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    cj = None
    if js.exists():
        try:
            cj = json.loads(js.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log(f"  WARN {cap_dir.name}: invalid caption.json ({e}); using caption.txt")
    if not dense and isinstance(cj, dict):
        dense = str(cj.get("temporal_caption", "")).strip()
    return dense, cj


def build_train_jsonl(out: Path) -> dict:
    split = out / "train"
    rows, drops = [], {}
    for vid in sorted((split / "videos").glob("*.mp4")):
        uuid = vid.stem
        dense, cj = read_captions(split / "captions" / uuid)
        if not dense and cj is None:
            drops[uuid] = "no_caption"
            continue
        m = probe(vid)
        if m["duration"] > MAX_DURATION_S:
            drops[uuid] = f"duration {m['duration']:.1f}s > {MAX_DURATION_S}s"
            continue
        if m["total_frames"] < MIN_WINDOW_FRAMES:
            drops[uuid] = f"{m['total_frames']} frames < {MIN_WINDOW_FRAMES}"
            continue
        win: dict = {"start_frame": 0, "end_frame": m["total_frames"] - 1, "temporal_interval": 1}
        if cj is not None:
            win["caption_json"] = cj
        if dense:
            win["caption"] = dense
        rows.append({"uuid": uuid, "duration": round(m["duration"], 3), "width": m["width"], "height": m["height"],
                     "vision_path": f"videos/{uuid}.mp4", "t2w_windows": [win]})
    split.mkdir(parents=True, exist_ok=True)
    with (split / JSONL_NAME).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"records": len(rows), "with_caption_json": sum("caption_json" in r["t2w_windows"][0] for r in rows),
            "dropped": drops}


def build_val_prompts(out: Path, ref: RefSpec) -> int:
    split = out / "val"
    layout = dict(PROMPT_DIRS_DEFAULT)
    if ref.prompt_names:
        layout = {d: (ref.prompt_names[d], ref.prompt_vision.get(d)) for d in ref.prompt_names}
    n = 0
    for vid in sorted((split / "videos").glob("*.mp4")):
        uuid = vid.stem
        dense, cj = read_captions(split / "captions" / uuid)
        prompt = caption_json_to_prompt(cj) if cj is not None else dense
        m = probe(vid)
        nf = min(ref.val_num_frames, (m["total_frames"] - 1) // 4 * 4 + 1)  # 4k+1 and fits the clip
        base = {"resolution": str(min(m["width"], m["height"])), "aspect_ratio": aspect_bucket(m["width"], m["height"]),
                "num_frames": nf, "fps": round(m["fps"])}
        for d, (kind, vis) in layout.items():
            rec = {"name": f"{kind}/{uuid}", "prompt": prompt, **base}
            if vis:
                rec["vision_path"] = f"../{vis}/{uuid}.{'jpg' if vis == 'images' else 'mp4'}"
            (split / d).mkdir(parents=True, exist_ok=True)
            (split / d / f"{uuid}.json").write_text(json.dumps(rec, indent=4, ensure_ascii=False), encoding="utf-8")
        n += 1
    return n


# --------------------------------------------------------------------------- validation
def validate(out: Path, ref: RefSpec) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    jl = out / "train" / JSONL_NAME
    if not jl.exists() or not jl.read_text(encoding="utf-8").strip():
        return [f"train/{JSONL_NAME} is missing or empty"], warnings
    recs = [json.loads(ln) for ln in jl.read_text(encoding="utf-8").splitlines() if ln.strip()]

    keys = set(recs[0].keys())
    if miss := [k for k in LOADER_RECORD_KEYS if k not in keys]:
        errors.append(f"JSONL records lack keys the Cosmos loader needs: {miss}")
    if miss := [k for k in ref.record_keys if k not in keys and k not in LOADER_RECORD_KEYS]:
        warnings.append(f"Reference records also carry {miss} (optional for the loader).")

    wk = set(recs[0]["t2w_windows"][0].keys())
    if miss := [k for k in LOADER_WINDOW_KEYS if k not in wk]:
        errors.append(f"t2w_windows lack keys the loader needs: {miss}")
    if not wk & {"caption", "caption_json"}:
        errors.append("t2w_windows have no caption")
    if "caption_json" in ref.window_keys and "caption_json" not in wk:
        warnings.append("Windows have 'caption' but no 'caption_json' (the reference has both). The Cosmos "
                        "loader falls back to 'caption'; add structured captions with caption_from_video and "
                        "--rebuild-only to match fully.")
    if extra := sorted(wk - set(ref.window_keys)):
        warnings.append(f"Window keys not in the reference: {extra}")

    if any((r["width"], r["height"]) != (ref.width, ref.height) for r in recs):
        warnings.append(f"Some clips are not {ref.width}x{ref.height} like the reference.")
    off_fps = [r["uuid"] for r in recs if abs(probe(out / "train" / r["vision_path"])["fps"] - ref.fps) > 1e-3]
    if off_fps:
        warnings.append(f"{len(off_fps)} of {len(recs)} clips are not {g(ref.fps)} fps like the reference "
                        f"(short episodes kept at a higher fps, or --fps was set); the Cosmos loader reads each "
                        f"clip's own fps.")
    for r in recs:
        if not (out / "train" / r["vision_path"]).exists():
            errors.append(f"Missing video {r['vision_path']}")

    vdir = out / "val"
    val_dirs = sorted(d.name for d in vdir.iterdir() if d.is_dir()) if vdir.exists() else []
    if val_dirs != ref.val_dirs:
        errors.append(f"val/ folders {val_dirs} != reference {ref.val_dirs}")
    for d, ref_keys in ref.prompt_keys.items():
        f = next((vdir / d).glob("*.json"), None) if (vdir / d).exists() else None
        if f is None:
            errors.append(f"No prompt files in val/{d}")
        elif miss := [k for k in ref_keys if k not in json.loads(f.read_text(encoding="utf-8"))]:
            errors.append(f"val/{d} prompt files lack {miss}")
    return errors, warnings


# --------------------------------------------------------------------------- orchestration
def convert(o: Options) -> int:
    """Run the whole pipeline. Returns a process exit code."""
    out = o.out
    summary_path = out / "conversion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    ref = load_reference(Repo(o.reference, o.download_dir))

    if not o.rebuild_only:
        src_repo = Repo(o.source, o.download_dir)
        s = inspect_source(src_repo)
        if s["format"] == "cosmos-sft":
            log("\nCOMPATIBLE: the source already uses the Cosmos video-SFT layout. Nothing to convert.")
            return 0
        if not s["format"].startswith("lerobot-v3"):
            raise ConversionError(f"Unsupported source format {s['format']} (LeRobot v3.x expected).")
        report = compatibility_report(s, ref)
        log("\nNOT COMPATIBLE. Differences:")
        for ln in report:
            log(f"  - {ln}")
        if o.check_only:
            return 0

        info, root = s["info"], s["root"]
        cams = select_cameras(s, o.cameras)
        target_fps = ref.fps if o.fps is None else o.fps   # 0 = keep source fps
        eff_fps = target_fps if target_fps > 0 else float(info["fps"])
        w, h = o.width or ref.width, o.height or ref.height
        log(f"\nTarget: {w}x{h} ({o.resize_mode}) @ {g(eff_fps)} fps, cameras {cams}")
        log(f"Downloading videos for {len(cams)} camera(s) from {src_repo.name} (resumable) ...")
        src_repo.fetch_patterns([f"{s['prefix']}meta/*"] + [f"{s['prefix']}videos/{c}/*" for c in cams])

        episodes, tasks = load_episodes(root)
        jobs, dropped = build_jobs(root, info, episodes, tasks, cams, s["depth_keys"], target_fps, o)
        if not jobs:
            raise ConversionError("No clips to export.")

        depth_cfg: dict = {}
        for cam in [c for c in cams if c in s["depth_keys"]]:
            quant = DepthQuant.from_feature(info["features"][cam])
            if o.depth_near is not None and o.depth_far is not None:
                near, far = o.depth_near, o.depth_far
            else:
                sample = [(j.src_video, j.start_s, j.duration_s) for j in jobs if j.camera == cam][:5]
                near, far = estimate_range(sample, quant, float(info["fps"]))
                near = o.depth_near if o.depth_near is not None else near
                far = o.depth_far if o.depth_far is not None else far
            depth_cfg[cam] = (quant, near, far)
            log(f"Depth {cam}: {quant}; rendering {near:.3f} m (white) to {far:.3f} m (black)")

        n_eps = len({j.episode_index for j in jobs})
        n_tr = sum("train" in j.splits for j in jobs)
        n_va = sum("val" in j.splits for j in jobs)
        n_raised = sum(j.fps_raised for j in jobs)
        log(f"Clips to export: {len(jobs)} from {n_eps} episodes x {len(cams)} camera(s) "
            f"(train {n_tr}, val {n_va}{', val also in train' if not o.val_holdout else ', val held out'})")
        if n_raised:
            log(f"  {n_raised} short clips use a higher fps so they keep the loader's {MIN_WINDOW_FRAMES}-frame minimum")
        if dropped:
            log(f"  {len(dropped)} clips are too short to keep at any fps (see summary)")
        clips, errors = [], []
        with ThreadPoolExecutor(max_workers=o.workers) as ex:
            futs = {ex.submit(process_clip, j, out, w, h, o, depth_cfg): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    clips.append(fut.result())
                except Exception as e:  # noqa: BLE001 - keep going; failures are reported at the end
                    errors.append({"uuid": futs[fut].uuid, "error": str(e)[-500:]})
                if i % 10 == 0 or i == len(jobs):
                    log(f"  exported {i}/{len(jobs)}")
        summary = {"source": o.source, "reference": o.reference, "compatibility_report": report,
                   "settings": {"cameras": cams, "fps": eff_fps, "width": w, "height": h,
                                "resize_mode": o.resize_mode, "max_clip_seconds": o.max_clip_seconds,
                                "val_holdout": o.val_holdout, "seed": o.seed,
                                "depth": {c: {"near_m": n, "far_m": f, "quantization": vars(q)}
                                          for c, (q, n, f) in depth_cfg.items()}},
                   "episodes_exported": n_eps,
                   "clips": sorted(clips, key=lambda c: c["uuid"]), "dropped_before_export": dropped,
                   "export_errors": errors}

    train_stats = build_train_jsonl(out)
    n_val = build_val_prompts(out, ref)
    errs, warns = validate(out, ref)
    summary.update({"train_jsonl": train_stats, "val_prompts": n_val, "validation": {"errors": errs, "warnings": warns}})
    out.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log(f"\ntrain/{JSONL_NAME}: {train_stats['records']} records "
        f"({train_stats['with_caption_json']} with caption_json, {len(train_stats['dropped'])} dropped)")
    log(f"val: {n_val} clips, prompts in {len(ref.prompt_keys) or 3} folders")
    if summary.get("export_errors"):
        log(f"WARNING: {len(summary['export_errors'])} clips failed to export; see {summary_path}")
    log(f"\nValidation against {ref.name}: {'PASS' if not errs else 'FAIL'}")
    for e in errs:
        log(f"  ERROR: {e}")
    for wmsg in warns:
        log(f"  note: {wmsg}")
    log(f"Summary: {summary_path}")
    return 1 if errs else 0
