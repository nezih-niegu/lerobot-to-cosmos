"""LeRobot depth streams -> 8-bit grayscale depth videos (brighter = closer).

LeRobot stores depth as 12-bit codes (gray12le, usually lossless HEVC) quantized in
log space (lerobot/datasets/depth_utils.py):

    code = rint((log(d + shift) - log(dmin + shift)) / (log(dmax + shift) - log(dmin + shift)) * 4095)

with ``d``, ``dmin``, ``dmax`` and ``shift`` in metres. The parameters live in the
feature's ``info`` as ``video.depth_min`` / ``video.depth_max`` / ``video.shift`` /
``video.use_log``. This module inverts that mapping and renders inverse depth between a
near and a far plane, the usual convention for depth-control videos in Cosmos Transfer.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .media import ffmpeg_exe

QMAX = 4095


@dataclass(frozen=True)
class DepthQuant:
    depth_min: float = 0.01
    depth_max: float = 10.0
    shift: float = 3.5
    use_log: bool = True

    @classmethod
    def from_feature(cls, feature: dict) -> DepthQuant:
        info = feature.get("info") or {}

        def get(name: str, default):
            return info.get(f"video.{name}", info.get(name, default))

        return cls(float(get("depth_min", 0.01)), float(get("depth_max", 10.0)),
                   float(get("shift", 3.5)), bool(get("use_log", True)))

    def to_metres(self, q: np.ndarray) -> np.ndarray:
        q = q.astype(np.float32)
        if self.use_log:
            lo, hi = math.log(self.depth_min + self.shift), math.log(self.depth_max + self.shift)
            d = np.exp(q * ((hi - lo) / QMAX) + lo) - self.shift
        else:
            d = q * ((self.depth_max - self.depth_min) / QMAX) + self.depth_min
        return np.clip(d, self.depth_min, self.depth_max)

    def to_code(self, d: np.ndarray) -> np.ndarray:
        """Forward quantization (used for test fixtures)."""
        d = np.asarray(d, dtype=np.float64)
        if self.use_log:
            lo, hi = math.log(self.depth_min + self.shift), math.log(self.depth_max + self.shift)
            norm = (np.log(d + self.shift) - lo) / (hi - lo)
        else:
            norm = (d - self.depth_min) / (self.depth_max - self.depth_min)
        return np.rint(norm * QMAX).clip(0, QMAX).astype(np.uint16)


def is_depth_feature(feature: dict) -> bool:
    info = feature.get("info") or {}
    return bool(info.get("is_depth_map") or info.get("video.is_depth_map")) or feature["shape"][-1] == 1


def _code_plane(frame: av.VideoFrame) -> np.ndarray:
    fmt = frame.format.name
    if fmt in ("gray12le", "gray12be", "gray16le", "gray16be", "gray10le", "gray"):
        return frame.to_ndarray(format=fmt).astype(np.uint16)
    return frame.to_ndarray(format="gray16le") >> 4  # unexpected layout: rescale 16 -> 12 bit


def iter_codes(video: Path, start_s: float, n_src_frames: int, step: int, src_fps: float) -> Iterator[np.ndarray]:
    """Yield the 12-bit code planes of source frames start, start+step, ... within the clip."""
    with av.open(str(video)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if start_s > 0:
            c.seek(max(0, int((start_s - 1.0) / st.time_base)), stream=st, backward=True)
        for frame in c.decode(st):
            t = frame.time if frame.time is not None else 0.0
            idx = round((t - start_s) * src_fps)
            if idx < 0:
                continue
            if idx >= n_src_frames:
                break
            if idx % step == 0:
                yield _code_plane(frame)


def render(q: np.ndarray, quant: DepthQuant, near: float, far: float) -> np.ndarray:
    """Inverse-depth grayscale: near -> 255, far -> 0; invalid/saturated codes -> 0."""
    d = quant.to_metres(q)
    inv_near, inv_far = 1.0 / near, 1.0 / far
    v = (1.0 / d - inv_far) / (inv_near - inv_far)
    img = np.rint(np.clip(v, 0.0, 1.0) * 255).astype(np.uint8)
    img[(q == 0) | (q >= QMAX)] = 0
    return img


def estimate_range(samples: list[tuple[Path, float, float]], quant: DepthQuant, src_fps: float,
                   every_s: float = 1.0) -> tuple[float, float]:
    """Robust near/far (2nd/98th percentile of valid depth) over a few (video, start, duration) clips."""
    vals = []
    step = max(1, round(every_s * src_fps))
    for video, start, dur in samples:
        for q in iter_codes(video, start, round(dur * src_fps), step, src_fps):
            valid = q[(q > 0) & (q < QMAX)]
            if valid.size:
                vals.append(quant.to_metres(valid[:: max(1, valid.size // 20000)]))
    if not vals:
        return quant.depth_min, quant.depth_max
    allv = np.concatenate(vals)
    near, far = (float(x) for x in np.percentile(allv, [2, 98]))
    near = max(quant.depth_min, near * 0.9)
    far = min(quant.depth_max, max(far * 1.1, near * 1.5))
    return round(near, 3), round(far, 3)


def encode_depth_clip(video: Path, start_s: float, duration_s: float, src_fps: float, step: int, out_fps: float,
                      quant: DepthQuant, near: float, far: float, vf: str, enc: list[str], out: Path) -> int:
    """Decode, render and encode one depth clip with ffmpeg reading raw gray frames from stdin."""
    n_src = round(duration_s * src_fps)
    frames = iter_codes(video, start_s, n_src, step, src_fps)
    first = next(frames, None)
    if first is None:
        raise RuntimeError(f"No depth frames decoded from {video} at {start_s:.3f}s")
    h, w = first.shape
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-framerate", f"{out_fps:g}", "-i", "-",
           "-vf", vf, "-an", *enc, "-movflags", "+faststart", str(out)]
    count = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            for q in _chain(first, frames):
                proc.stdin.write(render(q, quant, near, far).tobytes())
                count += 1
        finally:
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg failed ({rc}) encoding {out}: {err.read().decode(errors='replace')[-2000:]}")
    return count


def _chain(first: np.ndarray, rest: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    yield first
    yield from rest
