"""Video helpers: a bundled ffmpeg binary (no Homebrew/apt needed) and PyAV probing."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

import av


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """ffmpeg to use: $LEROBOT_TO_COSMOS_FFMPEG, else the binary shipped in the imageio-ffmpeg wheel."""
    override = os.environ.get("LEROBOT_TO_COSMOS_FFMPEG")
    if override:
        return override
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-v", "error", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr[-2000:]}")


def probe(path: Path) -> dict:
    """width, height, fps, total_frames (packet count, like ffprobe -count_packets), duration (s)."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = s.average_rate or s.guessed_rate or s.base_rate
        fps = float(rate) if rate else 0.0
        n = sum(1 for pkt in c.demux(s) if pkt.size > 0)
        if c.duration:
            duration = c.duration / 1_000_000
        elif s.duration and s.time_base:
            duration = float(s.duration * s.time_base)
        else:
            duration = n / fps if fps else 0.0
        return {"width": int(s.codec_context.width), "height": int(s.codec_context.height),
                "fps": fps, "total_frames": n, "duration": duration}
