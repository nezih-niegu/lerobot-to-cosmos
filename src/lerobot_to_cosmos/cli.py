"""Command-line entry point: ``lerobot-to-cosmos``."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .converter import (
    DEFAULT_CAMERAS,
    DEFAULT_CAPTION_TEMPLATE,
    DEFAULT_DEPTH_CAPTION_TEMPLATE,
    DEFAULT_REFERENCE,
    DEFAULT_SOURCE,
    ConversionError,
    Options,
    convert,
)
from .hub import DatasetLinkError
from .hub_upload import (
    UploadError,
    check_path_in_repo,
    check_repo_id,
    get_token,
    upload,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Fetch a LeRobot v3.0 dataset and a Cosmos-format reference dataset from their "
        "Hugging Face links, report whether they are compatible, and convert the first into "
        "the reference's Cosmos3 video-SFT layout (videos, captions, video_dataset_file.jsonl, "
        "val prompts). Defaults: source = SO101cupsRGBD, reference = NVIDIA BridgeData2 subset."
    ),
)


class ResizeMode(str, Enum):
    crop = "crop"
    pad = "pad"
    stretch = "stretch"
    keep = "keep"


@app.command()
def run(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output folder, e.g. sft_dataset_so101.")],
    source: Annotated[str, typer.Option("--source", help="Link (or local folder) of the dataset to convert.")] = DEFAULT_SOURCE,
    reference: Annotated[str, typer.Option("--reference", help="Link (or local folder) of the Cosmos dataset to match.")] = DEFAULT_REFERENCE,
    download_dir: Annotated[Path, typer.Option("--download-dir", help="Where downloads are stored.")] = Path("hf_downloads"),
    check_only: Annotated[bool, typer.Option("--check-only", help="Only report compatibility (downloads metadata only).")] = False,
    rebuild_only: Annotated[bool, typer.Option("--rebuild-only", help="Skip export; rebuild JSONL + val prompts from --out (after captioning).")] = False,
    camera: Annotated[
        list[str] | None,
        typer.Option("--camera", "-c", help="Camera key to export (repeatable), 'all' (every camera incl. depth) or 'all-rgb'. Default: all."),
    ] = None,
    fps: Annotated[float | None, typer.Option("--fps", help="Target fps. Default: reference fps. 0 keeps the source fps.")] = None,
    width: Annotated[int | None, typer.Option("--width", help="Target width. Default: reference.")] = None,
    height: Annotated[int | None, typer.Option("--height", help="Target height. Default: reference.")] = None,
    resize_mode: Annotated[ResizeMode, typer.Option("--resize-mode", help="crop/pad/stretch to the target size, or keep the aspect ratio.")] = ResizeMode.crop,
    max_clip_seconds: Annotated[float, typer.Option("--max-clip-seconds", help="Split longer episodes (loader limit 61 s).")] = 60.0,
    val_ratio: Annotated[float, typer.Option("--val-ratio", help="Share of episodes for val/.")] = 0.05,
    val_episodes: Annotated[int | None, typer.Option("--val-episodes", help="Exact number of val episodes (overrides --val-ratio).")] = None,
    val_holdout: Annotated[bool, typer.Option("--val-holdout", help="Keep val episodes out of train. Default: every episode trains; val/ holds copies for inference previews.")] = False,
    max_episodes: Annotated[int, typer.Option("--max-episodes", help="Only convert the first N episodes (quick test).")] = 0,
    caption_template: Annotated[str, typer.Option("--caption-template", help="Placeholder caption; {task} and {camera} are filled in.")] = DEFAULT_CAPTION_TEMPLATE,
    depth_caption_template: Annotated[str, typer.Option("--depth-caption-template", help="Placeholder caption for depth clips.")] = DEFAULT_DEPTH_CAPTION_TEMPLATE,
    depth_near: Annotated[float | None, typer.Option("--depth-near", help="Depth (m) rendered white. Default: estimated from the data.")] = None,
    depth_far: Annotated[float | None, typer.Option("--depth-far", help="Depth (m) rendered black. Default: estimated from the data.")] = None,
    crf: Annotated[int, typer.Option("--crf", help="x264 quality (lower = better).")] = 18,
    workers: Annotated[int, typer.Option("--workers", help="Parallel ffmpeg jobs.")] = 4,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the train/val split.")] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Re-encode videos that already exist.")] = False,
    overwrite_captions: Annotated[bool, typer.Option("--overwrite-captions", help="Rewrite existing caption.txt files.")] = False,
    push_to_hub: Annotated[str | None, typer.Option("--push-to-hub", help="After a successful conversion, upload --out to this Hugging Face dataset repo (e.g. your-name/so101-cups-cosmos) with the hf CLI. Asks for a fine-grained hf_ token (or reads HF_TOKEN).")] = None,
    public: Annotated[bool, typer.Option("--public", help="Create the Hub repo as public (default: private). Ignored if it already exists.")] = False,
    path_in_repo: Annotated[str | None, typer.Option("--path-in-repo", help="Upload into this folder of the Hub repo (e.g. 30fps), so several versions can share one repo. The repo's top-level card becomes an index of the versions.")] = None,
    clean_root: Annotated[bool, typer.Option("--clean-root", help="With --path-in-repo: delete a dataset previously uploaded to the repo root (train/, val/, summaries) from the Hub.")] = False,
    upload_only: Annotated[bool, typer.Option("--upload-only", help="Skip conversion; just upload an existing --out folder (needs --push-to-hub).")] = False,
) -> None:
    """Check compatibility and convert SOURCE into the REFERENCE's Cosmos video-SFT format."""
    opts = Options(
        out=out, source=source, reference=reference, download_dir=download_dir, check_only=check_only,
        rebuild_only=rebuild_only, cameras=list(camera or DEFAULT_CAMERAS), fps=fps, width=width, height=height,
        resize_mode=resize_mode.value, max_clip_seconds=max_clip_seconds, val_ratio=val_ratio,
        val_episodes=val_episodes, val_holdout=val_holdout, max_episodes=max_episodes,
        caption_template=caption_template, depth_caption_template=depth_caption_template,
        depth_near=depth_near, depth_far=depth_far,
        crf=crf, workers=workers, seed=seed, overwrite=overwrite, overwrite_captions=overwrite_captions,
    )
    try:
        token = repo_id = None
        if (path_in_repo or clean_root) and not push_to_hub:
            raise UploadError("--path-in-repo / --clean-root only apply with --push-to-hub.")
        if upload_only and not push_to_hub:
            raise UploadError("--upload-only needs --push-to-hub <your-name/repo-name>.")
        if push_to_hub:
            if check_only:
                raise UploadError("--push-to-hub can't be combined with --check-only (nothing is written).")
            repo_id = check_repo_id(push_to_hub)
            path_in_repo = check_path_in_repo(path_in_repo)
            # Ask up front, so a long conversion doesn't end up waiting at a prompt.
            token = get_token(typer.prompt)
        code = 0 if upload_only else convert(opts)
        if repo_id and token:
            if code != 0:
                typer.echo("Not uploading: the conversion did not pass validation.", err=True)
            else:
                upload(out, repo_id, token, private=not public, path_in_repo=path_in_repo, clean_root=clean_root)
    except (ConversionError, DatasetLinkError, UploadError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(2) from None
    raise typer.Exit(code)


def main() -> None:
    """Entry point declared in [project.scripts]."""
    app()


if __name__ == "__main__":
    main()
