# lerobot-to-cosmos

![lerobot-to-cosmos: a LeRobot v3.0 dataset (100 episodes, 4 cameras including depth) is cut into one clip per episode and camera, resampled to 5 fps and 256x256, written in the Cosmos3 video-SFT layout, checked with NVIDIA's own loader, and uploaded to the Hugging Face Hub](docs/abstract.svg)

Converts a **LeRobot v3.0** robot dataset into the **NVIDIA Cosmos3 video-SFT format**, fetching both datasets straight from their Hugging Face links:

| Role | Default link |
|---|---|
| Source (converted) | https://huggingface.co/datasets/omarcavazosgzz/SO101cupsRGBD |
| Reference (format to match) | https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions |

One command does all of this:

1. Downloads a small sample of the reference (its JSONL, one clip, one prompt per val folder) and reads the target spec from it: folder layout, JSONL keys, resolution, fps.
2. Downloads the source's `meta/` and prints a **compatibility report**. If the source is already in the Cosmos format, it stops there.
3. Downloads the videos of **all 100 episodes from all 4 cameras** (`realsense`, `realsense_depth`, `logitech`, `usb`). The `data/` parquet files are skipped because the format has no place for them. It then cuts every episode out of LeRobot's shared video files and resamples it to the reference's size and fps, keeping exactly every Nth source frame.
4. Writes the reference layout and **validates it against the reference**.

```
sft_dataset_so101/
├── train/
│   ├── videos/episode_000000_clip000_realsense.mp4
│   │         episode_000000_clip000_realsense_depth.mp4
│   │         episode_000000_clip000_logitech.mp4
│   │         episode_000000_clip000_usb.mp4 ...
│   ├── captions/episode_000000_clip000_realsense/caption.txt
│   └── video_dataset_file.jsonl
├── val/
│   ├── videos/  videos_5frames/  images/  captions/
│   ├── inference_prompt/  inference_prompt_i2v/  inference_prompt_v2v/
└── conversion_summary.json
```

No ffmpeg install is needed: a static ffmpeg ships in the `imageio-ffmpeg` wheel and video probing uses PyAV.

## Install and run (macOS / Linux)

Requires [uv](https://docs.astral.sh/uv/). uv downloads Python 3.12 by itself if needed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once; or: brew install uv
```

From this folder, convert with the default links:

```bash
uv run lerobot-to-cosmos --out sft_dataset_so101
```

Once the repo is on GitHub, the same thing works without cloning (replace `USER`):

```bash
uvx --from git+https://github.com/USER/lerobot-to-cosmos lerobot-to-cosmos --out sft_dataset_so101
```

To install it as a regular command, run `uv tool install .` in this folder, or `uv tool install git+https://github.com/USER/lerobot-to-cosmos`. If `lerobot-to-cosmos` isn't found afterwards, run `uv tool update-shell` and open a new terminal.

Supported platforms: macOS 14+ on Apple Silicon, macOS 11+ on Intel, and Linux x86_64/aarch64 (glibc). Those are the platforms PyAV ships wheels for.

## What happens to the whole dataset

- **Every episode trains.** All 100 episodes x 4 cameras go into `train/`. `val/` holds copies of 5 episodes (all 4 cameras), used for the Cosmos inference previews. Add `--val-holdout` to keep those 5 out of training instead.
- **Every camera is its own clip.** Each camera of each episode becomes a separate clip named `_<camera>` (Bridge has one view per clip).
- **Depth becomes a viewable video.** `realsense_depth` holds 12-bit, log-quantized depth (0.01 to 10 m, shift 3.5), which can't go into an RGB video as-is. It is decoded back to metres with LeRobot's exact formula and rendered as grayscale inverse depth: brighter = closer, black = no reading. The near/far range is estimated once for the whole dataset so brightness means the same distance in every clip; set it yourself with `--depth-near 0.2 --depth-far 1.5`. The range used is recorded in `conversion_summary.json`. Depth clips get their own placeholder caption that says they are depth maps.
- **Short episodes are kept.** The Cosmos loader skips clips under 61 frames, which at 5 fps means 12.2 s. Shorter episodes are exported at the lowest exact rate that still gives 61 frames (for example 10 or 15 fps); the loader reads each clip's own fps. Only episodes under ~2 s (fewer than 61 frames even at 30 fps) are dropped, and they are listed in the summary.
- Episodes longer than 60 s are split into equal clips (`_clip000`, `_clip001`, ...).

Expect the full 7.34 GB download. The output is much smaller (256x256 at 5 fps).

## Common options

```bash
uv run lerobot-to-cosmos --out sft_dataset_so101 --check-only        # compatibility report only (meta/ download)
uv run lerobot-to-cosmos --out test_run --max-episodes 5             # quick trial
uv run lerobot-to-cosmos --out sft_dataset_so101 -c all-rgb          # only the 3 RGB cameras
uv run lerobot-to-cosmos --out sft_dataset_so101 -c observation.images.realsense   # one camera
uv run lerobot-to-cosmos --out sft_dataset_so101 --fps 0 --resize-mode keep   # keep 30 fps and aspect ratio
uv run lerobot-to-cosmos --out out --source <hf link or folder> --reference <hf link or folder>
uv run lerobot-to-cosmos --help
```

Size and fps follow the reference: 256x256 (center crop) at 5 fps. Downloads go to `hf_downloads/` and are resumable; reruns skip videos that already exist. Private or gated repos use your Hugging Face login (`hf auth login` or `HF_TOKEN`).

## Captions

LeRobot only stores a task string per episode, so `caption.txt` starts as a template built from it. The Cosmos loader accepts that, but real captions train much better. Run NVIDIA's captioner from [cosmos-framework](https://github.com/nvidia/cosmos-framework) on the exported videos, then rebuild:

```bash
python -m cosmos_framework.scripts.caption_from_video --video sft_dataset_so101/train/videos \
    -o sft_dataset_so101/train/captions --server http://localhost:8000/v1
python -m cosmos_framework.scripts.caption_from_video --video sft_dataset_so101/val/videos \
    -o sft_dataset_so101/val/captions --server http://localhost:8000/v1
uv run lerobot-to-cosmos --out sft_dataset_so101 --rebuild-only
```

The captioner treats the depth clips as ordinary grayscale videos, so check a few of their captions. To keep the depth template captions, delete the captioner's output for the `*_realsense_depth` folders before rebuilding.

`--rebuild-only` adds `caption_json` (the structured caption the Cosmos3 SFT loader prefers) to every JSONL window and to the val prompts.

## Train with Cosmos3

In cosmos-framework, point the `vision_sft_nano` recipe at the output folder:

```bash
export DATASET_PATH=$PWD/sft_dataset_so101
```

## Upload to Hugging Face

Add `--push-to-hub` to upload the result with the `hf` CLI once the conversion passes validation:

```bash
uv run lerobot-to-cosmos --out sft_dataset_so101 --push-to-hub your-name/so101-cups-cosmos
uv run lerobot-to-cosmos --out sft_dataset_so101 --upload-only --push-to-hub your-name/so101-cups-cosmos   # upload an existing folder, no reconversion
```

To keep several versions in **one** repo (e.g. 5 fps and 30 fps), give each its own folder with `--path-in-repo`:

```bash
uv run lerobot-to-cosmos --out sft_dataset_so101 --upload-only --push-to-hub your-name/so101-cups-cosmos --path-in-repo 5fps
uv run lerobot-to-cosmos --out sft_dataset_so101_30fps --upload-only --push-to-hub your-name/so101-cups-cosmos --path-in-repo 30fps
```

Each folder is a complete dataset with its own card. The repo's top-level card becomes an index of the versions, and is refreshed after every upload. To train on one version, point `DATASET_PATH` at its folder. If the repo already has a version at its root from an earlier upload without `--path-in-repo`, the tool stops instead of mixing the two layouts. Upload that version into a folder too, adding `--clean-root` to remove the root copy from the Hub; your local files aren't touched.

It asks for a **fine-grained token** (`hf_...`) with hidden input, at the start, so a long conversion never stops at a prompt. Create one at <https://huggingface.co/settings/tokens/new?tokenType=fineGrained> with *Write access to contents/settings of all repos under your personal namespace*, or the same for an organization you upload to. For unattended runs, set `HF_TOKEN` instead of typing it.

The token is used for this run only. It's handed to `hf` through the environment: never saved to disk, never placed on a command line (where shell history or `ps` would show it), and masked if `hf` prints it. Before uploading, the tool checks that the token works, that the repo's namespace is your account or one of your organizations, and that the token can write there. It accepts a classic write token too, with a warning.

The upload:
- creates the dataset repo, **private by default** (`--public` to make it public; an existing repo keeps its visibility)
- uploads the folder in resumable commits, skipping `.DS_Store`; if it's interrupted, rerun the same command
- adds a dataset card (`README.md`) generated from `conversion_summary.json` (clip counts, cameras, depth range, caption status, and the `cosmos-check loader` result if present), with the license copied from the source dataset's card when it declares one; a README you wrote yourself is never overwritten
- refuses to upload a folder whose last conversion failed validation

## Test it with the Cosmos pipeline

`cosmos-check` runs your converted dataset through NVIDIA's own code from [cosmos-framework](https://github.com/nvidia/cosmos-framework), pinned to commit `a3fb346` (2026-11-20). Pass `--cosmos-ref main` to test against the latest version instead.

### 1. Data-loader check (macOS or Linux, no GPU)

```bash
uv run cosmos-check loader -d sft_dataset_so101
```

This runs NVIDIA's actual SFT data loader (`get_sft_dataset` / `SFTDataset`) on every clip, using the `vision_sft_nano` recipe's settings, which are read straight from the recipe's source. For each clip it:

- loads the metadata with the same filters training uses
- decodes the video, selects frames, and resizes and crops to the recipe's 256 bucket
- builds the caption and tokenizes it with the recipe's Qwen3-VL tokenizer
- sets up the text/image/video conditioning plan

It then checks the result: frame counts match the JSONL, frames aren't blank, captions aren't truncated, and every sample fits the 45,056-token packing limit. It also pulls a few samples through the loader's training iterator.

It writes two files into the dataset folder (Cosmos ignores both):
- `cosmos_loader_check.json`: the per-clip report
- `cosmos_loader_preview.png`: first, middle and last frame of 8 clips, exactly as the model receives them

The first run takes a few minutes. It fetches cosmos-framework (~45 MB), builds a CPU-only PyTorch environment in `~/.cache/lerobot-to-cosmos`, and downloads the Qwen tokenizer files. Later runs reuse all of that, and the full 400-clip check takes about 2-3 minutes. The only change from training is that caption dropout is off, so every caption gets checked; training randomly blanks 10% of them. `ffmpeg`/`ffprobe` come from your system if installed, otherwise from a one-time automatic download.

### 2. Training smoke run (Linux + NVIDIA GPUs)

```bash
uv run cosmos-check train -d sft_dataset_so101 --dry-run      # show every step, run nothing (works anywhere)
uv run cosmos-check train -d sft_dataset_so101                # Cosmos3-Nano, all GPUs (8 recommended)
uv run cosmos-check train -d sft_dataset_so101 --model edge --gpus 4   # Cosmos3-Edge (2B) fits 4 GPUs
```

This follows NVIDIA's documented setup ([setup.md](https://github.com/nvidia/cosmos-framework/blob/main/docs/setup.md), [training.md](https://github.com/nvidia/cosmos-framework/blob/main/docs/training.md)) in `./cosmos_work/`:

1. clone cosmos-framework
2. `uv sync --all-extras --group=cu130-train`, using the exact uv version the repo requires (`--cuda 12.8` for drivers older than 580)
3. download the Wan2.2 VAE
4. convert the base checkpoint to DCP
5. run the data-loader check above
6. train for 5 iterations (`--iters`). The recipe is shortened the way NVIDIA's own smoke test (`tests/vision_sft_nano_5iter.toml`) does it: 5 iterations, a 1-step warm-up, and a checkpoint at the end. It keeps the recipe's real learning rate; NVIDIA's test lowers it so the loss visibly drops.

The run passes the way NVIDIA's smoke test does: training exits cleanly, every iteration logs a finite loss, and a checkpoint is saved. The loss values and their trend are printed for reference. Steps that are already done are skipped on reruns.

Requirements are NVIDIA's: Ampere or newer GPUs, CUDA ≥ 12.8, `git`, and `ffmpeg` (`sudo apt-get install -y git ffmpeg`). You also need plenty of disk space; base checkpoints and training checkpoints take 100+ GB. If Hugging Face asks for a login while downloading checkpoints, set `HF_TOKEN`. On a Mac, `train` stops at the preflight check and points you to `loader`.

### 3. Full fine-tune + validation video (Linux + NVIDIA GPUs)

```bash
uv run cosmos-check train -d sft_dataset_so101 --model edge --full --dry-run   # plan, time and disk estimate
uv run cosmos-check train -d sft_dataset_so101 --model edge --full
```

`--full` trains with NVIDIA's recipe exactly as shipped: 500 iterations, a checkpoint every 100, a 50-step warm-up, learning rate 1e-4. The run is named `so101_full_<model>`. It reuses the environment, VAE and base checkpoint from the smoke run. Before starting, it uses the smoke run's measurements to estimate how long training will take and how much disk the checkpoints need.

- **Interrupted?** Run the same command again. NVIDIA's trainer resumes from the run's latest checkpoint, including optimizer state and iteration count. A run that already finished skips straight to the video.
- **Longer or shorter:** `--iters 2000` changes the length. The learning-rate schedule is stretched so it still decays by the end.
- **Less disk:** `--save-every 250` keeps fewer checkpoints.
- **Start over / keep runs separate:** `--run-name my_run` starts a new run, e.g. `-d sft_dataset_so101_30fps --run-name so101_full_edge_30fps`.

After training, it exports the final checkpoint to Hugging Face format (`<run>/model`) and makes a **validation video**. It takes the first frame and caption of a held-out validation episode (realsense camera by default) and generates a 5-second image-to-video clip with both the base model and your fine-tuned model. It then writes `<run>/val_samples/<episode>_comparison.mp4`: real episode | base model | fine-tuned, side by side with labels.

To make more videos, or videos from an earlier checkpoint, without retraining:

```bash
uv run cosmos-check sample -d sft_dataset_so101 --model edge --clips 3 --seconds 8
uv run cosmos-check sample -d sft_dataset_so101 --model edge --checkpoint-iter 200 --camera logitech
```

NVIDIA's content guardrails stay on during generation, as their model license expects.

## Development

```bash
uv sync
uv run pytest        # builds synthetic LeRobot + Bridge fixtures and stubs the HF Hub
# also run NVIDIA's loader in the tests (needs a Python with the loader deps and a checkout):
COSMOS_CHECK_PYTHON=~/.cache/lerobot-to-cosmos/loader-env-*/bin/python \
  COSMOS_REPO=~/.cache/lerobot-to-cosmos/cosmos-framework-a3fb34690e15 uv run pytest
uv run ruff check .
```

## License

[MIT](LICENSE). This covers the code in this repository only. A converted dataset keeps its source dataset's license; the dataset card copies it over when the source declares one. `cosmos-check` downloads NVIDIA's cosmos-framework at run time, under NVIDIA's own license, and it isn't included here.
