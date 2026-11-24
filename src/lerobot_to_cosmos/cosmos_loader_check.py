#!/usr/bin/env python3
"""Run NVIDIA cosmos-framework's own SFT data loader over a converted dataset (CPU only).

This file runs inside an environment that has cosmos-framework's data dependencies
(torch, transformers, boto3, ...), *not* inside lerobot-to-cosmos's own environment, so it
imports nothing from this package. ``cosmos-check loader`` and ``cosmos-check train``
launch it with the right interpreter and ``PYTHONPATH`` pointing at a cosmos-framework
checkout.

What it does:

1. Reads the dataset settings of the recipe (default ``vision_sft_nano``) straight from
   the recipe's source in the checkout (the ``L(get_sft_dataset)(...)`` and
   ``L(PackingDataLoader)(...)`` calls) plus the ``[dataloader_train]`` overrides from the
   recipe TOML, so nothing is copied by hand.
2. Builds the dataset with ``cosmos_framework...sft_dataset.get_sft_dataset`` (metadata
   loading and filtering exactly as in training).
3. Runs ``SFTDataset.process_one_sample`` on every clip (video decode, frame selection,
   resize/crop, caption assembly, tokenization, conditioning plan) and checks the output.
4. Pulls a few samples through ``SFTDataset.__iter__`` (the path training uses).

Only change from the recipe: CFG caption dropout is switched off so every caption gets
checked (training randomly blanks 10% of them).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import sys
import time
import tomllib
from pathlib import Path

RECIPE_DIR = "cosmos_framework/configs/base/experiment/sft"
TOML_DIR = "examples/toml/sft_config"


# --------------------------------------------------------------------------- recipe extraction
def _lazy_call_kwargs(tree: ast.AST, target: str) -> dict:
    """kwargs (literals only) of the first ``L(<target>)(...)`` call in source order."""
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
                and isinstance(node.func.func, ast.Name) and node.func.func.id == "L"
                and node.func.args and isinstance(node.func.args[0], ast.Name)
                and node.func.args[0].id == target):
            calls.append(node)
    if not calls:
        raise SystemExit(f"Could not find L({target})(...) in the recipe source.")
    call = min(calls, key=lambda n: (n.lineno, n.col_offset))
    out = {}
    for kw in call.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            out[kw.arg] = ast.unparse(kw.value)  # non-literal (e.g. a nested L(...) call)
    return out


def recipe_settings(repo: Path, recipe: str) -> dict:
    src = (repo / RECIPE_DIR / f"{recipe}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ds = _lazy_call_kwargs(tree, "get_sft_dataset")
    pack = _lazy_call_kwargs(tree, "PackingDataLoader")
    toml_path = repo / TOML_DIR / f"{recipe}.toml"
    overrides = {}
    if toml_path.exists():
        overrides = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("dataloader_train", {})
    if "max_caption_tokens" in overrides:
        ds["max_caption_tokens"] = overrides["max_caption_tokens"]
    if "max_sequence_length" in overrides:
        pack["max_sequence_length"] = overrides["max_sequence_length"]
    return {"dataset": ds, "packing": pack, "toml_overrides": overrides}


def _qwen_processor(name: str):
    """Same as cosmos_framework.configs.base.defaults.reasoner.create_qwen2_tokenizer_with_download
    with config_variant="hf" (which returns ``LLMTokenizerProcessor(Qwen2Tokenizer.from_pretrained(name))``),
    without importing the training-only modules that file pulls in."""
    from types import SimpleNamespace

    from transformers import Qwen2Tokenizer

    return SimpleNamespace(tokenizer=Qwen2Tokenizer.from_pretrained(name))


def ensure_ffmpeg() -> str:
    """NVIDIA's loader shells out to ``ffmpeg`` and ``ffprobe``. Use the system ones, else real
    static builds from the static-ffmpeg package (downloaded once)."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return f"system ({shutil.which('ffprobe')})"
    try:
        from static_ffmpeg import run

        ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "FAIL: NVIDIA's loader needs ffmpeg and ffprobe on PATH and the automatic download failed "
            f"({type(e).__name__}: {e}).\nInstall them, then rerun: macOS `brew install ffmpeg`, "
            "Ubuntu/Debian `sudo apt-get install -y ffmpeg`."
        ) from None
    dirs = list(dict.fromkeys([os.path.dirname(ffmpeg), os.path.dirname(ffprobe)]))
    os.environ["PATH"] = os.pathsep.join([*dirs, os.environ.get("PATH", "")])
    return f"static-ffmpeg ({ffprobe})"


# --------------------------------------------------------------------------- checks
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="cosmos-framework checkout")
    ap.add_argument("--dataset", type=Path, required=True, help="Converted dataset root (contains train/)")
    ap.add_argument("--recipe", default="vision_sft_nano")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct",
                    help="HF id or local folder of the Qwen tokenizer the recipe uses")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = every clip")
    ap.add_argument("--iter-samples", type=int, default=3, help="Samples pulled through SFTDataset.__iter__")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--preview", type=Path, default=None, help="Write a PNG contact sheet of loader output")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo, dataset = args.repo.resolve(), args.dataset.resolve()
    sys.path.insert(0, str(repo))
    random.seed(args.seed)
    ff = ensure_ffmpeg()

    import boto3
    import numpy as np
    import torch
    from cosmos_framework.data.generator.local_datasets import sft_dataset as S
    from cosmos_framework.data.generator.local_datasets.helper import client_config
    from cosmos_framework.utils.lazy_config import LazyCall as L

    settings = recipe_settings(repo, args.recipe)
    kw = dict(settings["dataset"])
    pack = settings["packing"]
    recipe_dropout = kw.get("cfg_dropout_rate")
    jsonl = dataset / "train" / "video_dataset_file.jsonl"
    if not jsonl.exists():
        print(f"FAIL: {jsonl} not found")
        return 1
    kw["jsonl_paths"] = [str(jsonl)]
    kw["tokenizer_config"] = L(_qwen_processor)(name=args.tokenizer)
    kw["cfg_dropout_rate"] = 0.0

    print(f"cosmos-framework: {repo}")
    print(f"ffmpeg/ffprobe: {ff}")
    print(f"recipe: {args.recipe}  (loader settings read from {RECIPE_DIR}/{args.recipe}.py"
          f"{' + ' + TOML_DIR + '/' + args.recipe + '.toml' if settings['toml_overrides'] else ''})")
    for k in ("resolution", "num_video_frames", "temporal_interval_mode", "frame_selection_mode",
              "conditioning_config", "conditioning_fps", "max_caption_tokens"):
        print(f"  {k} = {kw.get(k)!r}")
    print(f"  cfg_dropout_rate = 0.0 for this check (recipe: {recipe_dropout})")
    print(f"  max_sequence_length = {pack.get('max_sequence_length')}")

    t0 = time.time()
    raw_records = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    ds = S.get_sft_dataset(**kw)
    kept = len(ds.metadata)
    print(f"\nMetadata: {raw_records} JSONL records -> {kept} kept by the Cosmos loader filters "
          f"(duration <= 61 s, windows >= 61 frames)")

    sp = int(pack.get("patch_spatial", 2))
    sc = int(pack.get("tokenizer_spatial_compression_factor", 16))
    tc = int(pack.get("tokenizer_temporal_compression_factor", 4))
    max_seq = int(pack.get("max_sequence_length", 45056))
    max_cap = int(kw.get("max_caption_tokens", 2048))

    ds.s3_client = boto3.client("s3", config=client_config)
    items = sorted(ds.metadata, key=lambda m: m["uuid"])
    if args.max_samples:
        items = items[: args.max_samples]

    errors: list[str] = []
    bad: set[str] = set()
    per_sample = []
    preview_frames = []
    styles: dict[str, int] = {}
    for i, md in enumerate(items, 1):
        uid = md["uuid"]
        win = md["t2w_windows"][0]
        try:
            s = ds.process_one_sample(md)
        except Exception as e:  # noqa: BLE001 - report every failing clip
            errors.append(f"{uid}: loader raised {type(e).__name__}: {e}")
            bad.add(uid)
            continue
        if s is None:
            errors.append(f"{uid}: loader returned None (sample skipped in training)")
            bad.add(uid)
            continue
        v = s["video"]
        c, t, h, w = v.shape
        tw, th = ds.output_sizes[md["aspect_ratio"]]
        n_window = min(win["end_frame"], s["n_orig_video_frames"] - 1) - win["start_frame"] + 1
        expect_t = (n_window - 1) // tc * tc + 1
        t_lat = 1 + (t - 1) // tc
        vision_tokens = t_lat * (h // sc // sp) * (w // sc // sp)
        text_tokens = int(s["text_token_ids"].numel())
        std = float(v.float().std())
        problems = []
        if v.dtype != torch.uint8 or c != 3:
            problems.append(f"video tensor {tuple(v.shape)} {v.dtype}, expected uint8 [3,T,H,W]")
        if (w, h) != (tw, th):
            problems.append(f"frame size {w}x{h} != recipe bucket {tw}x{th} ({md['aspect_ratio']})")
        if t != expect_t:
            problems.append(f"{t} frames decoded, expected {expect_t} from the JSONL window")
        if std < 1.0:
            problems.append(f"frames look blank (pixel std {std:.2f})")
        if not s["ai_caption"]:
            problems.append("empty caption")
        if text_tokens >= max_cap:
            problems.append(f"caption truncated at {max_cap} tokens")
        if vision_tokens + text_tokens > max_seq:
            problems.append(f"{vision_tokens + text_tokens} tokens > max_sequence_length {max_seq} (can't be packed)")
        if "sequence_plan" not in s and kw.get("conditioning_config"):
            problems.append("no conditioning sequence_plan")
        errors += [f"{uid}: {p}" for p in problems]
        if problems:
            bad.add(uid)
        styles[s["sampled_caption_style"]] = styles.get(s["sampled_caption_style"], 0) + 1
        per_sample.append({"uuid": uid, "frames": t, "latent_frames": t_lat, "size": [w, h],
                           "fps": round(float(s["conditioning_fps"]), 3), "caption_style": s["sampled_caption_style"],
                           "text_tokens": text_tokens, "total_tokens": vision_tokens + text_tokens})
        if args.preview and len(preview_frames) < 8:
            frames = v.permute(1, 2, 3, 0).numpy()
            preview_frames.append(np.concatenate([frames[0], frames[t // 2], frames[-1]], axis=1))
        if i % 50 == 0 or i == len(items):
            print(f"  processed {i}/{len(items)} clips")

    # The path training actually uses: SFTDataset.__iter__ (sharding, shuffling, retries).
    iter_ok = 0
    if args.iter_samples and not errors:
        ds2 = S.get_sft_dataset(**kw)
        ds2.shard_world_size, ds2.shard_rank = 1, 0
        it = iter(ds2)
        for _ in range(args.iter_samples):
            sample = next(it)
            assert sample["video"].dtype == torch.uint8
            iter_ok += 1

    elapsed = time.time() - t0
    example = per_sample[0] if per_sample else None
    report = {
        "recipe": args.recipe, "repo": str(repo), "dataset": str(dataset), "tokenizer": args.tokenizer,
        "loader_settings": {k: v for k, v in kw.items() if k not in ("tokenizer_config", "jsonl_paths")},
        "recipe_cfg_dropout_rate": recipe_dropout, "packing": pack,
        "jsonl_records": raw_records, "kept_by_loader": kept, "checked": len(items),
        "passed": len(items) - len(bad), "iter_samples": iter_ok,
        "caption_styles": styles, "errors": errors, "seconds": round(elapsed, 1),
        "max_total_tokens": max((p["total_tokens"] for p in per_sample), default=0),
        "samples": per_sample,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.preview and preview_frames:
        from PIL import Image

        Image.fromarray(np.concatenate(preview_frames, axis=0)).save(args.preview)

    print(f"\nChecked {len(items)} clips in {elapsed:.0f}s with NVIDIA's loader:")
    if example:
        print(f"  e.g. {example['uuid']}: {example['frames']} frames ({example['latent_frames']} latent) "
              f"at {example['size'][0]}x{example['size'][1]}, {example['fps']} fps, "
              f"{example['text_tokens']} caption tokens, {example['total_tokens']} tokens total")
    print(f"  caption styles used: {styles}  (caption_json is preferred when present)")
    if per_sample:
        print(f"  largest sample: {report['max_total_tokens']} tokens (packing limit {max_seq})")
    if iter_ok:
        print(f"  {iter_ok} samples pulled through SFTDataset.__iter__ (training path)")
    if kept < raw_records:
        errors.insert(0, f"{raw_records - kept} JSONL records were dropped by the loader filters")
    if args.report:
        print(f"  report: {args.report}")
    if args.preview and preview_frames:
        print(f"  preview: {args.preview}")
    if errors:
        print(f"\nFAIL: {len(errors)} problem(s)")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more in the report")
        return 1
    print("\nPASS: every clip loads through Cosmos's SFT data loader as it would in training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
