#!/usr/bin/env python
"""
Batch syllable segmentation with **Sylber**
───────────────────────────────────────────
For every .wav under --in_path
→ writes Praat TextGrids to   --out_dir/sub_<ID>/run_<N>/<stem>.TextGrid

Example
-------
python seg_wav.py \
    --in_path  "/path/to/wavs" \
    --out_dir  "/path/to/TextGrids" \
    --model_ckpt sylber \
    --merge_thr 0.30
"""

# ────────────────────────── Imports ──────────────────────────
from __future__ import annotations
import argparse, pathlib, warnings
import numpy as np, soundfile as sf, librosa, torch
from sylber import Segmenter


# ────────────────  filename-parsing helpers  ────────────────
def _run_number(stem: str) -> int | None:
    """Extract integer run number from '…_run-02_…'   (returns None if absent)."""
    for part in stem.split("_"):
        if part.startswith("run-"):
            try:
                return int(part.split("-", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def _sub_number(stem: str) -> int | None:
    """Extract integer subject ID from '…_sub-09_…'   (returns None if absent)."""
    for part in stem.split("_"):
        if part.startswith("sub-"):
            try:
                return int(part.split("-", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def _tokens_after_stim(stem: str) -> list[str]:
    """
    Return stimulus tokens appearing after the 4th underscore in the filename
    e.g. 'behav_sub-09_run-02_stim-23_ne ne ne' → ['ne','ne','ne']
    """
    parts = stem.split("_", 4)
    return parts[4].split() if len(parts) == 5 else []


def _has_run_of(tokens: list[str], n: int = 3) -> bool:
    """True if *tokens* contains ≥ n identical items consecutively."""
    run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        if run >= n:
            return True
    return False


# ────────────────  merge-adjacent helpers  ────────────────
def _adjacent_sims(feat: np.ndarray) -> np.ndarray:
    """Cosine similarity between consecutive rows of feat [N,D] → [N-1]."""
    if len(feat) < 2:
        return np.empty(0, dtype=float)
    a = feat[:-1] / np.linalg.norm(feat[:-1], axis=1, keepdims=True)
    b = feat[1:] / np.linalg.norm(feat[1:], axis=1, keepdims=True)
    return (a * b).sum(1)


def _merge_pairs(segs: list[list[float]], feat: np.ndarray, thr: float | None):
    """
    Iteratively merge neighbouring intervals whose embedding similarity ≥ *thr*.
    *segs*  : [[start,end], …]   (seconds)
    *feat*  : [N,D] segment-level embeddings from Sylber
    """
    if thr is None or len(segs) < 3:
        return segs
    segs = [list(s) for s in segs]  # make mutable copy
    sims = _adjacent_sims(feat)
    while len(sims):
        i = int(np.argmax(sims))
        if sims[i] < thr or len(segs) < 3:
            break
        # merge seg i and i+1
        segs[i:i + 2] = [[segs[i][0], segs[i + 1][1]]]
        feat = np.delete(feat, i + 1, axis=0)
        sims = _adjacent_sims(feat)
    return segs


# ────────────────  TextGrid helpers  ────────────────
def _intervalize(segs: list[list[float]], audio_len: float):
    """
    Convert contiguous syllable segments to full interval list incl. silent gaps.
    Returns list[(xmin,xmax), …] covering [0,audio_len].
    """
    intervals, prev = [], 0.0
    for s, e in segs:
        if s > prev:  # preceding silence
            intervals.append((prev, s))
        intervals.append((s, e))  # syllable itself
        prev = e
    if prev < audio_len:  # trailing silence
        intervals.append((prev, audio_len))
    return intervals


def _write_tg(intervals, xmax, path: pathlib.Path):
    """Write a 1-tier empty-label TextGrid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as tg:
        tg.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
        tg.write('xmin = 0\n');
        tg.write(f'xmax = {xmax:.9f}\n')
        tg.write('tiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n')
        tg.write('        class = "IntervalTier"\n        name = "syll"\n')
        tg.write('        xmin = 0\n');
        tg.write(f'        xmax = {xmax:.9f}\n')
        tg.write(f'        intervals: size = {len(intervals)}\n')
        for i, (xmin, xmax) in enumerate(intervals, 1):
            tg.write(f'        intervals [{i}]:\n')
            tg.write(f'            xmin = {xmin:.9f}\n')
            tg.write(f'            xmax = {xmax:.9f}\n')
            tg.write('            text = ""\n')


# ───────────────────────────  main  ───────────────────────────
def main(args):
    wav_root = pathlib.Path(args.in_path).expanduser()
    wav_paths = [wav_root] if wav_root.is_file() else sorted(wav_root.rglob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"No .wav files found under {wav_root}")

    out_root = pathlib.Path(args.out_dir).expanduser()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    segger = Segmenter(model_ckpt=args.model_ckpt, device=device)  # <-- key change
    for wav in wav_paths:
        # --- load & mono ---
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(-1)

        # --- resample (librosa ≥0.11: kw-only) ---
        if sr != 16_000:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16_000,
                                     res_type="kaiser_best")
            sr = 16_000

        # --- Sylber segmentation ---
        out = segger(str(wav), in_second=True)
        segs = out["segments"]  # list[[s,e], …]  (seconds)
        feat = out["segment_features"]  # [N,D] ndarray
        segs = _merge_pairs(segs, feat, args.merge_thr)

        # --- build path ---
        sub = _sub_number(wav.stem) or "unk"
        run = _run_number(wav.stem) or "unk"
        stem = "_".join(wav.stem.split())  # collapse any spaces
        tg_p = out_root / f"sub_{sub:02}" / f"run_{run}" / (stem + ".TextGrid")

        # --- save TextGrid ---
        _write_tg(_intervalize(segs, len(audio) / sr), len(audio) / sr, tg_p)

        if (not args.quiet) and _has_run_of(_tokens_after_stim(wav.stem), 3):
            print(f"{wav.name}: {len(segs)} syllables")


# ───────────────────────────  CLI  ───────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # silence librosa & torch hints
    p = argparse.ArgumentParser()
    p.add_argument("--model_ckpt", default="sylber",
                   help="HF repo or local folder with Sylber checkpoint")
    p.add_argument("--in_path", required=True,
                   help="WAV file **or** directory")
    p.add_argument("--out_dir", required=True,
                   help="Destination folder for TextGrids")
    p.add_argument("--merge_thr", type=float, default=None,
                   help="Merge adjacent syllables above cosine-sim threshold "
                        "(e.g. 0.30). Omit to disable merging.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-file progress lines")
    main(p.parse_args())
