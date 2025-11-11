# video_glaze.py
# Usage:
#   python video_glaze.py --in input.mp4 --out protected.mp4 --seed 42 \
#       --epsilon 3.0 --words 12 --dict-path words.txt --fps 0
#
# Notes
# - Processes frame-by-frame. Keeps original FPS unless --fps > 0.
# - Embeds a fresh rotating payload every N frames (default 12) to vary noise over time.

import argparse, os, random, struct
from typing import List
import numpy as np
import cv2

def load_words(dict_path: str | None, count: int, rng: random.Random) -> List[str]:
    fallback = [
        "orchid","nebula","granite","whisper","cipher","lumen","quartz","saffron",
        "isotope","tangent","aurora","ember","matrix","quiver","spectrum","zephyr",
        "basilisk","topaz","harbor","mycelium","galaxy","silica","fjord","plasma"
    ]
    pool = fallback
    if dict_path and os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
            pool = [w.strip() for w in f if w.strip()]
        if not pool:
            pool = fallback
    return [rng.choice(pool) for _ in range(count)]

def glaze_like_perturbation_bgr(bgr: np.ndarray, epsilon: float, rng: random.Random) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    noise = rng.normalvariate
    n = np.array([noise(0, 1) for _ in range(h*w*c)], dtype=np.float32).reshape(h, w, c)
    low = cv2.GaussianBlur(n, (0,0), sigmaX=1.2)
    high = n - low
    low2 = cv2.GaussianBlur(n, (0,0), sigmaX=0.5)
    high2 = n - low2
    hf = 0.6*high + 0.4*high2
    hf = hf / (np.std(hf) + 1e-8)
    hf = hf * (epsilon / 3.0)
    out = np.clip(rgb.astype(np.float32) + hf, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    jitter = rng.uniform(-0.6, 0.6)
    hsv[...,0] = (hsv[...,0] + jitter) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def _bytes_iter(b: bytes):
    for byte in b:
        for bit in range(8):
            yield (byte >> (7 - bit)) & 1

def lsb_embed_text_bgr(bgr: np.ndarray, text: str, seed: int) -> np.ndarray:
    payload = text.encode("utf-8")
    length_prefix = struct.pack(">I", len(payload))
    data = length_prefix + payload

    h, w, c = bgr.shape
    total_slots = h*w*c
    needed_bits = len(data) * 8
    if needed_bits > total_slots:
        raise ValueError(f"Message too large for frame capacity ({needed_bits} bits > {total_slots} bits).")

    rng = np.random.default_rng(seed)
    positions = rng.permutation(total_slots)[:needed_bits]

    flat = bgr.reshape(-1)
    for pos, bit in zip(positions, _bytes_iter(data)):
        flat[pos] = (flat[pos] & 0xFE) | bit
    return flat.reshape(bgr.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--epsilon", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--words", type=int, default=12)
    ap.add_argument("--dict-path", type=str, default=None)
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--payload-interval", type=int, default=12,
                    help="embed a fresh random payload every N frames")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fps = args.fps if args.fps > 0 else orig_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    frame_idx = 0
    words_pool = load_words(args.dict_path, max(50, args.words*3), rng)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # glaze-like noise
            frame = glaze_like_perturbation_bgr(frame, epsilon=args.epsilon, rng=rng)

            # rotate payload periodically
            if args.words > 0 and (frame_idx % args.payload-interval == 0):
                sample = rng.sample(words_pool, k=min(args.words, len(words_pool)))
                payload = " ".join(sample)
                try:
                    frame = lsb_embed_text_bgr(frame, payload, seed=(args.seed ^ 0xFEEDFACE ^ frame_idx))
                except ValueError:
                    # If the frame is tiny, skip embedding to avoid errors.
                    pass

            out.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        out.release()

if __name__ == "__main__":
    main()
