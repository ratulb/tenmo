#!/usr/bin/env python3
"""Fetch the TinyStories subset.

Downloads a byte-bounded prefix of HF `roneneldan/TinyStories` train text
(TinyStories-train.txt), cuts it at a story boundary, splits stories 90/10
into train/val, and writes the two text files plus counts. Tokenization
(mbpe gpt2 → cached id stream) happens Mojo-side later — this
script moves text bytes only, so it needs nothing but the stdlib.

Story separator: TinyStories txt files delimit stories with
`<|endoftext|>`; if the marker is absent (format drift), fall back to
blank-line chunking so the script still produces a clean boundary cut.

Usage:
    python3 scripts/fetch_tinystories.py [--bytes 134217728]
        [--out-dir examples/data] [--url <hf-resolve-url>]
    # smoke test (validates parsing/split without a 100MB+ download):
    python3 scripts/fetch_tinystories.py --bytes 1048576 --out-dir /tmp/ts_smoke

Outputs (in --out-dir):
    tinystories_train.txt / tinystories_val.txt + stdout token estimate.
License note: dataset is CDLA-Sharing-1.0, not for
redistribution — fetched bytes stay in the gitignored data/ dir.
"""

import argparse
import os
import sys
import urllib.request

DEFAULT_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories"
    "/resolve/main/TinyStories-train.txt"
)
SEPARATOR = "<|endoftext|>"
# ~4 chars/token heuristic — estimate only; the true count is
# measured at mbpe-encode time and recorded in the run log.
CHARS_PER_TOKEN = 4


def stream_prefix(url: str, budget: int) -> str:
    """Stream at most `budget` raw bytes, decoded incrementally as UTF-8."""
    chunks = []
    received = 0
    req = urllib.request.Request(url, headers={"User-Agent": "tenmo-ep18"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        while received < budget:
            piece = resp.read(min(1 << 20, budget - received))
            if not piece:
                break
            chunks.append(piece)
            received += len(piece)
    # A multi-byte char may straddle the cut — drop the ragged tail.
    return b"".join(chunks).decode("utf-8", errors="ignore")


def split_stories(blob: str) -> "list[str]":
    """Cut the blob into stories at separator boundaries.

    The download is truncated mid-story, so the trailing fragment (no
    closing separator) is discarded — every kept story is complete.
    """
    if SEPARATOR in blob:
        parts = blob.split(SEPARATOR)
        # split() leaves the unterminated tail as the last part: drop it.
        stories = [p.strip() for p in parts[:-1]]
    else:  # format drift fallback: blank-line separated paragraphs
        stories = [p.strip() for p in blob.split("\n\n")]
        stories = stories[:-1]  # same ragged-tail rule
    return [s for s in stories if s]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bytes", type=int, default=134217728,
                    help="download budget in bytes (default 128MB)")
    ap.add_argument("--out-dir", default="examples/data")
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    print(f"[fetch] GET {args.url} (budget {args.bytes} bytes)", flush=True)
    blob = stream_prefix(args.url, args.bytes)
    stories = split_stories(blob)
    if not stories:
        print("[fetch] ERROR: no stories parsed — separator may have changed",
              file=sys.stderr)
        return 1

    # Contiguous 90/10 split on the STORY stream: no window drawn
    # from one side may cross into the other at WindowLoader time, and a
    # prefix split keeps the subset deterministic across re-fetches.
    cut = int(len(stories) * 0.9)
    train, val = stories[cut * 0:cut], stories[cut:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "tinystories_train.txt")
    val_path = os.path.join(args.out_dir, "tinystories_val.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val) + "\n")

    # Report the numbers in the run log: story counts are
    # exact, token counts are estimates until mbpe-encode time.
    t_chars = sum(len(s) for s in train)
    v_chars = sum(len(s) for s in val)
    print(f"[fetch] stories: train {len(train)}, val {len(val)}")
    print(f"[fetch] chars:   train {t_chars}, val {v_chars}")
    print(f"[fetch] ~tokens: train {t_chars // CHARS_PER_TOKEN}, "
          f"val {v_chars // CHARS_PER_TOKEN} (@~{CHARS_PER_TOKEN} ch/tok)")
    print(f"[fetch] wrote {train_path}, {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
