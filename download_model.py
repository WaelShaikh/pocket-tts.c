#!/usr/bin/env python3
"""Download Supertonic 2 TTS model weights (ONNX)."""

import argparse
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
except ImportError:
    print("error: huggingface_hub is required (pip install huggingface_hub)")
    sys.exit(1)

REPO_ID = "Supertone/supertonic-2"
ONNX_FILES = [
    "text_encoder.onnx",
    "duration_predictor.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
]
JSON_FILES = [
    "tts.json",
    "unicode_indexer.json",
]

def download(repo_id: str, filename: str, subfolder: str, out_dir: str):
    try:
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"error: download failed for {filename}: {exc}")
        sys.exit(1)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="supertonic-model", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Downloading model from {REPO_ID} to {args.out}...")

    # Download ONNX files
    for f in ONNX_FILES:
        download(REPO_ID, f, "onnx", args.out)

    # Download JSON config files
    for f in JSON_FILES:
        download(REPO_ID, f, "onnx", args.out)

    print(f"Done. Model files saved to: {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
