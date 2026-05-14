"""
Download SongEval dataset directly via huggingface_hub,
bypassing the datasets library's torchcodec requirement.

Saves to ./data/songeval/ with:
  - metadata.json (list of dicts with annotations)
  - mp3/*.mp3 (audio files)

Usage:
    python download_songeval.py [--output-dir ./data/songeval]
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm

REPO_ID = "ASLP-lab/SongEval"


def main():
    parser = argparse.ArgumentParser(description="Download SongEval dataset")
    parser.add_argument("--output-dir", type=str, default="./data/songeval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir = output_dir / "mp3"
    mp3_dir.mkdir(exist_ok=True)

    # Download metadata
    print("Downloading metadata...")
    meta_path = hf_hub_download(REPO_ID, "metadata.jsonl", repo_type="dataset")
    with open(meta_path, encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    print(f"Found {len(metadata)} entries")

    # Save local copy of metadata
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Download audio files
    print("Downloading audio files...")
    api = HfApi()
    repo_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    mp3_files = [f for f in repo_files if f.startswith("mp3/") and f.endswith(".mp3")]
    print(f"Found {len(mp3_files)} audio files")

    # Check which files already exist
    to_download = []
    for mp3_file in mp3_files:
        local_path = output_dir / mp3_file
        if not local_path.exists():
            to_download.append(mp3_file)

    if to_download:
        print(f"Downloading {len(to_download)} files ({len(mp3_files) - len(to_download)} already cached)...")
        for mp3_file in tqdm(to_download, desc="Downloading"):
            downloaded = hf_hub_download(REPO_ID, mp3_file, repo_type="dataset")
            # Copy to local directory
            local_path = output_dir / mp3_file
            import shutil
            shutil.copy2(downloaded, local_path)
    else:
        print("All audio files already downloaded.")

    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"  Metadata: {output_dir / 'metadata.json'}")
    print(f"  Audio: {mp3_dir}/ ({len(mp3_files)} files)")


if __name__ == "__main__":
    main()
