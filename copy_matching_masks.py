#!/usr/bin/env python3
"""
Recursively copy matching target mask files into folders that contain target overlay files.

Example naming:
    0_1-target-overlay.png
    0_1-target-mask.png

For every '*-target-overlay.png' found under the target parent directory, this script
looks for the matching '*-target-mask.png' in the source mask directory and copies it
into the overlay file's folder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_MASK_DIR = Path(
    "/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/data/dev_images/"
    "dev_annot_trans_260529_renamed"
)


def copy_matching_masks(
    parent_dir: Path,
    source_mask_dir: Path = DEFAULT_SOURCE_MASK_DIR,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    if not parent_dir.is_dir():
        raise NotADirectoryError(f"Parent directory does not exist: {parent_dir}")
    if not source_mask_dir.is_dir():
        raise NotADirectoryError(f"Source mask directory does not exist: {source_mask_dir}")

    overlay_files = sorted(parent_dir.rglob("*-target-overlay.png"))

    copied = 0
    skipped_existing = 0
    missing_masks = 0

    for overlay_path in overlay_files:
        sample_rep_id = overlay_path.name.removesuffix("-target-overlay.png")
        mask_name = f"{sample_rep_id}-target-mask.png"
        source_mask_path = source_mask_dir / mask_name
        destination_mask_path = overlay_path.parent / mask_name

        if not source_mask_path.is_file():
            print(f"[missing mask] {mask_name} for overlay: {overlay_path}")
            missing_masks += 1
            continue

        if destination_mask_path.exists() and not overwrite:
            print(f"[skip exists] {destination_mask_path}")
            skipped_existing += 1
            continue

        print(f"[copy] {source_mask_path} -> {destination_mask_path}")
        if not dry_run:
            shutil.copy2(source_mask_path, destination_mask_path)
        copied += 1

    print("\nDone")
    print(f"Overlay files found: {len(overlay_files)}")
    print(f"Copied masks: {copied}")
    print(f"Skipped existing masks: {skipped_existing}")
    print(f"Missing source masks: {missing_masks}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy matching '*-target-mask.png' files into folders containing '*-target-overlay.png'."
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Target parent folder to search recursively for overlay files.",
    )
    parser.add_argument(
        "--source-mask-dir",
        type=Path,
        default=DEFAULT_SOURCE_MASK_DIR,
        help="Folder containing the original '*-target-mask.png' files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite mask files if they already exist in the destination folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying files.",
    )
    args = parser.parse_args()

    copy_matching_masks(
        parent_dir=args.parent_dir,
        source_mask_dir=args.source_mask_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
