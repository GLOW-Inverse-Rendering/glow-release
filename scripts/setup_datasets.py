#!/usr/bin/env python3
"""Download and install GLOW release datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/"
    "glow_dataset_release.zip"
)
ARCHIVE_NAME = "glow_dataset_release"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up GLOW release datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and extract the dataset archive.")
    download.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where glow_dataset_release.zip and glow_dataset_release/ will be placed.",
    )
    download.add_argument("--force", action="store_true", help="Replace an existing extraction.")

    install = subparsers.add_parser("install", help="Move extracted dataset folders into this project.")
    install.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the extracted glow_dataset_release directory.",
    )
    install.add_argument("--force", action="store_true", help="Replace existing dataset folders.")

    return parser.parse_args()


def remove_existing(path: Path, force: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not force:
        raise SystemExit(
            f"Refusing to overwrite existing path: {path}\n"
            "Pass --force to replace it."
        )
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def download(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    zip_path = output_dir / f"{ARCHIVE_NAME}.zip"
    extracted = output_dir / ARCHIVE_NAME

    output_dir.mkdir(parents=True, exist_ok=True)

    if extracted.exists():
        if not args.force:
            raise SystemExit(
                f"Refusing to overwrite existing directory: {extracted}\n"
                "Pass --force to replace it."
            )
        shutil.rmtree(extracted)

    if zip_path.exists():
        print(f"Using existing archive: {zip_path}")
    else:
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    if not extracted.is_dir():
        raise SystemExit(f"Archive did not contain expected directory: {extracted}")

    print(f"Ready: {extracted}")


def move_one(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise SystemExit(f"Missing source path: {src}")
    remove_existing(dst, force)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"{dst}")


def install(args: argparse.Namespace) -> None:
    source = args.source
    if not source.is_dir():
        raise SystemExit(f"Missing extracted dataset directory: {source}")

    datasets_dir = project_root() / "datasets"
    move_one(source / "data" / "real", datasets_dir / "real", args.force)
    move_one(source / "data" / "synthetic", datasets_dir / "synthetic", args.force)
    move_one(source / "mitsuba3_scenes", datasets_dir / "mitsuba3_scenes", args.force)

    print(f"Installed datasets under: {datasets_dir}")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "download":
            download(args)
        elif args.command == "install":
            install(args)
        else:
            raise AssertionError(args.command)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
