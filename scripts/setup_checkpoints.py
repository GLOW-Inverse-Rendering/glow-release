#!/usr/bin/env python3
"""Download, inspect, and install GLOW release checkpoints."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


CHECKPOINT_URL = (
    "https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/"
    "glow-checkpoint-release.zip"
)
ARCHIVE_NAME = "glow-checkpoint-release"

STAGE_DIRS = {
    "stage1": {
        "neus": "01_wildlight",
        "mitsuba": None,
    },
    "stage2": {
        "neus": "02_refinement2_neus",
        "mitsuba": "02_refinement2_mitsuba",
    },
    "stage3": {
        "neus": "03_material_neus",
        "mitsuba": "03_material_mitsuba",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_checkpoints_dir() -> Path:
    return project_root() / "checkpoints"


def archive_root(checkpoints_dir: Path) -> Path:
    return checkpoints_dir / ARCHIVE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up GLOW release checkpoints."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=default_checkpoints_dir(),
        help="Directory containing or receiving glow-checkpoint-release/.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and extract checkpoints.")
    download.add_argument("--force", action="store_true", help="Replace an existing extraction.")

    subparsers.add_parser("list", help="List available scenes and stages.")

    install = subparsers.add_parser("install", help="Install one scene/stage into an experiment dir.")
    install.add_argument("--scene", required=True, help="Scene name, e.g. coffee_table_colocated.")
    install.add_argument(
        "--stage",
        required=True,
        choices=sorted(STAGE_DIRS.keys()),
        help="Checkpoint stage to install.",
    )
    install.add_argument(
        "--exp-dir",
        required=True,
        type=Path,
        help="Runtime experiment directory to receive checkpoints.",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoint files in the destination.",
    )

    return parser.parse_args()


def require_archive(root: Path) -> Path:
    archive = archive_root(root)
    if not archive.is_dir():
        raise SystemExit(
            f"Missing checkpoint archive: {archive}\n"
            "Run: python3 scripts/setup_checkpoints.py download"
        )
    return archive


def download(args: argparse.Namespace) -> None:
    checkpoints_dir = args.checkpoints_dir
    zip_path = checkpoints_dir / f"{ARCHIVE_NAME}.zip"
    extracted = archive_root(checkpoints_dir)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if extracted.exists():
        if not args.force:
            raise SystemExit(
                f"Refusing to overwrite existing directory: {extracted}\n"
                "Pass --force to replace it."
            )
        shutil.rmtree(extracted)

    if not zip_path.exists():
        print(f"Downloading {CHECKPOINT_URL}")
        urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)
    else:
        print(f"Using existing archive: {zip_path}")

    print(f"Extracting to {checkpoints_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(checkpoints_dir)

    if not extracted.is_dir():
        raise SystemExit(f"Archive did not contain expected directory: {extracted}")

    print(f"Ready: {extracted}")


def scene_names(archive: Path) -> list[str]:
    return sorted(
        path.name
        for path in archive.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def available_stage_names(scene_dir: Path) -> list[str]:
    stages = []
    for stage, dirs in STAGE_DIRS.items():
        neus_dir = scene_dir / dirs["neus"]
        mitsuba_name = dirs["mitsuba"]
        mitsuba_dir = scene_dir / mitsuba_name if mitsuba_name else None
        if neus_dir.is_dir() and (mitsuba_dir is None or mitsuba_dir.is_dir()):
            stages.append(stage)
    return stages


def list_checkpoints(args: argparse.Namespace) -> None:
    archive = require_archive(args.checkpoints_dir)
    for scene in scene_names(archive):
        stages = ", ".join(available_stage_names(archive / scene))
        print(f"{scene}: {stages}")


def copy_files(src_dir: Path, dst_dir: Path, force: bool) -> list[Path]:
    if not src_dir.is_dir():
        raise SystemExit(f"Missing checkpoint directory: {src_dir}")

    files = sorted(path for path in src_dir.iterdir() if path.is_file())
    if not files:
        raise SystemExit(f"No checkpoint files found in: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in files:
        dst = dst_dir / src.name
        if dst.exists() and not force:
            raise SystemExit(
                f"Refusing to overwrite existing file: {dst}\n"
                "Pass --force to overwrite destination checkpoint files."
            )
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def install(args: argparse.Namespace) -> None:
    archive = require_archive(args.checkpoints_dir)
    scene_dir = archive / args.scene
    if not scene_dir.is_dir():
        choices = ", ".join(scene_names(archive))
        raise SystemExit(f"Unknown scene: {args.scene}\nAvailable scenes: {choices}")

    stage_dirs = STAGE_DIRS[args.stage]
    exp_dir = args.exp_dir

    copied_neus = copy_files(
        scene_dir / stage_dirs["neus"],
        exp_dir / "checkpoints",
        args.force,
    )

    copied_mitsuba: list[Path] = []
    if stage_dirs["mitsuba"]:
        copied_mitsuba = copy_files(
            scene_dir / stage_dirs["mitsuba"],
            exp_dir / "mitsuba" / "checkpoints",
            args.force,
        )

    print(f"Installed scene={args.scene} stage={args.stage}")
    for path in copied_neus + copied_mitsuba:
        print(f"  {path}")

    print("\nRun notes:")
    print("  is_continue=true")
    if copied_mitsuba:
        print(f"  mitsuba_renderer.out_dir={exp_dir / 'mitsuba'}")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "download":
            download(args)
        elif args.command == "list":
            list_checkpoints(args)
        elif args.command == "install":
            install(args)
        else:
            raise AssertionError(args.command)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
