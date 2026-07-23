from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_test_subset(zip_path: Path, raw_dir: Path) -> Path:
    test_dir = raw_dir / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        print(f"Using existing extracted test directory: {test_dir}")
        return test_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting test subset from {zip_path} to {raw_dir}")
    subprocess.run(["unzip", "-q", str(zip_path), "test/*", "-d", str(raw_dir)], check=True)
    return test_dir


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_flat_dirs(test_dir: Path, flat_input: Path, flat_target: Path) -> int:
    ensure_clean_dir(flat_input)
    ensure_clean_dir(flat_target)

    count = 0
    for seq_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        blur_dir = seq_dir / "blur"
        sharp_dir = seq_dir / "sharp"
        if not blur_dir.is_dir() or not sharp_dir.is_dir():
            continue
        for blur_path in sorted(blur_dir.glob("*.png")):
            sharp_path = sharp_dir / blur_path.name
            if not sharp_path.exists():
                raise FileNotFoundError(f"Missing sharp image for {blur_path}: {sharp_path}")
            key_name = f"{seq_dir.name}_{blur_path.stem}.png"
            link_or_copy(blur_path.resolve(), flat_input / key_name)
            link_or_copy(sharp_path.resolve(), flat_target / key_name)
            count += 1

    if count == 0:
        raise RuntimeError(f"No GoPro test pairs found under {test_dir}")
    print(f"Prepared {count} flat GoPro test pairs.")
    return count


def create_lmdb(nafnet_root: Path, flat_input: Path, flat_target: Path, input_lmdb: Path, target_lmdb: Path) -> None:
    sys.path.insert(0, str(nafnet_root))
    from basicsr.utils.create_lmdb import prepare_keys
    from basicsr.utils.lmdb_util import make_lmdb_from_imgs

    if input_lmdb.exists():
        print(f"Removing existing LMDB: {input_lmdb}")
        shutil.rmtree(input_lmdb)
    if target_lmdb.exists():
        print(f"Removing existing LMDB: {target_lmdb}")
        shutil.rmtree(target_lmdb)

    img_paths, keys = prepare_keys(str(flat_input), "png")
    make_lmdb_from_imgs(str(flat_input), str(input_lmdb), img_paths, keys, multiprocessing_read=False)

    img_paths, keys = prepare_keys(str(flat_target), "png")
    make_lmdb_from_imgs(str(flat_target), str(target_lmdb), img_paths, keys, multiprocessing_read=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GoPro test LMDBs for NAFNet from GOPRO_Large.zip.")
    parser.add_argument("--zip", required=True, help="Path to GOPRO_Large.zip.")
    parser.add_argument("--raw-dir", default="data/gopro_raw", help="Directory for extracted test images.")
    parser.add_argument("--nafnet-root", default="third_party/NAFNet")
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    nafnet_root = Path(args.nafnet_root).resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"GoPro zip not found: {zip_path}")
    if not nafnet_root.is_dir():
        raise FileNotFoundError(f"NAFNet root not found: {nafnet_root}")

    test_dir = extract_test_subset(zip_path, raw_dir)

    dataset_dir = nafnet_root / "datasets" / "GoPro" / "test"
    flat_input = dataset_dir / "input"
    flat_target = dataset_dir / "target"
    input_lmdb = dataset_dir / "input.lmdb"
    target_lmdb = dataset_dir / "target.lmdb"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    prepare_flat_dirs(test_dir, flat_input, flat_target)
    create_lmdb(nafnet_root, flat_input, flat_target, input_lmdb, target_lmdb)

    print(f"Prepared LMDBs:\n  {input_lmdb}\n  {target_lmdb}")


if __name__ == "__main__":
    main()

