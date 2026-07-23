from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_lmdb_dir(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and ((path / "data.mdb").exists() or (path / "lock.mdb").exists())


def strip_known_suffix(name: str, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> str:
    lowered = name.lower()
    for ext in sorted(extensions, key=len, reverse=True):
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def normalize_image_id(value: str | Path) -> str:
    raw = str(value).replace("\\", "/").strip()
    raw = raw.split()[0] if raw.split() else raw
    raw = raw.strip("/")
    raw = strip_known_suffix(raw)
    return raw.replace("/", "__")


def alias_ids(value: str | Path) -> Set[str]:
    raw = str(value).replace("\\", "/").strip()
    raw = raw.split()[0] if raw.split() else raw
    raw = raw.strip("/")
    no_ext = strip_known_suffix(raw)
    base = strip_known_suffix(Path(raw).name)
    aliases = {normalize_image_id(no_ext), normalize_image_id(base)}
    aliases.discard("")
    return aliases


def safe_filename(image_id: str, suffix: str = ".png") -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in image_id)
    if not safe.lower().endswith(suffix.lower()):
        safe += suffix
    return safe


def read_image_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def scan_image_files(root: str | Path, extensions: Sequence[str] | None = None) -> List[Path]:
    root = Path(root)
    exts = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


class ImageCollection:
    """Read images from either a normal image directory or a BasicSR-style LMDB."""

    def __init__(self, path: str | Path, image_extensions: Sequence[str] | None = None):
        self.path = Path(path)
        self.image_extensions = tuple(image_extensions or sorted(IMAGE_EXTENSIONS))
        self.kind = "lmdb" if is_lmdb_dir(self.path) else "directory"
        self._items: Dict[str, object] = {}
        self._aliases: Dict[str, str] = {}
        self._env = None

        if not self.path.exists():
            raise FileNotFoundError(f"Image collection path does not exist: {self.path}")
        if self.kind == "lmdb":
            self._index_lmdb()
        else:
            self._index_directory()

    def keys(self) -> Set[str]:
        return set(self._aliases.keys())

    def primary_keys(self) -> Set[str]:
        return set(self._items.keys())

    def has(self, image_id: str) -> bool:
        return image_id in self._aliases

    def read(self, image_id: str) -> np.ndarray:
        if image_id not in self._aliases:
            raise KeyError(f"Image id '{image_id}' was not found in {self.path}")
        primary = self._aliases[image_id]
        item = self._items[primary]
        if self.kind == "directory":
            return read_image_rgb(Path(item))
        return self._read_lmdb_value(item)

    def describe(self, image_id: str) -> str:
        if image_id not in self._aliases:
            raise KeyError(f"Image id '{image_id}' was not found in {self.path}")
        primary = self._aliases[image_id]
        item = self._items[primary]
        if self.kind == "directory":
            return str(Path(item))
        key = bytes(item).decode("utf-8", errors="ignore")
        return f"{self.path}:{key}"

    def sample_keys(self, limit: int = 5) -> List[str]:
        return sorted(self.keys())[:limit]

    def _add_item(self, primary: str, item: object, aliases: Iterable[str]) -> None:
        if primary not in self._items:
            self._items[primary] = item
        for alias in aliases:
            self._aliases.setdefault(alias, primary)

    def _index_directory(self) -> None:
        if not self.path.is_dir():
            raise NotADirectoryError(f"Expected an image directory: {self.path}")
        for file_path in scan_image_files(self.path, self.image_extensions):
            rel = file_path.relative_to(self.path)
            primary = normalize_image_id(rel)
            self._add_item(primary, file_path, alias_ids(rel) | alias_ids(file_path.name))
        if not self._items:
            raise FileNotFoundError(f"No image files found under {self.path}")

    def _index_lmdb(self) -> None:
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("lmdb is required to read GoPro LMDB files. Install requirements_extra.txt.") from exc

        self._env = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key_bytes in cursor.iternext(values=False):
                key = key_bytes.decode("utf-8", errors="ignore")
                primary = normalize_image_id(key)
                self._add_item(primary, key_bytes, alias_ids(key))
        if not self._items:
            raise FileNotFoundError(f"No LMDB entries found under {self.path}")

    def _read_lmdb_value(self, key_bytes: bytes) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("LMDB environment is not open")
        with self._env.begin(write=False) as txn:
            value = txn.get(key_bytes)
        if value is None:
            key = key_bytes.decode("utf-8", errors="ignore")
            raise KeyError(f"LMDB key not found: {key}")
        arr = np.frombuffer(value, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            key = key_bytes.decode("utf-8", errors="ignore")
            raise ValueError(f"Could not decode LMDB image for key: {key}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def common_image_ids(*collections: ImageCollection) -> List[str]:
    if not collections:
        return []
    common = set(collections[0].keys())
    for collection in collections[1:]:
        common &= collection.keys()
    return sorted(common)


def assert_same_shape(image_id: str, *images: np.ndarray) -> None:
    shapes = [img.shape for img in images]
    if len(set(shapes)) != 1:
        raise ValueError(f"Image shape mismatch for {image_id}: {shapes}")
