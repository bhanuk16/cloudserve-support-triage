"""One-time install of the embedding model, with integrity checking.

    python -m src.setup_models                     # download (about 80 MB)
    python -m src.setup_models --archive onnx.tar.gz   # install from a file you already have

The archive is the ONNX export of all-MiniLM-L6-v2 that Chroma distributes. Its
SHA-256 is pinned below; a file that does not match is rejected, so a corrupted
or tampered download can never be loaded. Extraction refuses absolute paths and
'..' entries (path-traversal protection).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .config import settings

URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
REQUIRED = ("model.onnx", "tokenizer.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_members(tar: tarfile.TarFile):
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk():
            raise RuntimeError(f"refusing unsafe archive entry: {m.name}")
        yield m


def install(archive: Path, target: Path) -> None:
    digest = sha256(archive)
    if digest != SHA256:
        raise RuntimeError(f"checksum mismatch for {archive}: got {digest}, expected {SHA256}")
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive, "r:gz") as tar:
        tar.extractall(tmp, members=list(_safe_members(tar)))
        found = {p.name: p for p in Path(tmp).rglob("*") if p.is_file()}
        missing = [n for n in REQUIRED if n not in found]
        if missing:
            raise RuntimeError(f"archive is missing {missing}")
        target.mkdir(parents=True, exist_ok=True)
        for name, src in found.items():
            shutil.copy2(src, target / name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="install from a local onnx.tar.gz instead of downloading")
    ap.add_argument("--force", action="store_true", help="reinstall even if already present")
    args = ap.parse_args(argv)

    target = settings.model_dir
    if not args.force and all((target / n).exists() for n in REQUIRED):
        print(f"Embedding model already installed in {target}")
        return 0

    if args.archive:
        install(args.archive, target)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "onnx.tar.gz"
            print(f"Downloading {URL} ...")
            try:
                urllib.request.urlretrieve(URL, dest)
            except Exception as exc:  # noqa: BLE001 - surface a readable message
                print(f"Download failed: {exc}\nDownload the file manually from {URL} "
                      f"and run: python -m src.setup_models --archive <path>", file=sys.stderr)
                return 1
            install(dest, target)
    print(f"Embedding model installed in {target} (checksum verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
