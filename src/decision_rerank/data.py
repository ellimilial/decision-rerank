"""Fetch missing public data, then reproduce the hash-pinned benchmark artifacts."""
from hashlib import sha256
import json
import logging
from pathlib import Path
import tempfile
import time
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def read_range(request: Request, offset: int, end: int, size: int) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                if response.status == 206:
                    if response.headers.get("Content-Range") != f"bytes {offset}-{end}/{size}":
                        raise ValueError("Unexpected download range")
                    expected = end - offset + 1
                elif response.status == 200 and offset == 0:
                    expected = size
                else:
                    raise ValueError("Source did not return the requested bytes")
                body = response.read(expected + 1)
            if len(body) != expected:
                raise OSError("Incomplete download")
            return body
        except (OSError, URLError, IncompleteRead) as error:
            if isinstance(error, HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(.25 * 2 ** attempt)
    raise AssertionError("Unreachable")


def check_file(path: Path, expected_sha256: str) -> None:
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"Artifact hash changed: {path}")


def download_file(root: Path, source: dict) -> Path:
    """Cache a pinned source atomically. Existing corrupt files are never overwritten."""
    relative = Path(source["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid data path: {relative}")
    path = Path(root) / relative
    if path.exists():
        check_file(path, source["sha256"])
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    size = source["bytes"]
    logger.info("Downloading %s (%s bytes)", relative, size)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            offset = 0
            while offset < size:
                end = min(offset + 4 * 1024 * 1024, size) - 1
                request = Request(source["url"], headers={
                    "User-Agent": "decision-rerank/0.1", "Range": f"bytes={offset}-{end}",
                })
                body = read_range(request, offset, end, size)
                stream.write(body)
                offset += len(body)
            stream.flush()
            check_file(temporary, source["sha256"])
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
    return path


def ensure_benchmark_data(root: Path, name: str, *, offline: bool = False) -> None:
    """Download/build only the requested benchmark, using the repository's manifests.

    Cached runtime files need no network or preparation imports. Missing files are
    rebuilt in a temporary directory, verified, then installed without changing
    the authoritative manifests or existing files.
    """
    root = Path(root).resolve()
    directory = root / "research/benchmarks"
    prepared = json.loads((directory / "prepared-manifest.json").read_text())
    evidence = json.loads((directory / "evidence-manifest.json").read_text())
    check_file(root / evidence["prepared_manifest"]["path"], evidence["prepared_manifest"]["sha256"])
    collections = [c for c in prepared["collections"] if c["benchmark"] == name]
    if not collections:
        raise ValueError(f"Unknown benchmark: {name}")
    shared = {c["name"]: c for c in evidence["collections"]}
    missing = []
    for collection in collections:
        descriptors = [collection["files"][key] for key in ["candidates", "qrels"]]
        descriptors += [shared[collection["name"]]["files"][key] for key in ["documents", "queries"]]
        for descriptor in descriptors:
            path = root / descriptor["path"]
            if path.exists():
                check_file(path, descriptor["sha256"])
            else:
                missing.append(collection["name"])
    if not missing:
        return
    if offline:
        raise FileNotFoundError(f"Missing benchmark data for {name}; offline=True prevents downloads")
    from .preparation import check_dependencies, prepare_collection
    check_dependencies(prepared["dependencies"], evidence["library_version"])
    for source in evidence["tokenizer_files"]:
        download_file(root, source)
    for collection in collections:
        if collection["name"] not in missing:
            continue
        logger.info("Preparing %s", collection["name"])
        for source in collection["sources"]:
            download_file(root, source)
        with tempfile.TemporaryDirectory(prefix="decision-rerank-prepare-", dir=root / "data") as folder:
            staged = Path(folder)
            prepare_collection(root, staged, collection, shared[collection["name"]])
            descriptors = [collection["files"][key] for key in ["candidates", "qrels"]]
            descriptors += [shared[collection["name"]]["files"][key] for key in ["documents", "queries"]]
            for descriptor in descriptors:
                check_file(staged / descriptor["path"], descriptor["sha256"])
            for descriptor in descriptors:
                target = root / descriptor["path"]
                if target.exists():
                    check_file(target, descriptor["sha256"])
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (staged / descriptor["path"]).replace(target)
