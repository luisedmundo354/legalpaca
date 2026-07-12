from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rankers import complete_bm25_scores_from_hits


BM25_RUNTIME_PROTOCOL = "pyserini_1_5_0_sparse_jni_only_v1"
BM25_K1 = 0.9
BM25_B = 0.4
PYSERINI_VERSION = "1.5.0"
PYJNIUS_VERSION = "1.7.0"
ANSERINI_JAR_NAME = "anserini-1.5.0-fatjar.jar"
ANSERINI_JAR_SIZE = 163_855_488
ANSERINI_JAR_SHA256 = "bb0761df51ef7db5be361199a40a45722cccf7f0b2271e2b25337e97dd578aea"
CORRETTO_JAVA_HOME = "/opt/amazon-corretto-21"
CORRETTO_VERSION_MARKER = "21.0.11"
BM25_INDEX_ARGUMENTS = (
    "--collection",
    "JsonCollection",
    "--generator",
    "DefaultLuceneDocumentGenerator",
    "--threads",
    "1",
    "--storePositions",
    "--storeDocvectors",
    "--storeRaw",
)


@dataclass(frozen=True)
class Bm25RuntimeIdentity:
    protocol: str
    java_home: str
    java_version: str
    pyserini: str
    pyjnius: str
    anserini_jar_size: int
    anserini_jar_sha256: str

    def to_payload(self) -> dict[str, str | int]:
        return {
            "protocol": self.protocol,
            "java_home": self.java_home,
            "java_version": self.java_version,
            "pyserini": self.pyserini,
            "pyjnius": self.pyjnius,
            "anserini_jar_size": self.anserini_jar_size,
            "anserini_jar_sha256": self.anserini_jar_sha256,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pyserini_root() -> Path:
    spec = importlib.util.find_spec("pyserini")
    locations = None if spec is None else spec.submodule_search_locations
    if locations is None:
        raise RuntimeError("Pinned Pyserini package is absent")
    roots = tuple(Path(value) for value in locations)
    if len(roots) != 1:
        raise RuntimeError(f"Pyserini package must have one root; found={roots}")
    root = roots[0]
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Pyserini package root must be a real directory: {root}")
    return root


def locate_anserini_jar() -> Path:
    jar_dir = _pyserini_root() / "resources/jars"
    if jar_dir.is_symlink() or not jar_dir.is_dir():
        raise RuntimeError("Pyserini Anserini JAR directory is absent or a symlink")
    jars = tuple(sorted(path for path in jar_dir.iterdir() if path.is_file()))
    if tuple(path.name for path in jars) != (ANSERINI_JAR_NAME,):
        raise RuntimeError(
            "Pyserini must contain exactly the frozen Anserini JAR: "
            f"actual={[path.name for path in jars]}"
        )
    jar = jars[0]
    if jar.is_symlink():
        raise RuntimeError("Pinned Anserini JAR must not be a symlink")
    if jar.stat().st_size != ANSERINI_JAR_SIZE:
        raise RuntimeError(
            f"Anserini JAR size changed: actual={jar.stat().st_size}, "
            f"expected={ANSERINI_JAR_SIZE}"
        )
    actual_sha256 = _sha256_file(jar)
    if actual_sha256 != ANSERINI_JAR_SHA256:
        raise RuntimeError(
            f"Anserini JAR hash changed: actual={actual_sha256}, "
            f"expected={ANSERINI_JAR_SHA256}"
        )
    return jar


def validate_bm25_runtime() -> Bm25RuntimeIdentity:
    if "jnius" in sys.modules or "pyserini.pyclass" in sys.modules:
        raise RuntimeError(
            "BM25 runtime validation must run before the process initializes PyJNIus"
        )
    java_home = os.environ.get("JAVA_HOME")
    if java_home != CORRETTO_JAVA_HOME:
        raise RuntimeError(
            f"JAVA_HOME changed: actual={java_home!r}, expected={CORRETTO_JAVA_HOME!r}"
        )
    java_binary = Path(CORRETTO_JAVA_HOME) / "bin/java"
    if java_binary.is_symlink() or not java_binary.is_file():
        raise RuntimeError(f"Pinned Corretto binary is absent or a symlink: {java_binary}")
    resolved_java = shutil.which("java")
    if resolved_java != str(java_binary):
        raise RuntimeError(
            f"PATH selected the wrong Java binary: actual={resolved_java!r}, "
            f"expected={str(java_binary)!r}"
        )
    completed = subprocess.run(
        [str(java_binary), "-version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    java_version = (completed.stderr or completed.stdout).strip()
    if completed.returncode != 0 or CORRETTO_VERSION_MARKER not in java_version:
        raise RuntimeError(
            "Pinned Corretto version check failed: "
            f"returncode={completed.returncode}, output={java_version!r}"
        )
    versions = {
        "pyserini": importlib.metadata.version("pyserini"),
        "pyjnius": importlib.metadata.version("pyjnius"),
    }
    if versions != {"pyserini": PYSERINI_VERSION, "pyjnius": PYJNIUS_VERSION}:
        raise RuntimeError(f"Pinned sparse package versions changed: {versions}")
    locate_anserini_jar()
    return Bm25RuntimeIdentity(
        protocol=BM25_RUNTIME_PROTOCOL,
        java_home=java_home,
        java_version=java_version,
        pyserini=versions["pyserini"],
        pyjnius=versions["pyjnius"],
        anserini_jar_size=ANSERINI_JAR_SIZE,
        anserini_jar_sha256=ANSERINI_JAR_SHA256,
    )


def _exact_text_mapping(
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    name: str,
) -> tuple[tuple[str, str], ...]:
    if (
        not isinstance(ids, Sequence)
        or isinstance(ids, (str, bytes))
        or not isinstance(texts, Sequence)
        or isinstance(texts, (str, bytes))
        or not ids
        or len(ids) != len(texts)
    ):
        raise ValueError(f"{name} IDs/texts must be aligned non-empty sequences")
    records: list[tuple[str, str]] = []
    for position, (identity, text) in enumerate(zip(ids, texts)):
        if (
            type(identity) is not str
            or not identity
            or identity.strip() != identity
            or type(text) is not str
            or not text.strip()
        ):
            raise ValueError(f"{name}[{position}] has an invalid identity or text")
        records.append((identity, text))
    if [identity for identity, _ in records] != sorted(identity for identity, _ in records):
        raise ValueError(f"{name} IDs must be lexicographically sorted")
    if len(records) != len({identity for identity, _ in records}):
        raise ValueError(f"{name} IDs contain duplicates")
    return tuple(records)


def build_bm25_index(
    *,
    passage_ids: Sequence[str],
    passage_texts: Sequence[str],
    scratch_dir: Path,
) -> Path:
    records = _exact_text_mapping(
        passage_ids,
        passage_texts,
        name="BM25 passages",
    )
    scratch_dir = Path(scratch_dir)
    if scratch_dir.exists() or scratch_dir.is_symlink():
        raise FileExistsError(f"BM25 scratch directory must be absent: {scratch_dir}")
    if scratch_dir.parent.is_symlink() or not scratch_dir.parent.is_dir():
        raise ValueError("BM25 scratch parent must be a real existing directory")
    scratch_dir.mkdir()
    collection_dir = scratch_dir / "collection"
    index_dir = scratch_dir / "index"
    collection_dir.mkdir()
    collection_path = collection_dir / "passages.jsonl"
    with collection_path.open("xb") as destination:
        for passage_id, text in records:
            payload = json.dumps(
                {"id": passage_id, "contents": text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            destination.write((payload + "\n").encode("utf-8"))
        destination.flush()
        os.fsync(destination.fileno())
    directory_fd = os.open(collection_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--input",
        str(collection_dir),
        "--index",
        str(index_dir),
        *BM25_INDEX_ARGUMENTS,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=dict(os.environ),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Pinned Pyserini indexing failed: "
            f"returncode={completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}"
        )
    if index_dir.is_symlink() or not index_dir.is_dir():
        raise RuntimeError("Pinned Pyserini did not create a real index directory")
    index_files = tuple(path for path in index_dir.iterdir() if path.is_file())
    if not index_files:
        raise RuntimeError("Pinned Pyserini created an empty index directory")
    return index_dir


class PinnedBm25Searcher:
    def __init__(self, index_dir: Path, *, expected_document_count: int) -> None:
        if type(expected_document_count) is not int or expected_document_count < 1:
            raise ValueError("expected_document_count must be a positive exact integer")
        index_dir = Path(index_dir)
        if index_dir.is_symlink() or not index_dir.is_dir():
            raise ValueError(f"BM25 index must be a real directory: {index_dir}")
        if "jnius" in sys.modules or "pyserini.pyclass" in sys.modules:
            raise RuntimeError(
                "PinnedBm25Searcher must be the first PyJNIus initialization in this process"
            )
        locate_anserini_jar()
        from pyserini.pyclass import autoclass

        searcher_class = autoclass("io.anserini.search.SimpleSearcher")
        self._searcher = searcher_class(str(index_dir))
        actual_count = int(self._searcher.get_total_num_docs())
        if actual_count != expected_document_count:
            self.close()
            raise RuntimeError(
                f"BM25 index document count changed: actual={actual_count}, "
                f"expected={expected_document_count}"
            )
        self._searcher.set_bm25(BM25_K1, BM25_B)
        self._closed = False

    def search(self, query: str, *, k: int) -> tuple[dict[str, str | float], ...]:
        if self._closed:
            raise RuntimeError("BM25 searcher is closed")
        if type(query) is not str or not query.strip():
            raise ValueError("BM25 query must be a non-empty exact string")
        if type(k) is not int or k < 1:
            raise ValueError("BM25 k must be a positive exact integer")
        hits = self._searcher.search(query, k)
        return tuple(
            {"passage_id": str(hit.docid), "score": float(hit.score)}
            for hit in hits
        )

    def close(self) -> None:
        if getattr(self, "_closed", False):
            raise RuntimeError("BM25 searcher was closed more than once")
        searcher = getattr(self, "_searcher", None)
        if searcher is None:
            raise RuntimeError("BM25 searcher was not initialized")
        searcher.close()
        self._closed = True

    def __enter__(self) -> "PinnedBm25Searcher":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def build_and_score_bm25(
    *,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    passage_ids: Sequence[str],
    passage_texts: Sequence[str],
    scratch_dir: Path,
    torch_module: Any,
):
    validate_bm25_runtime()
    queries = _exact_text_mapping(query_ids, query_texts, name="BM25 queries")
    passages = _exact_text_mapping(passage_ids, passage_texts, name="BM25 passages")
    index_dir = build_bm25_index(
        passage_ids=[identity for identity, _ in passages],
        passage_texts=[text for _, text in passages],
        scratch_dir=scratch_dir,
    )
    hits_by_query: dict[str, tuple[Mapping[str, object], ...]] = {}
    with PinnedBm25Searcher(
        index_dir,
        expected_document_count=len(passages),
    ) as searcher:
        for query_id, query_text in queries:
            hits_by_query[query_id] = searcher.search(query_text, k=len(passages))
    return complete_bm25_scores_from_hits(
        query_ids=[identity for identity, _ in queries],
        passage_ids=[identity for identity, _ in passages],
        hits_by_query=hits_by_query,
        torch_module=torch_module,
    )
