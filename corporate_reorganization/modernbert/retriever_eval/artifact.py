from __future__ import annotations

import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelArtifactRef:
    source: str
    local_dir: Path


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {s3_uri}")
    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def _download_s3_to_path(s3_uri: str, dst_path: Path) -> None:
    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dst_path))


def extract_model_tar_gz(model_tar_gz: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar_gz, "r:gz") as tar:
        tar.extractall(path=dst_dir)

    model_safetensors = dst_dir / "model.safetensors"
    if model_safetensors.exists():
        return dst_dir

    nested_model = next(dst_dir.glob("**/model.safetensors"), None)
    if nested_model is None:
        raise FileNotFoundError(f"Could not find model.safetensors under: {dst_dir}")
    return nested_model.parent


def resolve_model_artifact(
    *,
    model_dir: Optional[str],
    model_s3_uri: Optional[str],
    work_dir: Optional[str],
) -> ModelArtifactRef:
    if model_dir:
        local_dir = Path(model_dir).expanduser().resolve()
        return ModelArtifactRef(source=f"dir:{local_dir}", local_dir=local_dir)

    if not model_s3_uri:
        raise ValueError("Provide --model_dir or --model_s3_uri")

    base_work_dir = Path(work_dir).expanduser().resolve() if work_dir else None
    if base_work_dir is None:
        base_work_dir = Path(tempfile.mkdtemp(prefix="cr_model_"))
    else:
        base_work_dir.mkdir(parents=True, exist_ok=True)

    model_tar_path = base_work_dir / "model.tar.gz"
    _download_s3_to_path(model_s3_uri, model_tar_path)

    extracted_root = base_work_dir / "extracted"
    model_root = extract_model_tar_gz(model_tar_path, extracted_root)
    return ModelArtifactRef(source=model_s3_uri, local_dir=model_root)


def cleanup_model_artifact(model_artifact: ModelArtifactRef) -> None:
    if model_artifact.source.startswith("dir:"):
        return
    local_dir = model_artifact.local_dir
    tmp_root = local_dir
    for _ in range(5):
        if tmp_root.parent == tmp_root:
            break
        tmp_root = tmp_root.parent
    if tmp_root.name.startswith("cr_model_"):
        shutil.rmtree(tmp_root, ignore_errors=True)

