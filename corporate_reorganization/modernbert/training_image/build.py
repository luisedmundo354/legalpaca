"""Checked no-cache builder for the frozen ARR retrieval training image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence


EXPECTED_TOOLCHAIN = {
    "builder_driver": "docker",
    "buildkit_version": "v0.29.0",
    "buildx_version": "v0.33.0",
}
EXPECTED_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
EXPECTED_CONFIG_DIGEST = (
    "sha256:24784672e3d1f8004fe6577069d6f01393239310276a570f5e8d0db1fe13b85f"
)
EXPECTED_MANIFEST_DIGEST = (
    "sha256:b44c9b182a2490329b25394568299420bcfbe85a8fb17df955378b1f3630d9be"
)
EXPECTED_RUNTIME_INVENTORY_SHA256 = (
    "1151907eb4c0c63a6a317ae11b909ceb7bbbe29d4a56c46d8bec91d8424d795c"
)
EXPECTED_TRAINING_CONTRACT_SHA256 = (
    "db4b2b307a56686054c2c04fbcebf5c133077765074ceef61a613c183a4b04ef"
)
EXPECTED_SOURCE_PARENT = {
    "commit": "b02aa697310c1512fc421d7e4c6c2f81d35ec2e7",
    "commit_epoch": 1783895427,
    "tree": "279db73d13e56087282cfde33bf9e3fc20e4b48d",
}
EXPECTED_SDK_BASE_IMAGE = {
    "resolved_digest_uri": (
        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
        "huggingface-pytorch-training@"
        "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
    ),
    "resolved_tag_uri": (
        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
        "huggingface-pytorch-training:"
        "2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04"
    ),
    "sagemaker_sdk_version": "2.248.2",
}


def _run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(arguments[:3])}): {completed.stderr.strip()}"
        )
    return completed


def validate_toolchain() -> dict[str, str]:
    buildx = _run(("docker", "buildx", "version"))
    version = re.fullmatch(
        r"github\.com/docker/buildx "
        r"(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)) "
        r"[0-9a-f]{40}",
        buildx.stdout.strip(),
    )
    if version is None:
        raise RuntimeError("Unexpected docker buildx version output")
    inspect = _run(("docker", "buildx", "inspect", "--bootstrap"))
    drivers = re.findall(r"^Driver:\s+(\S+)$", inspect.stdout, flags=re.MULTILINE)
    buildkit = re.findall(
        r"^BuildKit version:\s+"
        r"(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))$",
        inspect.stdout,
        flags=re.MULTILINE,
    )
    actual = {
        "builder_driver": drivers[0] if len(drivers) == 1 else "",
        "buildkit_version": buildkit[0] if len(buildkit) == 1 else "",
        "buildx_version": version.group(1),
    }
    if actual != EXPECTED_TOOLCHAIN:
        raise RuntimeError(
            f"Training-image build toolchain changed: actual={actual}, "
            f"expected={EXPECTED_TOOLCHAIN}"
        )
    return actual


def load_build_identity(modernbert_dir: Path) -> dict[str, object]:
    root = Path(modernbert_dir).resolve(strict=True)
    path = root / "training_image/build_identity.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Training build identity must be a regular non-symlink file")
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8"):
        raise ValueError("Training build identity must be canonical pretty JSON")
    expected_keys = {
        "builder",
        "build_exporter",
        "manifest_type",
        "replicas",
        "runtime_inventory_sha256",
        "schema_version",
        "sdk_base_image",
        "source_inventory",
        "source_parent",
        "toolchain",
        "training_contract_sha256",
    }
    if set(value) != expected_keys or value["schema_version"] != 1:
        raise ValueError("Training build identity schema changed")
    if (
        value["manifest_type"] != "arr_retrieval_training_image_build"
        or value["runtime_inventory_sha256"]
        != EXPECTED_RUNTIME_INVENTORY_SHA256
        or value["training_contract_sha256"]
        != EXPECTED_TRAINING_CONTRACT_SHA256
        or value["source_parent"] != EXPECTED_SOURCE_PARENT
        or value["sdk_base_image"] != EXPECTED_SDK_BASE_IMAGE
    ):
        raise ValueError("Training build identity provenance changed")
    if value["toolchain"] != EXPECTED_TOOLCHAIN:
        raise ValueError("Recorded training build toolchain changed")
    expected_replicas = [
        {
            "build_replica": replica,
            "config_digest": EXPECTED_CONFIG_DIGEST,
            "manifest_digest": EXPECTED_MANIFEST_DIGEST,
            "media_type": EXPECTED_MEDIA_TYPE,
        }
        for replica in (1, 2)
    ]
    if value["replicas"] != expected_replicas:
        raise ValueError("Training build replica identities changed")
    builder = value["builder"]
    if type(builder) is not dict or set(builder) != {"path", "sha256", "size"}:
        raise ValueError("Training builder identity schema changed")
    builder_path = root / builder["path"]
    if builder_path.is_symlink() or not builder_path.is_file():
        raise ValueError("Training builder is absent or a symlink")
    builder_bytes = builder_path.read_bytes()
    if (
        builder["path"] != "training_image/build.py"
        or builder["size"] != len(builder_bytes)
        or builder["sha256"] != hashlib.sha256(builder_bytes).hexdigest()
    ):
        raise ValueError("Training builder identity changed")
    source_inventory = value["source_inventory"]
    if type(source_inventory) is not dict or set(source_inventory) != {
        "files",
        "inventory_sha256",
    }:
        raise ValueError("Training source inventory schema changed")
    records = source_inventory["files"]
    if type(records) is not list or not records:
        raise ValueError("Training source inventory must be non-empty")
    if records != sorted(records, key=lambda record: record["path"]):
        raise ValueError("Training source inventory is not sorted")
    for record in records:
        if type(record) is not dict or set(record) != {"path", "sha256", "size"}:
            raise ValueError("Training source record schema changed")
        source = root / record["path"]
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Training source is absent or a symlink: {record['path']}")
        payload = source.read_bytes()
        if (
            len(payload) != record["size"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Training source identity changed: {record['path']}")
    inventory_bytes = (
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(inventory_bytes).hexdigest() != source_inventory[
        "inventory_sha256"
    ]:
        raise ValueError("Training source inventory digest changed")
    return value


def render_build_command(
    modernbert_dir: Path,
    metadata_file: Path,
    *,
    build_replica: int,
    identity: dict[str, object],
) -> tuple[list[str], str]:
    if type(build_replica) is not int or build_replica not in {1, 2}:
        raise ValueError("build_replica must be exact integer 1 or 2")
    root = Path(modernbert_dir).resolve(strict=True)
    metadata = Path(metadata_file)
    if not metadata.is_absolute():
        raise ValueError("metadata_file must be absolute")
    if os.path.lexists(metadata):
        raise FileExistsError(f"Refusing to overwrite build metadata: {metadata}")
    if metadata.parent.is_symlink() or not metadata.parent.is_dir():
        raise ValueError("Build metadata parent must be a real directory")
    exporter = identity["build_exporter"]
    expected_exporter = {
        "compression": "gzip",
        "compression_level": 6,
        "force_compression": False,
        "oci_mediatypes": False,
        "platform": "linux/amd64",
        "provenance": False,
        "rewrite_timestamp": True,
        "sbom": False,
        "type": "image",
        "unpack": False,
    }
    if exporter != expected_exporter:
        raise ValueError("Training build exporter changed")
    parent = identity["source_parent"]
    if type(parent) is not dict or set(parent) != {"commit", "commit_epoch", "tree"}:
        raise ValueError("Training source-parent schema changed")
    replica = identity["replicas"][build_replica - 1]
    if replica.get("build_replica") != build_replica:
        raise ValueError("Training build replica order changed")
    image_name = f"arr-retrieval-train:step10a-bootstrap-build{build_replica}"
    output = ",".join(
        (
            "type=image",
            f"name={image_name}",
            "push=false",
            "rewrite-timestamp=true",
            "unpack=false",
            "compression=gzip",
            "compression-level=6",
            "force-compression=false",
            "oci-mediatypes=false",
        )
    )
    return [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--pull",
        "--no-cache",
        "--provenance=false",
        "--sbom=false",
        "--output",
        output,
        "--build-arg",
        f"SOURCE_DATE_EPOCH={parent['commit_epoch']}",
        "--metadata-file",
        str(metadata),
        "--file",
        str(root / "training_image/Dockerfile"),
        str(root),
    ], image_name


def build_image(
    modernbert_dir: Path,
    metadata_file: Path,
    *,
    build_replica: int,
) -> dict[str, str]:
    root = Path(modernbert_dir).resolve(strict=True)
    validate_toolchain()
    identity = load_build_identity(root)
    command, image_name = render_build_command(
        root,
        Path(metadata_file),
        build_replica=build_replica,
        identity=identity,
    )
    _run(command)
    load_build_identity(root)
    inspected = json.loads(_run(("docker", "image", "inspect", image_name)).stdout)
    if type(inspected) is not list or len(inspected) != 1:
        raise RuntimeError("Docker inspect did not return exactly one training image")
    descriptor = inspected[0].get("Descriptor")
    expected = identity["replicas"][build_replica - 1]
    if (
        type(descriptor) is not dict
        or descriptor.get("mediaType") != EXPECTED_MEDIA_TYPE
        or descriptor.get("digest") != expected["manifest_digest"]
        or descriptor.get("annotations", {}).get("config.digest")
        != expected["config_digest"]
    ):
        raise RuntimeError("Rebuilt training image differs from its accepted replica")
    return {
        "config_digest": expected["config_digest"],
        "image_name": image_name,
        "manifest_digest": expected["manifest_digest"],
        "media_type": EXPECTED_MEDIA_TYPE,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one exact ARR retrieval training-image replica",
        allow_abbrev=False,
    )
    parser.add_argument("--modernbert-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--build-replica", type=int, choices=(1, 2), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_image(
        args.modernbert_dir,
        args.metadata_file,
        build_replica=args.build_replica,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
