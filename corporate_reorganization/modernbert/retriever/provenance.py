from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Mapping


EXPECTED_BASE_RUNTIME_VERSIONS = {
    "python": "3.11.10",
    "torch": "2.5.1+cu124",
    "transformers": "4.49.0",
    "accelerate": "1.4.0",
    "numpy": "1.26.4",
    "flash-attn": "2.7.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "huggingface-hub": "0.29.1",
}
EXPECTED_RUNTIME_VERSIONS = {
    **EXPECTED_BASE_RUNTIME_VERSIONS,
    "deepspeed": "0.17.1",
    "hjson": "3.1.0",
    "nvidia-ml-py": "13.590.48",
    "py-cpuinfo": "9.0.0",
}

EXPECTED_BASE_TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
EXPECTED_DERIVED_TRAINING_IMAGE = (
    "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@"
    "sha256:b44c9b182a2490329b25394568299420bcfbe85a8fb17df955378b1f3630d9be"
)
EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256 = (
    "1151907eb4c0c63a6a317ae11b909ceb7bbbe29d4a56c46d8bec91d8424d795c"
)
EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256 = (
    "db4b2b307a56686054c2c04fbcebf5c133077765074ceef61a613c183a4b04ef"
)
EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL = "arr_retrieval_training_source_bootstrap_v1"
EXPECTED_SNAPSHOT_TREE_SHA256 = (
    "aca85feea4adb60c4b021eb1a439aff47c844495005f2acdee1baef9d611d63d"
)
EXPECTED_SNAPSHOT_MANIFEST_SHA256 = (
    "0807d16ba5b49a5e30c8b09b72acef7d8c6326823a850640027cc1363ee446b5"
)
EXPECTED_EXPERIMENT_CONFIG_SHA256 = (
    "e51f4e8097f8888adda0382dd5c9377d7fd7417e0356b176f50ab37f7002aa96"
)
EXPECTED_DEEPSPEED_CONFIG_SHA256 = (
    "a4731d98bc8b191761e5ee4cfc451ccef71fa028aa3153a6b0d8204b6b833823"
)
EXPECTED_DATASET_MANIFEST_LOGICAL_PATH = (
    "corporate_reorganization/data/final_annotations_gold/"
    "processed_retrieval_v2/dataset_manifest.json"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be"
)
EXPECTED_DATASET_OUTPUT_SHA256 = {
    "cases.jsonl": "313b53fe32be512c7a4a94ecf9a21b718fa1ee50b92b6877a11c1c89289f443f",
    "corpus.jsonl": "f0abc16886727a3c818201fc4888224edf281c3c711b15685d86fd5d63137474",
    "pools/candidates_by_case.json": (
        "75c33c3fa56e7983532f54e3ac2f6969648c9363bb09cd5a1812073c542b3c5f"
    ),
    "pools/candidates_global.json": (
        "39fb2f3360c66ac33cb1aca6cded3f192c15cb06d4494333ebd2a17d1ffc894d"
    ),
    "queries/all.jsonl": (
        "bcc6e7573009329f50aaa42a483981e9e30c6e3060984dd840f1c0d7e6f66279"
    ),
}
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "469858f2f8e42d0b19e53ee71af690f722482120348a2fe9719b99104758e00d"
)
EXPECTED_FOLD_MANIFEST_LOGICAL_PATH = (
    "corporate_reorganization/modernbert/experiments/retrieval_cv/configs/folds.json"
)
EXPECTED_PASSAGE_INDEX_SHA256 = (
    "641b7a6f9f77d308b9b2b4b38ab2318ffdbc61af4b4ad718caf0d3ad571ec43d"
)
EXPECTED_FOLD_ROTATION_SHA256_BY_OUTER_FOLD = {
    0: "87c919ca238fff044008e4c28887492667f7e71d071a7a8886be03d694df7d17",
    1: "e34788eff4a73c44c1152a229ce73e13bf4872ba5a17e9a446d8cc1ea895cd71",
    2: "e823b3a001c77b26932b0523d875f7c7b46b2acdcddf710378372da270659d25",
    3: "3b8c7a503e605d4cccf014c9e2e6937e59c3667ef1b8ef0d5d2ce4da4f754669",
    4: "6f28da2312ca70323cebec321f59495229e9535ff11c361a03d4b255abe393c6",
}
EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD = {
    0: "402e46f78a3fff52b25cf9fbac1ade6626e6b337c9d705c90b2484f46255f120",
    1: "ec72f31ab89677f405d67e1e4c6e5280ee9d51a84458e986b0913265b0adec16",
    2: "955e124fa756c7b6add85ec61a018aebce3612620734245ae6dcec6660b245a4",
    3: "2a9b13d9806d29ea0dd4511ca6d5484ac65ebddf595a4a2e5bb09003a62f8f5b",
    4: "a1cc62b39dbc4a2e6e6b76ce5a1e04ddd03cc847d017d47502917938fc4d0fab",
}
EXPECTED_VALIDATION_IDENTITY_BY_CELL = {
    (0, "structured"): {
        "case_ids_sha256": "4a0924447060a12c5ba33f3c862124f3348b31f8a1c9f74f0088b88f3d8329ea",
        "query_ids_sha256": "aa01a40b798375a8670434d5d30e0684596e3ee084fbdc7cd5a21156028060ec",
        "passage_ids_sha256": "3bf4c9efb3e1f22c4938fa0d003725e3ce29f33a5e32e151e8f561ac894a5606",
        "contract_sha256": "14dc03cb597e33f7b71526a20f4a09d23b3ea821005f2c08265284058c010770",
    },
    (0, "flat_masked"): {
        "case_ids_sha256": "4a0924447060a12c5ba33f3c862124f3348b31f8a1c9f74f0088b88f3d8329ea",
        "query_ids_sha256": "aa01a40b798375a8670434d5d30e0684596e3ee084fbdc7cd5a21156028060ec",
        "passage_ids_sha256": "3bf4c9efb3e1f22c4938fa0d003725e3ce29f33a5e32e151e8f561ac894a5606",
        "contract_sha256": "3484604032fce5507bc4112d329d023c0b3290afa517fc4d460e215c29fa9cea",
    },
    (1, "structured"): {
        "case_ids_sha256": "224b64bd85153f56133d027b2bc985bf01a1b5abbb2774902ed5695c86d4a250",
        "query_ids_sha256": "b596c20badb9c249bc04e7a73346e6decec53a15bee33a5522b08ef6a2e191a3",
        "passage_ids_sha256": "0748f63691e56a3a0a3c22fceafde5341ae5522b8b8b4cbbf54b10ab49f4c8d5",
        "contract_sha256": "be7272663e29440ddf849c935a331e343069aa3d3fc983e7f820b5303c762149",
    },
    (1, "flat_masked"): {
        "case_ids_sha256": "224b64bd85153f56133d027b2bc985bf01a1b5abbb2774902ed5695c86d4a250",
        "query_ids_sha256": "b596c20badb9c249bc04e7a73346e6decec53a15bee33a5522b08ef6a2e191a3",
        "passage_ids_sha256": "0748f63691e56a3a0a3c22fceafde5341ae5522b8b8b4cbbf54b10ab49f4c8d5",
        "contract_sha256": "a4f64a7a4d507526882852351bf49765218edfba03fe57da42a95e55dce6461a",
    },
    (2, "structured"): {
        "case_ids_sha256": "07ce49a5ffd2a8faa7beb9a613f5ba81941077c06e3d3bd8ef3b91485518f3a4",
        "query_ids_sha256": "14a61d76862bff28a71b366c1bc29871afd46e61f41282d7ce7eeaa81ab47201",
        "passage_ids_sha256": "25d826cf0c93ab71f5fa22a94c66240122e3a8b04c3a1195665d22b09b4306f4",
        "contract_sha256": "e0701a4bfa71749398dc532be889e50fd42db51992f353e1041f49e22eb33c4e",
    },
    (2, "flat_masked"): {
        "case_ids_sha256": "07ce49a5ffd2a8faa7beb9a613f5ba81941077c06e3d3bd8ef3b91485518f3a4",
        "query_ids_sha256": "14a61d76862bff28a71b366c1bc29871afd46e61f41282d7ce7eeaa81ab47201",
        "passage_ids_sha256": "25d826cf0c93ab71f5fa22a94c66240122e3a8b04c3a1195665d22b09b4306f4",
        "contract_sha256": "dc1d07ada4319da0d1c60c2f326957b36074d20c69bce3004cfb1f3bbf05e374",
    },
    (3, "structured"): {
        "case_ids_sha256": "83ccb7dc548206bf501bb951a1352302dd025256125ef0a4da60b449c7921cd8",
        "query_ids_sha256": "a7f5201b6022b82f1744fd359f0bda7116ed73c642d4dca041f975051c602367",
        "passage_ids_sha256": "0b394ebf7300e26203258c65c0e47db615c4b73245c91d96f334f5a7a263c020",
        "contract_sha256": "de7107984e9563640ecc515e900c74e273cc9a3c85d05f0f4bd506c7409e8c00",
    },
    (3, "flat_masked"): {
        "case_ids_sha256": "83ccb7dc548206bf501bb951a1352302dd025256125ef0a4da60b449c7921cd8",
        "query_ids_sha256": "a7f5201b6022b82f1744fd359f0bda7116ed73c642d4dca041f975051c602367",
        "passage_ids_sha256": "0b394ebf7300e26203258c65c0e47db615c4b73245c91d96f334f5a7a263c020",
        "contract_sha256": "8438ec9976729ea5fd9dce1c18215c44ce2704da0678627dc321809864279a8d",
    },
    (4, "structured"): {
        "case_ids_sha256": "bd11b0cfd2ee549a9563ff3c97eef1cabd8a3493e55b46015fc8ed6d26894369",
        "query_ids_sha256": "bff10ad25e147b98a1fb86d35723e19fe4347ee9fc2892f19ab19562114efbef",
        "passage_ids_sha256": "1f73506fecbd6d7eb0e12a3b56b1cedb1627b7aef940387916615c6a144a8e67",
        "contract_sha256": "5c369ec578187fc04f030bdcb78f5e43fd8fcb68d135a0d34d3ddc007aec82a9",
    },
    (4, "flat_masked"): {
        "case_ids_sha256": "bd11b0cfd2ee549a9563ff3c97eef1cabd8a3493e55b46015fc8ed6d26894369",
        "query_ids_sha256": "bff10ad25e147b98a1fb86d35723e19fe4347ee9fc2892f19ab19562114efbef",
        "passage_ids_sha256": "1f73506fecbd6d7eb0e12a3b56b1cedb1627b7aef940387916615c6a144a8e67",
        "contract_sha256": "5fd20c9bf25c598830e63664920e0e5687500354a6f184661d19d6915df4372c",
    },
}


def _is_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None

_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_version_inventory() -> dict[str, str]:
    inventory = {"python": platform.python_version()}
    for package in EXPECTED_RUNTIME_VERSIONS:
        if package == "python":
            continue
        try:
            inventory[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required runtime package is absent: {package}") from exc
    return inventory


def validate_runtime_versions(actual: Mapping[str, str] | None = None) -> dict[str, str]:
    inventory = dict(runtime_version_inventory() if actual is None else actual)
    if inventory != EXPECTED_RUNTIME_VERSIONS:
        missing = sorted(set(EXPECTED_RUNTIME_VERSIONS) - set(inventory))
        extra = sorted(set(inventory) - set(EXPECTED_RUNTIME_VERSIONS))
        mismatched = {
            name: {"expected": EXPECTED_RUNTIME_VERSIONS[name], "actual": inventory[name]}
            for name in sorted(set(inventory).intersection(EXPECTED_RUNTIME_VERSIONS))
            if inventory[name] != EXPECTED_RUNTIME_VERSIONS[name]
        }
        raise RuntimeError(
            "Training runtime does not match the frozen inventory: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    return inventory


def validate_preimport_environment(experiment_seed: int) -> None:
    if type(experiment_seed) is not int or experiment_seed < 0:
        raise ValueError("experiment_seed must be a non-negative exact int")
    expected = {
        "PYTHONHASHSEED": str(experiment_seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
    }
    mismatched = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatched:
        raise RuntimeError(
            "Required pre-import deterministic/offline environment is not exact: "
            f"{mismatched}"
        )


def validate_training_image_environment() -> dict[str, Any]:
    expected = {
        "ARR_TRAINING_IMAGE_URI": EXPECTED_DERIVED_TRAINING_IMAGE,
        "ARR_TRAINING_BASE_IMAGE_URI": EXPECTED_BASE_TRAINING_IMAGE,
        "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256": (
            EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        ),
    }
    mismatched = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatched:
        raise RuntimeError(
            "Training image provenance environment is not exact: "
            f"{mismatched}"
        )
    verified_pairs = {
        "name": ("ARR_SOURCE_BUNDLE_NAME", "ARR_VERIFIED_SOURCE_BUNDLE_NAME"),
        "size": ("ARR_SOURCE_BUNDLE_SIZE", "ARR_VERIFIED_SOURCE_BUNDLE_SIZE"),
        "sha256": ("ARR_SOURCE_BUNDLE_SHA256", "ARR_VERIFIED_SOURCE_BUNDLE_SHA256"),
        "inventory_sha256": (
            "ARR_SOURCE_INVENTORY_SHA256",
            "ARR_VERIFIED_SOURCE_INVENTORY_SHA256",
        ),
        "commit_epoch": (
            "ARR_SOURCE_COMMIT_EPOCH",
            "ARR_VERIFIED_SOURCE_COMMIT_EPOCH",
        ),
    }
    source = {
        name: os.environ.get(requested)
        for name, (requested, verified) in verified_pairs.items()
        if os.environ.get(requested) == os.environ.get(verified)
    }
    if set(source) != set(verified_pairs):
        changed = {
            name: {
                "requested": os.environ.get(requested),
                "verified": os.environ.get(verified),
            }
            for name, (requested, verified) in verified_pairs.items()
            if os.environ.get(requested) != os.environ.get(verified)
        }
        raise RuntimeError(f"Verified source bootstrap identity changed: {changed}")
    source_name = source["name"]
    source_sha256 = source["sha256"]
    source_inventory_sha256 = source["inventory_sha256"]
    if (
        type(source_name) is not str
        or type(source_sha256) is not str
        or source_name != f"source-{source_sha256}.tar.gz"
        or not _is_sha256(source_sha256)
        or not _is_sha256(source_inventory_sha256)
    ):
        raise RuntimeError("Verified source bundle name/hash/inventory is invalid")
    for name in ("size", "commit_epoch"):
        value = source[name]
        if (
            type(value) is not str
            or not value.isascii()
            or not value.isdecimal()
            or value.startswith("0")
            or int(value) < 1
        ):
            raise RuntimeError(f"Verified source {name} is not a positive canonical integer")
    verified_runtime = {
        "ARR_VERIFIED_TRAINING_BOOTSTRAP_PROTOCOL": (
            EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL
        ),
        "ARR_VERIFIED_TRAINING_CONTRACT_SHA256": (
            EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256
        ),
        "ARR_VERIFIED_TRAINING_RUNTIME_INVENTORY_SHA256": (
            EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        ),
    }
    changed_runtime = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in verified_runtime.items()
        if os.environ.get(name) != value
    }
    if changed_runtime:
        raise RuntimeError(
            f"Verified training bootstrap/runtime identity changed: {changed_runtime}"
        )
    plan_sha256 = os.environ.get("ARR_TRAINING_PLAN_SHA256")
    staging_sha256 = os.environ.get("ARR_TRAINING_STAGING_RECEIPT_SHA256")
    if not _is_sha256(plan_sha256) or not _is_sha256(staging_sha256):
        raise RuntimeError("Training plan/staging receipt identity is not lowercase SHA-256")
    return {
        "bootstrap_protocol": EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
        "source_bundle": {
            "commit_epoch": int(source["commit_epoch"]),
            "inventory_sha256": source_inventory_sha256,
            "name": source_name,
            "sha256": source_sha256,
            "size": int(source["size"]),
        },
        "training_image_contract_sha256": (
            EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256
        ),
        "training_plan_sha256": plan_sha256,
        "training_staging_receipt_sha256": staging_sha256,
    }


def load_snapshot_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Snapshot manifest must be a regular file: {path}")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    canonical = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if raw != canonical:
        raise ValueError(f"Snapshot manifest is not canonical JSON: {path}")
    if type(manifest) is not dict:
        raise TypeError("Snapshot manifest must be a JSON object")

    expected_keys = {
        "schema_version",
        "manifest_type",
        "model_id",
        "revision",
        "tree_sha256",
        "files",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"Snapshot manifest fields mismatch: missing={sorted(expected_keys - set(manifest))}, "
            f"extra={sorted(set(manifest) - expected_keys)}"
        )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Snapshot manifest schema_version must be exact integer 1")
    if manifest["manifest_type"] != "huggingface_model_snapshot":
        raise ValueError("Unexpected snapshot manifest_type")
    if manifest["model_id"] != "answerdotai/ModernBERT-base":
        raise ValueError("Unexpected snapshot model_id")
    if manifest["revision"] != "8949b909ec900327062f0ebf497f51aef5e6f0c8":
        raise ValueError("Unexpected snapshot revision")
    if (
        type(manifest["tree_sha256"]) is not str
        or _LOWER_SHA256.fullmatch(manifest["tree_sha256"]) is None
    ):
        raise ValueError("Snapshot tree_sha256 must be lowercase 64-hex")
    if type(manifest["files"]) is not list or not manifest["files"]:
        raise ValueError("Snapshot files must be a non-empty JSON list")

    expected_file_keys = {"path", "size", "sha256"}
    paths: list[str] = []
    for record in manifest["files"]:
        if type(record) is not dict or set(record) != expected_file_keys:
            raise ValueError("Every snapshot file record must contain exactly path, size, and sha256")
        relative_path = record["path"]
        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path != Path(relative_path).name
            or relative_path.strip() != relative_path
        ):
            raise ValueError(f"Snapshot file path must be one root-level filename: {relative_path!r}")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"Snapshot size must be a non-negative exact int: {relative_path}")
        if (
            type(record["sha256"]) is not str
            or _LOWER_SHA256.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Snapshot sha256 must be lowercase 64-hex: {relative_path}")
        paths.append(relative_path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Snapshot file records must be unique and sorted by path")

    expected_tree_hash = hashlib.sha256(_canonical_json_bytes(manifest["files"])).hexdigest()
    if manifest["tree_sha256"] != expected_tree_hash:
        raise ValueError(
            f"Snapshot tree hash mismatch: recorded={manifest['tree_sha256']}, "
            f"expected={expected_tree_hash}"
        )
    if manifest["tree_sha256"] != EXPECTED_SNAPSHOT_TREE_SHA256:
        raise ValueError(
            "Snapshot manifest is not the frozen ModernBERT tree: "
            f"actual={manifest['tree_sha256']}, expected={EXPECTED_SNAPSHOT_TREE_SHA256}"
        )
    return manifest


def validate_snapshot_directory(snapshot_dir: Path, manifest: Mapping[str, Any]) -> None:
    if not snapshot_dir.is_dir() or snapshot_dir.is_symlink():
        raise ValueError(f"Snapshot directory must be a real directory: {snapshot_dir}")
    expected_records = {record["path"]: record for record in manifest["files"]}
    actual_names = sorted(path.name for path in snapshot_dir.iterdir())
    expected_names = sorted(expected_records)
    if actual_names != expected_names:
        raise ValueError(
            f"Snapshot directory inventory mismatch: actual={actual_names}, expected={expected_names}"
        )

    for name in expected_names:
        path = snapshot_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Snapshot entry must be a regular non-symlink file: {path}")
        record = expected_records[name]
        actual_size = path.stat().st_size
        if actual_size != record["size"]:
            raise ValueError(
                f"Snapshot size mismatch for {name}: actual={actual_size}, expected={record['size']}"
            )
        actual_hash = _sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(
                f"Snapshot SHA-256 mismatch for {name}: actual={actual_hash}, expected={record['sha256']}"
            )
