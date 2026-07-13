from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


BASE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
BASE_IMAGE_DIGEST = BASE_IMAGE_URI.rsplit("@", 1)[1]
ENTRYPOINT = ["bash", "-m", "start_with_right_hostname.sh"]
CMD = ["/bin/bash"]
ENTRYPOINT_SCRIPT = "/usr/local/bin/start_with_right_hostname.sh"
ENTRYPOINT_SCRIPT_SHA256 = (
    "680a39c5aa0797febfd91c3bc0cef0a7125ef95f80db385c55762696ef845fc9"
)
TRAINING_MODULE = "sagemaker_pytorch_container.training:main"
ALLOWED_PYTHON_HASH_SEEDS = ["17", "29", "43"]
BOOTSTRAP_PATH = "/opt/training_bootstrap/bootstrap.py"
BOOTSTRAP_PROTOCOL = "arr_retrieval_training_source_bootstrap_v1"
BOOTSTRAP_ENTRYPOINT = "train_sm.py"
BOOTSTRAP_SHA256 = (
    "a0f47d63e01209432bcaba2241ec93be77d2b81c05802c97f570843eddc7d5e1"
)
EXPECTED_BOOTSTRAP = {
    "entrypoint": BOOTSTRAP_ENTRYPOINT,
    "path": BOOTSTRAP_PATH,
    "protocol": BOOTSTRAP_PROTOCOL,
    "sha256": BOOTSTRAP_SHA256,
}

EXPECTED_PACKAGES = {
    "accelerate": "1.4.0",
    "deepspeed": "0.17.1",
    "flash-attn": "2.7.3",
    "hjson": "3.1.0",
    "huggingface-hub": "0.29.1",
    "numpy": "1.26.4",
    "nvidia-ml-py": "13.590.48",
    "packaging": "24.1",
    "py-cpuinfo": "9.0.0",
    "safetensors": "0.5.3",
    "sagemaker_pytorch_training": "2.8.1",
    "sagemaker_training": "5.0.0",
    "tokenizers": "0.21.4",
    "torch": "2.5.1+cu124",
    "transformers": "4.49.0",
}
EXPECTED_REQUIREMENTS = [
    {
        "name": "deepspeed",
        "sha256": "6d6e21796982b9e024f489e1c211666cc6c0be6e344751368610b9d2da285d6e",
        "url": (
            "https://files.pythonhosted.org/packages/38/10/"
            "a7f63e086c1e1c12e290c98363c748ef5ddd6313fde739d2aeccd5ed0cd4/"
            "deepspeed-0.17.1.tar.gz"
        ),
        "version": "0.17.1",
    },
    {
        "name": "hjson",
        "sha256": "65713cdcf13214fb554eb8b4ef803419733f4f5e551047c9b711098ab7186b89",
        "url": (
            "https://files.pythonhosted.org/packages/1f/7f/"
            "13cd798d180af4bf4c0ceddeefba2b864a63c71645abc0308b768d67bb81/"
            "hjson-3.1.0-py3-none-any.whl"
        ),
        "version": "3.1.0",
    },
    {
        "name": "nvidia-ml-py",
        "sha256": "fd43d30ee9cd0b7940f5f9f9220b68d42722975e3992b6c21d14144c48760e43",
        "url": (
            "https://files.pythonhosted.org/packages/fd/72/"
            "fb2af0d259a651affdce65fd6a495f0e07a685a0136baf585c5065204ee7/"
            "nvidia_ml_py-13.590.48-py3-none-any.whl"
        ),
        "version": "13.590.48",
    },
    {
        "name": "py-cpuinfo",
        "sha256": "859625bc251f64e21f077d099d4162689c762b5d6a4c3c97553d56241c9674d5",
        "url": (
            "https://files.pythonhosted.org/packages/e0/a9/"
            "023730ba63db1e494a271cb018dcd361bd2c917ba7004c3e49d5daf795a2/"
            "py_cpuinfo-9.0.0-py3-none-any.whl"
        ),
        "version": "9.0.0",
    },
]
FIXED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "DS_BUILD_OPS": "0",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
EXPECTED_CUDA = {
    "cuda_version": "12.4.1",
    "cudart_package_version": "12.4.127-1",
    "cudnn_package": {"name": "libcudnn9-cuda-12", "version": "9.1.0.70-1"},
    "nccl_library": "/usr/local/lib/libnccl.so.2.23.4",
    "torch_cuda": "12.4",
    "torch_cudnn": 90100,
    "torch_nccl": [2, 23, 4],
}
_CONTRACT_KEYS = {
    "base_image",
    "bootstrap",
    "cuda",
    "environment",
    "packages",
    "python",
    "requirements",
    "sagemaker",
    "schema_version",
}
_INVENTORY_KEYS = {
    "bootstrap",
    "contract_sha256",
    "cuda",
    "environment",
    "packages",
    "python",
    "sagemaker",
    "schema_version",
}
_POSITIVE_DECIMAL_RE = re.compile(r"[1-9][0-9]*\Z")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_contract(path: Path) -> tuple[dict[str, object], str]:
    path = Path(path)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"Training-image contract must be a regular non-symlink: {path}")
    if stat.S_IMODE(mode) != 0o644:
        raise ValueError("Training-image contract mode must be 0644")
    raw = path.read_bytes()
    contract = json.loads(raw)
    if type(contract) is not dict or raw != _canonical_bytes(contract):
        raise ValueError("Training-image contract must be compact canonical JSON")
    validate_contract(contract)
    return contract, _sha256_bytes(raw)


def validate_contract(contract: Mapping[str, object]) -> None:
    if set(contract) != _CONTRACT_KEYS:
        raise ValueError("Training-image contract schema changed")
    expected = {
        "base_image": {"digest": BASE_IMAGE_DIGEST, "uri": BASE_IMAGE_URI},
        "bootstrap": EXPECTED_BOOTSTRAP,
        "cuda": EXPECTED_CUDA,
        "environment": {
            "fixed": FIXED_ENVIRONMENT,
            "python_hash_seed": {
                "allowed": ALLOWED_PYTHON_HASH_SEEDS,
                "image_default": "17",
                "name": "PYTHONHASHSEED",
            },
            "source_date_epoch": {
                "name": "SOURCE_DATE_EPOCH",
                "protocol": "positive_decimal_no_leading_zero_v1",
            },
        },
        "packages": EXPECTED_PACKAGES,
        "python": {"implementation": "CPython", "version": "3.11.10"},
        "requirements": EXPECTED_REQUIREMENTS,
        "sagemaker": {
            "cmd": CMD,
            "entrypoint": ENTRYPOINT,
            "script_path": ENTRYPOINT_SCRIPT,
            "script_sha256": ENTRYPOINT_SCRIPT_SHA256,
            "training_module": TRAINING_MODULE,
        },
        "schema_version": 2,
    }
    if dict(contract) != expected:
        raise ValueError("Training-image contract values changed")


def _dpkg_version(package_name: str) -> str:
    completed = subprocess.run(
        ["dpkg-query", "-W", "-f=${Version}", package_name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"dpkg-query failed for {package_name}: {completed.stderr.strip()}"
        )
    value = completed.stdout
    if not value or value != value.strip() or "\n" in value:
        raise RuntimeError(f"Unexpected dpkg version for {package_name}: {value!r}")
    return value


def collect_inventory(
    contract_sha256: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    import torch

    environment = os.environ if environ is None else environ
    selected_environment = {
        name: environment.get(name) for name in sorted(FIXED_ENVIRONMENT)
    }
    selected_environment["PYTHONHASHSEED"] = environment.get("PYTHONHASHSEED")
    selected_environment["SOURCE_DATE_EPOCH"] = environment.get("SOURCE_DATE_EPOCH")

    nccl_link = Path("/usr/local/lib/libnccl.so.2")
    if not nccl_link.is_symlink():
        raise RuntimeError(f"NCCL SONAME must be a symlink: {nccl_link}")
    entrypoint_script = Path(ENTRYPOINT_SCRIPT)
    if entrypoint_script.is_symlink() or not entrypoint_script.is_file():
        raise RuntimeError(
            f"SageMaker entrypoint script must be a regular non-symlink: {entrypoint_script}"
        )
    bootstrap = Path(BOOTSTRAP_PATH)
    bootstrap_mode = bootstrap.lstat().st_mode
    if (
        stat.S_ISLNK(bootstrap_mode)
        or not stat.S_ISREG(bootstrap_mode)
        or stat.S_IMODE(bootstrap_mode) != 0o555
    ):
        raise RuntimeError(
            f"Training bootstrap must be a 0555 regular non-symlink: {bootstrap}"
        )

    inventory = {
        "bootstrap": {
            **EXPECTED_BOOTSTRAP,
            "sha256": _sha256_bytes(bootstrap.read_bytes()),
        },
        "contract_sha256": contract_sha256,
        "cuda": {
            "cuda_version": environment.get("CUDA_VERSION"),
            "cudart_package_version": environment.get("NV_CUDA_CUDART_VERSION"),
            "cudnn_package": {
                "name": EXPECTED_CUDA["cudnn_package"]["name"],
                "version": _dpkg_version(EXPECTED_CUDA["cudnn_package"]["name"]),
            },
            "nccl_library": str(nccl_link.resolve(strict=True)),
            "torch_cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
            "torch_nccl": list(torch.cuda.nccl.version()),
        },
        "environment": selected_environment,
        "packages": {
            name: importlib.metadata.version(name) for name in sorted(EXPECTED_PACKAGES)
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "sagemaker": {
            "cmd": CMD,
            "entrypoint": ENTRYPOINT,
            "script_path": ENTRYPOINT_SCRIPT,
            "script_sha256": _sha256_bytes(entrypoint_script.read_bytes()),
            "training_module": environment.get("SAGEMAKER_TRAINING_MODULE"),
        },
        "schema_version": 2,
    }
    validate_inventory(inventory, contract_sha256=contract_sha256)
    return inventory


def validate_inventory(
    inventory: Mapping[str, object],
    *,
    contract_sha256: str,
) -> None:
    if set(inventory) != _INVENTORY_KEYS:
        raise RuntimeError("Training-image inventory schema changed")
    if inventory["schema_version"] != 2:
        raise RuntimeError("Training-image inventory schema_version changed")
    if inventory["contract_sha256"] != contract_sha256:
        raise RuntimeError("Training-image contract digest changed")
    if inventory["bootstrap"] != EXPECTED_BOOTSTRAP:
        raise RuntimeError(
            f"Training bootstrap identity changed: {inventory['bootstrap']!r}"
        )
    if inventory["python"] != {"implementation": "CPython", "version": "3.11.10"}:
        raise RuntimeError(f"Python runtime changed: {inventory['python']!r}")
    if inventory["packages"] != EXPECTED_PACKAGES:
        raise RuntimeError(f"Python package inventory changed: {inventory['packages']!r}")
    if inventory["cuda"] != EXPECTED_CUDA:
        raise RuntimeError(f"CUDA/cuDNN/NCCL identity changed: {inventory['cuda']!r}")
    expected_sagemaker = {
        "cmd": CMD,
        "entrypoint": ENTRYPOINT,
        "script_path": ENTRYPOINT_SCRIPT,
        "script_sha256": ENTRYPOINT_SCRIPT_SHA256,
        "training_module": TRAINING_MODULE,
    }
    if inventory["sagemaker"] != expected_sagemaker:
        raise RuntimeError(
            f"SageMaker training entrypoint contract changed: {inventory['sagemaker']!r}"
        )
    actual_environment = inventory["environment"]
    if type(actual_environment) is not dict:
        raise RuntimeError("Training-image environment inventory changed type")
    for name, expected in FIXED_ENVIRONMENT.items():
        if actual_environment.get(name) != expected:
            raise RuntimeError(
                f"Training-image environment changed: {name}={actual_environment.get(name)!r}"
            )
    seed = actual_environment.get("PYTHONHASHSEED")
    if seed not in ALLOWED_PYTHON_HASH_SEEDS:
        raise RuntimeError(
            "PYTHONHASHSEED must be one of the frozen controlled seeds: "
            f"actual={seed!r}, expected={ALLOWED_PYTHON_HASH_SEEDS}"
        )
    source_date_epoch = actual_environment.get("SOURCE_DATE_EPOCH")
    if type(source_date_epoch) is not str or _POSITIVE_DECIMAL_RE.fullmatch(
        source_date_epoch
    ) is None:
        raise RuntimeError(
            "SOURCE_DATE_EPOCH must be a positive canonical decimal integer: "
            f"actual={source_date_epoch!r}"
        )
    expected_environment_keys = set(FIXED_ENVIRONMENT) | {
        "PYTHONHASHSEED",
        "SOURCE_DATE_EPOCH",
    }
    if set(actual_environment) != expected_environment_keys:
        raise RuntimeError("Training-image environment inventory schema changed")


def _write_absent(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Runtime inventory output must be absent: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"Runtime inventory parent must be a real directory: {path.parent}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen training image")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    _, contract_sha256 = load_contract(arguments.contract)
    inventory = collect_inventory(contract_sha256)
    payload = _canonical_bytes(inventory)
    if arguments.output is None:
        os.write(1, payload)
    else:
        _write_absent(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
