from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from training_image import build, runtime_contract  # noqa: E402


TRAINING_IMAGE_DIR = MODERNBERT_DIR / "training_image"
CONTRACT_SHA256 = "6df440464cb8a317b2703e885f1efc8431edfdf16a29f729aef4f62c36fbfe09"
BASE_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
LOCKED_ARTIFACTS = [
    (
        "https://files.pythonhosted.org/packages/38/10/"
        "a7f63e086c1e1c12e290c98363c748ef5ddd6313fde739d2aeccd5ed0cd4/"
        "deepspeed-0.17.1.tar.gz",
        "6d6e21796982b9e024f489e1c211666cc6c0be6e344751368610b9d2da285d6e",
    ),
    (
        "https://files.pythonhosted.org/packages/1f/7f/"
        "13cd798d180af4bf4c0ceddeefba2b864a63c71645abc0308b768d67bb81/"
        "hjson-3.1.0-py3-none-any.whl",
        "65713cdcf13214fb554eb8b4ef803419733f4f5e551047c9b711098ab7186b89",
    ),
    (
        "https://files.pythonhosted.org/packages/fd/72/"
        "fb2af0d259a651affdce65fd6a495f0e07a685a0136baf585c5065204ee7/"
        "nvidia_ml_py-13.590.48-py3-none-any.whl",
        "fd43d30ee9cd0b7940f5f9f9220b68d42722975e3992b6c21d14144c48760e43",
    ),
    (
        "https://files.pythonhosted.org/packages/e0/a9/"
        "023730ba63db1e494a271cb018dcd361bd2c917ba7004c3e49d5daf795a2/"
        "py_cpuinfo-9.0.0-py3-none-any.whl",
        "859625bc251f64e21f077d099d4162689c762b5d6a4c3c97553d56241c9674d5",
    ),
]


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


def _valid_inventory(seed: str = "17") -> dict[str, object]:
    return {
        "contract_sha256": CONTRACT_SHA256,
        "cuda": copy.deepcopy(runtime_contract.EXPECTED_CUDA),
        "environment": {
            **runtime_contract.FIXED_ENVIRONMENT,
            "PYTHONHASHSEED": seed,
            "SOURCE_DATE_EPOCH": "1783881756",
        },
        "packages": copy.deepcopy(runtime_contract.EXPECTED_PACKAGES),
        "python": {"implementation": "CPython", "version": "3.11.10"},
        "sagemaker": {
            "cmd": ["/bin/bash"],
            "entrypoint": ["bash", "-m", "start_with_right_hostname.sh"],
            "script_path": "/usr/local/bin/start_with_right_hostname.sh",
            "script_sha256": (
                "680a39c5aa0797febfd91c3bc0cef0a7125ef95f80db385c55762696ef845fc9"
            ),
            "training_module": "sagemaker_pytorch_container.training:main",
        },
        "schema_version": 1,
    }


class RetrievalTrainingImageContractTest(unittest.TestCase):
    def test_checked_build_command_binds_toolchain_exporter_and_absent_metadata(self) -> None:
        identity = build.load_build_identity(MODERNBERT_DIR)
        self.assertEqual(identity["toolchain"], build.EXPECTED_TOOLCHAIN)
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / "replica-1.json"
            command, image_name = build.render_build_command(
                MODERNBERT_DIR,
                metadata,
                build_replica=1,
                identity=identity,
            )
            self.assertEqual(image_name, "arr-retrieval-train:step8-parent-build1")
            self.assertEqual(command[:3], ["docker", "buildx", "build"])
            self.assertIn("--pull", command)
            self.assertIn("--no-cache", command)
            self.assertIn("--provenance=false", command)
            self.assertIn("--sbom=false", command)
            self.assertIn(
                "type=image,name=arr-retrieval-train:step8-parent-build1,"
                "push=false,rewrite-timestamp=true,unpack=false,compression=gzip,"
                "compression-level=6,force-compression=false,oci-mediatypes=false",
                command,
            )
            self.assertIn("SOURCE_DATE_EPOCH=1783895427", command)
            metadata.write_text("occupied\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build.render_build_command(
                    MODERNBERT_DIR,
                    metadata,
                    build_replica=1,
                    identity=identity,
                )

    def test_two_builds_and_source_inventory_are_frozen(self) -> None:
        path = TRAINING_IMAGE_DIR / "build_identity.json"
        raw = path.read_bytes()
        identity = json.loads(raw)
        self.assertEqual(raw, json.dumps(identity, sort_keys=True, indent=2).encode() + b"\n")
        self.assertEqual(identity["schema_version"], 1)
        self.assertEqual(
            {record["manifest_digest"] for record in identity["replicas"]},
            {"sha256:78221762d9cc7dd24a3f9958f1caabfe6d06eb5668fe82d5055038b803291712"},
        )
        self.assertEqual(
            {record["config_digest"] for record in identity["replicas"]},
            {"sha256:aff8c9ca06b1a2aa8c78480cc8c27476eb32761176efe244eb5563dc37cfc9ad"},
        )
        rows = identity["source_inventory"]["files"]
        rebuilt = []
        for record in rows:
            source = MODERNBERT_DIR / record["path"]
            payload = source.read_bytes()
            self.assertEqual(len(payload), record["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
            rebuilt.append(record)
        inventory_payload = (
            json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        self.assertEqual(
            hashlib.sha256(inventory_payload).hexdigest(),
            identity["source_inventory"]["inventory_sha256"],
        )

    def test_contract_is_canonical_and_exact(self) -> None:
        path = TRAINING_IMAGE_DIR / "image_contract.json"
        raw = path.read_bytes()
        contract = json.loads(raw)
        self.assertEqual(raw, _canonical_bytes(contract))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CONTRACT_SHA256)
        loaded, digest = runtime_contract.load_contract(path)
        self.assertEqual(loaded, contract)
        self.assertEqual(digest, CONTRACT_SHA256)
        self.assertEqual(contract["base_image"]["uri"], BASE_IMAGE)
        self.assertEqual(
            [(row["url"], row["sha256"]) for row in contract["requirements"]],
            LOCKED_ARTIFACTS,
        )

    def test_requirements_lock_contains_only_exact_direct_artifacts(self) -> None:
        lock = (TRAINING_IMAGE_DIR / "requirements.lock").read_text(encoding="utf-8")
        self.assertEqual(lock.count("https://"), 4)
        self.assertEqual(lock.count("--hash=sha256:"), 4)
        for url, digest in LOCKED_ARTIFACTS:
            self.assertIn(url, lock)
            self.assertIn(f"--hash=sha256:{digest}", lock)
        self.assertNotIn("==", lock)
        self.assertNotIn("git+", lock)
        self.assertNotIn("latest", lock.lower())

    def test_dockerfile_preserves_base_and_sagemaker_descriptor(self) -> None:
        dockerfile = (TRAINING_IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(dockerfile.count(f"FROM {BASE_IMAGE}"), 1)
        self.assertIn(
            "# syntax=docker/dockerfile:1.7@sha256:"
            "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e",
            dockerfile,
        )
        self.assertIn("ARG SOURCE_DATE_EPOCH", dockerfile)
        self.assertIn("DS_BUILD_OPS=0", dockerfile)
        self.assertIn("--no-build-isolation", dockerfile)
        self.assertIn("--no-cache-dir", dockerfile)
        self.assertIn("--no-compile", dockerfile)
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn(
            f'org.opencontainers.image.training-contract.sha256="{CONTRACT_SHA256}"',
            dockerfile,
        )
        self.assertIn('WORKDIR /\nENTRYPOINT ["bash","-m","start_with_right_hostname.sh"]', dockerfile)
        self.assertTrue(dockerfile.rstrip().endswith('CMD ["/bin/bash"]'))
        self.assertNotIn(":latest", dockerfile)
        self.assertNotIn("||", dockerfile)

    def test_docker_context_is_an_exact_allowlist(self) -> None:
        dockerignore = (
            TRAINING_IMAGE_DIR / "Dockerfile.dockerignore"
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            dockerignore,
            [
                "**",
                "!training_image/",
                "!training_image/Dockerfile",
                "!training_image/Dockerfile.dockerignore",
                "!training_image/image_contract.json",
                "!training_image/requirements.lock",
                "!training_image/runtime_contract.py",
            ],
        )

    def test_contract_mutations_fail(self) -> None:
        contract = json.loads((TRAINING_IMAGE_DIR / "image_contract.json").read_bytes())
        mutations = []
        changed_base = copy.deepcopy(contract)
        changed_base["base_image"]["digest"] = "sha256:" + "0" * 64
        mutations.append(changed_base)
        changed_package = copy.deepcopy(contract)
        changed_package["packages"]["deepspeed"] = "0.17.2"
        mutations.append(changed_package)
        changed_requirement = copy.deepcopy(contract)
        changed_requirement["requirements"][0]["sha256"] = "0" * 64
        mutations.append(changed_requirement)
        changed_entrypoint = copy.deepcopy(contract)
        changed_entrypoint["sagemaker"]["entrypoint"] = ["train"]
        mutations.append(changed_entrypoint)
        changed_seed_set = copy.deepcopy(contract)
        changed_seed_set["environment"]["python_hash_seed"]["allowed"] = ["17"]
        mutations.append(changed_seed_set)
        changed_schema = copy.deepcopy(contract)
        changed_schema["extra"] = True
        mutations.append(changed_schema)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    runtime_contract.validate_contract(mutation)

    def test_inventory_accepts_only_frozen_seed_set(self) -> None:
        for seed in ("17", "29", "43"):
            with self.subTest(seed=seed):
                runtime_contract.validate_inventory(
                    _valid_inventory(seed), contract_sha256=CONTRACT_SHA256
                )
        for seed in ("0", "1", "42", "043", "random", None):
            with self.subTest(seed=seed):
                inventory = _valid_inventory()
                inventory["environment"]["PYTHONHASHSEED"] = seed
                with self.assertRaises(RuntimeError):
                    runtime_contract.validate_inventory(
                        inventory, contract_sha256=CONTRACT_SHA256
                    )

    def test_runtime_identity_mutations_fail(self) -> None:
        mutations = []
        changed_python = _valid_inventory()
        changed_python["python"]["version"] = "3.11.11"
        mutations.append(changed_python)
        changed_package = _valid_inventory()
        changed_package["packages"]["torch"] = "2.6.0"
        mutations.append(changed_package)
        changed_cuda = _valid_inventory()
        changed_cuda["cuda"]["torch_nccl"] = [2, 23, 5]
        mutations.append(changed_cuda)
        changed_environment = _valid_inventory()
        changed_environment["environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        mutations.append(changed_environment)
        changed_entrypoint = _valid_inventory()
        changed_entrypoint["sagemaker"]["script_sha256"] = "0" * 64
        mutations.append(changed_entrypoint)
        changed_schema = _valid_inventory()
        changed_schema["unexpected"] = True
        mutations.append(changed_schema)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(RuntimeError):
                    runtime_contract.validate_inventory(
                        mutation, contract_sha256=CONTRACT_SHA256
                    )

    def test_source_date_epoch_and_absent_output_are_strict(self) -> None:
        for invalid in (None, "", "0", "01", "-1", "1.0", " 1"):
            with self.subTest(invalid=invalid):
                inventory = _valid_inventory()
                inventory["environment"]["SOURCE_DATE_EPOCH"] = invalid
                with self.assertRaises(RuntimeError):
                    runtime_contract.validate_inventory(
                        inventory, contract_sha256=CONTRACT_SHA256
                    )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "inventory.json"
            inventory = _valid_inventory("29")
            payload = _canonical_bytes(inventory)
            runtime_contract._write_absent(output, payload)
            self.assertEqual(output.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                runtime_contract._write_absent(output, payload)


if __name__ == "__main__":
    unittest.main()
