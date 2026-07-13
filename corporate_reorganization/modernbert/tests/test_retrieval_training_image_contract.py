from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from training_image import bootstrap, build, runtime_contract  # noqa: E402


TRAINING_IMAGE_DIR = MODERNBERT_DIR / "training_image"
CONTRACT_SHA256 = "db4b2b307a56686054c2c04fbcebf5c133077765074ceef61a613c183a4b04ef"
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
        "bootstrap": copy.deepcopy(runtime_contract.EXPECTED_BOOTSTRAP),
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
        "schema_version": 2,
    }


SOURCE_EPOCH = 1_700_000_000
SOURCE_FILES = {
    "train_sm.py": b"print('verified training')\n",
    "trainer.py": b"VALUE = 1\n",
}


def _source_inventory() -> list[dict[str, object]]:
    return [
        {
            "mode": "0644",
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "type": "file",
        }
        for name, payload in sorted(SOURCE_FILES.items())
    ]


def _source_inventory_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(_source_inventory())).hexdigest()


def _tar_gzip(
    members: list[dict[str, object]],
    *,
    epoch: int = SOURCE_EPOCH,
) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=6,
        fileobj=output,
        mtime=epoch,
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for record in members:
                information = tarfile.TarInfo(str(record["name"]))
                information.uid = int(record.get("uid", 0))
                information.gid = int(record.get("gid", 0))
                information.uname = str(record.get("uname", ""))
                information.gname = str(record.get("gname", ""))
                information.mtime = int(record.get("mtime", epoch))
                information.mode = int(record.get("mode", 0o644))
                entry_type = record.get("type", "file")
                if entry_type == "file":
                    payload = bytes(record.get("payload", b""))
                    information.type = tarfile.REGTYPE
                    information.size = len(payload)
                    archive.addfile(information, io.BytesIO(payload))
                elif entry_type == "directory":
                    information.type = tarfile.DIRTYPE
                    information.mode = int(record.get("mode", 0o755))
                    information.size = 0
                    archive.addfile(information)
                elif entry_type == "symlink":
                    information.type = tarfile.SYMTYPE
                    information.linkname = str(record.get("linkname", "train_sm.py"))
                    archive.addfile(information)
                elif entry_type == "hardlink":
                    information.type = tarfile.LNKTYPE
                    information.linkname = str(record.get("linkname", "train_sm.py"))
                    archive.addfile(information)
                elif entry_type == "fifo":
                    information.type = tarfile.FIFOTYPE
                    archive.addfile(information)
                else:
                    raise AssertionError(f"Unsupported test member type: {entry_type}")
    return output.getvalue()


def _valid_source_archive() -> bytes:
    return _tar_gzip(
        [
            {"name": name, "payload": payload}
            for name, payload in sorted(SOURCE_FILES.items())
        ]
    )


def _write(path: Path, payload: bytes, mode: int) -> None:
    path.write_bytes(payload)
    path.chmod(mode)


def _bootstrap_fixture(
    root: Path,
    *,
    rank: int = 0,
    live_seed: str = "29",
) -> dict[str, object]:
    baked = root / "baked"
    baked.mkdir()
    runtime_path = baked / "runtime_contract.py"
    bootstrap_path = baked / "bootstrap.py"
    active_bootstrap_path = baked / "active-bootstrap.py"
    contract_path = baked / "image_contract.json"
    inventory_path = baked / "runtime_inventory.json"
    shutil.copyfile(TRAINING_IMAGE_DIR / "runtime_contract.py", runtime_path)
    runtime_path.chmod(0o644)
    shutil.copyfile(TRAINING_IMAGE_DIR / "bootstrap.py", bootstrap_path)
    bootstrap_path.chmod(0o555)
    shutil.copyfile(TRAINING_IMAGE_DIR / "bootstrap.py", active_bootstrap_path)
    active_bootstrap_path.chmod(0o555)
    shutil.copyfile(TRAINING_IMAGE_DIR / "image_contract.json", contract_path)
    contract_path.chmod(0o644)
    inventory = _valid_inventory("17")
    inventory_raw = _canonical_bytes(inventory)
    _write(inventory_path, inventory_raw, 0o644)

    source_channel = root / "source"
    source_channel.mkdir()
    source_raw = _valid_source_archive()
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    source_name = f"source-{source_sha256}.tar.gz"
    _write(source_channel / source_name, source_raw, 0o644)
    extraction_parent = root / "code"
    extraction_parent.mkdir()
    environment = {
        **runtime_contract.FIXED_ENVIRONMENT,
        "PYTHONHASHSEED": live_seed,
        "SOURCE_DATE_EPOCH": "1783881756",
        "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256": hashlib.sha256(
            inventory_raw
        ).hexdigest(),
        "ARR_SOURCE_BUNDLE_NAME": source_name,
        "ARR_SOURCE_BUNDLE_SIZE": str(len(source_raw)),
        "ARR_SOURCE_BUNDLE_SHA256": source_sha256,
        "ARR_SOURCE_INVENTORY_SHA256": _source_inventory_sha256(),
        "ARR_SOURCE_COMMIT_EPOCH": str(SOURCE_EPOCH),
        "SM_CHANNEL_SOURCE": str(source_channel),
        "OMPI_COMM_WORLD_LOCAL_RANK": str(rank),
        "OMPI_COMM_WORLD_RANK": str(rank),
        "OMPI_COMM_WORLD_SIZE": "4",
    }
    return {
        "bootstrap_path": bootstrap_path,
        "active_bootstrap_path": active_bootstrap_path,
        "contract_path": contract_path,
        "environment": environment,
        "extraction_parent": extraction_parent,
        "inventory_path": inventory_path,
        "runtime_path": runtime_path,
        "source_channel": source_channel,
        "source_raw": source_raw,
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
            self.assertEqual(
                image_name, "arr-retrieval-train:step10a-bootstrap-build1"
            )
            self.assertEqual(command[:3], ["docker", "buildx", "build"])
            self.assertIn("--pull", command)
            self.assertIn("--no-cache", command)
            self.assertIn("--provenance=false", command)
            self.assertIn("--sbom=false", command)
            self.assertIn(
                "type=image,name=arr-retrieval-train:step10a-bootstrap-build1,"
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
            {"sha256:b44c9b182a2490329b25394568299420bcfbe85a8fb17df955378b1f3630d9be"},
        )
        self.assertEqual(
            {record["config_digest"] for record in identity["replicas"]},
            {"sha256:24784672e3d1f8004fe6577069d6f01393239310276a570f5e8d0db1fe13b85f"},
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
        self.assertEqual(contract["bootstrap"], runtime_contract.EXPECTED_BOOTSTRAP)
        self.assertEqual(contract["schema_version"], 2)
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
            "COPY --chmod=0555 training_image/bootstrap.py "
            "/opt/training_bootstrap/bootstrap.py",
            dockerfile,
        )
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
                "!training_image/bootstrap.py",
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
        changed_bootstrap = copy.deepcopy(contract)
        changed_bootstrap["bootstrap"]["sha256"] = "0" * 64
        mutations.append(changed_bootstrap)
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
        changed_bootstrap = _valid_inventory()
        changed_bootstrap["bootstrap"]["protocol"] = "changed"
        mutations.append(changed_bootstrap)
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


class NetworkIsolatedTrainingBootstrapTest(unittest.TestCase):
    @staticmethod
    def _run(fixture: dict[str, object], environment: dict[str, str], recorder):
        return bootstrap.run(
            ["bootstrap.py", "--outer-fold", "2", "--sampler", "global_uniform"],
            environ=environment,
            runtime_contract_path=fixture["runtime_path"],
            image_contract_path=fixture["contract_path"],
            runtime_inventory_path=fixture["inventory_path"],
            bootstrap_path=fixture["bootstrap_path"],
            active_bootstrap_path=fixture["active_bootstrap_path"],
            source_channel_path=fixture["source_channel"],
            extraction_parent=fixture["extraction_parent"],
            execvpe=recorder,
        )

    def test_every_mpi_rank_verifies_extracts_and_execs_exact_entrypoint(self) -> None:
        for rank in range(4):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary), rank=rank)
                calls = []

                def recorder(executable, command, environment):
                    calls.append((executable, list(command), dict(environment)))
                    return "exec-replaced"

                result = self._run(fixture, fixture["environment"], recorder)
                self.assertEqual(result, "exec-replaced")
                self.assertEqual(len(calls), 1)
                executable, command, child_environment = calls[0]
                source_root = fixture["extraction_parent"] / f"arr-source-rank-{rank}"
                self.assertEqual(executable, sys.executable)
                self.assertEqual(
                    command,
                    [
                        sys.executable,
                        str(source_root / "train_sm.py"),
                        "--outer-fold",
                        "2",
                        "--sampler",
                        "global_uniform",
                    ],
                )
                for relative, payload in SOURCE_FILES.items():
                    extracted = source_root / relative
                    self.assertEqual(extracted.read_bytes(), payload)
                    self.assertEqual(extracted.stat().st_mode & 0o777, 0o644)
                self.assertEqual(
                    child_environment[
                        "ARR_VERIFIED_TRAINING_BOOTSTRAP_PROTOCOL"
                    ],
                    bootstrap.BOOTSTRAP_PROTOCOL,
                )
                self.assertEqual(
                    child_environment["ARR_VERIFIED_SOURCE_INVENTORY_SHA256"],
                    _source_inventory_sha256(),
                )
                self.assertEqual(
                    child_environment["ARR_VERIFIED_TRAINING_CONTRACT_SHA256"],
                    CONTRACT_SHA256,
                )
                self.assertEqual(child_environment["PYTHONHASHSEED"], "29")

    def test_only_the_three_controlled_live_seeds_are_normalized(self) -> None:
        for seed in ("17", "29", "43"):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary), live_seed=seed)
                self._run(fixture, fixture["environment"], lambda *args: None)
        for seed in ("0", "42", "043", "random"):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary), live_seed=seed)
                with self.assertRaises(RuntimeError):
                    self._run(fixture, fixture["environment"], lambda *args: None)

    def test_live_environment_and_baked_identity_mutations_fail(self) -> None:
        environment_mutations = {
            "fixed": ("HF_HUB_OFFLINE", "0"),
            "source_epoch": ("SOURCE_DATE_EPOCH", "1783881757"),
            "runtime_digest": (
                "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256",
                "0" * 64,
            ),
            "channel": ("SM_CHANNEL_SOURCE", "/different/source"),
        }
        for name, (key, value) in environment_mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary))
                environment = dict(fixture["environment"])
                environment[key] = value
                with self.assertRaises((RuntimeError, ValueError)):
                    self._run(fixture, environment, lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            path = fixture["bootstrap_path"]
            path.chmod(0o644)
            path.write_bytes(path.read_bytes() + b"\n")
            path.chmod(0o555)
            with self.assertRaises(RuntimeError):
                self._run(fixture, fixture["environment"], lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            path = fixture["active_bootstrap_path"]
            path.chmod(0o644)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(RuntimeError):
                self._run(fixture, fixture["environment"], lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            inventory = json.loads(fixture["inventory_path"].read_bytes())
            inventory["packages"]["torch"] = "different"
            raw = _canonical_bytes(inventory)
            _write(fixture["inventory_path"], raw, 0o644)
            environment = dict(fixture["environment"])
            environment["ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256"] = (
                hashlib.sha256(raw).hexdigest()
            )
            with self.assertRaises(RuntimeError):
                self._run(fixture, environment, lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            inventory = json.loads(fixture["inventory_path"].read_bytes())
            inventory["environment"]["PYTHONHASHSEED"] = "29"
            raw = _canonical_bytes(inventory)
            _write(fixture["inventory_path"], raw, 0o644)
            environment = dict(fixture["environment"])
            environment["ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256"] = (
                hashlib.sha256(raw).hexdigest()
            )
            with self.assertRaises(RuntimeError):
                self._run(fixture, environment, lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            contract = json.loads(fixture["contract_path"].read_bytes())
            contract["bootstrap"]["entrypoint"] = "other.py"
            _write(fixture["contract_path"], _canonical_bytes(contract), 0o644)
            with self.assertRaises(ValueError):
                self._run(fixture, fixture["environment"], lambda *args: None)

    def test_source_identity_environment_is_exact(self) -> None:
        keys = list(bootstrap.SOURCE_IDENTITY_ENV.values())
        for key in keys:
            with self.subTest(missing=key), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary))
                environment = dict(fixture["environment"])
                del environment[key]
                with self.assertRaises(RuntimeError):
                    self._run(fixture, environment, lambda *args: None)

        mutations = {
            "name": ("ARR_SOURCE_BUNDLE_NAME", "source-" + "0" * 64 + ".tar.gz"),
            "size": ("ARR_SOURCE_BUNDLE_SIZE", "01"),
            "sha": ("ARR_SOURCE_BUNDLE_SHA256", "A" * 64),
            "inventory": ("ARR_SOURCE_INVENTORY_SHA256", "0" * 63),
            "epoch": ("ARR_SOURCE_COMMIT_EPOCH", "0"),
            "large_epoch": ("ARR_SOURCE_COMMIT_EPOCH", str(0x1_0000_0000)),
        }
        for name, (key, value) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary))
                environment = dict(fixture["environment"])
                environment[key] = value
                with self.assertRaises(ValueError):
                    self._run(fixture, environment, lambda *args: None)

    def test_source_channel_requires_one_exact_regular_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            _write(fixture["source_channel"] / "extra", b"extra", 0o644)
            with self.assertRaises(ValueError):
                self._run(fixture, fixture["environment"], lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            archive = next(fixture["source_channel"].iterdir())
            archive.unlink()
            archive.symlink_to(fixture["bootstrap_path"])
            with self.assertRaises(ValueError):
                self._run(fixture, fixture["environment"], lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary))
            archive = next(fixture["source_channel"].iterdir())
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                self._run(fixture, fixture["environment"], lambda *args: None)

    def test_archive_traversal_links_duplicates_types_and_metadata_fail(self) -> None:
        invalid_members = {
            "traversal": [{"name": "../train_sm.py", "payload": b"x"}],
            "absolute": [{"name": "/train_sm.py", "payload": b"x"}],
            "backslash": [{"name": "dir\\train_sm.py", "payload": b"x"}],
            "symlink": [{"name": "train_sm.py", "type": "symlink"}],
            "hardlink": [{"name": "train_sm.py", "type": "hardlink"}],
            "fifo": [{"name": "train_sm.py", "type": "fifo"}],
            "duplicate": [
                {"name": "train_sm.py", "payload": b"a"},
                {"name": "train_sm.py", "payload": b"b"},
            ],
            "mode": [{"name": "train_sm.py", "payload": b"x", "mode": 0o600}],
            "uid": [{"name": "train_sm.py", "payload": b"x", "uid": 1}],
            "member_epoch": [
                {"name": "train_sm.py", "payload": b"x", "mtime": SOURCE_EPOCH + 1}
            ],
            "unsorted": [
                {"name": "trainer.py", "payload": b"x"},
                {"name": "train_sm.py", "payload": b"x"},
            ],
            "missing_entrypoint": [{"name": "trainer.py", "payload": b"x"}],
            "implicit_parent": [{"name": "pkg/train_sm.py", "payload": b"x"}],
        }
        for name, members in invalid_members.items():
            with self.subTest(name=name):
                raw = _tar_gzip(members)
                with self.assertRaises(ValueError):
                    bootstrap._read_normalized_archive(
                        raw,
                        expected_epoch=SOURCE_EPOCH,
                        expected_inventory_sha256=_source_inventory_sha256(),
                    )
        with self.assertRaises(ValueError):
            bootstrap._read_normalized_archive(
                _tar_gzip(
                    [
                        {"name": name, "payload": payload}
                        for name, payload in sorted(SOURCE_FILES.items())
                    ],
                    epoch=SOURCE_EPOCH + 1,
                ),
                expected_epoch=SOURCE_EPOCH,
                expected_inventory_sha256=_source_inventory_sha256(),
            )
        valid_raw = _valid_source_archive()
        for name, index, replacement in (
            ("xfl", 8, 2),
            ("os", 9, 3),
        ):
            changed_header = bytearray(valid_raw)
            changed_header[index] = replacement
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "gzip header"
            ):
                bootstrap._read_normalized_archive(
                    bytes(changed_header),
                    expected_epoch=SOURCE_EPOCH,
                    expected_inventory_sha256=_source_inventory_sha256(),
                )
        with self.assertRaises(ValueError):
            bootstrap._read_normalized_archive(
                valid_raw + valid_raw,
                expected_epoch=SOURCE_EPOCH,
                expected_inventory_sha256=_source_inventory_sha256(),
            )

    def test_rank_topology_mutations_and_existing_destination_fail(self) -> None:
        mutations = {
            "missing": ("OMPI_COMM_WORLD_RANK", None),
            "world": ("OMPI_COMM_WORLD_SIZE", "8"),
            "range": ("OMPI_COMM_WORLD_RANK", "4"),
            "local": ("OMPI_COMM_WORLD_LOCAL_RANK", "1"),
            "leading_zero": ("OMPI_COMM_WORLD_RANK", "00"),
            "conflict": ("RANK", "1"),
        }
        for name, (key, value) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = _bootstrap_fixture(Path(temporary))
                environment = dict(fixture["environment"])
                if value is None:
                    del environment[key]
                else:
                    environment[key] = value
                with self.assertRaises((RuntimeError, ValueError)):
                    self._run(fixture, environment, lambda *args: None)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bootstrap_fixture(Path(temporary), rank=2)
            (fixture["extraction_parent"] / "arr-source-rank-2").mkdir()
            with self.assertRaises(FileExistsError):
                self._run(fixture, fixture["environment"], lambda *args: None)

    def test_inventory_digest_and_entrypoint_are_fixed(self) -> None:
        raw = _valid_source_archive()
        inventory, contents = bootstrap._read_normalized_archive(
            raw,
            expected_epoch=SOURCE_EPOCH,
            expected_inventory_sha256=_source_inventory_sha256(),
        )
        self.assertEqual(inventory, _source_inventory())
        self.assertEqual(contents, SOURCE_FILES)
        wrong = "0" * 64
        self.assertNotEqual(wrong, _source_inventory_sha256())
        with self.assertRaises(ValueError):
            bootstrap._read_normalized_archive(
                raw,
                expected_epoch=SOURCE_EPOCH,
                expected_inventory_sha256=wrong,
            )

    def test_extraction_modes_do_not_depend_on_process_umask(self) -> None:
        contents = {
            "pkg/helper.py": b"HELPER = True\n",
            "train_sm.py": SOURCE_FILES["train_sm.py"],
        }
        inventory = [
            {"mode": "0755", "path": "pkg", "type": "directory"},
            {
                "mode": "0644",
                "path": "pkg/helper.py",
                "sha256": hashlib.sha256(contents["pkg/helper.py"]).hexdigest(),
                "size": len(contents["pkg/helper.py"]),
                "type": "file",
            },
            {
                "mode": "0644",
                "path": "train_sm.py",
                "sha256": hashlib.sha256(contents["train_sm.py"]).hexdigest(),
                "size": len(contents["train_sm.py"]),
                "type": "file",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            previous_umask = os.umask(0o077)
            try:
                destination = bootstrap._extract_absent(
                    parent,
                    rank=3,
                    inventory=inventory,
                    contents=contents,
                )
            finally:
                os.umask(previous_umask)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o755)
            self.assertEqual((destination / "pkg").stat().st_mode & 0o777, 0o755)
            self.assertEqual(
                (destination / "pkg/helper.py").stat().st_mode & 0o777,
                0o644,
            )
            self.assertEqual(
                (destination / "train_sm.py").stat().st_mode & 0o777,
                0o644,
            )


if __name__ == "__main__":
    unittest.main()
