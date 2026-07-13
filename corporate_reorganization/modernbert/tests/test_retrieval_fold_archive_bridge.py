from __future__ import annotations

import base64
import copy
import dataclasses
import gzip
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from processing_fold_eval import archive_bridge as bridge  # noqa: E402
from retriever.artifacts import (  # noqa: E402
    ArtifactFileRecord,
    CONTROLLED_ARTIFACT_PROTOCOL,
    ControlledArtifactExpectation,
)
from retriever.provenance import (  # noqa: E402
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
)


_EPOCH = 1_783_947_659
_ZERO_SHA = "0" * 64
_ONE_SHA = "1" * 64
_TWO_SHA = "2" * 64
_KMS_ARN = "arn:aws:kms:us-east-1:371087393859:key/test-key"


def _pax_record(key: str, value: str) -> bytes:
    suffix = f" {key}={value}\n".encode("utf-8")
    length = len(suffix) + 1
    while True:
        candidate = str(length).encode("ascii") + suffix
        if len(candidate) == length:
            return candidate
        length = len(candidate)


def _pax_payload(
    *,
    key_order: tuple[str, ...] = bridge._PAX_KEYS,
    values: dict[str, str] | None = None,
) -> bytes:
    defaults = {
        "atime": f"{_EPOCH}.0000000",
        "ctime": f"{_EPOCH}.0000000",
        "mtime": f"{_EPOCH}.0000000",
        "LIBARCHIVE.creationtime": f"{_EPOCH}.0000000",
    }
    if values is not None:
        defaults.update(values)
    payload = b"".join(_pax_record(key, defaults[key]) for key in key_order)
    if key_order == bridge._PAX_KEYS and values is None:
        assert len(payload) == 130
    return payload


def _nul_field(value: str, width: int) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) >= width:
        raise ValueError("fixture field is too long")
    return raw + b"\0" * (width - len(raw))


def _octal(value: int, width: int) -> bytes:
    raw = f"{value:0{width - 1}o} ".encode("ascii")
    if len(raw) != width:
        raise ValueError("fixture octal field overflow")
    return raw


def _ustar_header(
    name: str,
    *,
    typeflag: bytes,
    mode: bytes,
    size: int,
    uname: str,
    gname: str,
    linkname: str = "",
    mtime: int = _EPOCH,
) -> bytes:
    header = bytearray(512)
    header[0:100] = _nul_field(name, 100)
    header[100:108] = mode
    header[108:116] = b"0000000 "
    header[116:124] = b"0000000 "
    header[124:136] = _octal(size, 12)
    header[136:148] = _octal(mtime, 12)
    header[148:156] = b"        "
    header[156:157] = typeflag
    header[157:257] = _nul_field(linkname, 100)
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[265:297] = _nul_field(uname, 32)
    header[297:329] = _nul_field(gname, 32)
    header[329:337] = b"0000000 "
    header[337:345] = b"0000000 "
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}".encode("ascii") + b"\0 "
    return bytes(header)


def _padded(payload: bytes) -> bytes:
    return payload + b"\0" * ((-len(payload)) % 512)


def _archive_bytes(
    members: list[tuple[str, str, bytes | None]],
    *,
    pax_payloads: list[bytes] | None = None,
    pax_names: list[str] | None = None,
    logical_types: list[bytes] | None = None,
    logical_linknames: list[str] | None = None,
    tail: bytes = b"",
) -> bytes:
    blocks: list[bytes] = []
    for index, (kind, name, payload) in enumerate(members):
        raw_name = name + "/" if kind == "directory" and not name.endswith("/") else name
        pax_payload = (
            _pax_payload() if pax_payloads is None else pax_payloads[index]
        )
        pax_name = (
            "./PaxHeaders.X/" + raw_name.replace("/", "_")
            if pax_names is None
            else pax_names[index]
        )
        blocks.append(
            _ustar_header(
                pax_name,
                typeflag=b"x",
                mode=b"0100644 ",
                size=len(pax_payload),
                uname="",
                gname="",
            )
        )
        blocks.append(_padded(pax_payload))
        if logical_types is None:
            typeflag = b"5" if kind == "directory" else b"0"
        else:
            typeflag = logical_types[index]
        body = b"" if payload is None else payload
        mode = b"0040755 " if kind == "directory" else b"0100644 "
        blocks.append(
            _ustar_header(
                raw_name,
                typeflag=typeflag,
                mode=mode,
                size=0 if kind == "directory" else len(body),
                uname="root",
                gname="root",
                linkname=("" if logical_linknames is None else logical_linknames[index]),
            )
        )
        if kind != "directory":
            blocks.append(_padded(body))
    tar_payload = b"".join(blocks) + b"\0" * 1024 + tail
    return gzip.compress(tar_payload, compresslevel=6, mtime=0)


def _write_archive(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _mutate_tar_header(
    payload: bytes, *, header_offset: int, field_offset: int, replacement: bytes
) -> bytes:
    tar_payload = bytearray(gzip.decompress(payload))
    header = bytearray(tar_payload[header_offset : header_offset + 512])
    header[field_offset : field_offset + len(replacement)] = replacement
    header[148:156] = b"        "
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}".encode("ascii") + b"\0 "
    tar_payload[header_offset : header_offset + 512] = header
    return gzip.compress(bytes(tar_payload), compresslevel=6, mtime=0)


def _object_record(
    *, key: str, version: str, size: int, full_sha256: str | None = None
) -> dict[str, object]:
    checksum = (
        {"algorithm": "CRC32", "type": "COMPOSITE", "value": "AAAAAA==-2"}
        if full_sha256 is None
        else {
            "algorithm": "SHA256",
            "type": "FULL_OBJECT",
            "value": base64.b64encode(bytes.fromhex(full_sha256)).decode("ascii"),
        }
    )
    return {
        "bucket": "ir-sagemaker",
        "key": key,
        "version_id": version,
        "size": size,
        "etag": '"0123456789abcdef0123456789abcdef-2"',
        "checksum": checksum,
        "encryption": {
            "algorithm": "aws:kms",
            "kms_key_id": _KMS_ARN,
            "bucket_key_enabled": True,
        },
    }


def _canonical_systems(outer_fold: int) -> list[dict[str, object]]:
    return bridge._expected_cells(outer_fold)


def _build_manifest(root: Path) -> dict[str, object]:
    systems: list[dict[str, object]] = []
    for ordinal, cell in enumerate(_canonical_systems(0)):
        system_id = bridge._system_id(cell)
        run_id = bridge._run_id(cell)
        archive_name = f"{ordinal:02d}-{system_id}.model.tar.gz"
        archive_path = root / archive_name
        artifact_manifest = (
            json.dumps({"system_id": system_id}, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        _write_archive(
            archive_path,
            _archive_bytes([("file", "artifact_manifest.json", artifact_manifest)]),
        )
        size = archive_path.stat().st_size
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        systems.append(
            {
                "ordinal": ordinal,
                "system_id": system_id,
                "run_id": run_id,
                "job_name": bridge._job_name(cell, "a3"),
                "cell": cell,
                "archive_path": str(archive_path),
                "source_object": _object_record(
                    key=f"outputs/{run_id}/job/output/model.tar.gz",
                    version=f"source-version-{ordinal}",
                    size=size,
                ),
                "destination_object": _object_record(
                    key=f"fold-0/archives/{archive_name}",
                    version=f"destination-version-{ordinal}",
                    size=size,
                    full_sha256=archive_sha256,
                ),
                "terminal_receipt_sha256": hashlib.sha256(
                    f"terminal-{ordinal}".encode()
                ).hexdigest(),
                "request_receipt_sha256": hashlib.sha256(
                    f"request-{ordinal}".encode()
                ).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "protocol": bridge.ARCHIVE_INPUT_MANIFEST_PROTOCOL,
        "experiment_id": "arr_retrieval_cv_v1",
        "outer_fold": 0,
        "attempt_id": "a3",
        "archive_root": str(root),
        "training_plan_sha256": _ZERO_SHA,
        "training_staging_receipt_sha256": _ONE_SHA,
        "source_bundle": {
            "name": f"source-{_TWO_SHA}.tar.gz",
            "size": 400_089,
            "sha256": _TWO_SHA,
            "inventory_sha256": "3" * 64,
            "commit_epoch": 1_783_917_519,
        },
        "copy_set_receipt_sha256": "4" * 64,
        "systems": systems,
    }


def _replace_system_archive(
    system: dict[str, object], members: list[tuple[str, str, bytes | None]]
) -> None:
    archive_path = Path(system["archive_path"])
    _write_archive(archive_path, _archive_bytes(members))
    size = archive_path.stat().st_size
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    system["source_object"]["size"] = size
    system["destination_object"]["size"] = size
    system["destination_object"]["checksum"] = {
        "algorithm": "SHA256",
        "type": "FULL_OBJECT",
        "value": base64.b64encode(bytes.fromhex(archive_sha256)).decode("ascii"),
    }


def _expectations(
    manifest: dict[str, object], receipt: dict[str, object]
) -> dict[str, ControlledArtifactExpectation]:
    result: dict[str, ControlledArtifactExpectation] = {}
    source = manifest["source_bundle"]
    for system, receipt_system in zip(manifest["systems"], receipt["systems"]):
        cell = system["cell"]
        result[system["system_id"]] = ControlledArtifactExpectation(
            artifact_manifest_sha256=receipt_system["archive_evidence"]["artifact"][
                "artifact_manifest_sha256"
            ],
            training_plan_sha256=manifest["training_plan_sha256"],
            training_staging_receipt_sha256=manifest[
                "training_staging_receipt_sha256"
            ],
            source_bundle_name=source["name"],
            source_bundle_size=source["size"],
            source_bundle_sha256=source["sha256"],
            source_bundle_inventory_sha256=source["inventory_sha256"],
            source_bundle_commit_epoch=source["commit_epoch"],
            experiment_id=manifest["experiment_id"],
            outer_fold=cell["outer_fold"],
            query_view=cell["query_view"],
            sampler=cell["sampler"],
            experiment_seed=cell["experiment_seed"],
            dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
            fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
            passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
            model_artifact_protocol=CONTROLLED_ARTIFACT_PROTOCOL,
        )
    return result


def _reseal(receipt: dict[str, object]) -> None:
    unsealed = copy.deepcopy(receipt)
    del unsealed["receipt_sha256"]
    receipt["receipt_sha256"] = bridge._document_sha256(unsealed)


def _refresh_resealed_receipt(
    receipt: dict[str, object], *, system_index: int = 0
) -> None:
    evidence = receipt["systems"][system_index]["archive_evidence"]
    members = evidence["members"]
    files = sorted(
        (record for record in members if record["kind"] == "file"),
        key=lambda record: record["path"],
    )
    directories = [record for record in members if record["kind"] == "directory"]
    tar = evidence["tar"]
    tar.update(
        {
            "physical_member_count": 2 * len(members),
            "logical_member_count": len(members),
            "pax_header_count": len(members),
            "file_count": len(files),
            "directory_count": len(directories),
            "file_bytes": sum(record["size"] for record in files),
            "pax_payload_bytes": 130 * len(members),
            "max_pax_payload_bytes": 130,
            "max_file_bytes": max(record["size"] for record in files),
            "max_path_bytes": max(
                len(record["path"].encode("utf-8")) for record in members
            ),
            "stream_inventory_sha256": bridge._document_sha256(members),
            "member_inventory_sha256": bridge._document_sha256(
                sorted(members, key=lambda record: record["path"])
            ),
        }
    )
    manifest = next(
        record for record in files if record["path"] == "artifact_manifest.json"
    )
    artifact = evidence["artifact"]
    artifact.update(
        {
            "artifact_manifest_size": manifest["size"],
            "artifact_manifest_sha256": manifest["sha256"],
            "artifact_manifest_capture_sha256": manifest["sha256"],
            "file_count": len(files),
            "file_bytes": sum(record["size"] for record in files),
            "file_inventory_sha256": bridge._document_sha256(files),
        }
    )
    uncompressed_size = (
        len(members) * 1536
        + sum(((record["size"] + 511) // 512) * 512 for record in files)
        + 1024
    )
    evidence["gzip"]["uncompressed_size"] = uncompressed_size
    evidence["gzip"]["isize_mod_2_32"] = uncompressed_size % (2**32)
    receipt["aggregate"] = bridge._aggregate_evidence(receipt["systems"])
    _reseal(receipt)


@dataclasses.dataclass(frozen=True)
class _FakeIdentity:
    system_id: str
    manifest_sha256: str


def _fake_artifact_validator(root: Path, *, expectation: ControlledArtifactExpectation):
    files: list[ArtifactFileRecord] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file():
            payload = path.read_bytes()
            files.append(
                ArtifactFileRecord(
                    path=path.relative_to(root).as_posix(),
                    size=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
    return SimpleNamespace(
        root=root,
        files=tuple(files),
        identity=_FakeIdentity(root.name, expectation.artifact_manifest_sha256),
    )


class ArchiveEnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _scan_payload(self, payload: bytes) -> dict[str, object]:
        path = self.root / "model.tar.gz"
        _write_archive(path, payload)
        return bridge.scan_controlled_archive(path, expected_size=len(payload))

    def test_exact_libarchive_envelope_scans(self) -> None:
        payload = _archive_bytes(
            [
                ("directory", "metadata", None),
                ("file", "metadata/value.json", b"{\"ok\":true}\n"),
                ("file", "artifact_manifest.json", b"{\"commit_marker\":true}\n"),
            ]
        )
        evidence = self._scan_payload(payload)
        self.assertEqual(evidence["archive"]["size"], len(payload))
        self.assertEqual(evidence["gzip"]["member_count"], 1)
        self.assertEqual(evidence["tar"]["physical_member_count"], 6)
        self.assertEqual(evidence["tar"]["logical_member_count"], 3)
        self.assertEqual(evidence["tar"]["pax_header_count"], 3)
        self.assertEqual(evidence["tar"]["trailing_zero_bytes"], 0)
        self.assertEqual(
            [record["path"] for record in evidence["members"]],
            ["metadata", "metadata/value.json", "artifact_manifest.json"],
        )

    def test_unsorted_siblings_and_differing_ctime_are_accepted(self) -> None:
        differing_ctime = _pax_payload(
            values={
                "ctime": f"{_EPOCH - 1}.7654321",
                "mtime": f"{_EPOCH}.9999999",
                "LIBARCHIVE.creationtime": f"{_EPOCH}.9999999",
            }
        )
        payload = _archive_bytes(
            [
                ("file", "zeta.json", b"z\n"),
                ("file", "artifact_manifest.json", b"{}\n"),
                ("file", "alpha.json", b"a\n"),
            ],
            pax_payloads=[differing_ctime] * 3,
        )
        evidence = self._scan_payload(payload)
        self.assertEqual(
            [record["path"] for record in evidence["members"]],
            ["zeta.json", "artifact_manifest.json", "alpha.json"],
        )

    def test_gzip_header_mutations_are_rejected(self) -> None:
        valid = _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        for offset, value in ((2, 9), (3, 4), (4, 1), (8, 2), (9, 0)):
            with self.subTest(offset=offset):
                changed = bytearray(valid)
                changed[offset] = value
                with self.assertRaises(ValueError):
                    self._scan_payload(bytes(changed))

    def test_raw_ustar_header_mutations_are_rejected(self) -> None:
        valid = _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        logical_header = 1024
        variants = (
            (0, 100, b"0100600 "),
            (0, 108, b"0000001 "),
            (logical_header, 100, b"0100600 "),
            (logical_header, 257, b"ustar "),
            (logical_header, 297, _nul_field("admin", 32)),
            (logical_header, 500, b"x"),
        )
        for header_offset, field_offset, replacement in variants:
            with self.subTest(
                header_offset=header_offset,
                field_offset=field_offset,
            ):
                changed = _mutate_tar_header(
                    valid,
                    header_offset=header_offset,
                    field_offset=field_offset,
                    replacement=replacement,
                )
                with self.assertRaises(ValueError):
                    self._scan_payload(changed)

    def test_truncated_concatenated_and_trailing_gzip_rejected(self) -> None:
        valid = _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        for label, payload in (
            ("truncated", valid[:-3]),
            ("concatenated", valid + valid),
            ("trailing", valid + b"x"),
        ):
            with self.subTest(label=label):
                path = self.root / f"{label}.tar.gz"
                _write_archive(path, payload)
                with self.assertRaises(ValueError):
                    bridge.scan_controlled_archive(path, expected_size=len(payload))

    def test_tar_tail_and_nonzero_padding_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._scan_payload(
                _archive_bytes(
                    [("file", "artifact_manifest.json", b"{}\n")], tail=b"\0" * 512
                )
            )
        payload = bytearray(
            _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        )
        # Rebuilding a stream with nonzero TAR padding is clearer than patching compressed bytes.
        pax = _pax_payload()
        blocks = [
            _ustar_header(
                "./PaxHeaders.X/artifact_manifest.json",
                typeflag=b"x",
                mode=b"0100644 ",
                size=len(pax),
                uname="",
                gname="",
            ),
            _padded(pax),
            _ustar_header(
                "artifact_manifest.json",
                typeflag=b"0",
                mode=b"0100644 ",
                size=3,
                uname="root",
                gname="root",
            ),
            b"{}\n" + b"x" + b"\0" * 508,
            b"\0" * 1024,
        ]
        with self.assertRaises(ValueError):
            self._scan_payload(gzip.compress(b"".join(blocks), compresslevel=6, mtime=0))
        del payload

    def test_pax_order_keys_times_and_name_are_exact(self) -> None:
        variants = [
            {
                "pax_payloads": [
                    _pax_payload(
                        key_order=("ctime", "atime", "mtime", "LIBARCHIVE.creationtime")
                    )
                ]
            },
            {
                "pax_payloads": [
                    _pax_payload(values={"mtime": f"{_EPOCH}.000000"})
                ]
            },
            {
                "pax_payloads": [
                    _pax_payload(
                        values={
                            "LIBARCHIVE.creationtime": f"{_EPOCH + 1}.0000000"
                        }
                    )
                ]
            },
            {"pax_names": ["./PaxHeaders.X/wrong"]},
        ]
        for index, kwargs in enumerate(variants):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self._scan_payload(
                        _archive_bytes(
                            [("file", "artifact_manifest.json", b"{}\n")], **kwargs
                        )
                    )

    def test_pax_epochs_have_exactly_ten_decimal_digits(self) -> None:
        for epoch in ("123456789", "0123456789", "12345678901"):
            with self.subTest(epoch=epoch):
                with self.assertRaises(ValueError):
                    bridge._parse_pax_payload(
                        _pax_payload(values={"atime": f"{epoch}.0000000"})
                    )

    def test_traversal_and_path_aliases_rejected(self) -> None:
        for name in ("../x", "/x", "./x", "a//x", "a\\x"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._scan_payload(_archive_bytes([("file", name, b"x")]))

    def test_links_special_sparse_and_duplicate_paths_rejected(self) -> None:
        for typeflag in (b"1", b"2", b"3", b"4", b"6", b"S", b"L"):
            with self.subTest(typeflag=typeflag):
                with self.assertRaises(ValueError):
                    self._scan_payload(
                        _archive_bytes(
                            [("file", "artifact_manifest.json", b"{}\n")],
                            logical_types=[typeflag],
                        )
                    )
        with self.assertRaises(ValueError):
            self._scan_payload(
                _archive_bytes(
                    [
                        ("file", "artifact_manifest.json", b"{}\n"),
                        ("file", "artifact_manifest.json", b"other\n"),
                    ]
                )
            )

    def test_parent_must_precede_child(self) -> None:
        with self.assertRaises(ValueError):
            self._scan_payload(
                _archive_bytes(
                    [
                        ("file", "d/artifact_manifest.json", b"{}\n"),
                        ("directory", "d", None),
                    ]
                )
            )

    def test_symlink_and_hardlinked_archive_rejected(self) -> None:
        payload = _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        original = self.root / "original.tar.gz"
        _write_archive(original, payload)
        symlink = self.root / "symlink.tar.gz"
        symlink.symlink_to(original)
        with self.assertRaises(OSError):
            bridge.scan_controlled_archive(symlink, expected_size=len(payload))
        hardlink = self.root / "hardlink.tar.gz"
        os.link(original, hardlink)
        with self.assertRaises(ValueError):
            bridge.scan_controlled_archive(original, expected_size=len(payload))

    def test_size_and_tree_caps_are_fail_closed(self) -> None:
        payload = _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        path = self.root / "model.tar.gz"
        _write_archive(path, payload)
        with mock.patch.object(bridge, "MAX_ARCHIVE_BYTES", len(payload)):
            with self.assertRaises(ValueError):
                bridge.scan_controlled_archive(path, expected_size=len(payload))
        with mock.patch.object(bridge, "MAX_TREE_BYTES", 2):
            with self.assertRaises(ValueError):
                bridge.scan_controlled_archive(path, expected_size=len(payload))

    def test_path_replacement_during_scan_rejected(self) -> None:
        payload = _archive_bytes([("file", "artifact_manifest.json", b"{}\n")])
        path = self.root / "model.tar.gz"
        replacement = self.root / "replacement.tar.gz"
        _write_archive(path, payload)
        _write_archive(replacement, payload)
        original_hash = bridge._hash_archive

        def hash_then_swap(snapshot):
            result = original_hash(snapshot)
            os.replace(replacement, path)
            return result

        with mock.patch.object(bridge, "_hash_archive", side_effect=hash_then_swap):
            with self.assertRaises(RuntimeError):
                bridge.scan_controlled_archive(path, expected_size=len(payload))


class StableJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_hardlinked_canonical_json_is_rejected(self) -> None:
        path = self.root / "document.json"
        path.write_bytes(bridge._canonical_bytes({"ok": True}))
        os.link(path, self.root / "alias.json")
        with self.assertRaises(ValueError):
            bridge._load_canonical_json(path, name="Test document")

    def test_path_replacement_during_canonical_json_read_is_rejected(self) -> None:
        path = self.root / "document.json"
        replacement = self.root / "replacement.json"
        path.write_bytes(bridge._canonical_bytes({"ok": True}))
        replacement.write_bytes(bridge._canonical_bytes({"ok": True}))
        original_read = bridge._read_descriptor_exact

        def read_then_swap(descriptor, size, *, name):
            raw = original_read(descriptor, size, name=name)
            os.replace(replacement, path)
            return raw

        with mock.patch.object(
            bridge, "_read_descriptor_exact", side_effect=read_then_swap
        ):
            with self.assertRaises(RuntimeError):
                bridge._load_canonical_json(path, name="Test document")


class FoldManifestReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.manifest = _build_manifest(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_twelve_cell_manifest_and_receipt(self) -> None:
        validated = bridge.validate_fold_archive_input_manifest(self.manifest)
        self.assertEqual(len(validated["systems"]), 12)
        receipt = bridge.build_fold_archive_inventory_receipt(validated)
        replay = bridge.validate_fold_archive_inventory_receipt(
            receipt, input_manifest=validated
        )
        self.assertEqual(replay, receipt)
        self.assertEqual(receipt["aggregate"]["archive_count"], 12)
        self.assertEqual(receipt["receipt_sha256"], bridge._document_sha256({
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }))

    def test_unsorted_archive_order_survives_receipt_validation(self) -> None:
        _replace_system_archive(
            self.manifest["systems"][0],
            [
                ("file", "zeta.json", b"z\n"),
                ("file", "artifact_manifest.json", b"{}\n"),
                ("file", "alpha.json", b"a\n"),
            ],
        )
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        validated = bridge.validate_fold_archive_inventory_receipt(
            receipt, input_manifest=self.manifest
        )
        self.assertEqual(
            [
                member["path"]
                for member in validated["systems"][0]["archive_evidence"]["members"]
            ],
            ["zeta.json", "artifact_manifest.json", "alpha.json"],
        )

    def test_manifest_missing_reordered_or_aliased_cell_rejected(self) -> None:
        missing = copy.deepcopy(self.manifest)
        missing["systems"].pop()
        reordered = copy.deepcopy(self.manifest)
        reordered["systems"][0], reordered["systems"][1] = (
            reordered["systems"][1],
            reordered["systems"][0],
        )
        aliased = copy.deepcopy(self.manifest)
        aliased["systems"][1]["destination_object"] = copy.deepcopy(
            aliased["systems"][0]["destination_object"]
        )
        for value in (missing, reordered, aliased):
            with self.assertRaises(ValueError):
                bridge.validate_fold_archive_input_manifest(value)

    def test_receipt_splice_and_reseal_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        pair: tuple[int, int] | None = None
        for left in range(len(receipt["systems"])):
            for right in range(left + 1, len(receipt["systems"])):
                left_archive = receipt["systems"][left]["archive_evidence"]["archive"]
                right_archive = receipt["systems"][right]["archive_evidence"]["archive"]
                if (
                    left_archive["size"] == right_archive["size"]
                    and left_archive["sha256"] != right_archive["sha256"]
                ):
                    pair = (left, right)
                    break
            if pair is not None:
                break
        self.assertIsNotNone(pair, "fixture needs two different same-size archives")
        left, right = pair
        spliced = copy.deepcopy(receipt)
        spliced["systems"][left]["archive_evidence"], spliced["systems"][right][
            "archive_evidence"
        ] = (
            spliced["systems"][right]["archive_evidence"],
            spliced["systems"][left]["archive_evidence"],
        )
        spliced["aggregate"] = bridge._aggregate_evidence(spliced["systems"])
        _reseal(spliced)
        with self.assertRaises(ValueError):
            bridge.validate_fold_archive_inventory_receipt(
                spliced, input_manifest=self.manifest
            )

    def test_resealed_receipt_numeric_type_confusion_is_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        for field, changed in (("member_count", True), ("compression_method", 8.0)):
            with self.subTest(field=field):
                forged = copy.deepcopy(receipt)
                forged["systems"][0]["archive_evidence"]["gzip"][field] = changed
                _reseal(forged)
                with self.assertRaises(ValueError):
                    bridge.validate_fold_archive_inventory_receipt(
                        forged, input_manifest=self.manifest
                    )

    def test_resealed_impossible_uncompressed_stream_size_is_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        forged = copy.deepcopy(receipt)
        gzip_evidence = forged["systems"][0]["archive_evidence"]["gzip"]
        gzip_evidence["uncompressed_size"] += 512
        gzip_evidence["isize_mod_2_32"] = gzip_evidence["uncompressed_size"] % (2**32)
        forged["aggregate"] = bridge._aggregate_evidence(forged["systems"])
        _reseal(forged)
        with self.assertRaises(ValueError):
            bridge.validate_fold_archive_inventory_receipt(
                forged, input_manifest=self.manifest
            )

    def test_resealed_unrepresentable_member_paths_are_rejected(self) -> None:
        _replace_system_archive(
            self.manifest["systems"][0],
            [
                ("directory", "a", None),
                ("file", "a/b", b"b\n"),
                ("file", "a_c", b"c\n"),
                ("file", "artifact_manifest.json", b"{}\n"),
            ],
        )
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        for forged_path in ("x" * 100, "x" * 85, "bad:path"):
            with self.subTest(path=forged_path):
                forged = copy.deepcopy(receipt)
                member = next(
                    record
                    for record in forged["systems"][0]["archive_evidence"]["members"]
                    if record["path"] == "a_c"
                )
                member["path"] = forged_path
                _refresh_resealed_receipt(forged)
                with self.assertRaises(ValueError):
                    bridge.validate_fold_archive_inventory_receipt(
                        forged, input_manifest=self.manifest
                    )

    def test_resealed_derived_pax_name_collision_is_rejected(self) -> None:
        _replace_system_archive(
            self.manifest["systems"][0],
            [
                ("directory", "a", None),
                ("file", "a/b", b"b\n"),
                ("file", "a_c", b"c\n"),
                ("file", "artifact_manifest.json", b"{}\n"),
            ],
        )
        forged = bridge.build_fold_archive_inventory_receipt(self.manifest)
        member = next(
            record
            for record in forged["systems"][0]["archive_evidence"]["members"]
            if record["path"] == "a_c"
        )
        member["path"] = "a_b"
        _refresh_resealed_receipt(forged)
        with self.assertRaises(ValueError):
            bridge.validate_fold_archive_inventory_receipt(
                forged, input_manifest=self.manifest
            )

    def test_resealed_ustar_file_size_overflow_is_rejected(self) -> None:
        _replace_system_archive(
            self.manifest["systems"][0],
            [
                ("file", "payload.bin", b"payload\n"),
                ("file", "artifact_manifest.json", b"{}\n"),
            ],
        )
        forged = bridge.build_fold_archive_inventory_receipt(self.manifest)
        member = next(
            record
            for record in forged["systems"][0]["archive_evidence"]["members"]
            if record["path"] == "payload.bin"
        )
        member["size"] = 9_000_000_000
        _refresh_resealed_receipt(forged)
        with self.assertRaises(ValueError):
            bridge.validate_fold_archive_inventory_receipt(
                forged, input_manifest=self.manifest
            )

    def test_resealed_impossible_allocated_bytes_are_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        archive_size = receipt["systems"][0]["archive_evidence"]["archive"]["size"]
        under_size = ((archive_size - 1) // 512) * 512
        for allocated_bytes in (1, under_size):
            with self.subTest(allocated_bytes=allocated_bytes):
                forged = copy.deepcopy(receipt)
                forged["systems"][0]["archive_evidence"]["archive"][
                    "allocated_bytes"
                ] = allocated_bytes
                _refresh_resealed_receipt(forged)
                with self.assertRaises(ValueError):
                    bridge.validate_fold_archive_inventory_receipt(
                        forged, input_manifest=self.manifest
                    )

    def test_receipt_manifest_cross_binding_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        changed_manifest = copy.deepcopy(self.manifest)
        changed_manifest["copy_set_receipt_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            bridge.validate_fold_archive_inventory_receipt(
                receipt, input_manifest=changed_manifest
            )

    def test_absent_only_receipt_write(self) -> None:
        output = self.root / "receipt.json"
        receipt = bridge.write_fold_archive_inventory_receipt(
            self.manifest, output_path=output
        )
        self.assertEqual(output.read_bytes(), bridge._canonical_bytes(receipt))
        with self.assertRaises(FileExistsError):
            bridge.write_fold_archive_inventory_receipt(
                self.manifest, output_path=output
            )

    def test_receipt_write_loops_over_legal_short_writes(self) -> None:
        output = self.root / "short-write-receipt.json"
        original_write = os.write
        call_count = 0

        def short_write(descriptor, payload):
            nonlocal call_count
            call_count += 1
            return original_write(descriptor, payload[: min(len(payload), 17)])

        with mock.patch.object(bridge.os, "write", side_effect=short_write):
            receipt = bridge.write_fold_archive_inventory_receipt(
                self.manifest, output_path=output
            )
        self.assertGreater(call_count, 1)
        self.assertEqual(output.read_bytes(), bridge._canonical_bytes(receipt))

    def test_receipt_parent_replacement_is_rejected(self) -> None:
        parent = self.root / "receipt-parent"
        moved = self.root / "receipt-parent-moved"
        parent.mkdir()
        output = parent / "receipt.json"
        original_open = bridge._open_absent_regular

        def swap_parent_then_open(parent_descriptor, name):
            os.rename(parent, moved)
            parent.mkdir()
            return original_open(parent_descriptor, name)

        with mock.patch.object(
            bridge, "_open_absent_regular", side_effect=swap_parent_then_open
        ):
            with self.assertRaises((RuntimeError, ValueError)):
                bridge.write_fold_archive_inventory_receipt(
                    self.manifest, output_path=output
                )
        self.assertFalse(output.exists())
        self.assertTrue((moved / "receipt.json").is_file())

    def test_successful_materialization_rescans_and_revalidates(self) -> None:
        _replace_system_archive(
            self.manifest["systems"][0],
            [
                ("directory", "metadata", None),
                ("file", "metadata/value.json", b"{}\n"),
                ("file", "artifact_manifest.json", b"{}\n"),
            ],
        )
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        expectations = _expectations(self.manifest, receipt)
        output = self.root / "materialized"
        with mock.patch.object(
            bridge, "validate_controlled_artifact", side_effect=_fake_artifact_validator
        ) as validator:
            materialized = bridge.materialize_fold_archives(
                self.manifest,
                receipt,
                output_root=output,
                expectations=expectations,
            )
        self.assertEqual(materialized.root, output)
        self.assertEqual(len(materialized.artifacts), 12)
        self.assertEqual(validator.call_count, 24)
        self.assertFalse((self.root / ".materialized.incomplete").exists())
        for system in self.manifest["systems"]:
            artifact = output / system["system_id"] / "artifact_manifest.json"
            self.assertTrue(artifact.is_file())
            self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o600)
        metadata = output / self.manifest["systems"][0]["system_id"] / "metadata"
        self.assertTrue(metadata.is_dir())
        self.assertEqual(stat.S_IMODE(metadata.stat().st_mode), 0o700)

    def test_partial_extraction_is_not_published_or_reused(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        expectations = _expectations(self.manifest, receipt)
        second = Path(self.manifest["systems"][1]["archive_path"])
        second.write_bytes(second.read_bytes()[:-1])
        output = self.root / "materialized"
        with mock.patch.object(
            bridge, "validate_controlled_artifact", side_effect=_fake_artifact_validator
        ):
            with self.assertRaises((ValueError, RuntimeError)):
                bridge.materialize_fold_archives(
                    self.manifest,
                    receipt,
                    output_root=output,
                    expectations=expectations,
                )
        self.assertFalse(output.exists())
        self.assertTrue((self.root / ".materialized.incomplete").is_dir())
        with self.assertRaises(FileExistsError):
            bridge.materialize_fold_archives(
                self.manifest,
                receipt,
                output_root=output,
                expectations=expectations,
            )

    def test_existing_output_is_never_overwritten(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        expectations = _expectations(self.manifest, receipt)
        output = self.root / "materialized"
        output.mkdir()
        marker = output / "owned.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            bridge.materialize_fold_archives(
                self.manifest,
                receipt,
                output_root=output,
                expectations=expectations,
            )
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_materialization_parent_replacement_at_rename_is_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        expectations = _expectations(self.manifest, receipt)
        parent = self.root / "publication"
        moved = self.root / "publication-moved"
        parent.mkdir()
        output = parent / "materialized"
        original_rename = bridge._rename_no_replace

        def swap_parent_then_rename(parent_descriptor, source_name, target_name):
            os.rename(parent, moved)
            parent.mkdir()
            return original_rename(parent_descriptor, source_name, target_name)

        with mock.patch.object(
            bridge, "validate_controlled_artifact", side_effect=_fake_artifact_validator
        ), mock.patch.object(
            bridge, "_rename_no_replace", side_effect=swap_parent_then_rename
        ):
            with self.assertRaises((RuntimeError, ValueError)):
                bridge.materialize_fold_archives(
                    self.manifest,
                    receipt,
                    output_root=output,
                    expectations=expectations,
                )
        self.assertFalse(output.exists())
        self.assertTrue((moved / "materialized").is_dir())

    def test_published_output_replacement_during_revalidation_is_rejected(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        expectations = _expectations(self.manifest, receipt)
        output = self.root / "materialized"
        moved = self.root / "materialized-moved"
        call_count = 0

        def validate_then_swap(root, *, expectation):
            nonlocal call_count
            artifact = _fake_artifact_validator(root, expectation=expectation)
            call_count += 1
            if call_count == 13:
                os.rename(output, moved)
                output.mkdir()
            return artifact

        with mock.patch.object(
            bridge, "validate_controlled_artifact", side_effect=validate_then_swap
        ):
            with self.assertRaises((RuntimeError, ValueError)):
                bridge.materialize_fold_archives(
                    self.manifest,
                    receipt,
                    output_root=output,
                    expectations=expectations,
                )
        self.assertTrue(output.is_dir())
        self.assertTrue(moved.is_dir())

    def test_external_expectation_mismatch_fails_before_extraction(self) -> None:
        receipt = bridge.build_fold_archive_inventory_receipt(self.manifest)
        expectations = _expectations(self.manifest, receipt)
        first = next(iter(expectations))
        expectations[first] = dataclasses.replace(
            expectations[first], artifact_manifest_sha256="f" * 64
        )
        output = self.root / "materialized"
        with self.assertRaises(ValueError):
            bridge.materialize_fold_archives(
                self.manifest,
                receipt,
                output_root=output,
                expectations=expectations,
            )
        self.assertFalse(output.exists())
        self.assertFalse((self.root / ".materialized.incomplete").exists())


if __name__ == "__main__":
    unittest.main()
