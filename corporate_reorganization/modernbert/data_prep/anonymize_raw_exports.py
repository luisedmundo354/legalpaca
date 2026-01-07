"""
Anonymizes Label Studio export metadata in final_annotations_gold/raw.

This script redacts annotator emails/names while preserving a stable per-annotator identifier
so experiments remain reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_json_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.glob("*.json") if p.is_file()])


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _extract_completed_by(data: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    completed_by = data.get("completed_by") if isinstance(data.get("completed_by"), dict) else {}
    user_id = completed_by.get("id")
    email = completed_by.get("email")
    return (int(user_id) if isinstance(user_id, int) else None, str(email) if isinstance(email, str) else None)


def _extract_assigned_to(data: Dict[str, Any]) -> Optional[str]:
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
    assigned_to = task_data.get("assigned_to")
    return str(assigned_to) if isinstance(assigned_to, str) else None


def _build_anon_maps(raw_dir: Path) -> Tuple[Dict[str, str], Dict[int, int]]:
    email_by_user_id: Dict[int, str] = {}
    for path in _iter_json_files(raw_dir):
        data = _read_json(path)
        user_id, email = _extract_completed_by(data)
        if user_id is not None and email:
            email_by_user_id[user_id] = email

    emails = sorted(set(email_by_user_id.values()))
    email_to_anon = {email: f"annotator_{idx + 1}" for idx, email in enumerate(emails)}
    user_id_to_anon_id = {
        user_id: (emails.index(email) + 1) for user_id, email in email_by_user_id.items() if email in emails
    }
    return email_to_anon, user_id_to_anon_id


def _apply_anonymization(data: Dict[str, Any], *, email_to_anon: Dict[str, str], user_id_to_anon_id: Dict[int, int]) -> None:
    user_id, email = _extract_completed_by(data)
    anon_label: Optional[str] = email_to_anon.get(email) if email else None
    anon_id: Optional[int] = user_id_to_anon_id.get(user_id) if user_id is not None else None

    if anon_label is None:
        assigned_to = _extract_assigned_to(data)
        anon_label = email_to_anon.get(assigned_to) if assigned_to else None
    if anon_id is None and anon_label is not None:
        try:
            anon_id = int(anon_label.split("_", 1)[1])
        except Exception:
            anon_id = None

    if "created_username" in data and anon_label is not None:
        data["created_username"] = anon_label

    if isinstance(data.get("completed_by"), dict) and anon_label is not None:
        completed_by = dict(data["completed_by"])
        completed_by["email"] = anon_label
        completed_by["first_name"] = ""
        completed_by["last_name"] = ""
        if anon_id is not None:
            completed_by["id"] = int(anon_id)
        data["completed_by"] = completed_by

    if anon_id is not None and isinstance(data.get("updated_by"), int):
        data["updated_by"] = int(anon_id)

    task = data.get("task")
    if isinstance(task, dict):
        task = dict(task)
        if isinstance(task.get("updated_by"), int) and anon_id is not None:
            task["updated_by"] = int(anon_id)

        task_data = task.get("data")
        if isinstance(task_data, dict):
            task_data = dict(task_data)
            if isinstance(task_data.get("assigned_to"), str) and anon_label is not None:
                task_data["assigned_to"] = anon_label
            task["data"] = task_data
        data["task"] = task


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    repo_root = _repo_root()
    default_raw_dir = repo_root / "corporate_reorganization/data/final_annotations_gold/raw"

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=default_raw_dir)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    paths = _iter_json_files(raw_dir)
    if not paths:
        raise SystemExit(f"No *.json files found under {raw_dir}")

    email_to_anon, user_id_to_anon_id = _build_anon_maps(raw_dir)
    if args.dry_run:
        print("email_to_anon:", json.dumps(email_to_anon, indent=2))
        print("user_id_to_anon_id:", json.dumps({str(k): v for k, v in user_id_to_anon_id.items()}, indent=2))
        return 0

    for path in paths:
        data = _read_json(path)
        _apply_anonymization(data, email_to_anon=email_to_anon, user_id_to_anon_id=user_id_to_anon_id)
        _write_json(path, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

