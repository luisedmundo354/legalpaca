from __future__ import annotations

import hashlib


def stable_int64_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)
