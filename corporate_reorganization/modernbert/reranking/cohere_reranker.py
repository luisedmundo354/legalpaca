from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float


class CohereReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-v4.0-pro",
        api_base_url: str = "https://api.cohere.ai",
        timeout_s: int = 120,
        max_documents_per_request: int = 100,
    ):
        self.api_key = str(api_key)
        self.model = str(model)
        self.api_base_url = str(api_base_url).rstrip("/")
        self.timeout_s = int(timeout_s)
        self.max_documents_per_request = int(max_documents_per_request)

    @staticmethod
    def from_env(
        *,
        api_key_env: str = "COHERE_API_KEY",
        model: str = "rerank-v4.0-pro",
        api_base_url_env: str = "COHERE_API_BASE_URL",
        timeout_s: int = 120,
        max_documents_per_request: int = 100,
    ) -> "CohereReranker":
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env var {api_key_env} for Cohere API key")
        api_base_url = os.environ.get(api_base_url_env) or "https://api.cohere.ai"
        return CohereReranker(
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
            timeout_s=timeout_s,
            max_documents_per_request=max_documents_per_request,
        )

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        documents = list(documents)
        if not documents:
            return []
        if top_n is None:
            top_n = len(documents)
        top_n = int(top_n)

        try:
            import cohere  # type: ignore

            client = cohere.Client(self.api_key)
            response = client.rerank(
                query=str(query),
                documents=list(documents),
                top_n=int(top_n),
                model=str(self.model),
            )
            out: List[RerankResult] = []
            for item in response.results:
                out.append(RerankResult(index=int(item.index), relevance_score=float(item.relevance_score)))
            return out
        except ImportError:
            return self._rerank_via_http(query=str(query), documents=documents, top_n=top_n)

    def _rerank_via_http(self, *, query: str, documents: List[str], top_n: int) -> List[RerankResult]:
        url = f"{self.api_base_url}/v1/rerank"
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": int(top_n),
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"Cohere rerank HTTP {e.code}: {body}") from e

        data = json.loads(raw)
        results = data.get("results") or []
        out: List[RerankResult] = []
        for item in results:
            out.append(RerankResult(index=int(item["index"]), relevance_score=float(item["relevance_score"])))
        return out

