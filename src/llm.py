"""Resilient client for any OpenAI-compatible chat API (OpenRouter, Groq, ...).

Failure handling (A11, risk R-06/R-07/R-08):
  * timeouts on every call; bounded retries with exponential backoff + jitter for
    timeouts, 429 and 5xx, honouring Retry-After when the provider sends it;
  * a circuit breaker: after repeated consecutive failures the client stops calling
    the provider for a cool-down period and fails fast, so an outage costs
    milliseconds per ticket instead of a full retry cycle each time;
  * an optional minimum interval between calls to stay under free-tier rate limits;
  * a SQLite response cache keyed on (model, messages, params): repeated runs are
    reproducible, free, and keep working for already-seen tickets during an outage.
Callers get an LLMUnavailable exception, never a crash, and route to a human.

The API key is read from the environment, sent only in the Authorization header,
and never logged or included in error messages.

LLM_PROVIDER=mock gives an offline, deterministic stand-in used by the tests and
CI (no network, no key). It is never selected implicitly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from dataclasses import dataclass

import httpx

from .config import settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The provider could not produce a response (outage, rate limit, timeout, breaker open)."""


@dataclass
class LLMResponse:
    content: str
    model: str
    from_cache: bool
    latency_ms: float
    throttle_ms: float = 0.0  # time spent waiting for the rate limiter (batch artefact, not service time)


class _Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, model TEXT, created REAL)")
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str, model: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)", (key, value, model, time.time()))
            self._conn.commit()


class CircuitBreaker:
    def __init__(self, threshold: int = 4, cooldown_s: float = 30.0):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self.failures, self.opened_at = 0, 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.failures < self.threshold:
                return True
            if time.monotonic() - self.opened_at >= self.cooldown_s:  # half-open: let one call probe
                self.failures = self.threshold - 1
                return True
            return False

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.failures = 0
            else:
                self.failures += 1
                if self.failures >= self.threshold:
                    self.opened_at = time.monotonic()

    @property
    def state(self) -> str:
        return "open" if self.failures >= self.threshold else "closed"


class LLMClient:
    RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, s=settings, transport: httpx.BaseTransport | None = None, use_cache: bool = True):
        self.s = s
        self.provider = s.llm_provider
        self.model = s.model_name
        self.cache = _Cache(s.cache_path) if use_cache else None
        self.breaker = CircuitBreaker()
        self._last_call = 0.0
        self._rate_lock = threading.Lock()
        self._http = None
        if self.provider != "mock":
            if not s.llm_api_key:
                log.warning("No LLM API key configured; generation will be unavailable and tickets will escalate.")
            headers = {"Authorization": f"Bearer {s.llm_api_key}", "Content-Type": "application/json"}
            if self.provider == "openrouter":
                headers["X-Title"] = "CloudServe support triage (capstone)"
            self._http = httpx.Client(base_url=s.llm_base_url, headers=headers, timeout=s.llm_timeout_s,
                                      transport=transport)

    # -- public ------------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600,
             json_mode: bool = True) -> LLMResponse:
        key = hashlib.sha256(json.dumps([self.model, messages, temperature, max_tokens, json_mode],
                                        sort_keys=True).encode()).hexdigest()
        if self.cache and (hit := self.cache.get(key)) is not None:
            return LLMResponse(hit, self.model, True, 0.0)
        if self.provider == "mock":
            content = _mock_completion(messages)
            return LLMResponse(content, "mock", False, 0.0)
        if not self.s.llm_api_key:
            raise LLMUnavailable("no API key configured")
        if not self.breaker.allow():
            raise LLMUnavailable("circuit breaker open after repeated provider failures")

        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        throttle = 0.0
        last_err = "unknown error"
        for attempt in range(self.s.llm_max_retries + 1):
            throttle += self._respect_rate_limit()
            try:
                r = self._http.post("/chat/completions", json=payload)
                if r.status_code == 400 and json_mode and "response_format" in r.text:
                    payload.pop("response_format", None)  # model without JSON mode: rely on the prompt
                    continue
                if r.status_code in self.RETRYABLE:
                    last_err = f"HTTP {r.status_code}"
                    self._sleep_backoff(attempt, r.headers.get("retry-after"))
                    continue
                if r.status_code >= 400:
                    last_err = f"HTTP {r.status_code}"
                    break  # 401/403/404 etc. will not fix themselves
                content = r.json()["choices"][0]["message"]["content"] or ""
                self.breaker.record(True)
                if self.cache:
                    self.cache.put(key, content, self.model)
                return LLMResponse(content, self.model, False, (time.perf_counter() - start) * 1000, throttle * 1000)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = type(exc).__name__
                self._sleep_backoff(attempt, None)
            except (KeyError, IndexError, ValueError) as exc:
                last_err = f"malformed provider response ({type(exc).__name__})"
                break
        self.breaker.record(False)
        raise LLMUnavailable(f"{self.provider} unavailable: {last_err}")

    # -- helpers -------------------------------------------------------------
    def _respect_rate_limit(self) -> float:
        if self.s.llm_min_interval_s <= 0:
            return 0.0
        with self._rate_lock:
            wait = self.s.llm_min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            return max(wait, 0.0)

    def _sleep_backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.s.llm_max_retries:
            return
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        delay = min(max(delay, (2 ** attempt) + random.uniform(0, 0.5)), 20.0)
        time.sleep(delay)


def _mock_completion(messages: list[dict]) -> str:
    """Deterministic, offline stand-in: answers with the first resolution steps of the
    first supplied passage and cites it. Exists only so tests and CI run without a key."""
    import re

    user = messages[-1]["content"]
    passages = re.findall(r'<passage id="([^"]+)">\s*(.*?)\s*</passage>', user, re.S)
    if not passages:
        return json.dumps({"insufficient_evidence": True, "answer": "", "citations": []})
    pid, text = next(((p, t) for p, t in passages if p.endswith("#resolution")), passages[0])
    text = text.split("\n", 1)[-1]  # drop the "title — section" header line
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"^\d+\.\s*", "", text.strip(), flags=re.M))
    claim = " ".join(f"{x.strip()} [{pid}]" for x in sentences[:2] if x.strip())
    return json.dumps({
        "insufficient_evidence": False,
        "answer": f"Thanks for getting in touch. {claim}",
        "citations": [{"passage_id": pid, "claim": claim}],
    })


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
