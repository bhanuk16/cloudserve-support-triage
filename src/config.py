"""Central configuration, read once from environment variables (.env supported).

Every tunable lives here so that behaviour can be changed without code changes,
and every path is resolved relative to the project root so the system runs the
same way regardless of the directory it is launched from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else (ROOT / p)


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- model provider (OpenAI-compatible: OpenRouter or Groq) ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter").lower()
    llm_api_key: str = field(default="", repr=False)  # never printed
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
    llm_timeout_s: float = _float("LLM_TIMEOUT_SECONDS", 20.0)
    llm_max_retries: int = _int("LLM_MAX_RETRIES", 3)
    llm_min_interval_s: float = _float("LLM_MIN_INTERVAL_SECONDS", 0.0)

    # --- paths ---
    data_dir: Path = _path("DATA_DIR", "data")
    docs_path: Path = _path("DOCS_PATH", "data/documentation.json")
    train_path: Path = _path("TRAIN_PATH", "data/development_tickets.json")
    model_dir: Path = _path("EMBEDDING_MODEL_DIR", "models/all-MiniLM-L6-v2")
    artifacts_dir: Path = _path("ARTIFACTS_DIR", "storage/artifacts")
    chroma_path: Path = _path("CHROMA_PATH", "storage/chroma")
    db_path: Path = _path("DECISION_DB_PATH", "storage/decisions.db")
    cache_path: Path = _path("LLM_CACHE_PATH", "storage/llm_cache.db")

    # --- decision thresholds (chosen from data; see docs/thresholds.md) ---
    confidence_threshold: float = _float("CONFIDENCE_THRESHOLD", 0.80)
    retrieval_top_k: int = _int("RETRIEVAL_TOP_K", 5)
    retrieval_min_score: float = _float("RETRIEVAL_MIN_SCORE", 0.35)

    # --- operations ---
    auto_respond_enabled: bool = _bool("AUTO_RESPOND_ENABLED", True)  # kill switch
    kill_switch_file: Path = _path("KILL_SWITCH_FILE", "storage/KILL_SWITCH")
    max_body_chars: int = _int("MAX_BODY_CHARS", 8000)
    metrics_port: int = _int("METRICS_PORT", 8001)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def llm_base_url(self) -> str:
        override = os.getenv("LLM_BASE_URL")
        if override:
            return override.rstrip("/")
        return {
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }.get(self.llm_provider, "https://openrouter.ai/api/v1")


def load_settings() -> Settings:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    return Settings(llm_api_key=key.strip())


settings = load_settings()
