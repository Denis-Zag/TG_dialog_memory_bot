"""Загрузка и проверка настроек приложения."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Ошибка отсутствующей или некорректной настройки."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки Telegram, ProxyAPI и Pinecone."""

    telegram_bot_token: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    openai_api_key: str
    openai_base_url: str
    embedding_model: str
    embedding_dimension: int
    chat_model: str
    duplicate_threshold: float
    retrieval_threshold: float
    search_top_k: int


def _read_positive_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} должен быть целым числом") from error
    if value <= 0:
        raise ConfigurationError(f"{name} должен быть больше нуля")
    return value


def _read_probability(name: str, default: str) -> float:
    raw_value = os.getenv(name, default).strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} должен быть числом") from error
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} должен находиться в диапазоне от 0 до 1")
    return value


def load_settings(*, require_telegram_token: bool = True) -> Settings:
    """Загрузить настройки из переменных окружения и локального файла .env."""

    load_dotenv()
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    missing: list[str] = []
    if require_telegram_token and not telegram_bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise ConfigurationError(
            "Заполните переменные в .env: " + ", ".join(missing)
        )

    return Settings(
        telegram_bot_token=telegram_bot_token,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=os.getenv(
            "PINECONE_INDEX_NAME", "tg-dialog-memory"
        ).strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        openai_api_key=openai_api_key,
        openai_base_url=os.getenv(
            "OPENAI_BASE_URL", "https://api.proxyapi.ru/v1"
        ).strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        ).strip(),
        embedding_dimension=_read_positive_int("EMBEDDING_DIMENSION", "1536"),
        chat_model=os.getenv("CHAT_MODEL", "openai/gpt-5-mini").strip(),
        duplicate_threshold=_read_probability(
            "MEMORY_DUPLICATE_THRESHOLD", "0.74"
        ),
        retrieval_threshold=_read_probability(
            "MEMORY_RETRIEVAL_THRESHOLD", "0.30"
        ),
        search_top_k=_read_positive_int("MEMORY_SEARCH_TOP_K", "3"),
    )
