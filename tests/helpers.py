"""Общие тестовые объекты без настоящих ключей и сетевых запросов."""

from __future__ import annotations

from config import Settings


def make_settings(**overrides: object) -> Settings:
    """Создать настройки и при необходимости заменить отдельные значения."""

    values: dict[str, object] = {
        "telegram_bot_token": "telegram-test",
        "pinecone_api_key": "pinecone-test",
        "pinecone_index_name": "test-index",
        "pinecone_cloud": "aws",
        "pinecone_region": "us-east-1",
        "openai_api_key": "openai-test",
        "openai_base_url": "https://example.test/v1",
        "embedding_model": "test-embedding-model",
        "embedding_dimension": 3,
        "chat_model": "test-chat-model",
        "duplicate_threshold": 0.74,
        "retrieval_threshold": 0.30,
        "search_top_k": 3,
    }
    values.update(overrides)
    return Settings(**values)
