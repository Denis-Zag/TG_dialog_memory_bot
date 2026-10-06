"""Менеджер долговременной векторной памяти в Pinecone."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors import NotFoundError

from config import Settings, load_settings
from logging_config import configure_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """Одна найденная запись памяти."""

    record_id: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class MemorySaveResult:
    """Результат попытки сохранить сообщение в память."""

    action: str
    similarity_score: float | None
    record_id: str | None
    existing_id: str | None


def decide_memory_action(
    similarity_score: float | None,
    duplicate_threshold: float,
) -> str:
    """Решить, сохранить новую запись или пропустить смысловой дубль."""

    if not 0 <= duplicate_threshold <= 1:
        raise ValueError("Порог сходства должен находиться в диапазоне от 0 до 1")
    if similarity_score is None:
        return "inserted"
    return "skipped" if similarity_score >= duplicate_threshold else "inserted"


def namespace_for_user(user_id: int | str) -> str:
    """Получить стабильное пространство имён без открытого Telegram ID."""

    normalized = str(user_id).strip()
    if not normalized:
        raise ValueError("user_id не должен быть пустым")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"user-{digest}"


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Прочитать поле как из SDK-объекта, так и из обычного словаря."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class PineconeManager:
    """Создаёт embeddings, сохраняет сообщения и ищет релевантную память."""

    def __init__(
        self,
        settings: Settings,
        *,
        pinecone_client: Any | None = None,
        openai_client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._pinecone = pinecone_client or Pinecone(
            api_key=settings.pinecone_api_key
        )
        self._openai = openai_client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=30.0,
        )
        self._index: Any | None = None

    def ensure_index(self) -> Any:
        """Создать индекс при отсутствии и проверить его размерность."""

        if self._index is not None:
            return self._index

        index_name = self._settings.pinecone_index_name
        existing_names = set(self._pinecone.list_indexes().names())
        if index_name not in existing_names:
            logger.info("Создание Pinecone-индекса %s", index_name)
            self._pinecone.create_index(
                name=index_name,
                dimension=self._settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
                timeout=120,
            )
        else:
            description = self._pinecone.describe_index(index_name)
            dimension = int(_field(description, "dimension", 0))
            metric = str(_field(description, "metric", ""))
            if dimension != self._settings.embedding_dimension:
                raise ValueError(
                    "Размерность Pinecone-индекса не совпадает с EMBEDDING_DIMENSION: "
                    f"{dimension} != {self._settings.embedding_dimension}"
                )
            if metric.lower() != "cosine":
                raise ValueError(
                    "Для проекта требуется Pinecone-индекс с метрикой cosine"
                )

        self._index = self._pinecone.Index(index_name)
        return self._index

    def create_embedding(self, text: str) -> list[float]:
        """Преобразовать непустой текст в embedding через ProxyAPI."""

        normalized = text.strip()
        if not normalized:
            raise ValueError("Нельзя создать embedding для пустого текста")
        response = self._openai.embeddings.create(
            model=self._settings.embedding_model,
            input=normalized,
        )
        embedding = list(response.data[0].embedding)
        if len(embedding) != self._settings.embedding_dimension:
            raise ValueError(
                "Модель вернула embedding неожиданной размерности: "
                f"{len(embedding)} != {self._settings.embedding_dimension}"
            )
        return embedding

    def query_by_vector(
        self,
        *,
        user_id: int | str,
        vector: list[float],
        top_k: int,
    ) -> list[MemoryMatch]:
        """Найти ближайшие записи в памяти конкретного пользователя."""

        if top_k <= 0:
            raise ValueError("top_k должен быть больше нуля")
        response = self.ensure_index().query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace_for_user(user_id),
        )
        matches: list[MemoryMatch] = []
        for item in _field(response, "matches", []) or []:
            metadata = _field(item, "metadata", {}) or {}
            text = str(_field(metadata, "text", "")).strip()
            if not text:
                continue
            matches.append(
                MemoryMatch(
                    record_id=str(_field(item, "id", "")),
                    text=text,
                    score=float(_field(item, "score", 0.0)),
                )
            )
        return matches

    def query_by_text(
        self,
        *,
        user_id: int | str,
        text: str,
        top_k: int,
    ) -> list[MemoryMatch]:
        """Создать embedding запроса и найти ближайшие записи памяти."""

        return self.query_by_vector(
            user_id=user_id,
            vector=self.create_embedding(text),
            top_k=top_k,
        )

    def save_memory(
        self,
        *,
        user_id: int | str,
        text: str,
        vector: list[float] | None = None,
    ) -> MemorySaveResult:
        """Сохранить новую мысль или пропустить смысловой дубль."""

        normalized = text.strip()
        if not normalized:
            raise ValueError("Нельзя сохранить пустое сообщение")

        memory_vector = vector or self.create_embedding(normalized)
        if len(memory_vector) != self._settings.embedding_dimension:
            raise ValueError("Размерность переданного embedding не совпадает с настройкой")
        nearest = self.query_by_vector(
            user_id=user_id,
            vector=memory_vector,
            top_k=1,
        )
        best_match = nearest[0] if nearest else None
        best_score = best_match.score if best_match else None
        action = decide_memory_action(
            best_score,
            self._settings.duplicate_threshold,
        )

        if action == "skipped":
            logger.info(
                "Память не изменена: action=skipped score=%.4f existing_id=%s",
                best_score,
                best_match.record_id,
            )
            return MemorySaveResult(
                action=action,
                similarity_score=best_score,
                record_id=None,
                existing_id=best_match.record_id,
            )

        record_id = f"memory-{uuid4().hex}"
        self.ensure_index().upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": memory_vector,
                    "metadata": {
                        "text": normalized,
                        "source": "telegram",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            ],
            namespace=namespace_for_user(user_id),
        )
        logger.info(
            "Память обновлена: action=inserted score=%s record_id=%s",
            f"{best_score:.4f}" if best_score is not None else "none",
            record_id,
        )
        return MemorySaveResult(
            action=action,
            similarity_score=best_score,
            record_id=record_id,
            existing_id=best_match.record_id if best_match else None,
        )

    def get_relevant_memories(
        self,
        *,
        user_id: int | str,
        text: str,
    ) -> list[MemoryMatch]:
        """Вернуть только достаточно близкие воспоминания для ответа модели."""

        matches = self.query_by_text(
            user_id=user_id,
            text=text,
            top_k=self._settings.search_top_k,
        )
        return self._filter_relevant(matches)

    def get_relevant_memories_by_vector(
        self,
        *,
        user_id: int | str,
        vector: list[float],
    ) -> list[MemoryMatch]:
        """Найти подходящую память по уже созданному embedding."""

        matches = self.query_by_vector(
            user_id=user_id,
            vector=vector,
            top_k=self._settings.search_top_k,
        )
        return self._filter_relevant(matches)

    def _filter_relevant(self, matches: list[MemoryMatch]) -> list[MemoryMatch]:
        """Оставить записи, прошедшие порог релевантности."""

        return [
            match
            for match in matches
            if match.score >= self._settings.retrieval_threshold
        ]

    def clear_user_memory(self, *, user_id: int | str) -> None:
        """Удалить записи только из пространства указанного пользователя."""

        try:
            self.ensure_index().delete(
                delete_all=True,
                namespace=namespace_for_user(user_id),
            )
        except NotFoundError:
            logger.info("Тестовая память уже пуста, очистка не требуется")


def run_manual_test() -> None:
    """Проверить PineconeManager без запуска Telegram polling."""

    configure_logging()
    settings = load_settings(require_telegram_token=False)
    manager = PineconeManager(settings)
    test_user_id = "vpg05-manual-test"
    manager.clear_user_memory(user_id=test_user_id)

    phrases = [
        "Я хочу полететь на Марс",
        "Я мечтаю отправиться в путешествие на Марс",
        "Я не хочу лететь на Марс",
        "Я люблю пиццу",
    ]
    print("Ручная проверка умной памяти")
    for phrase in phrases:
        result = manager.save_memory(user_id=test_user_id, text=phrase)
        score = (
            f"{result.similarity_score:.4f}"
            if result.similarity_score is not None
            else "нет"
        )
        print(f"action={result.action}; similarity_score={score}")


if __name__ == "__main__":
    run_manual_test()
