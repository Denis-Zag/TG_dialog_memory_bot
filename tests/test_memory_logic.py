"""Офлайн-проверки менеджера и изоляции векторной памяти."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from pinecone.errors import NotFoundError

from pinecone_manager import (
    MemoryMatch,
    PineconeManager,
    decide_memory_action,
    namespace_for_user,
)
from tests.helpers import make_settings


class FakeIndex:
    """Минимальная замена Pinecone-индекса для офлайн-проверки."""

    def __init__(
        self,
        matches: list[dict[str, Any]] | None = None,
        *,
        missing_on_delete: bool = False,
    ) -> None:
        self.matches = matches or []
        self.missing_on_delete = missing_on_delete
        self.queries: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        return {"matches": self.matches}

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def delete(self, **kwargs: Any) -> None:
        self.deletes.append(kwargs)
        if self.missing_on_delete:
            raise NotFoundError("Namespace not found")


class FakePinecone:
    """Имитирует управляющий клиент и один Pinecone-индекс."""

    def __init__(
        self,
        index: FakeIndex,
        *,
        index_exists: bool = True,
        dimension: int = 3,
        metric: str = "cosine",
    ) -> None:
        self.index = index
        self.index_exists = index_exists
        self.dimension = dimension
        self.metric = metric
        self.created: list[dict[str, Any]] = []

    def list_indexes(self) -> SimpleNamespace:
        names = ["test-index"] if self.index_exists else []
        return SimpleNamespace(names=lambda: names)

    def describe_index(self, _: str) -> SimpleNamespace:
        return SimpleNamespace(dimension=self.dimension, metric=self.metric)

    def create_index(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def Index(self, _: str) -> FakeIndex:
        return self.index


class FakeEmbeddings:
    """Возвращает заданный embedding и запоминает параметры запроса."""

    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=self.embedding)]
        )


class FakeOpenAI:
    """Содержит тестовый ресурс embeddings."""

    def __init__(self, embedding: list[float]) -> None:
        self.embeddings = FakeEmbeddings(embedding)


def make_manager(
    index: FakeIndex | None = None,
    *,
    pinecone: FakePinecone | None = None,
    openai: FakeOpenAI | None = None,
) -> PineconeManager:
    """Создать менеджер со всеми внешними зависимостями-заглушками."""

    fake_index = index or FakeIndex()
    return PineconeManager(
        make_settings(),
        pinecone_client=pinecone or FakePinecone(fake_index),
        openai_client=openai or FakeOpenAI([0.1, 0.2, 0.3]),
    )


class MemoryDecisionTests(unittest.TestCase):
    """Проверяет границу определения смыслового дубля."""

    def test_inserts_when_memory_is_empty(self) -> None:
        self.assertEqual(decide_memory_action(None, 0.80), "inserted")

    def test_skips_score_equal_to_threshold(self) -> None:
        self.assertEqual(decide_memory_action(0.80, 0.80), "skipped")

    def test_skips_score_above_threshold(self) -> None:
        self.assertEqual(decide_memory_action(0.91, 0.80), "skipped")

    def test_inserts_score_below_threshold(self) -> None:
        self.assertEqual(decide_memory_action(0.7999, 0.80), "inserted")

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "диапазоне"):
            decide_memory_action(0.5, 1.5)


class NamespaceTests(unittest.TestCase):
    """Проверяет стабильность и разделение памяти пользователей."""

    def test_user_namespaces_are_stable_and_isolated(self) -> None:
        first = namespace_for_user(100)
        self.assertEqual(first, namespace_for_user("100"))
        self.assertNotEqual(first, namespace_for_user(101))
        self.assertNotIn("100", first)

    def test_rejects_empty_user_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_id"):
            namespace_for_user("   ")


class IndexTests(unittest.TestCase):
    """Проверяет создание и защиту конфигурации Pinecone-индекса."""

    def test_uses_existing_cosine_index(self) -> None:
        index = FakeIndex()
        pinecone = FakePinecone(index)
        manager = make_manager(index, pinecone=pinecone)

        self.assertIs(manager.ensure_index(), index)
        self.assertEqual(pinecone.created, [])

    def test_creates_missing_index_with_required_parameters(self) -> None:
        index = FakeIndex()
        pinecone = FakePinecone(index, index_exists=False)
        manager = make_manager(index, pinecone=pinecone)

        manager.ensure_index()

        self.assertEqual(len(pinecone.created), 1)
        creation = pinecone.created[0]
        self.assertEqual(creation["name"], "test-index")
        self.assertEqual(creation["dimension"], 3)
        self.assertEqual(creation["metric"], "cosine")

    def test_rejects_wrong_index_dimension(self) -> None:
        pinecone = FakePinecone(FakeIndex(), dimension=4)
        manager = make_manager(pinecone=pinecone)

        with self.assertRaisesRegex(ValueError, "Размерность"):
            manager.ensure_index()

    def test_rejects_wrong_index_metric(self) -> None:
        pinecone = FakePinecone(FakeIndex(), metric="dotproduct")
        manager = make_manager(pinecone=pinecone)

        with self.assertRaisesRegex(ValueError, "cosine"):
            manager.ensure_index()


class EmbeddingAndQueryTests(unittest.TestCase):
    """Проверяет создание векторов и параметры поиска."""

    def test_create_embedding_uses_model_and_stripped_text(self) -> None:
        openai = FakeOpenAI([0.1, 0.2, 0.3])
        manager = make_manager(openai=openai)

        vector = manager.create_embedding("  Проверка  ")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertEqual(
            openai.embeddings.calls,
            [{"model": "test-embedding-model", "input": "Проверка"}],
        )

    def test_create_embedding_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "пустого"):
            make_manager().create_embedding("  ")

    def test_create_embedding_rejects_wrong_dimension(self) -> None:
        manager = make_manager(openai=FakeOpenAI([0.1, 0.2]))

        with self.assertRaisesRegex(ValueError, "размерности"):
            manager.create_embedding("Проверка")

    def test_query_uses_user_namespace_and_metadata(self) -> None:
        index = FakeIndex(
            [{"id": "m1", "score": 0.7, "metadata": {"text": "Факт"}}]
        )
        manager = make_manager(index)

        matches = manager.query_by_vector(
            user_id=100,
            vector=[0.1, 0.2, 0.3],
            top_k=2,
        )

        self.assertEqual(matches, [MemoryMatch("m1", "Факт", 0.7)])
        self.assertEqual(index.queries[0]["top_k"], 2)
        self.assertTrue(index.queries[0]["include_metadata"])
        self.assertEqual(
            index.queries[0]["namespace"],
            namespace_for_user(100),
        )

    def test_query_ignores_record_without_text(self) -> None:
        index = FakeIndex(
            [{"id": "m1", "score": 0.7, "metadata": {"source": "telegram"}}]
        )

        matches = make_manager(index).query_by_vector(
            user_id=100,
            vector=[0.1, 0.2, 0.3],
            top_k=1,
        )

        self.assertEqual(matches, [])

    def test_query_rejects_non_positive_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            make_manager().query_by_vector(
                user_id=100,
                vector=[0.1, 0.2, 0.3],
                top_k=0,
            )

    def test_relevant_memory_filter_removes_weak_matches(self) -> None:
        index = FakeIndex(
            [
                {"id": "strong", "score": 0.7, "metadata": {"text": "Да"}},
                {"id": "weak", "score": 0.2, "metadata": {"text": "Нет"}},
            ]
        )

        matches = make_manager(index).get_relevant_memories_by_vector(
            user_id=100,
            vector=[0.1, 0.2, 0.3],
        )

        self.assertEqual([match.record_id for match in matches], ["strong"])


class SaveAndDeleteTests(unittest.TestCase):
    """Проверяет запись, пропуск дублей, метаданные и очистку."""

    def test_save_uses_provided_vector_without_embedding_request(self) -> None:
        index = FakeIndex()
        openai = FakeOpenAI([9.0, 9.0, 9.0])
        manager = make_manager(index, openai=openai)

        result = manager.save_memory(
            user_id=100,
            text="Новая мысль",
            vector=[0.1, 0.2, 0.3],
        )

        self.assertEqual(result.action, "inserted")
        self.assertEqual(openai.embeddings.calls, [])
        self.assertEqual(len(index.upserts), 1)

    def test_saved_metadata_contains_only_expected_fields(self) -> None:
        index = FakeIndex()
        manager = make_manager(index)

        manager.save_memory(
            user_id=100,
            text="  Только пользовательский текст  ",
            vector=[0.1, 0.2, 0.3],
        )

        metadata = index.upserts[0]["vectors"][0]["metadata"]
        self.assertEqual(metadata["text"], "Только пользовательский текст")
        self.assertEqual(set(metadata), {"text", "source", "created_at"})
        self.assertNotIn("bot_response", metadata)
        self.assertNotIn("first_name", metadata)
        self.assertNotIn("last_name", metadata)

    def test_save_skips_close_duplicate_without_upsert(self) -> None:
        index = FakeIndex(
            [
                {
                    "id": "memory-old",
                    "score": 0.91,
                    "metadata": {"text": "Похожая мысль"},
                }
            ]
        )
        manager = make_manager(index)

        result = manager.save_memory(
            user_id=100,
            text="Переформулированная мысль",
            vector=[0.1, 0.2, 0.3],
        )

        self.assertEqual(result.action, "skipped")
        self.assertEqual(result.existing_id, "memory-old")
        self.assertEqual(index.upserts, [])

    def test_save_rejects_empty_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "пустое"):
            make_manager().save_memory(user_id=100, text="  ")

    def test_save_rejects_provided_vector_with_wrong_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "Размерность"):
            make_manager().save_memory(
                user_id=100,
                text="Факт",
                vector=[0.1, 0.2],
            )

    def test_clear_uses_only_selected_user_namespace(self) -> None:
        index = FakeIndex()
        make_manager(index).clear_user_memory(user_id=100)

        self.assertEqual(
            index.deletes,
            [
                {
                    "delete_all": True,
                    "namespace": namespace_for_user(100),
                }
            ],
        )

    def test_clear_ignores_missing_namespace(self) -> None:
        manager = make_manager(FakeIndex(missing_on_delete=True))
        manager.clear_user_memory(user_id="new-user")


if __name__ == "__main__":
    unittest.main()
