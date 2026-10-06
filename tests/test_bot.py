"""Офлайн-проверки Telegram-логики без запуска polling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from bot import get_display_name, process_user_message
from pinecone_manager import MemoryMatch, MemorySaveResult


class FakeMemory:
    """Имитирует память и сохраняет параметры вызовов."""

    def __init__(self) -> None:
        self.vector = [0.1, 0.2, 0.3]
        self.memories = [MemoryMatch("m1", "Я люблю пиццу", 0.8)]
        self.embedding_inputs: list[str] = []
        self.search_calls: list[dict[str, object]] = []
        self.save_calls: list[dict[str, object]] = []

    def create_embedding(self, text: str) -> list[float]:
        self.embedding_inputs.append(text)
        return self.vector

    def get_relevant_memories_by_vector(
        self,
        *,
        user_id: int,
        vector: list[float],
    ) -> list[MemoryMatch]:
        self.search_calls.append({"user_id": user_id, "vector": vector})
        return self.memories

    def save_memory(
        self,
        *,
        user_id: int,
        text: str,
        vector: list[float],
    ) -> MemorySaveResult:
        self.save_calls.append(
            {"user_id": user_id, "text": text, "vector": vector}
        )
        return MemorySaveResult("inserted", None, "m2", None)


class FakeChat:
    """Формирует ответ и запоминает полученный контекст."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reply(
        self,
        *,
        user_message: str,
        memories: list[MemoryMatch],
    ) -> str:
        self.calls.append(
            {"user_message": user_message, "memories": memories}
        )
        return "Ответ бота"


class DisplayNameTests(unittest.TestCase):
    """Проверяет отсутствие необязательных полей Telegram."""

    def test_uses_first_name_when_available(self) -> None:
        user = SimpleNamespace(first_name=" Денис ", username="denis")
        self.assertEqual(get_display_name(user), "Денис")

    def test_uses_username_when_first_name_is_missing(self) -> None:
        user = SimpleNamespace(first_name=None, username="denis")
        self.assertEqual(get_display_name(user), "denis")

    def test_uses_fallback_when_names_are_missing(self) -> None:
        user = SimpleNamespace(first_name=None, username=None)
        self.assertEqual(get_display_name(user), "пользователь")
        self.assertEqual(get_display_name(None), "пользователь")


class MessagePipelineTests(unittest.TestCase):
    """Проверяет полную цепочку сообщения без Telegram и внешних API."""

    def test_pipeline_reuses_one_embedding_and_saves_only_user_text(self) -> None:
        memory = FakeMemory()
        chat = FakeChat()

        answer, result = process_user_message(
            user_id=123,
            user_text="  Я люблю пиццу  ",
            memory=memory,
            chat=chat,
        )

        self.assertEqual(answer, "Ответ бота")
        self.assertEqual(result.action, "inserted")
        self.assertEqual(memory.embedding_inputs, ["Я люблю пиццу"])
        self.assertEqual(memory.search_calls[0]["vector"], memory.vector)
        self.assertEqual(memory.save_calls[0]["vector"], memory.vector)
        self.assertEqual(memory.save_calls[0]["text"], "Я люблю пиццу")
        self.assertNotIn("Ответ бота", str(memory.save_calls))
        self.assertEqual(chat.calls[0]["memories"], memory.memories)

    def test_pipeline_rejects_empty_message_before_external_calls(self) -> None:
        memory = FakeMemory()
        chat = FakeChat()

        with self.assertRaisesRegex(ValueError, "пустым"):
            process_user_message(
                user_id=123,
                user_text="  ",
                memory=memory,
                chat=chat,
            )

        self.assertEqual(memory.embedding_inputs, [])
        self.assertEqual(chat.calls, [])


if __name__ == "__main__":
    unittest.main()
