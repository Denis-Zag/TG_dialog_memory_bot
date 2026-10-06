"""Офлайн-проверки формирования запроса к текстовой модели."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from api_client import ChatClient
from pinecone_manager import MemoryMatch
from tests.helpers import make_settings


class FakeCompletions:
    """Возвращает заданный текст и сохраняет отправленные параметры."""

    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """Содержит тестовый ресурс chat.completions."""

    def __init__(self, content: str | None) -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class ChatClientTests(unittest.TestCase):
    """Проверяет контекст памяти и обработку ответа модели."""

    def test_reply_includes_memories_and_current_message(self) -> None:
        fake = FakeOpenAI("  Готовый ответ  ")
        client = ChatClient(make_settings(), client=fake)

        answer = client.reply(
            user_message="  Что мне нравится?  ",
            memories=[MemoryMatch("m1", "Я люблю пиццу", 0.8)],
        )

        self.assertEqual(answer, "Готовый ответ")
        request = fake.completions.calls[0]
        self.assertEqual(request["model"], "test-chat-model")
        user_content = request["messages"][1]["content"]
        self.assertIn("Я люблю пиццу", user_content)
        self.assertIn("Что мне нравится?", user_content)

    def test_reply_marks_absence_of_memories(self) -> None:
        fake = FakeOpenAI("Ответ")
        client = ChatClient(make_settings(), client=fake)

        client.reply(user_message="Привет", memories=[])

        user_content = fake.completions.calls[0]["messages"][1]["content"]
        self.assertIn("Подходящих воспоминаний пока нет", user_content)

    def test_reply_rejects_empty_user_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "пустым"):
            ChatClient(make_settings(), client=FakeOpenAI("Ответ")).reply(
                user_message="  ",
                memories=[],
            )

    def test_reply_rejects_empty_model_response(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "пустой ответ"):
            ChatClient(make_settings(), client=FakeOpenAI("  ")).reply(
                user_message="Привет",
                memories=[],
            )


if __name__ == "__main__":
    unittest.main()

