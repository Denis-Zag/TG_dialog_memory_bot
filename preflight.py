"""Безопасная проверка внешних сервисов без запуска Telegram polling."""

from __future__ import annotations

from api_client import ChatClient
from config import ConfigurationError, load_settings
from logging_config import configure_logging
from pinecone_manager import PineconeManager


def main() -> None:
    """Проверить embeddings, Pinecone и генерацию короткого ответа."""

    configure_logging()
    try:
        settings = load_settings(require_telegram_token=False)
        manager = PineconeManager(settings)
        vector = manager.create_embedding("Проверка подключения")
        manager.ensure_index()
        answer = ChatClient(settings).reply(
            user_message="Ответь одним словом: работает?",
            memories=[],
        )
    except ConfigurationError as error:
        raise SystemExit(f"Ошибка настроек: {error}") from error

    print(f"Embedding: успешно, размерность {len(vector)}")
    print(f"Pinecone: индекс {settings.pinecone_index_name} доступен")
    print(f"Текстовая модель: успешно, ответ: {answer}")


if __name__ == "__main__":
    main()

