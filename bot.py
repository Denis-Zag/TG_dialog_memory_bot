"""Telegram-интерфейс диалогового бота с долговременной памятью."""

from __future__ import annotations

import logging
from typing import Any

from telebot import TeleBot
from telebot.types import Message

from api_client import ChatClient
from config import ConfigurationError, load_settings
from logging_config import configure_logging
from pinecone_manager import MemorySaveResult, PineconeManager

logger = logging.getLogger(__name__)


def get_display_name(telegram_user: Any | None) -> str:
    """Выбрать доступное имя без предположения о заполненных полях Telegram."""

    if telegram_user is None:
        return "пользователь"
    first_name = str(getattr(telegram_user, "first_name", "") or "").strip()
    username = str(getattr(telegram_user, "username", "") or "").strip()
    return first_name or username or "пользователь"


def process_user_message(
    *,
    user_id: int,
    user_text: str,
    memory: PineconeManager,
    chat: ChatClient,
) -> tuple[str, MemorySaveResult]:
    """Выполнить одну цепочку ответа и сохранения пользовательского текста."""

    normalized = user_text.strip()
    if not normalized:
        raise ValueError("Сообщение пользователя не должно быть пустым")

    vector = memory.create_embedding(normalized)
    memories = memory.get_relevant_memories_by_vector(
        user_id=user_id,
        vector=vector,
    )
    answer = chat.reply(user_message=normalized, memories=memories)
    save_result = memory.save_memory(
        user_id=user_id,
        text=normalized,
        vector=vector,
    )
    return answer, save_result


def create_bot() -> TeleBot:
    """Создать Telegram-бота и зарегистрировать обработчики."""

    settings = load_settings()
    memory = PineconeManager(settings)
    chat = ChatClient(settings)
    bot = TeleBot(settings.telegram_bot_token)

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: Message) -> None:
        display_name = get_display_name(message.from_user)
        bot.reply_to(
            message,
            (
                f"Привет, {display_name}! Я диалоговый бот с долговременной "
                "памятью. Напиши сообщение, и я смогу учитывать его в будущих беседах."
            ),
        )

    @bot.message_handler(content_types=["text"])
    def handle_text(message: Message) -> None:
        if message.from_user is None or not message.text:
            return

        user_id = message.from_user.id
        user_text = message.text.strip()
        if not user_text:
            bot.reply_to(message, "Напиши непустое текстовое сообщение.")
            return

        try:
            answer, save_result = process_user_message(
                user_id=user_id,
                user_text=user_text,
                memory=memory,
                chat=chat,
            )
            logger.info(
                "Сообщение обработано: user_namespace=%s action=%s score=%s",
                "isolated",
                save_result.action,
                (
                    f"{save_result.similarity_score:.4f}"
                    if save_result.similarity_score is not None
                    else "none"
                ),
            )
            bot.reply_to(message, answer)
        except Exception:
            logger.exception("Не удалось обработать сообщение пользователя")
            bot.reply_to(
                message,
                "Не получилось обработать сообщение. Попробуй немного позже.",
            )

    return bot


def main() -> None:
    """Запустить Telegram polling."""

    configure_logging()
    try:
        bot = create_bot()
    except ConfigurationError as error:
        logger.error("Ошибка настроек: %s", error)
        raise SystemExit(1) from error

    logger.info("Telegram-бот запущен")
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)


if __name__ == "__main__":
    main()
