"""Единая настройка безопасного логирования приложения."""

from __future__ import annotations

import logging


def configure_logging() -> None:
    """Настроить вывод технических событий без текстов и секретов пользователя."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

