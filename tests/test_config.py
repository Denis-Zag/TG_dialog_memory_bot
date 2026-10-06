"""Проверки настроек приложения."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import ConfigurationError, load_settings


class SettingsTests(unittest.TestCase):
    """Проверяет обязательные значения и числовые ограничения."""

    def test_loads_settings_without_telegram_for_preflight(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
            "MEMORY_DUPLICATE_THRESHOLD": "0.88",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            settings = load_settings(require_telegram_token=False)

        self.assertEqual(settings.duplicate_threshold, 0.88)
        self.assertEqual(settings.embedding_dimension, 1536)
        self.assertEqual(settings.openai_base_url, "https://api.proxyapi.ru/v1")

    def test_uses_calibrated_duplicate_threshold_by_default(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            settings = load_settings(require_telegram_token=False)

        self.assertEqual(settings.duplicate_threshold, 0.74)

    def test_requires_telegram_token_for_bot(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "TELEGRAM_BOT_TOKEN"):
                load_settings()

    def test_rejects_invalid_threshold(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
            "MEMORY_DUPLICATE_THRESHOLD": "1.2",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "диапазоне"):
                load_settings(require_telegram_token=False)

    def test_requires_pinecone_key(self) -> None:
        values = {"OPENAI_API_KEY": "openai-test"}
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "PINECONE_API_KEY"):
                load_settings(require_telegram_token=False)

    def test_requires_openai_key(self) -> None:
        values = {"PINECONE_API_KEY": "pinecone-test"}
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                load_settings(require_telegram_token=False)

    def test_rejects_non_numeric_embedding_dimension(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
            "EMBEDDING_DIMENSION": "many",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "целым числом"):
                load_settings(require_telegram_token=False)

    def test_rejects_zero_search_top_k(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
            "MEMORY_SEARCH_TOP_K": "0",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "больше нуля"):
                load_settings(require_telegram_token=False)

    def test_rejects_invalid_retrieval_threshold(self) -> None:
        values = {
            "PINECONE_API_KEY": "pinecone-test",
            "OPENAI_API_KEY": "openai-test",
            "MEMORY_RETRIEVAL_THRESHOLD": "-0.1",
        }
        with patch("config.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "диапазоне"):
                load_settings(require_telegram_token=False)


if __name__ == "__main__":
    unittest.main()
