import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pages_export


FIELDS = [
    "timestamp", "token_address", "symbol", "price_usd", "liquidity_usd",
    "volume_24h", "price_change_5m", "price_change_1h", "source", "url",
]


class PagesExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.market_file = Path(self.temp_dir.name) / "market.csv"
        self.market_patch = patch.object(pages_export, "MARKET_FILE", self.market_file)
        self.market_patch.start()
        self.addCleanup(self.market_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def write_rows(self, rows):
        with self.market_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, now, token, age_seconds=30, **overrides):
        row = {
            "timestamp": (now - timedelta(seconds=age_seconds)).isoformat(),
            "token_address": token,
            "symbol": token,
            "price_usd": "1.25",
            "liquidity_usd": "25000",
            "volume_24h": "60000",
            "price_change_5m": "1.5",
            "price_change_1h": "2.5",
            "source": "dexscreener",
            "url": f"https://dexscreener.com/solana/{token.lower()}",
        }
        row.update(overrides)
        return row

    def test_exports_only_fresh_rows_from_dexscreener_urls(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.write_rows([
            self.row(now, "FreshToken"),
            self.row(now, "OldToken", age_seconds=3600),
            self.row(now, "WrongHost", url="https://example.com/pair"),
            self.row(now, "WrongSource", source="other"),
        ])

        result = pages_export.build_export(now)

        self.assertEqual(result["market_status"], "fresh")
        self.assertEqual(result["known_pairs"], 2)
        self.assertEqual([pair["token"] for pair in result["pairs"]], ["FreshToken"])

    def test_stale_market_data_is_not_exported_as_current_prices(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.write_rows([self.row(now, "OldToken", age_seconds=3600)])

        result = pages_export.build_export(now)

        self.assertEqual(result["market_status"], "stale")
        self.assertEqual(result["pairs"], [])

    def test_missing_market_file_returns_unavailable_without_rows(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        result = pages_export.build_export(now)

        self.assertEqual(result["market_status"], "unavailable")
        self.assertEqual(result["pairs"], [])


if __name__ == "__main__":
    unittest.main()
