import hashlib
import tempfile
import unittest
from pathlib import Path

from data_quality import DataQualityError
from train import _load_training_snapshot


class TrainingSnapshotTests(unittest.TestCase):
    def test_hash_and_parsed_rows_come_from_same_bytes(self):
        original = (
            "timestamp,token_address,price_usd,source,url\n"
            "2026-10-10T12:00:00Z,token-a,0.25,dexscreener,"
            "https://dexscreener.com/solana/pair-a\n"
        ).encode("utf-8")
        replacement = original.replace(b"0.25", b"0.50")

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "market.csv"
            source.write_bytes(original)
            market, quality, digest = _load_training_snapshot(source)
            source.write_bytes(replacement)

        self.assertEqual(digest, hashlib.sha256(original).hexdigest())
        self.assertEqual(float(market.loc[0, "price_usd"]), 0.25)
        self.assertEqual(quality["valid_rows"], 1)

    def test_rejects_incomplete_csv_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "market.csv"
            source.write_bytes(b"timestamp,token_address,price_usd\n2026-10-10")

            with self.assertRaises(DataQualityError):
                _load_training_snapshot(source)


if __name__ == "__main__":
    unittest.main()
