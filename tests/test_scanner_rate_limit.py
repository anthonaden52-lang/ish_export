import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scanner


class RateStateReplaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.target = self.root / "rate.json"
        self.tmp = self.root / "rate.tmp"
        self.tmp.write_text('{"discovery": [1]}', encoding="utf-8")
        self.addCleanup(self.temp_dir.cleanup)

    def test_retries_transient_permission_error_then_replaces(self):
        original_replace = Path.replace
        calls = 0

        def replace_with_transient_lock(path, target):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise PermissionError("temporary Windows file lock")
            return original_replace(path, target)

        with patch.object(Path, "replace", replace_with_transient_lock), \
                patch.object(scanner.time, "sleep") as sleep:
            scanner._replace_rate_state(self.tmp, self.target)

        self.assertEqual(calls, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(self.target.read_text(encoding="utf-8"), '{"discovery": [1]}')
        self.assertFalse(self.tmp.exists())

    def test_persistent_permission_error_fails_closed(self):
        self.target.write_text('{"pairs": [2]}', encoding="utf-8")
        with patch.object(Path, "replace", side_effect=PermissionError("persistent lock")), \
                patch.object(scanner.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                scanner._replace_rate_state(self.tmp, self.target)

        self.assertEqual(sleep.call_count, scanner.RATE_REPLACE_ATTEMPTS - 1)
        self.assertTrue(self.tmp.exists())
        self.assertEqual(self.target.read_text(encoding="utf-8"), '{"pairs": [2]}')


if __name__ == "__main__":
    unittest.main()
