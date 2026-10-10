import unittest

import pandas as pd

from dashboard import count_labelable_rows


class DashboardLabelCountTests(unittest.TestCase):
    def make_market(self, rows):
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        return pd.DataFrame({
            "timestamp": [start + pd.Timedelta(minutes=i) for i in range(rows)],
            "token_address": ["TOKEN"] * rows,
            "price_usd": [1.0 + i * 0.01 for i in range(rows)],
            "pair_age_minutes": [float(i) for i in range(rows)],
        })

    def test_counts_only_feature_ready_rows_with_observed_future_prices(self):
        market = self.make_market(41)

        count = count_labelable_rows(market, horizon_minutes=15, target_return=0.10)

        self.assertEqual(count, 11)

    def test_old_rows_without_horizon_observations_are_not_labelable(self):
        market = self.make_market(21)

        count = count_labelable_rows(market, horizon_minutes=15, target_return=0.10)

        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
