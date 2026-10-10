import unittest

import pandas as pd

from backtest import simulate_portfolio


class SimulatePortfolioTests(unittest.TestCase):
    def test_open_position_is_marked_when_another_entry_arrives(self):
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        market = pd.DataFrame([
            {"timestamp": start, "token_address": "A", "price_usd": 1.0},
            {"timestamp": start + pd.Timedelta(minutes=1), "token_address": "B", "price_usd": 1.0},
            {"timestamp": start + pd.Timedelta(minutes=2), "token_address": "A", "price_usd": 1.0},
            {"timestamp": start + pd.Timedelta(minutes=3), "token_address": "B", "price_usd": 1.0},
        ])
        entries = market.iloc[:2].copy()

        trades = simulate_portfolio(
            entries,
            market,
            horizon_minutes=1,
            max_hold_minutes=2,
            side_cost=0.0,
        )

        self.assertEqual(len(trades), 2)
        self.assertEqual([trade["token"] for trade in trades], ["A", "B"])
        self.assertTrue(all(trade["entry_price"] == 1.0 for trade in trades))


if __name__ == "__main__":
    unittest.main()
