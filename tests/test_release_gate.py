import copy
import json
import tempfile
import unittest
from pathlib import Path

import release_evidence
import release_gate
import release_gate_report


def passing_evidence():
    """Evidence that satisfies every requirement in MODEL_RELEASE_GATE.md."""
    return {
        "generated_at": "2026-10-10T12:00:00+00:00",
        "model_sha256": "a" * 64,
        "dataset": {"span_days": 9.5, "span_source": "labeled_rows", "usable_rows": 5000},
        "training": {
            "trained_at": "2026-10-10T12:00:00+00:00",
            "rows": 5000,
            "test_rows": 800,
            "calibration_rows": 300,
            "class_counts": {
                "fit": {"positive": 400, "negative": 1600},
                "calibration": {"positive": 60, "negative": 240},
                "test": {"positive": 120, "negative": 680},
            },
            "test_auc": 0.66,
            "test_brier": 0.14,
            "baseline_brier": 0.16,
            "calibration_method": "temporal_platt_logistic",
            "training_data_sha256": "b" * 64,
            "walk_forward": [
                {"fold": 1, "test_rows": 150, "test_positives": 40, "test_negatives": 110,
                 "auc": 0.61, "brier": 0.10, "baseline_brier": 0.12},
                {"fold": 2, "test_rows": 180, "test_positives": 55, "test_negatives": 125,
                 "auc": 0.63, "brier": 0.11, "baseline_brier": 0.13},
                {"fold": 3, "test_rows": 200, "test_positives": 70, "test_negatives": 130,
                 "auc": 0.65, "brier": 0.12, "baseline_brier": 0.14},
            ],
        },
        "backtest": {
            "closed_trades": 45,
            "mean_net_return": 0.012,
            "final_balance": 1042.0,
            "return_pct": 0.042,
            "max_drawdown": -0.08,
            "risk": {"worst_day_loss_pct": -0.02, "max_open_positions": 3, "daily_halt_events": 0},
            "risk_limits": {"max_daily_loss": 0.05, "max_positions": 3},
        },
    }


class NumberHelperTests(unittest.TestCase):
    def test_rejects_booleans_and_non_numbers(self):
        self.assertIsNone(release_gate.number(True))
        self.assertIsNone(release_gate.number("0.5"))
        self.assertIsNone(release_gate.number(float("nan")))
        self.assertIsNone(release_gate.number(float("inf")))

    def test_keeps_real_numbers(self):
        self.assertEqual(release_gate.number(0), 0)
        self.assertEqual(release_gate.number(0.25), 0.25)


class ClassPairTests(unittest.TestCase):
    def test_reads_a_complete_count_pair(self):
        self.assertEqual(release_gate.class_pair({"positive": 3, "negative": 7}), (3, 7))

    def test_reports_incomplete_or_missing_counts(self):
        self.assertIsNone(release_gate.class_pair(None))
        self.assertIsNone(release_gate.class_pair({"positive": 3}))
        self.assertIsNone(release_gate.class_pair({"positive": 3.0, "negative": 7.0}))


class GateEvaluationTests(unittest.TestCase):
    def test_fully_supported_evidence_promotes_the_release(self):
        report = release_gate.evaluate(passing_evidence())

        self.assertTrue(report["eligible_for_2_0"])
        self.assertEqual(report["release_name"], "Acorn 2.0")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["summary"]["total"], len(report["checks"]))

    def test_every_check_carries_a_requirement_and_detail(self):
        for check in release_gate.evaluate(passing_evidence())["checks"]:
            self.assertTrue(check["requirement"])
            self.assertTrue(check["detail"])

    def test_missing_evidence_holds_the_release(self):
        report = release_gate.evaluate({})

        self.assertFalse(report["eligible_for_2_0"])
        self.assertEqual(report["release_name"], "Acorn 1.0")
        self.assertEqual(report["summary"]["passed"], 0)

    def test_thin_test_block_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["training"]["test_rows"] = 84
        evidence["training"]["class_counts"]["test"] = {"positive": 13, "negative": 71}

        failing = release_gate.evaluate(evidence)["summary"]["failing"]

        self.assertIn("test_block_rows", failing)
        self.assertIn("test_block_classes", failing)

    def test_brier_worse_than_baseline_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["training"]["test_brier"] = 0.1391
        evidence["training"]["baseline_brier"] = 0.1338

        self.assertIn("test_brier", release_gate.evaluate(evidence)["summary"]["failing"])

    def test_brier_equal_to_baseline_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["training"]["baseline_brier"] = evidence["training"]["test_brier"]

        self.assertIn("test_brier", release_gate.evaluate(evidence)["summary"]["failing"])

    def test_walk_forward_folds_without_class_counts_do_not_qualify(self):
        evidence = passing_evidence()
        evidence["training"]["walk_forward"] = [
            {"fold": 1, "test_rows": 150, "auc": 0.9, "brier": 0.01, "baseline_brier": 0.5},
        ]

        report = release_gate.evaluate(evidence)

        self.assertEqual(report["metrics"]["folds_qualifying"], 0)
        self.assertIn("walk_forward_folds", report["summary"]["failing"])

    def test_losing_fold_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["training"]["walk_forward"][1]["brier"] = 0.20

        self.assertIn("walk_forward_brier", release_gate.evaluate(evidence)["summary"]["failing"])

    def test_short_dataset_span_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["dataset"] = {"span_days": 0.72, "span_source": "source_rows"}

        self.assertIn("dataset_span", release_gate.evaluate(evidence)["summary"]["failing"])

    def test_uncalibrated_artifact_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["training"]["calibration_method"] = "uncalibrated"

        self.assertIn("calibration_fitted", release_gate.evaluate(evidence)["summary"]["failing"])

    def test_daily_loss_beyond_tolerance_blocks_promotion(self):
        evidence = passing_evidence()
        evidence["backtest"]["risk"]["worst_day_loss_pct"] = -0.40

        self.assertIn("backtest_risk_limits", release_gate.evaluate(evidence)["summary"]["failing"])

    def test_metrics_snapshot_mirrors_the_evidence(self):
        metrics = release_gate.evaluate(passing_evidence())["metrics"]

        self.assertEqual(metrics["closed_trades"], 45)
        self.assertEqual(metrics["test_positives"], 120)
        self.assertEqual(metrics["median_fold_auc"], 0.63)
        self.assertEqual(metrics["dataset_span_days"], 9.5)

    def test_original_evidence_is_not_mutated(self):
        evidence = passing_evidence()
        before = copy.deepcopy(evidence)

        release_gate.evaluate(evidence)

        self.assertEqual(evidence, before)


class EvidenceReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)

    def write_history(self, lines):
        path = self.root / "model_history.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_last_record_skips_a_truncated_final_line(self):
        path = self.write_history([json.dumps({"rows": 1}), '{"rows": 2'])

        self.assertEqual(release_evidence.last_training_record(path), {"rows": 1})

    def test_last_record_returns_the_newest_complete_entry(self):
        path = self.write_history([json.dumps({"rows": 1}), json.dumps({"rows": 2})])

        self.assertEqual(release_evidence.last_training_record(path), {"rows": 2})

    def test_missing_history_file_is_not_an_error(self):
        self.assertIsNone(release_evidence.last_training_record(self.root / "absent.jsonl"))

    def test_dataset_span_prefers_labelled_rows(self):
        span = release_evidence.dataset_span({
            "first_labeled": "2026-10-01T00:00:00+00:00",
            "last_labeled": "2026-10-03T12:00:00+00:00",
            "data_quality": {"first_timestamp": "2026-09-01T00:00:00+00:00",
                             "last_timestamp": "2026-10-09T00:00:00+00:00"},
        })

        self.assertEqual(span["span_source"], "labeled_rows")
        self.assertAlmostEqual(span["span_days"], 2.5)

    def test_dataset_span_falls_back_to_source_rows(self):
        span = release_evidence.dataset_span({
            "data_quality": {"first_timestamp": "2026-10-01T00:00:00Z",
                             "last_timestamp": "2026-10-02T00:00:00Z"},
        })

        self.assertEqual(span["span_source"], "source_rows")
        self.assertAlmostEqual(span["span_days"], 1.0)

    def test_dataset_span_is_unavailable_without_timestamps(self):
        span = release_evidence.dataset_span({})

        self.assertEqual(span["span_source"], "unavailable")
        self.assertIsNone(span["span_days"])

    def test_collect_reports_missing_artifacts_instead_of_raising(self):
        payload = release_evidence.collect(
            history_path=self.root / "absent.jsonl",
            backtest_path=self.root / "absent.json",
            manifest_path=self.root / "absent_meta.json")

        self.assertIsNone(payload["training"])
        self.assertEqual(payload["artifacts"],
                         {"retrain_log": False, "backtest_summary": False, "model_manifest": False})


class ReportWriterTests(unittest.TestCase):
    def test_write_report_replaces_the_file_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "release_gate.json"
            path.write_text("{}", encoding="utf-8")

            release_gate_report.write_report({"release_name": "Acorn 1.0"}, path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"release_name": "Acorn 1.0"})
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_report_uses_supplied_evidence_without_touching_disk(self):
        payload = release_gate_report.report(passing_evidence())

        self.assertTrue(payload["eligible_for_2_0"])


if __name__ == "__main__":
    unittest.main()
