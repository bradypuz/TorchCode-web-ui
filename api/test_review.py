"""Review ranking and durable scheduling transitions, without real learner data."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from api.practice import PracticeStore
from api.review import INTERVALS, RECALL_TARGET, recent_mistakes, review_metrics


NOW = datetime(2026, 10, 10, 18, tzinfo=timezone.utc)
PASS = {"success": True, "passed": 4, "total": 4, "tests": []}
FAIL = {"success": False, "passed": 1, "total": 4, "tests": []}


class ReviewRankingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.store = PracticeStore(self.path / "practice.db", self.path / "absent.json")

    def tearDown(self):
        self.directory.cleanup()

    def dashboard(self, **kwargs):
        return self.store.dashboard(now=kwargs.pop("now", NOW), **kwargs)

    def state(self, task_id):
        with self.store.connection() as connection:
            return dict(connection.execute("SELECT * FROM progress WHERE user_id = 'local' AND task_id = ?",
                                           (task_id,)).fetchone())

    def review(self, task_id, **kwargs):
        return next(entry for entry in self.dashboard(**kwargs)["reviews"] if entry["id"] == task_id)

    def test_marking_today_solve_keeps_done_and_credit_in_a_full_cached_plan(self):
        self.dashboard()
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        self.assertEqual(self.dashboard()["goal"]["completed"], 1)
        response = self.store.mark_review("local", "relu", True)
        self.assertEqual(response, {"ok": True, "manual_review": True, "needs_review": True})
        dashboard = self.dashboard()
        self.assertEqual(len(dashboard["plan"]), 3)
        card = next(entry for entry in dashboard["plan"] if entry["id"] == "relu")
        self.assertEqual(card["kind"], "new")
        self.assertTrue(card["completed"])
        self.assertTrue(card["review_factors"]["manual"])
        self.assertEqual(dashboard["goal"]["completed"], 1)
        self.assertEqual(dashboard["stats"]["today_completed"], 1)
        self.assertNotIn("relu", [entry["id"] for entry in dashboard["reviews"]])
        self.assertEqual(dashboard["stats"]["solved"], 1)
        self.assertEqual(dashboard["curriculum"][0]["ready"], 0)

    def test_today_flagged_solve_returns_as_unfinished_review_tomorrow(self):
        self.dashboard()
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        self.store.mark_review("local", "relu", True)
        tomorrow = self.dashboard(now=NOW + timedelta(days=1))
        review = next(entry for entry in tomorrow["plan"] if entry["id"] == "relu")
        self.assertEqual(review["kind"], "review")
        self.assertFalse(review["completed"])
        self.assertTrue(review["manual_review"])
        self.assertEqual(tomorrow["goal"]["completed"], 0)
        self.assertEqual(tomorrow["stats"]["today_completed"], 0)

    def test_toggling_confidence_flag_does_not_change_earned_daily_credit(self):
        self.dashboard()
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        for flagged in (True, False, True):
            self.store.mark_review("local", "relu", flagged)
            dashboard = self.dashboard()
            card = next(entry for entry in dashboard["plan"] if entry["id"] == "relu")
            self.assertTrue(card["completed"])
            self.assertEqual(card["manual_review"], flagged)
            self.assertEqual(dashboard["goal"], {"target": 3, "completed": 1})
            self.assertEqual(dashboard["stats"]["today_completed"], 1)

    def test_confidence_flag_on_unsolved_lesson_does_not_earn_done(self):
        self.store.mark_review("local", "embedding", True)
        dashboard = self.dashboard()
        card = next(entry for entry in dashboard["plan"] if entry["id"] == "embedding")
        self.assertFalse(card["completed"])
        self.assertEqual(dashboard["goal"]["completed"], 0)
        self.assertEqual(dashboard["stats"]["today_completed"], 0)

    def test_scheduled_review_must_be_solved_on_the_new_day_to_earn_done(self):
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=3))
        before = self.dashboard()
        card = next(entry for entry in before["plan"] if entry["id"] == "relu")
        self.assertEqual(card["kind"], "review")
        self.assertFalse(card["completed"])
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        self.store.mark_review("local", "relu", True)
        after = self.dashboard()
        card = next(entry for entry in after["plan"] if entry["id"] == "relu")
        self.assertEqual(card["kind"], "review")
        self.assertTrue(card["completed"])
        self.assertEqual(after["goal"]["completed"], 1)

    def test_flagged_completion_uses_local_day_across_dst_fall_back(self):
        at = datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc)  # 00:30 PDT
        self.store.record_attempt("local", "relu", PASS, at=at)
        self.store.mark_review("local", "relu", True)
        today = self.dashboard(now=at + timedelta(hours=24))  # 23:30 PST same day
        self.assertEqual(today["day"], "2026-11-01")
        self.assertEqual(today["stats"]["today_completed"], 1)
        self.assertEqual(today["reviews"], [])
        tomorrow = self.dashboard(now=at + timedelta(hours=25))  # 00:30 PST next day
        self.assertEqual(tomorrow["day"], "2026-11-02")
        self.assertEqual(tomorrow["stats"]["today_completed"], 0)
        self.assertEqual(tomorrow["reviews"][0]["id"], "relu")
        self.assertFalse(tomorrow["reviews"][0]["completed"])

    def test_new_flag_replaces_incomplete_choice_and_clearing_removes_stale_review(self):
        self.assertNotIn("embedding", [entry["id"] for entry in self.dashboard()["plan"]])
        self.store.mark_review("local", "embedding", True)
        dashboard = self.dashboard()
        self.assertEqual(dashboard["plan"][0]["id"], "embedding")
        self.assertEqual(dashboard["plan"][0]["kind"], "review")
        self.assertIn("Needs more practice", dashboard["plan"][0]["reason"])
        self.store.mark_review("local", "embedding", False)
        refreshed = self.dashboard()
        self.assertEqual(refreshed["reviews"], [])
        self.assertEqual([entry["id"] for entry in refreshed["plan"]], ["relu", "softmax", "cross_entropy"])

    def test_failure_severity_uses_test_fraction_and_handles_zero_test_runs(self):
        self.dashboard()
        self.store.record_attempt("local", "relu", FAIL, at=NOW)
        self.store.record_attempt("local", "dropout", {**FAIL, "passed": 3}, at=NOW)
        self.store.record_attempt("local", "softmax", {**FAIL, "passed": 0, "total": 0}, at=NOW)
        dashboard = self.dashboard()
        self.assertEqual([entry["id"] for entry in dashboard["reviews"]], ["softmax", "relu", "dropout"])
        self.assertEqual(dashboard["plan"][0]["id"], "softmax")
        self.assertEqual(self.review("softmax")["review_factors"]["mistakes"], 1)
        self.assertEqual(self.review("relu")["review_factors"]["mistakes"], 0.75)
        self.assertEqual(self.review("dropout")["review_factors"]["mistakes"], 0.25)
        self.assertIsNone(self.review("softmax")["estimated_retention"])

    def test_elapsed_time_increases_priority_for_equal_memory_strength(self):
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=7))
        self.store.record_attempt("local", "softmax", PASS, at=NOW - timedelta(days=2))
        older, newer = self.review("relu"), self.review("softmax")
        self.assertEqual(older["half_life_days"], newer["half_life_days"])
        self.assertGreater(older["review_score"], newer["review_score"])
        self.assertLess(older["estimated_retention"], newer["estimated_retention"])

    def test_successful_spaced_recalls_strengthen_memory(self):
        for offset in (11, 9, 5):
            self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=offset))
        self.store.record_attempt("local", "softmax", PASS, at=NOW - timedelta(days=5))
        self.store.mark_review("local", "relu", True)
        self.store.mark_review("local", "softmax", True)
        stronger, weaker = self.review("relu"), self.review("softmax")
        self.assertGreater(stronger["half_life_days"], weaker["half_life_days"])
        self.assertGreater(stronger["estimated_retention"], weaker["estimated_retention"])
        self.assertLess(stronger["review_score"], weaker["review_score"])

    def test_repeated_same_day_passes_do_not_fake_spaced_recalls(self):
        for minute in range(12):
            self.store.record_attempt("local", "relu", PASS, at=NOW + timedelta(minutes=minute))
        self.assertEqual(self.state("relu")["clean_passes"], 1)
        self.assertEqual(self.state("relu")["review_due_days"], 2)
        self.store.record_attempt("local", "relu", PASS, at=NOW + timedelta(days=2))
        self.assertEqual(self.state("relu")["clean_passes"], 2)
        self.assertEqual(self.state("relu")["review_due_days"], 4)

    def test_old_inflated_pass_counter_does_not_overstate_memory_strength(self):
        at = NOW - timedelta(days=3)
        for minute in range(6):
            self.store.record_attempt("local", "relu", PASS, at=at + timedelta(minutes=minute))
        with self.store.connection() as connection:
            connection.execute("""UPDATE progress SET clean_passes = 6, review_due_days = 30,
                                  review_due_at = ? WHERE task_id = 'relu'""",
                               ((at + timedelta(days=30)).isoformat(),))
        card = self.review("relu")
        self.assertLess(card["half_life_days"], 7)
        self.assertLess(card["estimated_retention"], 0.8)
        self.store.record_attempt("local", "relu", PASS, at=at + timedelta(minutes=6))
        self.assertEqual(self.state("relu")["clean_passes"], 1)
        self.assertEqual(self.state("relu")["review_due_days"], 2)

    def test_failures_do_not_refresh_successful_recall_timestamp(self):
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=5))
        before = self.review("relu")
        self.store.record_attempt("local", "relu", FAIL, at=NOW)
        after = self.review("relu")
        self.assertEqual(after["last_attempt_at"], NOW.isoformat())
        self.assertLess(after["estimated_retention"], before["estimated_retention"])
        self.assertGreater(after["review_score"], before["review_score"])

    def test_clean_pass_clears_manual_flag_and_reschedules_with_lower_priority(self):
        self.store.record_attempt("local", "relu", FAIL, at=NOW - timedelta(days=1))
        self.store.mark_review("local", "relu", True)
        before = self.review("relu")["review_score"]
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        self.assertFalse(self.state("relu")["manual_review"])
        self.assertFalse(self.state("relu")["needs_review"])
        self.assertEqual(self.dashboard()["reviews"], [])
        after = self.review("relu", now=NOW + timedelta(days=2))
        self.assertLess(after["review_score"], before)
        self.assertFalse(after["review_factors"]["manual"])

    def test_unmark_does_not_clear_mistakes_or_reset_timestamps(self):
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=4))
        self.store.record_attempt("local", "relu", FAIL, at=NOW)
        before = self.state("relu")
        self.store.mark_review("local", "relu", True)
        response = self.store.mark_review("local", "relu", False)
        after = self.state("relu")
        self.assertTrue(response["needs_review"])
        self.assertFalse(response["manual_review"])
        for key in ("review_anchor_at", "review_due_at", "review_due_days", "clean_passes", "review_reason"):
            self.assertEqual(before[key], after[key])
        self.assertGreater(self.review("relu")["review_factors"]["mistakes"], 0)

    def test_live_rank_replaces_stale_review_without_losing_completed_goal(self):
        self.dashboard()
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        self.store.record_attempt("local", "softmax", {**FAIL, "passed": 3}, at=NOW)
        self.assertEqual([entry["id"] for entry in self.dashboard()["plan"]], ["relu", "softmax", "cross_entropy"])
        self.store.mark_review("local", "embedding", True)
        refreshed = self.dashboard()
        self.assertEqual([entry["id"] for entry in refreshed["plan"]], ["relu", "embedding", "cross_entropy"])
        self.assertEqual(refreshed["goal"], {"target": 3, "completed": 1})
        self.assertEqual(len({entry["id"] for entry in refreshed["plan"]}), 3)
        self.assertEqual(len(self.dashboard(count=10)["plan"]), 10)
        self.assertEqual(self.dashboard(count=1)["goal"], {"target": 1, "completed": 1})

    def test_legacy_solved_marker_has_unknown_recall_without_fabricated_history(self):
        with patch("api.practice.utc_now", return_value=NOW):
            self.store.import_solved("local", ["relu"])
        card = self.review("relu")
        self.assertIsNone(card["estimated_retention"])
        self.assertIsNone(card["half_life_days"])
        self.assertIsNone(card["last_attempt_at"])
        self.assertEqual(card["review_factors"]["forgetting"], 1)
        self.assertEqual(self.dashboard()["history"], [])

    def test_manual_toggle_preserves_undated_legacy_attempted_signal(self):
        with self.store.connection() as connection:
            connection.execute("INSERT INTO progress (user_id, task_id, needs_review, review_due_at) VALUES ('local', 'relu', 1, ?)",
                               (NOW.isoformat(),))
        self.store.mark_review("local", "relu", True)
        self.store.mark_review("local", "relu", False)
        self.assertTrue(self.state("relu")["needs_review"])
        self.assertEqual(self.review("relu")["id"], "relu")

    def test_manual_migration_recovers_real_schedule_and_preserves_flag(self):
        # Re-create only the older progress table; profiles/attempts already
        # have authentic local test data and are untouched by the migration.
        self.store.record_attempt("local", "relu", PASS, at=NOW)
        with sqlite3.connect(self.store.path) as connection:
            connection.execute("ALTER TABLE progress RENAME TO progress_new")
            connection.execute("""CREATE TABLE progress (
                user_id TEXT, task_id TEXT, solved INTEGER DEFAULT 0, needs_review INTEGER DEFAULT 0,
                clean_passes INTEGER DEFAULT 0, review_due_at TEXT, review_anchor_at TEXT,
                review_due_days INTEGER, review_reason TEXT, imported_attempts INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, task_id))""")
            connection.execute("""INSERT INTO progress VALUES ('local', 'relu', 1, 1, 1, ?, ?, 0, 'manual', 0)""",
                               ((NOW + timedelta(hours=3)).isoformat(), (NOW + timedelta(hours=3)).isoformat()))
            connection.execute("DROP TABLE progress_new")
        self.store = PracticeStore(self.store.path, self.path / "absent.json")
        state = self.state("relu")
        self.assertTrue(state["manual_review"])
        self.assertEqual(state["review_anchor_at"], NOW.isoformat())
        self.assertEqual(state["review_due_days"], 2)
        self.store.mark_review("local", "relu", False)
        self.assertEqual(self.dashboard()["reviews"], [])

    def test_current_day_explicit_date_uses_live_clock_for_forgetting(self):
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=2))
        with patch("api.practice.utc_now", return_value=NOW):
            early = self.store.dashboard(day="2026-10-10")["reviews"][0]
        with patch("api.practice.utc_now", return_value=NOW + timedelta(hours=5)):
            later = self.store.dashboard(day="2026-10-10")["reviews"][0]
        self.assertGreater(later["review_score"], early["review_score"])

    def test_historical_day_previews_use_stable_date_anchor(self):
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=2))
        with patch("api.practice.utc_now", return_value=NOW + timedelta(days=2)):
            one = self.store.dashboard(day="2026-10-10")["reviews"][0]
        with patch("api.practice.utc_now", return_value=NOW + timedelta(days=3)):
            two = self.store.dashboard(day="2026-10-10")["reviews"][0]
        self.assertEqual(one["review_score"], two["review_score"])

    def test_curve_uses_elapsed_hours_across_dst_and_calendar_queue_stays_correct(self):
        anchor = datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc)  # 00:30 PST
        self.store.record_attempt("local", "relu", PASS, hint_used=True, at=anchor)
        self.assertEqual(self.dashboard(now=anchor)["reviews"], [])
        next_day = anchor + timedelta(hours=23)  # 00:30 PDT next calendar day
        card = self.review("relu", now=next_day)
        expected = 2 ** (-(23 / 24) / card["half_life_days"])
        self.assertAlmostEqual(card["estimated_retention"], expected, places=4)

    def test_review_signals_are_scoped_to_profile(self):
        other = self.store.create_profile("Ada")
        self.store.record_attempt(other["id"], "relu", FAIL, at=NOW)
        self.store.mark_review(other["id"], "embedding", True)
        self.assertEqual(self.dashboard()["reviews"], [])
        reviews = self.dashboard(user_id=other["id"])["reviews"]
        self.assertEqual([entry["id"] for entry in reviews], ["embedding", "relu"])

    def test_backfilled_success_preserves_latest_failure_and_ever_solved_history(self):
        self.store.record_attempt("local", "relu", FAIL, at=NOW)
        self.store.record_attempt("local", "relu", PASS, at=NOW - timedelta(days=3))
        state = self.state("relu")
        self.assertTrue(state["solved"])
        self.assertTrue(state["needs_review"])
        self.assertEqual(state["review_reason"], "failed")
        self.assertEqual(self.review("relu")["last_attempt_at"], NOW.isoformat())


class ReviewMetricTests(unittest.TestCase):
    def test_recall_target_and_weighted_score(self):
        attempt = {"success": True, "hint_used": False, "created_at": (NOW - timedelta(days=2)).isoformat()}
        metric = review_metrics({"manual_review": True, "solved": True, "clean_passes": 1}, [attempt], NOW)
        self.assertEqual(metric["estimated_retention"], RECALL_TARGET)
        self.assertEqual(metric["review_score"], 45)
        self.assertIs(metric["review_factors"]["manual"], True)
        self.assertEqual(INTERVALS, (2, 4, 7, 14, 30))

    def test_recent_clean_attempts_reduce_mistakes_and_only_last_five_count(self):
        failure = {"success": False, "passed": 0, "total": 4}
        success = {"success": True, "hint_used": False}
        self.assertEqual(recent_mistakes([failure]), 1)
        self.assertAlmostEqual(recent_mistakes([failure, success]), 1 / 3)
        self.assertEqual(recent_mistakes([failure] * 100 + [success] * 5), 0)

    def test_future_anchor_clamps_elapsed_and_naive_timestamps_are_utc(self):
        state = {"solved": True, "clean_passes": 1}
        future = {"success": True, "created_at": (NOW + timedelta(days=1)).isoformat()}
        metric = review_metrics(state, [future], NOW)
        self.assertEqual(metric["estimated_retention"], 1)
        naive = {"success": True, "created_at": NOW.replace(tzinfo=None).isoformat()}
        aware = {"success": True, "created_at": NOW.isoformat()}
        as_of = (NOW + timedelta(days=2)).astimezone(ZoneInfo("America/Los_Angeles"))
        self.assertEqual(review_metrics(state, [naive], as_of), review_metrics(state, [aware], as_of))


if __name__ == "__main__":
    unittest.main()
