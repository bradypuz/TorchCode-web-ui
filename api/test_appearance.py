"""Run with: .venv/bin/python -m unittest api.test_appearance"""

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from api import practice
from api.practice import AppearanceRequest, PracticeStore


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.db_path = self.path / "practice.db"
        self.legacy_path = self.path / "progress.json"
        self.store = PracticeStore(self.db_path, self.legacy_path)

    def tearDown(self):
        self.directory.cleanup()

    def assert_http_error(self, status, operation):
        with self.assertRaises(HTTPException) as caught:
            operation()
        self.assertEqual(caught.exception.status_code, status)

    def test_existing_and_new_learners_default_to_dark_without_writing_preferences(self):
        learner = self.store.create_profile("Ada")
        self.assertEqual(self.store.appearance("local"), {"theme": "dark"})
        self.assertEqual(self.store.appearance(learner["id"]), {"theme": "dark"})
        with self.store.connection() as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM appearance_preferences").fetchone()[0], 0)

    def test_preferences_are_isolated_and_survive_reopening_and_updates(self):
        learner = self.store.create_profile("Ada")
        self.assertEqual(self.store.set_appearance("local", "light"), {"theme": "light"})
        self.assertEqual(self.store.appearance(learner["id"]), {"theme": "dark"})
        self.store.set_appearance(learner["id"], "dark")
        reopened = PracticeStore(self.db_path, self.legacy_path)
        self.assertEqual(reopened.appearance("local"), {"theme": "light"})
        self.assertEqual(reopened.appearance(learner["id"]), {"theme": "dark"})
        reopened.set_appearance("local", "dark")
        reopened.set_appearance(learner["id"], "light")
        again = PracticeStore(self.db_path, self.legacy_path)
        self.assertEqual(again.appearance("local"), {"theme": "dark"})
        self.assertEqual(again.appearance(learner["id"]), {"theme": "light"})
        with again.connection() as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM appearance_preferences").fetchone()[0], 2)

    def test_missing_profile_reads_and_writes_return_404(self):
        self.assert_http_error(404, lambda: self.store.appearance("missing"))
        self.assert_http_error(404, lambda: self.store.set_appearance("missing", "light"))
        self.assertEqual(self.store.appearance("local"), {"theme": "dark"})
        with self.store.connection() as connection:
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM appearance_preferences WHERE user_id = 'missing'").fetchone())

    def test_invalid_themes_do_not_overwrite_the_saved_preference(self):
        self.store.set_appearance("local", "light")
        for invalid in ("system", "Dark", "", " light ", None):
            with self.subTest(theme=invalid):
                self.assert_http_error(422, lambda: self.store.set_appearance("local", invalid))
                self.assertEqual(self.store.appearance("local"), {"theme": "light"})

    def test_database_also_rejects_invalid_themes_and_nonexistent_profiles(self):
        with self.assertRaises(sqlite3.IntegrityError), self.store.connection() as connection:
            connection.execute("INSERT INTO appearance_preferences VALUES ('local', 'system')")
        with self.assertRaises(sqlite3.IntegrityError), self.store.connection() as connection:
            connection.execute("INSERT INTO appearance_preferences VALUES ('missing', 'light')")

    def test_preference_table_is_added_to_an_existing_database_without_losing_history(self):
        old_path = self.path / "old-practice.db"
        with sqlite3.connect(old_path) as connection:
            connection.executescript("""
                CREATE TABLE profiles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES profiles(id), task_id TEXT NOT NULL,
                    success INTEGER NOT NULL, passed INTEGER NOT NULL, total INTEGER NOT NULL,
                    hint_used INTEGER NOT NULL, failed_tests TEXT NOT NULL,
                    created_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'web',
                    submitted_code TEXT
                );
                INSERT INTO profiles VALUES ('local', 'Existing learner', '2026-10-10T18:00:00+00:00');
                INSERT INTO attempts VALUES (
                    42, 'local', 'relu', 1, 4, 4, 0, '[]', '2026-10-10T18:00:00+00:00',
                    'web', 'def relu(x):\n    return x.clamp(min=0)'
                );
            """)
            original_profile = connection.execute("SELECT * FROM profiles").fetchall()
            original_attempts = connection.execute("SELECT * FROM attempts").fetchall()

        upgraded = PracticeStore(old_path, self.legacy_path)
        self.assertEqual(upgraded.appearance("local"), {"theme": "dark"})
        upgraded.set_appearance("local", "light")
        reopened = PracticeStore(old_path, self.legacy_path)
        self.assertEqual(reopened.appearance("local"), {"theme": "light"})
        with reopened.connection() as connection:
            self.assertEqual([tuple(row) for row in connection.execute("SELECT * FROM profiles")],
                             original_profile)
            self.assertEqual([tuple(row) for row in connection.execute("SELECT * FROM attempts")],
                             original_attempts)

    def test_api_contract_and_default_learner(self):
        learner = self.store.create_profile("Lin")
        with patch.object(practice, "get_store", return_value=self.store):
            self.assertEqual(practice.get_appearance(), {"theme": "dark"})
            self.assertEqual(practice.set_appearance(AppearanceRequest(theme="light")),
                             {"theme": "light"})
            self.assertEqual(practice.get_appearance("local"), {"theme": "light"})
            self.assertEqual(practice.set_appearance(
                AppearanceRequest(user_id=learner["id"], theme="dark")), {"theme": "dark"})
            self.assertEqual(practice.get_appearance(learner["id"]), {"theme": "dark"})
            self.assert_http_error(404, lambda: practice.get_appearance("missing"))
            self.assert_http_error(404, lambda: practice.set_appearance(
                AppearanceRequest(user_id="missing", theme="light")))

    def test_request_validation_requires_a_supported_theme(self):
        for values in ({}, {"theme": "system"}, {"theme": "LIGHT"}, {"theme": None},
                       {"theme": 1}, {"theme": "dark", "user_id": None}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                AppearanceRequest(**values)


if __name__ == "__main__":
    unittest.main()
