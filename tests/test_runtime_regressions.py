from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core import knowledge_base as kb
from layers.strategic import build_strategic_report


class RuntimeRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = kb.DB_PATH
        kb.DB_PATH = Path(self.temp_dir.name) / "knowledge.db"

    def tearDown(self) -> None:
        kb.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_init_db_migrates_legacy_signal_table_before_indexes(self) -> None:
        db = sqlite3.connect(kb.DB_PATH)
        try:
            db.execute(
                """CREATE TABLE signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    features TEXT NOT NULL,
                    reasons TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    target_050_move_pct REAL NOT NULL,
                    target_100_move_pct REAL NOT NULL,
                    target_200_move_pct REAL NOT NULL,
                    finalized INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )"""
            )
            db.commit()
        finally:
            db.close()

        asyncio.run(kb.init_db())

        db = sqlite3.connect(kb.DB_PATH)
        try:
            columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(signal_outcomes)").fetchall()
            }
            indexes = {
                row[1]
                for row in db.execute("PRAGMA index_list(signal_outcomes)").fetchall()
            }
        finally:
            db.close()

        self.assertIn("decision_group", columns)
        self.assertIn("hit_configured", columns)
        self.assertIn("idx_signal_decision_group", indexes)
        self.assertIn("idx_signal_hit_rate", indexes)

    def test_strategic_report_supports_empty_database(self) -> None:
        async def scenario():
            await kb.init_db()
            return await build_strategic_report(30)

        report = asyncio.run(scenario())
        self.assertEqual(report.total_trades, 0)
        self.assertEqual(
            report.statistical_tests["win_rate_confidence"]["verdict"],
            "INSUFFICIENT_EVIDENCE",
        )


if __name__ == "__main__":
    unittest.main()
