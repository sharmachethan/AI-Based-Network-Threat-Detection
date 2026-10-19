
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "threat_detection.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_id TEXT NOT NULL,
                prediction TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                attack_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Add the new field without deleting existing prediction records.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        if "explanation" not in columns:
            connection.execute(
                "ALTER TABLE predictions "
                "ADD COLUMN explanation TEXT NOT NULL DEFAULT ''"
            )


def save_prediction(
    flow_id,
    prediction,
    predicted_class,
    attack_probability,
    risk_level,
    model_name,
    created_at,
    explanation="",
):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                flow_id,
                prediction,
                predicted_class,
                attack_probability,
                risk_level,
                model_name,
                created_at,
                explanation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                prediction,
                predicted_class,
                attack_probability,
                risk_level,
                model_name,
                created_at,
                explanation,
            ),
        )

        return cursor.lastrowid


def get_recent_predictions(limit=50):
    limit = max(1, min(int(limit), 200))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


initialize_database()
