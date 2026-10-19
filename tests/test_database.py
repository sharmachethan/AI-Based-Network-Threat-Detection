
import sqlite3

import src.database as database


def test_database_initialization_and_migration(tmp_path, monkeypatch):
    """Initialization creates the table and explanation column."""
    test_db = tmp_path / "test_threat_detection.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)

    # Simulate an older database schema without explanation.
    with sqlite3.connect(test_db) as connection:
        connection.execute("""
            CREATE TABLE predictions (
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

    database.initialize_database()

    with sqlite3.connect(test_db) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            )
        }

    assert "explanation" in columns


def test_save_and_retrieve_prediction(tmp_path, monkeypatch):
    """A saved prediction can be retrieved with its explanation."""
    test_db = tmp_path / "test_threat_detection.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.initialize_database()

    record_id = database.save_prediction(
        flow_id="unit-test-flow",
        prediction="attack",
        predicted_class=1,
        attack_probability=0.95,
        risk_level="high",
        model_name="test-model",
        created_at="2026-10-19T10:00:00+00:00",
        explanation="Test explanation",
    )

    records = database.get_recent_predictions(limit=10)

    assert record_id == 1
    assert len(records) == 1
    assert records[0]["flow_id"] == "unit-test-flow"
    assert records[0]["prediction"] == "attack"
    assert records[0]["predicted_class"] == 1
    assert records[0]["attack_probability"] == 0.95
    assert records[0]["risk_level"] == "high"
    assert records[0]["explanation"] == "Test explanation"


def test_recent_predictions_limit(tmp_path, monkeypatch):
    """The requested limit is respected."""
    test_db = tmp_path / "test_threat_detection.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.initialize_database()

    for number in range(3):
        database.save_prediction(
            flow_id=f"flow-{number}",
            prediction="benign",
            predicted_class=0,
            attack_probability=0.1,
            risk_level="low",
            model_name="test-model",
            created_at=f"2026-10-19T10:0{number}:00+00:00",
        )

    records = database.get_recent_predictions(limit=2)

    assert len(records) == 2
    assert records[0]["flow_id"] == "flow-2"
    assert records[1]["flow_id"] == "flow-1"
