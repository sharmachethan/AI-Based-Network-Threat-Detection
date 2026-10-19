
import json
import sqlite3

import pandas as pd
import pytest

import src.dashboard as dashboard


def test_parse_json_explanation():
    explanation = {
        "summary": "The flow is classified as an attack.",
        "method": "Standardized logistic-regression contributions",
        "top_features": [
            {
                "feature": "Flow Duration",
                "contribution": 1.25,
                "direction": "attack",
            }
        ],
    }

    result = dashboard.parse_explanation(json.dumps(explanation))

    assert result == explanation
    assert len(result["top_features"]) == 1


def test_parse_legacy_text_explanation():
    result = dashboard.parse_explanation(
        "The flow is classified as benign."
    )

    assert result["summary"] == "The flow is classified as benign."
    assert result["method"] is None
    assert result["top_features"] == []


def test_parse_empty_explanation():
    assert dashboard.parse_explanation("") is None
    assert dashboard.parse_explanation(None) is None


def test_parse_invalid_json_as_legacy_text():
    result = dashboard.parse_explanation("{invalid json")

    assert result["summary"] == "{invalid json"
    assert result["method"] is None
    assert result["top_features"] == []


def test_load_predictions_returns_empty_when_database_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        dashboard,
        "DB_PATH",
        tmp_path / "missing.db",
    )

    result = dashboard.load_predictions()

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_predictions_reads_records(monkeypatch, tmp_path):
    db_path = tmp_path / "test_predictions.db"

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                flow_id TEXT NOT NULL,
                prediction TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                attack_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                explanation TEXT NOT NULL DEFAULT ''
            )
            """
        )

        connection.execute(
            """
            INSERT INTO predictions (
                id, flow_id, prediction, predicted_class,
                attack_probability, risk_level, model_name,
                created_at, explanation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "dashboard-test-flow",
                "attack",
                1,
                0.95,
                "high",
                "LogisticRegression",
                "2026-10-19T12:00:00",
                '{"summary":"Attack detected","method":"test",'
                '"top_features":[]}',
            ),
        )

    monkeypatch.setattr(dashboard, "DB_PATH", db_path)

    result = dashboard.load_predictions()

    assert len(result) == 1
    assert result.iloc[0]["flow_id"] == "dashboard-test-flow"
    assert result.iloc[0]["prediction"] == "attack"
    assert result.iloc[0]["attack_probability"] == pytest.approx(0.95)
    assert "explanation" in result.columns


def test_load_predictions_handles_sqlite_read_error(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "invalid.db"
    db_path.write_text("not a valid SQLite database", encoding="utf-8")

    monkeypatch.setattr(dashboard, "DB_PATH", db_path)

    result = dashboard.load_predictions()

    assert isinstance(result, pd.DataFrame)
    assert result.empty
