
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from pandas.errors import DatabaseError


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "threat_detection.db"

CLASSIFICATION_THRESHOLD = 0.60


st.set_page_config(
    page_title="Network Threat Detection",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# Database
# --------------------------------------------------

def load_predictions():
    """Load prediction records while handling database read errors."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH, timeout=5) as connection:
            return pd.read_sql_query(
                """
                SELECT
                    id,
                    flow_id,
                    prediction,
                    predicted_class,
                    attack_probability,
                    risk_level,
                    model_name,
                    created_at,
                    explanation
                FROM predictions
                ORDER BY id DESC
                """,
                connection,
            )

    except (sqlite3.Error, DatabaseError) as exc:
        st.error(f"Unable to read the database: {exc}")
        return pd.DataFrame()


def parse_explanation(raw_explanation):
    """Support JSON explanations and older summary-only records."""
    if raw_explanation is None:
        return None

    if not isinstance(raw_explanation, str):
        raw_explanation = str(raw_explanation)

    if not raw_explanation.strip():
        return None

    try:
        parsed = json.loads(raw_explanation)

        if isinstance(parsed, dict):
            return parsed

    except (json.JSONDecodeError, TypeError):
        pass

    # Older records may contain plain-text explanations.
    return {
        "summary": raw_explanation,
        "method": None,
        "top_features": [],
    }


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ AI-Based Network Threat Detection")
st.caption("Network security monitoring and prediction analysis")

st.info(
    f"Active classification threshold: "
    f"{CLASSIFICATION_THRESHOLD:.2f}. "
    "Flows with attack probability at or above this value "
    "are classified as attacks. Risk levels use separate "
    "probability bands."
)

if st.button("🔄 Refresh dashboard"):
    st.rerun()


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = load_predictions()

if df.empty:
    st.info(
        "No prediction records found. Submit a prediction "
        "through the FastAPI endpoint to populate the dashboard."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total predictions", 0)
    col2.metric("Detected attacks", 0)
    col3.metric("High-risk records", 0)

else:
    df["attack_probability"] = pd.to_numeric(
        df["attack_probability"],
        errors="coerce",
    )
    df["explanation"] = df["explanation"].fillna("")

    total = len(df)
    attacks = int((df["prediction"] == "attack").sum())
    benign = int((df["prediction"] == "benign").sum())
    high_risk = int((df["risk_level"] == "high").sum())
    medium_risk = int((df["risk_level"] == "medium").sum())
    low_risk = int((df["risk_level"] == "low").sum())

    # --------------------------------------------------
    # Summary metrics
    # --------------------------------------------------

    st.subheader("Overview")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total predictions", total)
    metric2.metric("Detected attacks", attacks)
    metric3.metric("Benign flows", benign)
    metric4.metric("High-risk records", high_risk)

    # --------------------------------------------------
    # Prediction and risk distributions
    # --------------------------------------------------

    st.subheader("Prediction analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Prediction distribution**")

        prediction_counts = (
            df["prediction"]
            .value_counts()
            .reindex(["attack", "benign"], fill_value=0)
        )

        st.bar_chart(prediction_counts)

    with chart_col2:
        st.markdown("**Risk distribution**")

        risk_counts = pd.Series(
            {
                "High": high_risk,
                "Medium": medium_risk,
                "Low": low_risk,
            }
        )

        st.bar_chart(risk_counts)

    st.markdown("**Attack probability by prediction record**")

    probability_df = df[
        ["id", "attack_probability"]
    ].dropna().sort_values("id")

    if not probability_df.empty:
        probability_df = probability_df.set_index("id")
        st.line_chart(probability_df)

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------

    st.subheader("Prediction records")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        prediction_filter = st.selectbox(
            "Prediction type",
            ["All", "Attack", "Benign"],
        )

    with filter_col2:
        risk_filter = st.selectbox(
            "Risk level",
            ["All", "High", "Medium", "Low"],
        )

    with filter_col3:
        search_text = st.text_input(
            "Search flow ID",
            placeholder="Enter a flow ID",
        )

    filtered = df.copy()

    if prediction_filter != "All":
        filtered = filtered[
            filtered["prediction"]
            == prediction_filter.lower()
        ]

    if risk_filter != "All":
        filtered = filtered[
            filtered["risk_level"]
            == risk_filter.lower()
        ]

    if search_text.strip():
        filtered = filtered[
            filtered["flow_id"]
            .astype(str)
            .str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    display_columns = [
        "id",
        "flow_id",
        "prediction",
        "attack_probability",
        "risk_level",
        "model_name",
        "created_at",
    ]

    st.caption(f"Showing {len(filtered)} matching records.")

    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        display_df = filtered[display_columns].copy()

        display_df["attack_probability"] = (
            display_df["attack_probability"] * 100
        ).round(2)

        display_df = display_df.rename(
            columns={
                "id": "Record ID",
                "flow_id": "Flow ID",
                "prediction": "Prediction",
                "attack_probability": "Attack Probability (%)",
                "risk_level": "Risk Level",
                "model_name": "Model",
                "created_at": "Created At",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------
    # Prediction explanation
    # --------------------------------------------------

    st.subheader("Prediction explanation")

    if filtered.empty:
        st.info("No records match the current filters.")

    else:
        selected_options = {
            f"Record {int(row['id'])} | {row['flow_id']}": int(row["id"])
            for _, row in filtered.iterrows()
        }

        selected_label = st.selectbox(
            "Select a prediction",
            options=list(selected_options.keys()),
        )

        selected_id = selected_options[selected_label]

        selected_row = filtered[
            filtered["id"] == selected_id
        ].iloc[0]

        probability = selected_row["attack_probability"]
        prediction = selected_row["prediction"]
        explanation = parse_explanation(
            selected_row["explanation"]
        )

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        detail_col1.metric(
            "Prediction",
            str(prediction).title(),
        )

        detail_col2.metric(
            "Attack probability",
            (
                f"{probability * 100:.2f}%"
                if pd.notna(probability)
                else "Unavailable"
            ),
        )

        detail_col3.metric(
            "Risk level",
            str(selected_row["risk_level"]).title(),
        )

        if explanation:
            st.markdown("**Model explanation summary**")

            st.write(
                explanation.get(
                    "summary",
                    "No explanation summary is available.",
                )
            )

            method = explanation.get("method")

            if method:
                st.caption(f"Explanation method: {method}")

            top_features = explanation.get("top_features", [])

            if top_features:
                st.markdown("**Top contributing features**")

                feature_df = pd.DataFrame(top_features)

                expected_columns = {
                    "feature",
                    "contribution",
                    "direction",
                }

                if expected_columns.issubset(feature_df.columns):
                    feature_df = feature_df[
                        [
                            "feature",
                            "contribution",
                            "direction",
                        ]
                    ]

                    feature_df = feature_df.rename(
                        columns={
                            "feature": "Feature",
                            "contribution": "Contribution",
                            "direction": "Direction",
                        }
                    )

                    st.dataframe(
                        feature_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.warning(
                        "The stored feature explanation has "
                        "an unexpected format."
                    )

            else:
                st.info(
                    "Detailed feature contributions are unavailable "
                    "for this record. Older predictions stored only "
                    "the explanation summary."
                )

        else:
            st.warning(
                "No explanation was stored for this prediction."
            )

        st.caption(
            "Feature contributions explain the model's prediction "
            "under the implemented explanation method. They do not "
            "establish that a feature caused malicious activity."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Network Threat Detection and Analysis | "
    "Model-assisted predictions are decision-support signals, "
    "not a guarantee that a flow is malicious or benign."
)
