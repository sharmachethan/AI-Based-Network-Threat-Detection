
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"

TEST_REPORT = REPORTS_DIR / "final_test_evaluation.json"
THRESHOLD_REPORT = REPORTS_DIR / "threshold_060_test_evaluation.json"
COMPARISON_REPORT = REPORTS_DIR / "threshold_comparison.json"


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)


def load_json_report(path):
    """Load a saved evaluation report without retraining the model."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        return result if isinstance(result, dict) else None

    except (OSError, json.JSONDecodeError):
        return None


st.title("📊 Model Evaluation")
st.caption(
    "Evaluation results for the Logistic Regression network "
    "threat detection baseline."
)

test_report = load_json_report(TEST_REPORT)
threshold_report = load_json_report(THRESHOLD_REPORT)
comparison_report = load_json_report(COMPARISON_REPORT)


if not test_report or not threshold_report or not comparison_report:
    st.error(
        "One or more evaluation reports could not be loaded. "
        "Check that the required JSON files exist in the reports folder."
    )
    st.stop()


# --------------------------------------------------
# Dataset and evaluation information
# --------------------------------------------------

st.subheader("Evaluation overview")

overview_col1, overview_col2, overview_col3 = st.columns(3)

overview_col1.metric(
    "Held-out test records",
    f"{test_report['test_records']:,}",
)

overview_col2.metric(
    "Model features",
    str(test_report["feature_count"]),
)

overview_col3.metric(
    "Classification threshold",
    f"{threshold_report['decision_threshold']:.2f}",
)

st.caption(
    "The results below use saved evaluation reports. "
    "Opening this page does not train or modify the model."
)


# --------------------------------------------------
# Held-out test metrics at threshold 0.60
# --------------------------------------------------

st.subheader("Held-out test performance")

classification = threshold_report["classification_report"]
attack = classification["Attack"]
benign = classification["Benign"]

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Accuracy",
    f"{threshold_report['accuracy']:.2%}",
)

metric_col2.metric(
    "Attack precision",
    f"{attack['precision']:.2%}",
)

metric_col3.metric(
    "Attack recall",
    f"{attack['recall']:.2%}",
)

metric_col4, metric_col5, metric_col6 = st.columns(3)

metric_col4.metric(
    "Attack F1-score",
    f"{attack['f1-score']:.2%}",
)

metric_col5.metric(
    "Benign F1-score",
    f"{benign['f1-score']:.2%}",
)

metric_col6.metric(
    "ROC-AUC",
    f"{test_report['roc_auc']:.2%}",
)

st.caption(
    "ROC-AUC is taken from the original held-out evaluation using "
    "the same model and test split. It is a threshold-independent "
    "reference; it was not recalculated in the threshold-0.60 run."
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

st.subheader("Confusion matrix")

matrix = threshold_report["confusion_matrix"]

tn, fp = matrix[0]
fn, tp = matrix[1]

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual benign", "Actual attack"],
    columns=["Predicted benign", "Predicted attack"],
)

st.dataframe(
    matrix_df,
    use_container_width=True,
)

conf_col1, conf_col2 = st.columns(2)

conf_col1.metric("True negatives", f"{tn:,}")
conf_col2.metric("True positives", f"{tp:,}")

conf_col3, conf_col4 = st.columns(2)

conf_col3.metric("False positives", f"{fp:,}")
conf_col4.metric("False negatives", f"{fn:,}")

st.caption(
    "False positives are benign flows classified as attacks. "
    "False negatives are attack flows classified as benign."
)


# --------------------------------------------------
# Threshold comparison on validation data
# --------------------------------------------------

st.subheader("Classification threshold comparison")

st.write(
    "These results come from the separate validation subset, "
    "not the held-out test set. Increasing the threshold generally "
    "reduces false positives but may increase missed attacks."
)

threshold_results = comparison_report.get("threshold_results", [])

if threshold_results:
    comparison_df = pd.DataFrame(threshold_results)

    display_columns = {
        "threshold": "Threshold",
        "accuracy": "Accuracy",
        "attack_precision": "Attack precision",
        "attack_recall": "Attack recall",
        "attack_f1": "Attack F1",
        "false_positives": "False positives",
        "false_negatives": "False negatives",
    }

    available_columns = [
        column
        for column in display_columns
        if column in comparison_df.columns
    ]

    display_df = comparison_df[available_columns].rename(
        columns=display_columns
    )

    percentage_columns = [
        "Accuracy",
        "Attack precision",
        "Attack recall",
        "Attack F1",
    ]

    for column in percentage_columns:
        if column in display_df.columns:
            display_df[column] = (
                display_df[column] * 100
            ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Percentage metrics are shown on a 0–100 scale. "
        "False-positive and false-negative columns are record counts."
    )

    chart_df = comparison_df.set_index("threshold")[
        [
            "attack_precision",
            "attack_recall",
            "attack_f1",
        ]
    ].rename(
        columns={
            "attack_precision": "Attack precision",
            "attack_recall": "Attack recall",
            "attack_f1": "Attack F1",
        }
    )

    st.markdown("**Attack metrics by validation threshold**")
    st.line_chart(chart_df)

    error_df = comparison_df.set_index("threshold")[
        ["false_positives", "false_negatives"]
    ].rename(
        columns={
            "false_positives": "False positives",
            "false_negatives": "False negatives",
        }
    )

    st.markdown("**Error counts by validation threshold**")
    st.line_chart(error_df)

else:
    st.warning("No threshold comparison results were found.")


# --------------------------------------------------
# Evaluation limitations
# --------------------------------------------------

st.subheader("Evaluation limitations")

st.warning(test_report.get("warning", "No evaluation warning is available."))

st.write(
    "The held-out test set uses a random split. Related network flows "
    "may therefore appear in both training and test sets. These results "
    "do not establish generalization to previously unseen attack families."
)

st.caption(
    "Threshold selection should reflect the intended balance between "
    "missed attacks and false alarms. The validation comparison should "
    "not be interpreted as proof of performance on unseen attack families."
)
