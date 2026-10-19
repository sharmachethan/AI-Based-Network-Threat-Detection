# AI-Based Network Threat Detection and Analysis

A machine-learning-based network traffic classification system that identifies network flows as **benign** or **attack** using the CIC-IDS2017 dataset. The project includes a trained Logistic Regression baseline, a FastAPI prediction service, a Streamlit dashboard, prediction persistence with SQLite, and model evaluation reports.

## Features

- Binary classification of network flows as benign or attack
- Data validation and preprocessing
- Logistic Regression pipeline with median imputation, feature scaling, and class weighting
- FastAPI endpoints for health checks, predictions, and prediction history
- SQLite storage for prediction records
- Streamlit dashboard for prediction analytics and risk visualization
- Model explanations based on feature contributions
- Held-out evaluation and classification-threshold comparison
- Automated tests for API, database, and dashboard functionality

## Technology Stack

- Python
- Pandas and NumPy
- scikit-learn
- FastAPI and Uvicorn
- Streamlit
- SQLite
- Pytest

## Project Structure

```text
NetworkThreatDetection/
├── data/             # Local datasets, database, and generated files
├── models/           # Trained model artifacts
├── reports/          # Evaluation reports and test results
├── src/
│   ├── api.py
│   ├── dashboard.py
│   ├── database.py
│   ├── pages/
│   │   └── 2_Model_Evaluation.py
│   └── ...           # Data preparation and evaluation scripts
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.12
- The project dataset and trained model artifacts
- Windows, macOS, or Linux with a compatible Python environment

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The dataset and trained model are not included in the dependency file. Make sure the required local data and model artifacts are available before running the application.

## Run the Application

Start the FastAPI service in one terminal:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Start the Streamlit dashboard in a second terminal:

```powershell
python -m streamlit run src/dashboard.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501.

The dashboard includes a separate **Model Evaluation** page.

## Model Evaluation

The saved held-out random test evaluation at classification threshold 0.60 reports:

| Metric | Result |
|---|---:|
| Test records | 566,149 |
| Accuracy | 91.96% |
| Attack precision | 73.08% |
| Attack recall | 93.67% |
| Attack F1-score | 82.11% |
| Benign F1-score | 94.81% |
| ROC-AUC reference | 97.61% |

The ROC-AUC value is referenced from the original evaluation using the same model and test split; it was not recalculated in the threshold-specific evaluation run.

### Evaluation limitations

- The train/test split is random, so related network flows may appear in both sets.
- These results do not establish generalization to previously unseen attack families.
- Threshold selection changes the trade-off between false alarms and missed attacks.
- Model predictions and feature contributions are decision-support signals, not proof that a flow is malicious or benign.

Detailed results are available in the `reports/` directory.

## Run Tests

Run the automated test suite:

```powershell
python -m pytest -v --junitxml=reports/test_results.xml
```

The latest recorded full test run completed with 25 passing tests. Rerun the command to verify the current code.

## API Overview

| Endpoint | Purpose |
|---|---|
| `GET /` | API information |
| `GET /health` | Service health |
| `POST /predict` | Classify a network flow |
| `GET /predictions` | Retrieve stored prediction records |

Prediction requests must provide the expected model feature names and numeric values. Refer to the API documentation at `/docs` for the request schema.

## Disclaimer

This is a machine-learning research and demonstration project. It does not replace a production intrusion detection system, network monitoring controls, or security analyst review.