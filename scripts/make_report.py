"""Generate the assignment report DOCX with clearly marked run-dependent results."""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MLOps_Assignment_01_Report.docx"
doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.7)
sec.bottom_margin = Inches(0.7)
sec.left_margin = Inches(0.8)
sec.right_margin = Inches(0.8)
styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(9.5)
styles["Title"].font.name = "Calibri"
styles["Title"].font.size = Pt(25)
styles["Title"].font.color.rgb = RGBColor(31, 78, 121)
for s in ["Heading 1", "Heading 2"]:
    styles[s].font.name = "Calibri"
    styles[s].font.color.rgb = RGBColor(31, 78, 121)


def title(text):
    doc.add_heading(text, level=1)


def para(text):
    doc.add_paragraph(text)


def bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Shading Accent 1"
    for c, h in zip(t.rows[0].cells, headers):
        c.text = h
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = str(v)


def page():
    doc.add_page_break()


p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("\n\nMLOps Experimental Learning Assignment\n").bold = True
p.add_run("End-to-End Heart Disease Classification\n").font.size = Pt(20)
p.add_run(
    "\nCourse: Machine Learning Operations (AIMLCZG523)\nAssignment 01 · Total Marks: 50\n\nStudent Name: __________________________\nStudent ID: ______________________________\nSubmission Date: _________________________\n"
).font.size = Pt(12)
para(
    "\nProject scope: data acquisition, EDA, model development, experiment tracking, reproducibility, CI/CD, containerization, deployment design, and monitoring."
)
para(
    "Important notice: This report describes an educational prototype. It is not a medical device, is not clinically validated, and must not be used for diagnosis or treatment."
)
page()

title("1. Executive Summary")
para(
    "This project implements a reproducible machine-learning workflow for binary classification on the UCI Heart Disease dataset. The target is transformed to 0 for the original target value 0 (no disease) and 1 for original values greater than 0 (disease present). The solution compares Logistic Regression and Random Forest, evaluates both with stratified five-fold cross-validation, and reports held-out test metrics. Preprocessing and the estimator are combined in a single scikit-learn Pipeline to keep training and serving transformations consistent."
)
para(
    "MLflow records each experiment’s model identity, split and cross-validation settings, cross-validation metric means and standard deviations, held-out metrics, classification report, confusion matrix, ROC curve, and serialized model. The selected model is saved as a joblib artifact and loaded by a FastAPI service. The service exposes prediction and health endpoints, request logs, and Prometheus metrics. GitHub Actions runs linting, tests, data acquisition, EDA, training, and artifact upload. Docker and Kubernetes manifests support local container and cluster deployment."
)
para(
    "Results must be populated from the actual execution of the repository. This document intentionally does not invent model scores, a public endpoint, screenshots, or a video recording."
)
table(
    ["Area", "Implementation"],
    [
        ["Dataset", "UCI Heart Disease repository, dataset ID 45"],
        ["Models", "Logistic Regression; Random Forest"],
        ["Selection", "Highest mean stratified 5-fold CV ROC-AUC"],
        ["Tracking", "MLflow local file store by default"],
        ["Serving", "FastAPI: /predict, /health, /metrics"],
        ["Automation", "GitHub Actions + pytest + Ruff"],
        ["Packaging", "Docker image"],
        ["Deployment", "Kubernetes Deployment and NodePort Service"],
    ],
)
page()

title("2. Problem Definition and Dataset")
para(
    "The assignment asks for a machine-learning classifier that estimates whether heart disease is present based on patient health features, then packages the workflow as a cloud-ready monitored API. The public UCI Heart Disease collection contains clinical attributes such as age, sex, chest-pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic result, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia-related category."
)
para(
    "The downloader uses the official UCI repository interface via the ucimlrepo Python package (dataset ID 45). It writes a CSV to data/raw/heart_disease.csv and prints row/column counts and missing-cell totals. The raw target is retained as target_original so the transformation is auditable. The data loader converts feature columns to numeric values, treats question-mark/malformed values as missing, drops rows whose target is absent, and binarizes the target as described above."
)
para(
    "The data are a relatively small benchmark dataset and should not be interpreted as representative of modern clinical populations. Variables, collection practices, and class distributions may differ across hospitals and demographic groups. Dataset access and successful download require network connectivity; the CSV generated during a run should be retained with the submitted project if permitted by course rules."
)
table(
    ["Field group", "Features"],
    [
        ["Numeric, scaled", "age, trestbps, chol, thalach, oldpeak"],
        ["Categorical, one-hot encoded", "sex, cp, fbs, restecg, exang, slope, ca, thal"],
        ["Target", "target_original → binary target"],
    ],
)
page()

title("3. Data Cleaning and Exploratory Analysis")
para(
    "The preprocessing pipeline imputes numeric missing values with the training-fold median and categorical missing values with the most frequent category. Numeric features are standardized; categorical fields are one-hot encoded with unknown categories ignored. All transformations are fitted only on the relevant training split/fold, preventing the test set from leaking information into preprocessing."
)
para(
    "The EDA script exports summary statistics and a missingness table, along with four key views: target class balance, feature histograms, and a feature/target correlation heatmap. These plots are saved under artifacts/eda/. The class-balance plot is used to inspect imbalance before model selection. Histograms help identify skew, outliers, and unusual value ranges. The correlation heatmap is a descriptive view of pairwise linear association; it does not establish causality."
)
para(
    "After executing the workflow, insert or attach the generated plots here and summarize the observed patterns. Do not assert a missingness percentage, distribution shape, or correlation value until it is confirmed from the produced artifacts."
)
table(
    ["EDA artifact", "Purpose", "Generated file"],
    [
        ["Summary statistics", "Ranges and quartiles", "summary_statistics.csv"],
        ["Missingness", "Missing count and percentage", "missingness.csv"],
        ["Class balance", "Target distribution", "class_balance.png"],
        ["Histograms", "Univariate feature distributions", "feature_histograms.png"],
        ["Correlation heatmap", "Pairwise numeric correlations", "correlation_heatmap.png"],
    ],
)
para(
    "Run commands: `python scripts/download_data.py` followed by `python scripts/eda.py`. Confirm the printed row count and inspect every generated artifact before final submission."
)
page()

title("4. Feature Engineering and Model Development")
para(
    "Two supervised classifiers are trained. Logistic Regression is a relatively interpretable linear baseline and benefits from scaled numeric variables. Random Forest models non-linear relationships and interactions using an ensemble of decision trees. The initial configuration uses class_weight=balanced to reduce the tendency to ignore a minority class. Random seeds are fixed where supported to improve repeatability."
)
para(
    "A stratified 80/20 train-test split reserves a holdout set for final evaluation. On the training portion, StratifiedKFold with five splits and shuffling estimates generalization performance while preserving class proportions as much as possible. Each candidate is evaluated on accuracy, precision, recall, and ROC-AUC. Mean and standard deviation are recorded for cross-validation. The model with the highest mean cross-validation ROC-AUC is selected; the holdout set is not used to choose the model."
)
para(
    "Accuracy measures the fraction of all predictions that are correct. Precision measures the fraction of positive predictions that are truly positive. Recall measures the fraction of positive cases detected. ROC-AUC evaluates ranking quality across thresholds. In a health-related context, metric trade-offs matter: a threshold that increases recall can also increase false positives. The default 0.5 threshold is a demonstration setting, not a clinically optimized threshold."
)
para("Actual result table — fill this from artifacts/model_comparison.csv after running the training script:")
table(
    ["Model", "CV ROC-AUC mean", "Test accuracy", "Test precision", "Test recall", "Test ROC-AUC"],
    [
        ["Logistic Regression", "[RUN RESULT]", "[RUN RESULT]", "[RUN RESULT]", "[RUN RESULT]", "[RUN RESULT]"],
        ["Random Forest", "[RUN RESULT]", "[RUN RESULT]", "[RUN RESULT]", "[RUN RESULT]", "[RUN RESULT]"],
    ],
)
para(
    "The training script saves per-model classification reports, confusion matrices, and ROC curves in artifacts/. These should be inspected for model-specific errors rather than relying on one headline score."
)
page()

title("5. Experiment Tracking and Reproducibility")
para(
    "MLflow is used to maintain an experiment named heart-disease-classification. Each candidate model has a separate run. The run records model name, test split fraction, random seed, cross-validation configuration, cross-validation means and standard deviations for four metrics, and holdout metrics. Artifacts include the classification report JSON, confusion matrix, ROC curve, and fitted scikit-learn pipeline. This makes comparison traceable and allows a reviewer to inspect how the final candidate was chosen."
)
para(
    "By default, MLflow uses a local file-backed tracking directory at mlruns/. The tracking UI can be started with `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000`. For team/cloud use, configure a shared tracking server and artifact store through MLFLOW_TRACKING_URI. Avoid committing credentials or sensitive data to the repository."
)
para(
    "The final joblib file contains the preprocessing transformer and classifier as one pipeline, not only the estimator. This is important because the same imputation, scaling, and encoding must occur during inference. artifacts/model_metadata.json records the selected model, expected feature order, threshold, target mapping, selection metric, and warning. The requirements file specifies dependency ranges; for strict production reproducibility, additionally generate a platform-specific lock file and record Python and operating-system versions."
)
para(
    "Repeatability is improved by fixed random seeds and explicit split/cross-validation settings. Some numeric variation can remain across package versions, processor architectures, or parallel execution. A clean-room verification should create a new virtual environment, install requirements, download the data, and rerun the pipeline."
)
page()

title("6. API Packaging, Testing, and CI/CD")
para(
    "FastAPI exposes POST /predict, GET /health, and GET /metrics. The request schema validates the 13 expected features and reasonable input ranges. /predict returns the binary prediction, a readable label, confidence for the returned class, disease probability, threshold, and a non-clinical-use disclaimer. If the model artifact is missing, the service reports model_loaded=false and prediction requests return HTTP 503. Request logs include method, path, status-related counters, and latency; input payloads are not written to logs."
)
para(
    "Automated tests cover target binarization, missing-column errors, pipeline fitting for both estimators, health response, and input schema validation. Run `pytest -q`. Ruff checks for common code-quality issues. GitHub Actions installs dependencies, lints, tests, downloads data, runs EDA and training, then uploads model/plot/MLflow artifacts for each workflow run. The pipeline should fail on lint or test failures and should not proceed to training if earlier steps fail."
)
para(
    "To enable CI, push the project to a GitHub repository and ensure Actions are enabled. Inspect the Actions run for a green status, confirm that uploaded artifacts exist, and capture a screenshot of the successful workflow. A failed data download may indicate network restrictions or a temporary repository issue; do not mark the workflow successful until it completes."
)
para(
    "Commands: `ruff check src scripts tests`, `pytest -q`, `python scripts/download_data.py`, `python scripts/eda.py`, and `python -m src.train`. The Docker image expects artifacts/model.joblib to exist before build time."
)
page()

title("7. Containerization and Deployment")
para(
    "The Dockerfile starts from Python 3.11 slim, installs dependencies, copies the API source and trained artifact, and starts Uvicorn on port 8000. Build after training: `docker build -t heart-disease-api .`. Run with `docker run --rm -p 8000:8000 heart-disease-api`. Validate GET /health, open /docs, and submit a valid example to POST /predict. Keep a screenshot of the container running and the API response as evidence."
)
para(
    "Kubernetes manifests define a two-replica Deployment with resource requests/limits, readiness and liveness probes, and a NodePort Service on port 30080. The container image must be available to the local cluster. For Minikube, build/load the image into the Minikube image environment and retrieve the service URL with `minikube service heart-disease-api --url`. For Docker Desktop Kubernetes, use the image available in its Docker daemon and verify the NodePort route. The optional Ingress manifest is only a template; it requires an installed ingress controller and a real hostname."
)
para(
    "Deployment verification sequence: `kubectl apply -f k8s/deployment.yaml`; `kubectl apply -f k8s/service.yaml`; `kubectl get pods,svc`; then call /health and /predict. Do not expose a public endpoint without access control, rate limits, TLS, privacy review, and an appropriate security assessment. This assignment prototype is not designed to process identifiable patient data."
)
table(
    ["Evidence", "Status to complete after running"],
    [
        ["Docker build and container run", "[Attach screenshot]"],
        ["/health and /predict response", "[Attach screenshot]"],
        ["Kubernetes pods and service", "[Attach screenshot]"],
        ["Public API URL or local access instructions", "[Insert verified URL/instructions]"],
    ],
)
page()

title("8. Monitoring, Limitations, and Risk")
para(
    "The service exposes Prometheus-compatible counters and a latency histogram at /metrics. prediction_requests_total counts requests by endpoint and status, while prediction_request_duration_seconds records request duration. Structured logs record request method, path, duration, model loading, prediction completion, and errors. A simple first monitoring pass can inspect `/metrics` and container logs; Prometheus can scrape the endpoint for a dashboard. For a production deployment, configure a Prometheus scrape target and Grafana panels for request volume, error rate, and latency percentiles."
)
para(
    "The current monitoring is operational, not clinical or statistical drift monitoring. It does not prove that predictions are accurate in deployment. A production system would need data validation, drift checks, model-version tracking, alert thresholds, access controls, audit requirements, privacy safeguards, rollback procedures, and human clinical oversight. The training dataset is small and historical, and the model has not been validated across populations or compared against clinical standards."
)
para(
    "Key limitations: (1) small benchmark dataset; (2) dataset version and missingness can affect results; (3) 0.5 threshold is illustrative; (4) confidence is a model probability and may be poorly calibrated; (5) no prospective or external validation; (6) no authentication or TLS in the local sample; (7) metrics are basic and no Grafana dashboard is included by default. The output must never be represented as medical advice."
)
page()

title("9. Conclusion and Submission Checklist")
para(
    "This repository provides the main components of a reproducible MLOps lifecycle: data acquisition, EDA, preprocessing, model comparison, MLflow tracking, model serialization, tests, CI automation, container packaging, Kubernetes deployment manifests, and basic monitoring. The key design decision is to serialize preprocessing with the estimator and choose the candidate based on cross-validation ROC-AUC rather than the holdout score. Actual metrics and deployment evidence must be generated by executing the workflow."
)
para("Before submitting, complete and verify every item below:")
bullets(
    [
        "Run the downloader, EDA, training, lint, and tests from a clean environment.",
        "Replace all [RUN RESULT] placeholders with values from artifacts/model_comparison.csv.",
        "Inspect and include the EDA, confusion-matrix, and ROC plots.",
        "Capture MLflow experiment runs and the successful GitHub Actions workflow.",
        "Build and run Docker; capture a successful /health and /predict response.",
        "Deploy to Minikube/Docker Desktop or a cloud cluster; capture pods and service output.",
        "Verify /metrics and capture API logs/monitoring evidence.",
        "Insert the real GitHub repository URL and deployed API URL or local access instructions.",
        "Record a short video showing data → training/tracking → tests/CI → container/API → deployment.",
        "Add student name/ID, review the report, and ensure you can explain the implementation.",
    ]
)
para(
    "Reference: UCI Machine Learning Repository, Heart Disease dataset (dataset ID 45), accessed through the ucimlrepo interface. Software references: scikit-learn, MLflow, FastAPI, Prometheus client, Docker, Kubernetes, pytest, and GitHub Actions documentation."
)
para(
    "Academic integrity statement: This implementation should be reviewed, executed, and personalized by the student. All reported metrics and evidence must reflect the student’s actual run. The student should be able to explain the workflow and design decisions independently."
)

# Footer with page field
for section in doc.sections:
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("MLOps Assignment 01 · Heart Disease Classification · Page ")
    fld = footer.add_run()
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    fld._r.append(begin)
    fld._r.append(instr)
    fld._r.append(separate)
    fld._r.append(text)
    fld._r.append(end)

doc.save(OUT)
print(OUT)
