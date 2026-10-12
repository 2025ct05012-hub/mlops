"""Create a lightweight notebook that documents the project workflow."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Heart Disease Classification — MLOps Workflow\n",
            "This notebook is a companion walkthrough. Run the scripts from the repository root for the full tracked workflow. Educational use only; not for clinical decisions.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "from src.data import load_dataset\n",
            "ROOT = Path.cwd()\n",
            "data_path = ROOT / 'data/raw/heart_disease.csv'\n",
            "X, y = load_dataset(data_path)\n",
            "print('features:', X.shape, 'target counts:', y.value_counts().to_dict())",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## EDA\n",
            "The EDA script exports missingness, summary statistics, histograms, class balance, and a correlation heatmap to `artifacts/eda/`.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["from scripts.eda import main as run_eda\n", "run_eda()"],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Training and experiment tracking\n",
            "Run the training module to compare Logistic Regression and Random Forest using stratified 5-fold cross-validation. MLflow stores run parameters, metrics, plots, and model artifacts.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.train import main as run_training\n",
            "run_training()\n",
            "comparison = pd.read_csv('artifacts/model_comparison.csv')\n",
            "comparison",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Interpretation and limitations\n",
            "ROC-AUC evaluates ranking across thresholds; precision measures the fraction of positive predictions that are correct; recall measures the fraction of positive cases found. For health-related screening, false negatives and false positives both matter. This public benchmark dataset is small and not representative of all populations; no clinical use is appropriate.",
        ],
    },
]
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "notebooks/heart_disease_mlops.ipynb").write_text(json.dumps(notebook, indent=2))
