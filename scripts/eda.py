"""Generate reproducible EDA summaries and plots."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data import FEATURES, load_dataset


def main() -> None:
    raw = ROOT / "data/raw/heart_disease.csv"
    out = ROOT / "artifacts/eda"
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(raw)
    df = X.copy(); df["target"] = y
    df.describe(include="all").to_csv(out / "summary_statistics.csv")
    pd.DataFrame({"missing_count": X.isna().sum(), "missing_pct": X.isna().mean() * 100}).to_csv(out / "missingness.csv")
    fig, ax = plt.subplots(figsize=(6, 4)); sns.countplot(x=y, ax=ax)
    ax.set(title="Target Class Balance", xlabel="Target (0 = no disease, 1 = disease)", ylabel="Count")
    fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=160); plt.close(fig)
    df[FEATURES].hist(figsize=(14, 10), bins=18, edgecolor="black")
    plt.suptitle("Feature Distributions", y=1.01); plt.tight_layout(); plt.savefig(out / "feature_histograms.png", dpi=160); plt.close()
    fig, ax = plt.subplots(figsize=(12, 9)); sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Feature and Target Correlation Heatmap"); fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=160); plt.close(fig)
    print(f"EDA outputs written to {out}")

if __name__ == "__main__": main()
