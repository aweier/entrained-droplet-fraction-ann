"""Builds results/model_comparison.csv and figures/model_comparison.png from recorded CV outputs."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tab = pd.read_csv(next((ROOT / "results" / "tabnet").glob("tabnet_cv_metrics_*.csv")))
classic = pd.read_csv(ROOT / "results" / "classic_ml" / "classic_ml_cv_results.csv")
def raw(name):
    return classic[(classic.Method == name) & (classic.Feature_Set == "Raw")].iloc[0]

lin, rf, gb = raw("Linear Regression"), raw("Random Forest"), raw("Gradient Boosting")

# ANN values are the 5-fold CV summary printed by notebooks/ANN_capstone.ipynb
rows = [
    ("Linear Regression", lin.R2_mean, lin.R2_std, lin.RMSE_mean, lin.RMSE_std),
    ("TabNet", tab.r2.mean(), tab.r2.std(ddof=1), tab.rmse.mean(), tab.rmse.std(ddof=1)),
    ("Random Forest", rf.R2_mean, rf.R2_std, rf.RMSE_mean, rf.RMSE_std),
    ("Gradient Boosting", gb.R2_mean, gb.R2_std, gb.RMSE_mean, gb.RMSE_std),
    ("ANN (32-16, tanh)", 0.896, 0.011, 0.093, 0.006),
]
df = pd.DataFrame(rows, columns=["model", "r2_mean", "r2_std", "rmse_mean", "rmse_std"]).round(3)
df.to_csv(ROOT / "results" / "model_comparison.csv", index=False)
print(df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = ["#9aa5b1", "#e8a33d", "#38a169", "#2f855a", "#2b6cb0"]
for ax, (m, s, title) in zip(axes, [("r2_mean", "r2_std", "R² (higher is better)"), ("rmse_mean", "rmse_std", "RMSE (lower is better)")]):
    ax.bar(df.model, df[m], yerr=df[s], color=colors, capsize=4)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=15)
    for i, v in enumerate(df[m]):
        ax.text(i, v / 2, f"{v:.2f}", ha="center", color="white", fontweight="bold")
fig.suptitle("5-fold cross-validated performance")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "model_comparison.png", dpi=150)
