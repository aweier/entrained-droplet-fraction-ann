"""Random vs. grouped-by-diameter cross-validation for Linear, RF, GBM and the ANN.

Random K-fold can leak near-duplicate experimental conditions between train and test.
Grouping by pipe diameter (D) tests generalization to unseen diameters.

Schemes: random KFold(5), GroupKFold(5) on D, and leave-one-diameter-out (LODO).
Usage:  python src/grouped_cv.py [--epochs 1500]
Outputs: results/grouped_cv_summary.csv, results/grouped_cv_by_diameter.csv,
         figures/grouped_cv_comparison.png
"""
import argparse

import torch  # import before sklearn/matplotlib to avoid Windows DLL conflicts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ann_cv import train_fold
from config import FEATURE_COLS, N_SPLITS, ROOT, SEED, TARGET_COL, load_dataset


def predict_sklearn(make, Xtr, ytr, Xte):
    return make().fit(Xtr, ytr.ravel()).predict(Xte)


def predict_ann(epochs):
    def f(Xtr, ytr, Xte):
        return train_fold(Xtr, ytr, Xte, epochs=epochs)[0]

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    args = ap.parse_args()

    df = load_dataset()
    X = df[FEATURE_COLS].to_numpy().astype(np.float32)
    y = df[TARGET_COL].to_numpy().astype(np.float32).reshape(-1, 1)
    groups = df["D"].to_numpy()

    models = {
        "Linear Regression": lambda Xtr, ytr, Xte: predict_sklearn(
            lambda: Pipeline([("s", StandardScaler()), ("m", LinearRegression())]), Xtr, ytr, Xte
        ),
        "Random Forest": lambda Xtr, ytr, Xte: predict_sklearn(
            lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1), Xtr, ytr, Xte
        ),
        "Gradient Boosting": lambda Xtr, ytr, Xte: predict_sklearn(
            lambda: GradientBoostingRegressor(n_estimators=200, random_state=SEED), Xtr, ytr, Xte
        ),
        "ANN (32-16, tanh)": predict_ann(args.epochs),
    }
    schemes = {
        "Random 5-fold": list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X)),
        "Group 5-fold (by D)": list(GroupKFold(N_SPLITS).split(X, y, groups)),
        f"Leave-one-diameter-out ({len(np.unique(groups))} folds)": [
            (np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)
        ],
    }

    summary, by_d = [], []
    for sname, splits in schemes.items():
        for mname, predict in models.items():
            oof = np.zeros(len(y))
            fold_r2, fold_rmse = [], []
            for tr, te in splits:
                pred = np.asarray(predict(X[tr], y[tr], X[te])).ravel()
                oof[te] = pred
                if len(te) >= 10:  # R2 is unstable on tiny folds
                    fold_r2.append(r2_score(y[te], pred))
                    fold_rmse.append(mean_squared_error(y[te], pred) ** 0.5)
                if sname.startswith("Leave"):
                    by_d.append({"model": mname, "D": groups[te][0], "n": len(te),
                                 "rmse": mean_squared_error(y[te], pred) ** 0.5})
            row = {
                "scheme": sname, "model": mname,
                "pooled_r2": r2_score(y, oof),
                "pooled_rmse": mean_squared_error(y, oof) ** 0.5,
                "fold_r2_mean": np.mean(fold_r2), "fold_r2_std": np.std(fold_r2, ddof=1),
                "fold_rmse_mean": np.mean(fold_rmse), "fold_rmse_std": np.std(fold_rmse, ddof=1),
            }
            summary.append(row)
            print(f"{sname:38s} {mname:20s} pooled R2={row['pooled_r2']:.3f} RMSE={row['pooled_rmse']:.3f}", flush=True)

    s = pd.DataFrame(summary).round(4)
    s.to_csv(ROOT / "results" / "grouped_cv_summary.csv", index=False)
    pd.DataFrame(by_d).round(4).to_csv(ROOT / "results" / "grouped_cv_by_diameter.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = list(models)
    w = 0.26
    colors = ["#2b6cb0", "#e8a33d", "#c05621"]
    for i, (sname, c) in enumerate(zip(schemes, colors)):
        v = [s[(s.scheme == sname) & (s.model == m)].pooled_rmse.iloc[0] for m in order]
        ax.bar(np.arange(len(order)) + (i - 1) * w, v, w, label=sname, color=c)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("Pooled out-of-fold RMSE (lower is better)")
    ax.set_title("Generalization to unseen pipe diameters")
    ax.axhline(0.2, color="gray", ls=":", lw=0.8)
    ax.set_ylim(0, 0.45)
    for p_ in ax.patches:
        if p_.get_height() > 0.45:
            ax.text(p_.get_x() + p_.get_width() / 2, 0.43, f"{p_.get_height():.1f}", ha="center", va="top", fontsize=8, color="white")

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "grouped_cv_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
