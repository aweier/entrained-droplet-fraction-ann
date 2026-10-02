"""Compact feed-forward ANN (8-32-16-1, tanh, sigmoid output) with 5-fold CV and SHAP.

Usage:  python src/ann_cv.py [--skip-shap]
Outputs: results/ann/ann_cv_metrics.csv, figures/ann_shap_*.png
"""
import argparse

import torch  # import before matplotlib/sklearn to avoid Windows DLL conflicts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, N_SPLITS, ROOT, SEED, TARGET_COL, load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim=8, hidden=(32, 16)):
        super().__init__()
        layers, prev = [], in_dim
        for k in hidden:
            fc = nn.Linear(prev, k)
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            layers += [fc, nn.Tanh()]
            prev = k
        out = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        self.body = nn.Sequential(*layers)
        self.out = nn.Sequential(out, nn.Sigmoid())  # keeps predictions in [0, 1]

    def forward(self, x):
        return self.out(self.body(x))


def train_fold(Xtr, ytr, Xte, epochs=1500, lr=1e-3, wd=1e-4, bs=128):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    Xt = torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xs = torch.tensor(Xte_s, dtype=torch.float32, device=DEVICE)

    model = MLP(in_dim=Xt.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    n = Xt.shape[0]
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for b in range(0, n, bs):
            sel = idx[b : b + bs]
            opt.zero_grad()
            lossf(model(Xt[sel]), yt[sel]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        yhat = model(Xs).cpu().numpy().ravel()
    return yhat, model, scaler, Xtr_s, Xte_s


def run_shap(model, scaler, Xtr_s, Xte_s, fig_dir):
    import shap

    bg = Xtr_s[np.random.default_rng(SEED).choice(len(Xtr_s), size=min(200, len(Xtr_s)), replace=False)]
    exp = Xte_s[:500]

    def predict_fn(x_np):
        with torch.no_grad():
            return model(torch.tensor(x_np, dtype=torch.float32, device=DEVICE)).cpu().numpy().ravel()

    shap_vals = shap.KernelExplainer(predict_fn, bg).shap_values(exp, nsamples="auto")
    # Plot against original (unscaled) feature values
    X_display = pd.DataFrame(scaler.inverse_transform(exp), columns=FEATURE_COLS)

    shap.summary_plot(shap_vals, X_display, plot_type="bar", show=False)
    plt.title("ANN Feature Importance (SHAP)")
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(fig_dir / "ann_shap_importance.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_vals, X_display, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "ann_shap_beeswarm.png", dpi=150)
    plt.close()

    shap.dependence_plot("Usg", shap_vals, X_display, interaction_index="rhog", show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "ann_shap_dependence_usg.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--epochs", type=int, default=1500)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = load_dataset()
    X = df[FEATURE_COLS].to_numpy().astype(np.float32)
    y = df[TARGET_COL].to_numpy().astype(np.float32).reshape(-1, 1)

    n_params = sum(p.numel() for p in MLP().parameters())
    print(f"ANN trainable parameters: {n_params}")

    rows, best = [], {"r2": -np.inf, "art": None}
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        yhat, model, scaler, Xtr_s, Xte_s = train_fold(X[tr], y[tr], X[te], epochs=args.epochs)
        yte = y[te].ravel()
        rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
        r2 = float(r2_score(yte, yhat))
        rows.append({"fold": fold, "rmse": rmse, "r2": r2})
        print(f"Fold {fold} | R2={r2:.4f}, RMSE={rmse:.4f}")
        if r2 > best["r2"]:
            best = {"r2": r2, "art": (model, scaler, Xtr_s, Xte_s)}

    res = pd.DataFrame(rows)
    print(
        f"ANN {N_SPLITS}-fold CV -> RMSE: {res.rmse.mean():.3f} +/- {res.rmse.std(ddof=1):.3f}, "
        f"R2: {res.r2.mean():.3f} +/- {res.r2.std(ddof=1):.3f}"
    )
    out = ROOT / "results" / "ann"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "ann_cv_metrics.csv", index=False)

    if not args.skip_shap:
        fig_dir = ROOT / "figures"
        fig_dir.mkdir(exist_ok=True)
        run_shap(*best["art"], fig_dir)


if __name__ == "__main__":
    main()

