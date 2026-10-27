"""
HW2: Multiple Linear Regression on Kaggle datasets (14-39 features), CRISP-DM
Problem 1: King County House Sales  -> predict price  (18 features)
Problem 2: Car Price Prediction     -> predict price  (24 features)

Usage:
    python hw2_analysis.py                     # run both problems
    python hw2_analysis.py --problem house     # run one problem
    python hw2_analysis.py --url <csv_url> --target <column> [--drop a,b] [--log]   # any dataset
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PROBLEMS = {
    "house": {
        "name": "King County House Sales",
        "kaggle": "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction",
        "url": "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv",
        "local": os.path.join(HERE, "data", "kc_house_data.csv"),
        "target": "price",
        # id: row identifier, date: sale timestamp, zipcode: a code, not a quantity
        "drop": ["id", "date", "zipcode"],
        "log": True,
        # size variables are right-skewed; log-log keeps exp() of large houses from exploding
        "log_features": ["sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15"],
    },
    "car": {
        "name": "Car Price Prediction",
        "kaggle": "https://www.kaggle.com/datasets/hellbuoy/car-price-prediction",
        "url": "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv",
        "local": os.path.join(HERE, "data", "CarPrice_Assignment.csv"),
        "target": "price",
        # car_ID: row identifier, CarName: 147 distinct model names for 205 rows
        "drop": ["car_ID", "CarName"],
        "log": True,
    },
}


# ---------- 1. Business / 2. Data Understanding ----------
def load(cfg):
    path = cfg.get("local")
    df = pd.read_csv(path if path and os.path.exists(path) else cfg["url"])
    print(f"\n=== {cfg['name']} ===  shape={df.shape}  "
          f"features={df.shape[1] - 1}  missing={int(df.isna().sum().sum())}")
    return df


# ---------- 3. Data Preparation ----------
def prepare(df, cfg):
    df = df.dropna()
    y = df[cfg["target"]].astype(float)
    X = df.drop(columns=[cfg["target"]] + [c for c in cfg.get("drop", []) if c in df.columns])
    # one-hot encode categorical columns, drop first level to avoid dummy-variable trap
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    for c in cfg.get("log_features", []):
        X[c] = np.log1p(X[c])
    return X, y


def drop_collinear(X, tol=1e-8):
    """Greedily keep columns that add rank; returns (kept, dropped)."""
    kept, dropped = [], []
    for c in X.columns:
        A = X[kept + [c]].values
        A = A - A.mean(axis=0)
        if np.linalg.matrix_rank(A, tol=tol * max(1.0, np.abs(A).max())) == len(kept) + 1:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def vif(X):
    out = {}
    for c in X.columns:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c]) if len(others.columns) else 0.0
        out[c] = float(1 / max(1 - r2, 1e-12))
    return out


# ---------- OLS with inference (intervals) ----------
class OLS:
    def __init__(self, X, y):
        self.cols = list(X.columns)
        A = np.column_stack([np.ones(len(X)), X.values])
        self.inv = np.linalg.pinv(A.T @ A)
        self.beta = self.inv @ A.T @ y.values
        resid = y.values - A @ self.beta
        self.n, self.k = A.shape
        self.df = self.n - self.k
        self.sigma2 = resid @ resid / self.df
        self.se = np.sqrt(np.diag(self.sigma2 * self.inv))
        self.p = 2 * stats.t.sf(np.abs(self.beta / self.se), self.df)
        sst = ((y.values - y.values.mean()) ** 2).sum()
        self.r2 = 1 - resid @ resid / sst
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / self.df
        self.aic = self.n * np.log(resid @ resid / self.n) + 2 * self.k
        self.resid = resid

    def predict(self, X, alpha=0.05):
        """Point prediction, confidence interval (mean) and prediction interval (new obs)."""
        A = np.column_stack([np.ones(len(X)), X[self.cols].values])
        yhat = A @ self.beta
        h = np.einsum("ij,jk,ik->i", A, self.inv, A)       # leverage x0' (X'X)^-1 x0
        t = stats.t.ppf(1 - alpha / 2, self.df)
        ci = t * np.sqrt(self.sigma2 * h)
        pi = t * np.sqrt(self.sigma2 * (1 + h))
        return yhat, (yhat - ci, yhat + ci), (yhat - pi, yhat + pi)


# ---------- Feature selection (training data only) ----------
def backward_elimination(X, y, sl=0.05):
    cols = list(X.columns)
    while len(cols) > 1:
        p = pd.Series(OLS(X[cols], y).p[1:], index=cols)
        if p.max() <= sl:
            break
        cols.remove(p.idxmax())
    return cols


def forward_stepwise(X, y):
    """Add the feature that lowers AIC most; returns (selected, path of (feature, aic))."""
    cols, path, best_aic = [], [], np.inf
    remaining = list(X.columns)
    while remaining:
        aic, c = min((OLS(X[cols + [c]], y).aic, c) for c in remaining)
        path.append((c, float(aic)))
        if aic < best_aic:
            best_aic, n_best = aic, len(path)
        cols.append(c)
        remaining.remove(c)
    return [c for c, _ in path[:n_best]], path


def cv_rmse(X, y, cols, log):
    """5-fold CV RMSE in original units of y."""
    kf = KFold(5, shuffle=True, random_state=SEED)
    errs = []
    for tr, va in kf.split(X):
        m = LinearRegression().fit(X.iloc[tr][cols], y.iloc[tr])
        p = m.predict(X.iloc[va][cols])
        yt = y.iloc[va].values
        if log:
            p, yt = np.exp(p), np.exp(yt)
        errs.append(np.sqrt(np.mean((yt - p) ** 2)))
    return float(np.mean(errs))


def best_k(X, y, ranked, log):
    """Pick k (number of top-ranked features) with the lowest CV RMSE."""
    scores = [cv_rmse(X, y, ranked[:k], log) for k in range(1, len(ranked) + 1)]
    return ranked[:int(np.argmin(scores)) + 1], scores


def select_features(X, y, log):
    methods = {"All-in": list(X.columns)}
    methods["Backward Elimination (p<0.05)"] = backward_elimination(X, y)
    methods["Forward Stepwise (AIC)"], path = forward_stepwise(X, y)

    f, _ = f_regression(X, y)
    ranked = list(X.columns[np.argsort(-f)])
    methods["SelectKBest (F-test, k by CV)"], _ = best_k(X, y, ranked, log)

    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(StandardScaler().fit_transform(X), y)
    ranked = list(X.columns[np.argsort(rfe.ranking_)])
    methods["RFE (k by CV)"], _ = best_k(X, y, ranked, log)

    lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=50000)).fit(X, y)
    methods["LassoCV (non-zero coef)"] = [c for c, b in zip(X.columns, lasso[-1].coef_) if abs(b) > 1e-10]
    return methods, path


# ---------- 5. Evaluation ----------
def metrics(y_true, y_pred):
    n = len(y_true)
    sse = float(((y_true - y_pred) ** 2).sum())
    r2 = 1 - sse / float(((y_true - y_true.mean()) ** 2).sum())
    return {"R2": round(r2, 4),
            "MAE": round(float(np.abs(y_true - y_pred).mean()), 2),
            "RMSE": round(float(np.sqrt(sse / n)), 2), "MSE": round(sse / n, 2)}


def assumption_tests(fit, X_tr):
    r = fit.resid
    A = np.column_stack([np.ones(len(X_tr)), X_tr[fit.cols].values])
    # Breusch-Pagan: regress squared residuals on X, LM = n * R^2
    u2 = r ** 2
    g, *_ = np.linalg.lstsq(A, u2, rcond=None)
    bp_r2 = 1 - ((u2 - A @ g) ** 2).sum() / ((u2 - u2.mean()) ** 2).sum()
    bp_p = stats.chi2.sf(len(r) * bp_r2, A.shape[1] - 1)
    jb = stats.jarque_bera(r)
    return {"Durbin_Watson": round(float((np.diff(r) ** 2).sum() / (r ** 2).sum()), 3),
            "Jarque_Bera_p": float(jb.pvalue), "Breusch_Pagan_p": float(bp_p)}


def run(key, cfg, outdir):
    df = load(cfg)
    target, log = cfg["target"], cfg.get("log", False)
    X, y_raw = prepare(df, cfg)
    y = np.log(y_raw) if log else y_raw

    # correlation with the target
    corr = X.corrwith(y_raw).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(corr))))
    ax.barh(corr.index[::-1], corr.values[::-1], color=["#2f6fdf" if v >= 0 else "#c2453b" for v in corr.values[::-1]])
    ax.set_xlabel(f"Pearson r with {target}")
    ax.set_title(f"{cfg['name']} - correlation with target")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{key}_corr.png"), dpi=110)
    plt.close(fig)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    # checked on the training set: a rare dummy level can be constant there
    kept, collinear = drop_collinear(X_tr)
    X, X_tr, X_te = X[kept], X_tr[kept], X_te[kept]
    if collinear:
        print("dropped perfectly collinear columns:", collinear)
    y_te_raw = np.exp(y_te) if log else y_te
    back = np.exp if log else (lambda v: v)

    # 4. Modeling: compare feature-selection methods
    methods, path = select_features(X_tr, y_tr, log)
    table = {}
    for name, cols in methods.items():
        fit = OLS(X_tr[cols], y_tr)
        pred = back(fit.predict(X_te)[0])
        table[name] = {"n_features": len(cols), "cv_RMSE": round(cv_rmse(X_tr, y_tr, cols, log), 2),
                       "train_Adj_R2": round(float(fit.adj_r2), 4), "aic": round(float(fit.aic), 1),
                       **metrics(y_te_raw.values, pred),
                       "features": cols}
    print(pd.DataFrame(table).T.drop(columns="features").to_string())

    # final model: lowest CV RMSE on training data; ties go to the simpler model
    best = min(table, key=lambda m: (table[m]["cv_RMSE"], table[m]["n_features"]))
    cols = methods[best]
    fit = OLS(X_tr[cols], y_tr)
    yhat, ci, pi = fit.predict(X_te)
    yhat, ci, pi = back(yhat), (back(ci[0]), back(ci[1])), (back(pi[0]), back(pi[1]))
    yt = y_te_raw.values
    coverage = float(((yt >= pi[0]) & (yt <= pi[1])).mean())
    print(f"final model: {best}  ({len(cols)} features)  95% PI coverage on test = {coverage:.1%}")

    # forward path: #features vs CV RMSE
    ks = range(1, len(path) + 1)
    order = [c for c, _ in path]
    cv_path = [cv_rmse(X_tr, y_tr, order[:k], log) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, cv_path, "o-", color="#2f6fdf")
    ax.axvline(len(methods["Forward Stepwise (AIC)"]), color="#c2453b", ls="--", label="AIC minimum")
    ax.set_xlabel("number of features (forward order)")
    ax.set_ylabel(f"5-fold CV RMSE ({target})")
    ax.set_title(f"{cfg['name']} - feature count vs error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{key}_fs_curve.png"), dpi=110)
    plt.close(fig)

    # prediction plot with 95% CI and PI (test samples sorted by prediction)
    o = np.argsort(yhat)
    xs = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(10, 5))
    big = len(o) > 200
    ax.scatter(xs, yt[o], s=3 if big else 16, color="#c2453b", alpha=0.3 if big else 0.7, label="actual")
    ax.fill_between(xs, pi[0][o], pi[1][o], color="#2f6fdf", alpha=0.25, label="95% prediction interval", zorder=2)
    ax.fill_between(xs, ci[0][o], ci[1][o], color="#2f6fdf", alpha=0.6, label="95% confidence interval", zorder=3)
    ax.plot(xs, yhat[o], color="#1d2433", lw=1.2, label="predicted", zorder=4)
    ax.set_xlabel("test samples (sorted by predicted value)")
    ax.set_ylabel(target)
    if log:
        ax.set_yscale("log")
    ax.set_title(f"{cfg['name']} - {best}: prediction with 95% CI / PI "
                 f"(PI coverage {coverage:.1%})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{key}_pred_interval.png"), dpi=110)
    plt.close(fig)

    # diagnostics: actual vs predicted, residuals, Q-Q
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].scatter(yt, yhat, s=8, alpha=0.5)
    lo, hi = min(yt.min(), yhat.min()), max(yt.max(), yhat.max())
    axs[0].plot([lo, hi], [lo, hi], "r--")
    axs[0].set(xlabel="actual", ylabel="predicted", title="actual vs predicted (test)")
    res_tr = fit.resid
    fit_tr = y_tr.values - res_tr
    axs[1].scatter(fit_tr, res_tr, s=6, alpha=0.4, color="#1a8f5a")
    axs[1].axhline(0, color="gray")
    axs[1].set(xlabel="fitted" + (" (log)" if log else ""), ylabel="residual", title="residuals vs fitted (train)")
    stats.probplot(res_tr, plot=axs[2])
    axs[2].set_title("normal Q-Q of residuals (train)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{key}_diagnostics.png"), dpi=110)
    plt.close(fig)

    v = vif(X_tr[cols])
    return {
        "name": cfg["name"], "kaggle": cfg.get("kaggle"), "url": cfg["url"], "target": target,
        "log_target": log, "n_rows": int(len(df)), "n_raw_features": int(df.shape[1] - 1),
        "dropped_columns": cfg.get("drop", []), "collinear_removed": collinear,
        "n_model_columns": int(X.shape[1]), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "methods": table, "final_model": best, "pi_coverage": round(coverage, 4),
        "assumptions": assumption_tests(fit, X_tr),
        "coef": {c: {"beta": float(b), "se": float(s), "p": float(p)}
                 for c, b, s, p in zip(["(Intercept)"] + cols, fit.beta, fit.se, fit.p)},
        "vif": {c: round(x, 2) for c, x in sorted(v.items(), key=lambda kv: -kv[1])},
        "forward_path": [{"feature": c, "aic": round(a, 1), "cv_RMSE": round(r, 2)}
                         for (c, a), r in zip(path, cv_path)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(PROBLEMS), help="run only one problem")
    ap.add_argument("--url", help="any CSV URL")
    ap.add_argument("--target", help="target column (with --url; default = last column)")
    ap.add_argument("--drop", default="", help="comma-separated columns to exclude (with --url)")
    ap.add_argument("--log", action="store_true", help="model log(target) (with --url)")
    args = ap.parse_args()

    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)

    if args.url:
        df_head = pd.read_csv(args.url, nrows=1)
        jobs = {"custom": {"name": "Custom", "url": args.url, "log": args.log,
                           "target": args.target or df_head.columns[-1],
                           "drop": [c for c in args.drop.split(",") if c]}}
    else:
        jobs = {args.problem: PROBLEMS[args.problem]} if args.problem else PROBLEMS

    summary = {k: run(k, cfg, outdir) for k, cfg in jobs.items()}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved figures and summary.json to {outdir}")


if __name__ == "__main__":
    main()
