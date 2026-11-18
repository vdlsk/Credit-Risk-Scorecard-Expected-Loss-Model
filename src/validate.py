"""
Step 3 - Validate the PD scorecard the way a bank's model-validation team would.

  Discrimination : ROC-AUC, Gini (= 2*AUC - 1), KS statistic
  Calibration    : Brier score, predicted vs actual default rate by decile,
                   Hosmer-Lemeshow test, calibration-in-the-large
  Stability      : Population Stability Index (PSI) of the score, train vs 2014
  Benchmark      : Gini of Lending Club's own sub-grade on the same loans

Outputs: results/validation_metrics.json, results/score_bands.csv, results/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
COLORS = {"train": "#2a78d6", "test_intime": "#eb6834", "oot_2014": "#1baf7a"}
LABELS = {"train": "Train (2012-13)", "test_intime": "In-time test (2012-13)", "oot_2014": "Out-of-time (2014)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def hosmer_lemeshow(y, p, g=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["dec"] = pd.qcut(d["p"], g, labels=False, duplicates="drop")
    t = d.groupby("dec").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = (((t["obs"] - t["exp"]) ** 2) / (t["exp"] * (1 - t["exp"] / t["n"]))).sum()
    return float(hl), float(1 - stats.chi2.cdf(hl, len(t) - 2))


def psi(expected, actual, edges):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    df = pd.read_parquet(ROOT / "data" / "scored_loans.parquet")
    grade_rank = df["sub_grade"].rank(method="dense")  # A1=1 ... G5=35, higher = riskier
    m = {}
    for s in ["train", "test_intime", "oot_2014"]:
        d = df[df["sample"] == s]
        auc = roc_auc_score(d["default"], d["pd"])
        hl, hl_p = hosmer_lemeshow(d["default"], d["pd"])
        m[s] = {"n": int(len(d)), "default_rate": float(d["default"].mean()),
                "mean_pd": float(d["pd"].mean()), "auc": float(auc), "gini": float(2 * auc - 1),
                "ks": ks_stat(d["default"], d["pd"]), "brier": float(brier_score_loss(d["default"], d["pd"])),
                "hosmer_lemeshow": hl, "hl_p_value": hl_p,
                "lc_subgrade_gini": float(2 * roc_auc_score(d["default"], grade_rank[d.index]) - 1)}

    # Stability: score PSI using train deciles as the reference bands
    tr, oot = df[df["sample"] == "train"]["score"], df[df["sample"] == "oot_2014"]["score"]
    edges = np.unique(np.quantile(tr, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    m["psi_score_train_vs_oot"] = psi(tr, oot, edges)

    # Recalibration check on 2014: refit only the intercept (calibration-in-the-large)
    d = df[df["sample"] == "oot_2014"]
    logit = np.log(d["pd"] / (1 - d["pd"]))
    import statsmodels.api as sm
    rc = sm.GLM(d["default"], np.ones((len(d), 1)), offset=logit, family=sm.families.Binomial()).fit()
    m["oot_intercept_shift"] = float(rc.params.iloc[0])
    json.dump(m, open(RES / "validation_metrics.json", "w"), indent=1)

    # Score bands table (rank ordering) on the out-of-time sample
    bands = pd.cut(df["score"], [-np.inf, 525, 535, 545, 555, 565, 575, np.inf])
    sb = (df.assign(band=bands).groupby(["band", "sample"], observed=True)["default"]
            .agg(["size", "mean"]).unstack("sample"))
    sb.to_csv(RES / "score_bands.csv", float_format="%.4f")

    # ---------- figures ----------
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for s in ["train", "test_intime", "oot_2014"]:
        d = df[df["sample"] == s]
        fpr, tpr, _ = roc_curve(d["default"], d["pd"])
        ax.plot(fpr, tpr, lw=2, color=COLORS[s], label=f"{LABELS[s]}  Gini {m[s]['gini']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    ax.set(xlabel="False positive rate (goods rejected)", ylabel="True positive rate (bads caught)",
           title="ROC curve - PD scorecard")
    ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for s in ["test_intime", "oot_2014"]:
        d = df[df["sample"] == s].copy()
        d["dec"] = pd.qcut(d["pd"], 10, labels=False)
        t = d.groupby("dec").agg(pred=("pd", "mean"), obs=("default", "mean"))
        ax.plot(t["pred"] * 100, t["obs"] * 100, marker="o", ms=6, lw=2, color=COLORS[s], label=LABELS[s])
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, ls="--", lw=1, color=INK2, label="Perfect calibration")
    ax.set(xlabel="Predicted PD (%)", ylabel="Actual default rate (%)", title="Calibration by PD decile")
    ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    d = df[df["sample"] == "oot_2014"]
    bins = np.linspace(df["score"].quantile(0.005), df["score"].quantile(0.995), 40)
    ax.hist(d.loc[d["default"] == 0, "score"], bins, density=True, alpha=0.55, color="#2a78d6", label="Repaid (good)")
    ax.hist(d.loc[d["default"] == 1, "score"], bins, density=True, alpha=0.55, color="#eb6834", label="Defaulted (bad)")
    ax.set(xlabel="Score (higher = safer)", ylabel="Share of loans", yticks=[],
           title=f"Score separation, 2014 out-of-time  (KS = {m['oot_2014']['ks']:.3f})")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "score_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    t = sb["mean"]["oot_2014"] * 100
    ax.bar(range(len(t)), t.values, color="#2a78d6", width=0.7)
    for i, v in enumerate(t.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", color=INK, fontsize=9)
    names = ["<=525", "525-535", "535-545", "545-555", "555-565", "565-575", ">575"]
    ax.set_xticks(range(len(t)), names)
    ax.set(xlabel="Score band", ylabel="Actual default rate (%)", title="Default rate falls as score rises (2014 loans)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "default_rate_by_band.png", dpi=150); plt.close(fig)

    print(json.dumps(m, indent=1))
    print(sb)


if __name__ == "__main__":
    main()
