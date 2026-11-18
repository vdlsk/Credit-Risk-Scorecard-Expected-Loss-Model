"""
Step 3b - Challenger model (LightGBM) on exactly the same train / test / 2014 samples.

Banks keep a simple, explainable scorecard as the champion and compare it with a
machine-learning challenger. The gap tells you how much predictive power the
scorecard gives up for transparency.
"""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from build_scorecard import CAT_VARS, NUM_VARS

ROOT = Path(__file__).resolve().parents[1]


def main():
    loans = pd.read_parquet(ROOT / "data" / "loans.parquet")
    samples = pd.read_parquet(ROOT / "data" / "scored_loans.parquet", columns=["sample", "pd"])
    df = pd.concat([loans.reset_index(drop=True), samples.reset_index(drop=True)], axis=1)  # same row order
    cats = {c: sorted(df[c].dropna().unique()) for c in CAT_VARS}
    X = df[NUM_VARS + CAT_VARS].assign(**{c: pd.Categorical(df[c], categories=cats[c]) for c in CAT_VARS})
    tr = df["sample"] == "train"
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(X[tr], df.loc[tr, "default"])
    out = {}
    for s in ["train", "test_intime", "oot_2014"]:
        k = df["sample"] == s
        out[s] = {"gini_challenger": 2 * roc_auc_score(df.loc[k, "default"], m.predict_proba(X[k])[:, 1]) - 1,
                  "gini_scorecard": 2 * roc_auc_score(df.loc[k, "default"], df.loc[k, "pd"]) - 1}
    imp = pd.Series(m.booster_.feature_importance("gain"), index=NUM_VARS + CAT_VARS).sort_values(ascending=False)
    out["top_features_by_gain"] = (imp / imp.sum()).head(10).round(3).to_dict()
    json.dump(out, open(ROOT / "results" / "challenger.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
