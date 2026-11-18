"""
Fine-tuning experiments: what raises out-of-time (2014) Gini?

Each experiment changes ONE thing versus the first scorecard (E0), and the
LightGBM rows show the ceiling a black-box model reaches on the same features.
Result table -> results/tuning_experiments.csv. Chosen setup = E6 (used in build_scorecard.py).
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import woe
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

df = pd.read_parquet(ROOT / "data" / "loans.parquet")
BASE_NUM = ["loan_amnt","annual_inc","dti","credit_hist_yrs","delinq_2yrs","inq_last_6mths","mths_since_last_delinq",
            "open_acc","pub_rec","revol_bal","revol_util","total_acc","pub_rec_bankruptcies","loan_to_income"]
BASE_CAT = ["emp_length","home_ownership","verification_status","purpose"]
BUREAU = ["mort_acc","acc_open_past_24mths","bc_util","bc_open_to_buy","tot_cur_bal","avg_cur_bal","num_actv_rev_tl",
          "mo_sin_rcnt_tl","mo_sin_old_rev_tl_op","pct_tl_nvr_dlq","percent_bc_gt_75","tot_hi_cred_lim","total_bc_limit",
          "num_tl_op_past_12m","mths_since_recent_inq","total_rev_hi_lim","num_rev_tl_bal_gt_0"]
oot = df[df.issue_year == 2014]
g = lambda y, p: 2 * roc_auc_score(y, p) - 1

def split(first_year):
    dev = df[(df.issue_year <= 2013) & (df.issue_year >= first_year)]
    m = np.random.default_rng(42).random(len(dev)) < 0.7
    return dev[m], dev[~m]

def scorecard(tr, te, num, cat, iv_min=0.02, min_share=0.05, corr_max=0.6):
    woe.MIN_SHARE = min_share
    b = woe.WoEBinner().fit(tr, tr["default"], num, cat)
    iv = b.iv(); cand = iv[iv >= iv_min].index.tolist()
    X = b.transform(tr, cand); c = X.corr().abs(); sel = []
    for v in cand:
        if all(c.loc[v, s] <= corr_max for s in sel): sel.append(v)
    while True:
        f = sm.Logit(tr["default"], sm.add_constant(b.transform(tr, sel))).fit(disp=0)
        bad = [v for v in sel if f.params[v] >= 0 or f.pvalues[v] > 0.05]
        if not bad: break
        sel.remove(max(bad, key=lambda v: f.pvalues[v]))
    pr = lambda d: f.predict(sm.add_constant(b.transform(d, sel), has_constant="add")[["const"] + sel])
    return sel, g(te["default"], pr(te)), g(oot["default"], pr(oot)), iv

def gbm(tr, te, num, cat):
    feats = num + cat
    prep = lambda d: d[feats].assign(**{c: d[c].astype("category") for c in cat})
    cats = {c: pd.api.types.union_categoricals([tr[c].astype("category"), oot[c].astype("category")]).categories for c in cat}
    prep = lambda d: d[feats].assign(**{c: pd.Categorical(d[c], categories=cats[c]) for c in cat})
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(prep(tr), tr["default"])
    return g(te["default"], m.predict_proba(prep(te))[:, 1]), g(oot["default"], m.predict_proba(prep(oot))[:, 1])


EXPERIMENTS = [
    ("E0 first version: 2007-13 loans, application vars, IV>=0.02", 2007, BASE_NUM, BASE_CAT, {}),
    ("E1 + finer bins (2.5%), IV>=0.01, + state", 2007, BASE_NUM, BASE_CAT + ["addr_state"], dict(iv_min=0.01, min_share=0.025)),
    ("E2 2012-13 loans only, application vars", 2012, BASE_NUM, BASE_CAT, {}),
    ("E3 2012-13 + bureau vars", 2012, BASE_NUM + BUREAU, BASE_CAT, {}),
    ("E4 2012-13 + bureau vars, IV>=0.01, + state", 2012, BASE_NUM + BUREAU, BASE_CAT + ["addr_state"], dict(iv_min=0.01, min_share=0.025)),
    ("E5 2012-13 + bureau vars, finer bins (2.5%), IV>=0.01", 2012, BASE_NUM + BUREAU, BASE_CAT, dict(iv_min=0.01, min_share=0.025)),
    ("E6 2012-13 + bureau vars, IV>=0.01  <- CHOSEN", 2012, BASE_NUM + BUREAU, BASE_CAT, dict(iv_min=0.01)),
]
rows = []
for name, start, num, cat, kw in EXPERIMENTS:
    tr, te = split(start)
    s = scorecard(tr, te, num, cat, **kw)
    rows.append((name, "scorecard", len(s[0]), s[1], s[2]))
    print(rows[-1])
for name, start, num, cat in [("C1 LightGBM, 2007-13, application vars", 2007, BASE_NUM, BASE_CAT),
                              ("C2 LightGBM, 2012-13, + bureau vars", 2012, BASE_NUM + BUREAU, BASE_CAT)]:
    tr, te = split(start)
    rows.append((name, "challenger", None, *gbm(tr, te, num, cat)))
    print(rows[-1])
out = pd.DataFrame(rows, columns=["experiment", "model", "n_vars", "gini_intime_test", "gini_oot_2014"]).round(3)
out.to_csv(ROOT / "results" / "tuning_experiments.csv", index=False)
print(out.to_string(index=False))
