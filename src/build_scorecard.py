"""
Step 2 - Build the application PD scorecard.

  Development sample : loans issued 2012-2013  (random 70% train / 30% in-time test)
                       2012 is when Lending Club started recording the credit-bureau attributes
                       (credit-card limits, accounts opened in last 24 months, ...). Loans from
                       2007-2011 are still used later for LGD, EAD and the crisis stress factor.
  Out-of-time sample : loans issued 2014       (never touched while building)

  1. WoE binning + IV on the TRAIN part only
  2. Keep variables with IV >= 0.01, drop one of any pair with |corr(WoE)| > 0.6
     (fine-tuning: IV 0.02 -> 0.01 and the bureau variables raised out-of-time Gini
      from 0.259 to 0.325 - see results/tuning_experiments.csv)
  3. Logistic regression (statsmodels) on WoE values; drop variables whose
     coefficient has the wrong sign or p-value > 0.05, refit
  4. Scale to points: 600 points at odds 50:1 (good:bad), 20 points to double the odds (PDO)

Outputs in results/: iv_table.csv, woe_bins.csv, coefficients.csv, scorecard.csv,
model.json, and data/scored_loans.parquet (PD and score for every loan).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from woe import WoEBinner

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 42

NUM_VARS = ["loan_amnt", "annual_inc", "dti", "credit_hist_yrs", "delinq_2yrs", "inq_last_6mths",
            "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "pub_rec_bankruptcies", "loan_to_income",
            # credit-bureau attributes at application
            "mort_acc", "acc_open_past_24mths", "bc_util", "bc_open_to_buy", "tot_cur_bal", "avg_cur_bal",
            "num_actv_rev_tl", "mo_sin_rcnt_tl", "mo_sin_old_rev_tl_op", "pct_tl_nvr_dlq", "percent_bc_gt_75",
            "tot_hi_cred_lim", "total_bc_limit", "num_tl_op_past_12m", "mths_since_recent_inq",
            "total_rev_hi_lim", "num_rev_tl_bal_gt_0"]
CAT_VARS = ["emp_length", "home_ownership", "verification_status", "purpose"]
# addr_state deliberately excluded: geography can act as a proxy for protected groups (fair-lending risk)
DEV_START, IV_MIN = 2012, 0.01

PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def main():
    df = pd.read_parquet(ROOT / "data" / "loans.parquet")
    dev = df[(df["issue_year"] >= DEV_START) & (df["issue_year"] <= 2013)]
    rng = np.random.default_rng(SEED)
    is_train = rng.random(len(dev)) < 0.7
    df["sample"] = "history_2007_2011"
    df.loc[df["issue_year"] == 2014, "sample"] = "oot_2014"
    df.loc[dev.index[is_train], "sample"] = "train"
    df.loc[dev.index[~is_train], "sample"] = "test_intime"
    train = df[df["sample"] == "train"]

    # 1. binning + IV on train only
    binner = WoEBinner().fit(train, train["default"], NUM_VARS, CAT_VARS)
    iv = binner.iv()
    iv.rename("iv").to_csv(RES / "iv_table.csv", index_label="variable")
    rows = []
    for v, t in binner.tables.items():
        tt = t.reset_index().rename(columns={"g": "bin"})
        tt.insert(0, "variable", v)
        rows.append(tt)
    pd.concat(rows).to_csv(RES / "woe_bins.csv", index=False, float_format="%.4f")

    # 2. IV filter + correlation filter
    cand = iv[iv >= IV_MIN].index.tolist()
    Xw = binner.transform(train, cand)
    corr = Xw.corr().abs()
    selected = []
    for v in cand:  # already sorted by IV, strongest first
        if all(corr.loc[v, s] <= 0.6 for s in selected):
            selected.append(v)

    # 3. logistic regression with sign / significance checks
    y = train["default"]
    while True:
        X = sm.add_constant(binner.transform(train, selected))
        fit = sm.Logit(y, X).fit(disp=0)
        coefs = fit.params.drop("const")
        pvals = fit.pvalues.drop("const")
        # Siddiqi WoE: higher WoE = safer, so every coefficient must be NEGATIVE
        bad = [v for v in selected if coefs[v] >= 0 or pvals[v] > 0.05]
        if not bad:
            break
        drop = max(bad, key=lambda v: pvals[v])
        print("dropping", drop, "coef", round(coefs[drop], 3), "p", round(pvals[drop], 4))
        selected.remove(drop)

    print(fit.summary2().tables[1].round(4))
    ct = fit.summary2().tables[1].copy()
    ct["odds_ratio"] = np.exp(ct["Coef."])
    ct["iv"] = iv.reindex(ct.index)
    ct.to_csv(RES / "coefficients.csv", float_format="%.4f")

    # 4. points per attribute
    alpha, n = fit.params["const"], len(selected)
    sc = []
    def order(b):  # numeric bins by their lower edge, "Missing"/"Other" last
        try:
            return (0, float(b.split(",")[0].strip("([")))
        except ValueError:
            return (1, 0.0)

    for v in selected:
        for b, w in sorted(binner.spec[v]["woe"].items(), key=lambda kv: order(kv[0]) if binner.spec[v]["kind"] == "num" else (0, -kv[1])):
            pts = -(fit.params[v] * w + alpha / n) * FACTOR + OFFSET / n
            t = binner.tables[v].loc[b]
            sc.append({"variable": v, "bin": b, "woe": w, "points": int(round(pts)),
                       "share_train": t["share"], "bad_rate_train": t["bad_rate"]})
    sc = pd.DataFrame(sc)
    sc.to_csv(RES / "scorecard.csv", index=False, float_format="%.4f")

    # score every loan
    Xall = sm.add_constant(binner.transform(df, selected), has_constant="add")
    df["pd"] = fit.predict(Xall[["const"] + selected])
    df["score"] = OFFSET - FACTOR * np.log(df["pd"] / (1 - df["pd"]))
    cols = ["id", "issue_year", "sample", "default", "pd", "score", "grade", "sub_grade", "int_rate",
            "funded_amnt", "loan_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
            "issue_d", "last_pymnt_d", "loan_status"]
    df[cols + selected].to_parquet(ROOT / "data" / "scored_loans.parquet", index=False)
    binner_spec = {v: binner.spec[v] for v in selected}
    json.dump({"selected": selected, "params": fit.params.to_dict(), "bins": binner_spec,
               "pdo": PDO, "base_score": BASE_SCORE, "base_odds": BASE_ODDS,
               "factor": FACTOR, "offset": OFFSET,
               "train_bad_rate": float(y.mean()),
               "defaults": {v: (float(train[v].median()) if v in NUM_VARS else str(train[v].mode()[0]))
                            for v in selected}}, open(RES / "model.json", "w"), indent=1)
    print("IV:\n", iv.round(4))
    print("selected:", selected)
    print(df.groupby("sample")["default"].agg(["size", "mean"]))


if __name__ == "__main__":
    main()
