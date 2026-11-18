"""
Streamlit app: score a new loan application with the PD scorecard and show its
expected credit loss and Basel capital.

Run:  streamlit run app/app.py      (after running the pipeline, see README)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.load(open(ROOT / "results" / "model.json"))
ECL = json.load(open(ROOT / "results" / "ecl_capital.json"))["parameters"]
CARD = pd.read_csv(ROOT / "results" / "scorecard.csv")
VAL = json.load(open(ROOT / "results" / "validation_metrics.json"))


def woe_of(var, value):
    spec = MODEL["bins"][var]
    if spec["kind"] == "num":
        if value is None or pd.isna(value):
            label = "Missing"
        else:
            label = str(pd.cut(pd.Series([value]), spec["edges"], include_lowest=True).astype(str)[0])
    else:
        label = value if value in spec["levels"] else "Other"
    return label, spec["woe"].get(label, 0.0)


def score_application(app: dict, recalibrate: bool):
    z = MODEL["params"]["const"]
    rows = []
    for v in MODEL["selected"]:
        label, w = woe_of(v, app[v])
        z += MODEL["params"][v] * w
        pts = CARD[(CARD["variable"] == v) & (CARD["bin"] == label)]["points"]
        rows.append({"variable": v, "value": (f"{app[v]:,.0f}" if app[v] >= 1000 else f"{app[v]:.3g}") if isinstance(app[v], float) else str(app[v]), "bin": label, "WoE": round(w, 3),
                     "points": int(pts.iloc[0]) if len(pts) else None})
    if recalibrate:
        z += VAL["oot_intercept_shift"]
    pd_life = 1 / (1 + np.exp(-z))
    score = MODEL["offset"] - MODEL["factor"] * np.log(pd_life / (1 - pd_life))
    return pd_life, score, pd.DataFrame(rows)


def irb_k(pd_, lgd):
    pd_ = max(pd_, 0.0005)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    r = 0.03 * w + 0.16 * (1 - w)
    return lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pd_ * lgd


st.set_page_config(page_title="Credit Scorecard & ECL", layout="wide")
st.title("Retail credit scorecard - PD, ECL and capital for one application")
st.caption("Scorecard built on 2012-13 Lending Club loans, tested on 2014; LGD/EAD from 2007-2012 defaults. For learning purposes only.")

LABELS = {
    "annual_inc": "Annual income ($)", "loan_to_income": "Loan amount / income",
    "dti": "Debt-to-income ratio (%)", "revol_util": "Revolving credit utilisation (0-1)",
    "inq_last_6mths": "Credit enquiries, last 6 months", "bc_open_to_buy": "Unused credit-card limit ($)",
    "tot_hi_cred_lim": "Total credit limit ($)", "acc_open_past_24mths": "Accounts opened, last 24 months",
    "mo_sin_rcnt_tl": "Months since most recent account opened", "num_rev_tl_bal_gt_0": "Revolving accounts with a balance",
    "total_acc": "Total credit accounts", "home_ownership": "Home ownership", "purpose": "Loan purpose",
    "emp_length": "Employment length",
}
with st.sidebar:
    st.header("Application")
    amount = st.number_input("Loan amount ($)", 1000, 35000, 12000, step=500)
    rate = st.slider("Interest rate (%)", 5.0, 26.0, 12.0, 0.5) / 100
    app = {}
    for v in MODEL["selected"]:
        spec, default = MODEL["bins"][v], MODEL["defaults"][v]
        if v == "loan_to_income":
            continue
        if spec["kind"] == "cat":
            opts = spec["levels"] + (["Other"] if "Other" in spec["woe"] else [])
            app[v] = st.selectbox(LABELS.get(v, v), opts, index=opts.index(default) if default in opts else 0)
        else:
            app[v] = st.number_input(LABELS.get(v, v), value=float(default), min_value=0.0)
    app["loan_to_income"] = amount / max(app.get("annual_inc", 1.0), 1.0)
    recal = st.checkbox("Use recalibrated PD (2014 annual review)", True)

pd_life, score, detail = score_application(app, recal)
pd12 = pd_life * ECL["share_defaults_in_first_12m"]
lgd = ECL["lgd_overall"]
ecl12 = pd12 * lgd * amount * ECL["ead_factor_12m"] * (1 + rate) ** (-ECL["avg_default_time_yrs_12m"])
ecl_life = pd_life * lgd * amount * ECL["ead_factor_lifetime"] * (1 + rate) ** (-ECL["avg_default_time_yrs_life"])
k = irb_k(pd12, lgd)

c = st.columns(5)
c[0].metric("Score", f"{score:.0f}")
c[1].metric("Lifetime PD (36m)", f"{pd_life:.1%}")
c[2].metric("12-month PD", f"{pd12:.2%}")
c[3].metric("Stage 1 ECL / Stage 2 ECL", f"${ecl12:,.0f} / ${ecl_life:,.0f}")
c[4].metric("Basel capital (8% of RWA)", f"${k * amount:,.0f}", f"RWA {12.5 * k:.0%} of EAD", delta_color="off")

st.subheader("How the score is built")
st.dataframe(detail, hide_index=True, width="stretch")
st.caption(f"Points add up to the score (rounding aside). 600 points = odds of 50:1 good:bad; "
           f"every {MODEL['pdo']} points doubles the odds. With recalibration on, the score shown is "
           f"about {MODEL['factor'] * VAL['oot_intercept_shift']:.0f} points below the sum of points.")

st.subheader("Model performance")
v = VAL
st.table(pd.DataFrame({
    "Sample": ["Train 2012-13", "In-time test 2012-13", "Out-of-time 2014"],
    "Gini": [v[s]["gini"] for s in ["train", "test_intime", "oot_2014"]],
    "KS": [v[s]["ks"] for s in ["train", "test_intime", "oot_2014"]],
    "Actual default rate": [v[s]["default_rate"] for s in ["train", "test_intime", "oot_2014"]],
    "Mean predicted PD": [v[s]["mean_pd"] for s in ["train", "test_intime", "oot_2014"]],
}).round(3))
st.image(str(ROOT / "results" / "figures" / "default_rate_by_band.png"), width=560)
