"""
Step 4 - Turn PDs into money: LGD, EAD, IFRS 9 expected credit loss, Basel IRB capital,
and a baseline vs adverse stress scenario, for the 2014 loan book (153K loans).

Parameters estimated from history (2007-2012 defaults, whose recovery process had
finished by the 2017 data snapshot), then applied to 2014:

  LGD  = 1 - (recoveries - collection fees) / exposure at default      (per grade)
  EAD factor = exposure at default / original loan amount               (amortising loan)
  Default timing: share of lifetime defaults that happen in the first 12 months
         (default month ~= month of last payment + 4, i.e. 120 days past due)
  PD_12m = PD_lifetime x share_in_first_12_months

  IFRS 9:  Stage 1 ECL = PD_12m      x LGD x EAD_12m     (all loans are Stage 1 on day one)
           Stage 2 ECL = PD_lifetime x LGD x EAD_life    (what it would be after a significant
                                                         increase in credit risk)
           Cash flows discounted at the loan's own interest rate.
  Basel IRB (other retail):
     R = 0.03*(1-e^(-35PD))/(1-e^(-35)) + 0.16*(1-(1-e^(-35PD))/(1-e^(-35)))
     K = LGD*N[ (G(PD) + sqrt(R)*G(0.999)) / sqrt(1-R) ] - PD*LGD
     RWA = 12.5 * K * EAD,   capital at 8% of RWA = K * EAD
     PD = 12-month PD floored at 0.05% (Basel III retail floor)
  Adverse scenario:
     PD odds multiplied by the "crisis vintage effect": a logistic regression on 2007-2013 loans with
     borrower-risk variables (WoE) plus a dummy for 2007-08 issue years. exp(dummy coefficient) = how much
     higher the default odds of crisis-era loans were for the SAME borrower quality. LGD +5 percentage points.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from woe import WoEBinner

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
LGD_ADDON = 0.05
PD_FLOOR = 0.0005


def odds_shift(pd_, mult):
    o = pd_ / (1 - pd_) * mult
    return o / (1 + o)


def irb_capital_k(pd_, lgd):
    pd_ = np.maximum(pd_, PD_FLOOR)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    r = 0.03 * w + 0.16 * (1 - w)
    k = lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pd_ * lgd
    return k


def main():
    df = pd.read_parquet(ROOT / "data" / "scored_loans.parquet")
    val = json.load(open(RES / "validation_metrics.json"))

    # ---------- parameters from history (2007-2012 charged-off loans) ----------
    h = df[(df["loan_status"] == "Charged Off") & (df["issue_year"] <= 2012)].copy()
    h["ead"] = (h["funded_amnt"] - h["total_rec_prncp"]).clip(lower=0)
    h["lgd"] = (1 - (h["recoveries"] - h["collection_recovery_fee"]) / h["ead"]).clip(0, 1)
    months_to_last_pay = ((h["last_pymnt_d"].dt.year - h["issue_d"].dt.year) * 12
                          + (h["last_pymnt_d"].dt.month - h["issue_d"].dt.month)).fillna(0)
    h["default_month"] = months_to_last_pay + 4
    h["ead_factor"] = h["ead"] / h["funded_amnt"]
    h["in_12m"] = h["default_month"] <= 12
    share_12m = float(h["in_12m"].mean())
    ead_f_12 = float(h.loc[h["in_12m"], "ead_factor"].mean())
    ead_f_life = float(h["ead_factor"].mean())
    t_12 = float(h.loc[h["in_12m"], "default_month"].mean()) / 12
    t_life = float(h["default_month"].mean()) / 12
    lgd_by_grade = h.groupby("grade")["lgd"].mean()
    lgd_all = float(h["lgd"].mean())

    # crisis stress multiplier: vintage effect of 2007-08 loans, controlling for borrower quality
    loans = pd.read_parquet(ROOT / "data" / "loans.parquet")
    hist = loans[loans["issue_year"] <= 2013]
    base_num = ["annual_inc", "loan_to_income", "dti", "revol_util", "inq_last_6mths"]
    base_cat = ["purpose", "home_ownership"]
    wb = WoEBinner().fit(hist, hist["default"], base_num, base_cat)
    Xc = wb.transform(hist, base_num + base_cat)
    Xc["crisis_2007_08"] = (hist["issue_year"] <= 2008).astype(float)
    cfit = sm.Logit(hist["default"], sm.add_constant(Xc)).fit(disp=0)
    stress_mult = float(np.exp(cfit.params["crisis_2007_08"]))
    stress_ci = [float(np.exp(x)) for x in cfit.conf_int().loc["crisis_2007_08"]]

    # ---------- the 2014 book ----------
    b = df[df["sample"] == "oot_2014"].copy()
    b["lgd"] = b["grade"].map(lgd_by_grade).fillna(lgd_all)
    # recalibrated PD: model PD with the 2014 intercept shift (annual-review recalibration)
    b["pd_recal"] = 1 / (1 + np.exp(-(np.log(b["pd"] / (1 - b["pd"])) + val["oot_intercept_shift"])))

    def run(pd_life, lgd, name):
        pd12 = pd_life * share_12m
        df12 = (1 + b["int_rate"]) ** (-t_12)
        dfl = (1 + b["int_rate"]) ** (-t_life)
        ecl12 = pd12 * lgd * b["funded_amnt"] * ead_f_12 * df12
        ecl_life = pd_life * lgd * b["funded_amnt"] * ead_f_life * dfl
        k = irb_capital_k(pd12, lgd)
        ead = b["funded_amnt"]
        return {"scenario": name, "exposure": float(ead.sum()), "avg_pd_lifetime": float(pd_life.mean()),
                "avg_pd_12m": float(pd12.mean()), "avg_lgd": float(lgd.mean()),
                "ecl_stage1_12m": float(ecl12.sum()), "ecl_stage2_lifetime": float(ecl_life.sum()),
                "rwa": float((12.5 * k * ead).sum()), "capital_8pct": float((k * ead).sum()),
                "rwa_density": float((12.5 * k * ead).sum() / ead.sum())}, ecl12, ecl_life

    base, e12, el = run(b["pd"], b["lgd"], "Baseline (model PD)")
    recal, _, el_recal = run(b["pd_recal"], b["lgd"], "Baseline (recalibrated PD)")
    adverse, _, _ = run(odds_shift(b["pd_recal"], stress_mult), (b["lgd"] + LGD_ADDON).clip(0, 1),
                        "Adverse (crisis PD x LGD +5pp)")
    scen = pd.DataFrame([base, recal, adverse])

    # back-test: realised loss on 2014 loans (recoveries may still be in progress -> upper bound)
    r = b[b["default"] == 1]
    realised = float(((r["funded_amnt"] - r["total_rec_prncp"]).clip(lower=0)
                      - (r["recoveries"] - r["collection_recovery_fee"])).clip(lower=0).sum())

    def undisc(pd_life):
        return float((pd_life * b["lgd"] * b["funded_amnt"] * ead_f_life).sum())

    # ECL by score band
    b["band"] = pd.cut(b["score"], [-np.inf, 525, 535, 545, 555, 565, 575, np.inf],
                       labels=["<=525", "525-535", "535-545", "545-555", "555-565", "565-575", ">575"])
    by_band = b.assign(ecl12=e12, ecl_life=el).groupby("band", observed=True).agg(
        loans=("id", "size"), exposure=("funded_amnt", "sum"), avg_pd=("pd", "mean"),
        actual_default=("default", "mean"), ecl_12m=("ecl12", "sum"), ecl_lifetime=("ecl_life", "sum"))
    by_band["ecl_12m_pct"] = by_band["ecl_12m"] / by_band["exposure"]
    by_band["ecl_life_pct"] = by_band["ecl_lifetime"] / by_band["exposure"]
    by_band.to_csv(RES / "ecl_by_score_band.csv", float_format="%.4f")
    scen.to_csv(RES / "scenario_results.csv", index=False, float_format="%.2f")

    params = {"lgd_overall": lgd_all, "lgd_by_grade": lgd_by_grade.round(4).to_dict(),
              "share_defaults_in_first_12m": share_12m, "ead_factor_12m": ead_f_12,
              "ead_factor_lifetime": ead_f_life, "avg_default_time_yrs_12m": t_12,
              "avg_default_time_yrs_life": t_life, "crisis_odds_multiplier": stress_mult, "crisis_multiplier_95ci": stress_ci,
              "lgd_addon_adverse": LGD_ADDON, "n_defaults_used": int(len(h)),
              "realised_loss_2014": realised,
              # back-test uses UNdiscounted lifetime loss so it is comparable with realised cash loss
              "undiscounted_loss_model": undisc(b["pd"]),
              "undiscounted_loss_recal": undisc(b["pd_recal"]),
              "backtest_model_vs_realised": undisc(b["pd"]) / realised,
              "backtest_recal_vs_realised": undisc(b["pd_recal"]) / realised}
    json.dump({"parameters": params, "scenarios": scen.to_dict("records")}, open(RES / "ecl_capital.json", "w"), indent=1)

    # figure: 12m vs lifetime ECL rate by band
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(by_band))
    ax.bar(x - 0.18, by_band["ecl_12m_pct"] * 100, 0.34, color="#2a78d6", label="Stage 1: 12-month ECL")
    ax.bar(x + 0.18, by_band["ecl_life_pct"] * 100, 0.34, color="#eb6834", label="Stage 2: lifetime ECL")
    ax.set_xticks(x, by_band.index.astype(str))
    ax.set(xlabel="Score band (2014 loans)", ylabel="ECL as % of loan amount",
           title="Expected credit loss by score band")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1"); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "ecl_by_band.png", dpi=150, facecolor="#fcfcfb"); plt.close(fig)

    pd.set_option("display.width", 200)
    print(json.dumps(params, indent=1))
    print(scen.round(4).T)
    print(by_band.round(4))


if __name__ == "__main__":
    main()
