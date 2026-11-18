"""
Step 1 - Prepare the Lending Club loan book for credit risk modelling.

Input : Lending Club issued-loan files for 2007-2014 (one .rda file per year,
        from github.com/kuhnrl30/LendingClubData - see README "Data").
Output: data/loans.parquet  (one row per loan, application-time features +
        outcome + the fields needed later for LGD / EAD).

Key choices (explain these in an interview):
  * Only 36-month loans  -> every loan issued up to Dec-2014 had reached its
    maturity date by the data snapshot (2017), so the outcome is known.
  * Loans marked "Does not meet the credit policy" are dropped: they were
    approved outside the normal policy, so they are not the population a
    scorecard is built for.
  * Bad  (default = 1): Charged Off, Default, Late (31-120 days)
    Good (default = 0): Fully Paid
    Indeterminate (dropped): Current, In Grace Period, Late (16-30 days)
  * Only information known on the application date is used as a feature.
    Payment history, recoveries etc. are kept ONLY for LGD/EAD, never as
    model inputs (that would be data leakage).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

RAW_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "loans.parquet"

KEEP = [
    # identifiers / timing
    "id", "issue_d", "issue_year", "term", "loan_status", "grade", "sub_grade", "int_rate",
    # application-time borrower and loan features
    "loan_amnt", "funded_amnt", "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status", "purpose", "earliest_cr_line", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "pub_rec_bankruptcies", "installment", "addr_state",
    # credit-bureau attributes pulled at application (recorded by Lending Club from 2012 onwards)
    "mort_acc", "acc_open_past_24mths", "bc_util", "bc_open_to_buy", "tot_cur_bal", "avg_cur_bal",
    "num_actv_rev_tl", "mo_sin_rcnt_tl", "mo_sin_old_rev_tl_op", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "tot_hi_cred_lim", "total_bc_limit", "num_tl_op_past_12m", "mths_since_recent_inq", "total_rev_hi_lim",
    "num_rev_tl_bal_gt_0",
    # post-origination fields: used ONLY for LGD / EAD, never as features
    "total_rec_prncp", "recoveries", "collection_recovery_fee", "last_pymnt_d",
]

BUREAU = ["mort_acc", "acc_open_past_24mths", "bc_util", "bc_open_to_buy", "tot_cur_bal", "avg_cur_bal",
          "num_actv_rev_tl", "mo_sin_rcnt_tl", "mo_sin_old_rev_tl_op", "pct_tl_nvr_dlq", "percent_bc_gt_75",
          "tot_hi_cred_lim", "total_bc_limit", "num_tl_op_past_12m", "mths_since_recent_inq",
          "total_rev_hi_lim", "num_rev_tl_bal_gt_0"]

BAD = {"Charged Off", "Default", "Late (31-120 days)"}
GOOD = {"Fully Paid"}


def main():
    frames = []
    for year in range(2007, 2015):
        f = RAW_DIR / f"LendingClubData_{year}.rda"
        df = list(pyreadr.read_r(str(f)).values())[0]
        frames.append(df[KEEP])
    df = pd.concat(frames, ignore_index=True)
    n_all = len(df)

    df = df[df["term"].astype(str).str.strip() == "36"]
    n_36 = len(df)
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)

    # --- types and derived application-time features ---
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["issue_year"] = df["issue_d"].dt.year.astype(int)
    for c in ["addr_state", "loan_status", "grade", "sub_grade", "emp_length", "home_ownership",
              "verification_status", "purpose", "id"]:
        df[c] = df[c].astype(object)
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"])
    df["credit_hist_yrs"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    num_cols = ["mths_since_last_delinq", "pub_rec_bankruptcies", "delinq_2yrs", "inq_last_6mths",
                "open_acc", "pub_rec", "total_acc", "revol_bal"] + BUREAU
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["installment_to_income"] = df["installment"] * 12 / df["annual_inc"].replace(0, np.nan)
    df["emp_length"] = df["emp_length"].replace({"n/a": "Unknown"}).fillna("Unknown")
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})

    df.to_parquet(OUT, index=False)
    print(f"all loans 2007-2014: {n_all:,}; 36-month: {n_36:,}; resolved & in policy: {len(df):,}")
    print(df.groupby("issue_year")["default"].agg(["size", "mean"]).round(3))


if __name__ == "__main__":
    main()
