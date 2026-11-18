"""
Weight of Evidence (WoE) binning and Information Value (IV), written from scratch
so every step can be explained.

  WoE(bin) = ln( %Good in bin / %Bad in bin )          (Siddiqi convention)
     WoE > 0 -> safer than average, WoE < 0 -> riskier than average
  IV       = sum over bins of (%Good - %Bad) * WoE
     rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong

Numeric variables: start from fine quantile bins ("fine classing"), then merge
neighbouring bins until (a) the bad rate moves in one direction only
(monotonic - needed so the scorecard makes business sense) and (b) every bin
holds at least MIN_SHARE of the sample ("coarse classing").
Missing values always get their own bin.
Categorical variables: rare levels (< MIN_SHARE_CAT) are pooled into "Other".
"""
import numpy as np
import pandas as pd

MIN_SHARE = 0.05       # numeric bins: at least 5% of the sample
MIN_SHARE_CAT = 0.01   # categorical levels: at least 1% (keeps rare but risky groups, e.g. small_business)
EPS = 0.5  # small count added to avoid log(0) in tiny bins


def _woe_table(groups: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"g": groups.values, "y": y.values}).groupby("g", observed=True)["y"].agg(["size", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    tot_g, tot_b = t["good"].sum(), t["bad"].sum()
    t["pct_good"] = (t["good"] + EPS) / (tot_g + EPS * len(t))
    t["pct_bad"] = (t["bad"] + EPS) / (tot_b + EPS * len(t))
    t["woe"] = np.log(t["pct_good"] / t["pct_bad"])
    t["iv"] = (t["pct_good"] - t["pct_bad"]) * t["woe"]
    t["bad_rate"] = t["bad"] / t["n"]
    t["share"] = t["n"] / t["n"].sum()
    return t


def _monotone_merge(edges, x, y):
    """Merge adjacent intervals until bad rate is monotonic and bins are big enough."""
    n = len(x)
    while True:
        b = pd.cut(x, edges, include_lowest=True, duplicates="drop")
        t = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=True)["y"].agg(["size", "mean"])
        rates = t["mean"].values
        sizes = t["size"].values / n
        direction = (np.sign(np.corrcoef(np.arange(len(rates)), rates)[0, 1]) or 1) if len(rates) > 2 else 1
        diffs = np.diff(rates) * direction
        bad_pairs = np.where(diffs < 0)[0]          # neighbours breaking monotonicity
        small = np.where(sizes < MIN_SHARE)[0]
        if len(t) == 1 or (len(bad_pairs) == 0 and len(small) == 0):
            return edges
        if len(bad_pairs):
            # merge the violating pair whose bad rates are closest
            i = bad_pairs[np.argmin(np.abs(np.diff(rates))[bad_pairs])]
        else:
            j = small[0]
            i = j - 1 if j == len(rates) - 1 else (j if j == 0 else (j - 1 if abs(rates[j] - rates[j - 1]) < abs(rates[j] - rates[j + 1]) else j))
        edges = np.delete(edges, i + 1)   # remove the boundary between bin i and i+1


class WoEBinner:
    def __init__(self, n_fine=20):
        self.n_fine = n_fine
        self.spec = {}    # var -> dict(kind, edges / mapping)
        self.tables = {}  # var -> WoE table

    def fit(self, df: pd.DataFrame, y: pd.Series, num_vars, cat_vars):
        for v in num_vars:
            x = df[v]
            ok = x.notna()
            qs = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, self.n_fine + 1)))
            if x[ok].nunique() <= self.n_fine:   # discrete counts (e.g. enquiries): one bin per value to start
                inner = qs[:-1]
            else:                                # continuous: quantile cut points
                inner = qs[1:-1]
            edges = np.concatenate([[-np.inf], inner, [np.inf]])
            edges = _monotone_merge(edges, x[ok], y[ok])
            self.spec[v] = {"kind": "num", "edges": [float(e) for e in edges]}
            self.tables[v] = _woe_table(self._group(v, x), y)
        for v in cat_vars:
            share = df[v].value_counts(normalize=True)
            keep = share[share >= MIN_SHARE_CAT].index.tolist()
            self.spec[v] = {"kind": "cat", "levels": keep}
            self.tables[v] = _woe_table(self._group(v, df[v]), y)
        for v, t in self.tables.items():
            self.spec[v]["woe"] = {str(k): float(w) for k, w in t["woe"].items()}
        return self

    def _group(self, v, x):
        s = self.spec[v]
        if s["kind"] == "num":
            g = pd.cut(x, s["edges"], include_lowest=True).astype(str)
            return g.where(x.notna(), "Missing")
        return x.where(x.isin(s["levels"]), "Other").astype(str)

    def transform(self, df: pd.DataFrame, vars_):
        out = {}
        for v in vars_:
            g = self._group(v, df[v])
            w = self.spec[v]["woe"]
            out[v] = g.map(w).fillna(0.0)   # unseen bin -> neutral WoE 0
        return pd.DataFrame(out, index=df.index)

    def iv(self):
        return pd.Series({v: t["iv"].sum() for v, t in self.tables.items()}).sort_values(ascending=False)
