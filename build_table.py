"""Step 1 — one row per close-approach event, built only from what was known
two days before closest approach.

    python3 build_table.py

Input   data/Collision Avoidance Challenge - Dataset/raw_data/raw_data_2015-2019.gz
        (199,082 CDMs, 15,321 events, 2015-2019) and the two competition files,
        which are used only to mark which events ESA put in its test set.
Output  data/processed/events.csv.gz   one row per eligible event
        data/processed/eda.json        counts quoted in the README

Definitions (ESA, kelvins.esa.int and Uriot et al. 2022):
  * the LABEL is the risk in the LAST CDM of the event (smallest time_to_tca),
    high risk = final log10 risk >= -6;
  * the INPUTS are the CDMs received at least 2 days before closest approach
    (time_to_tca >= 2), because a manoeuvre is planned at least 2 days out;
  * an event is ELIGIBLE when it has at least one usable CDM and its last CDM
    is within 1 day of closest approach (otherwise the "final" risk is not a
    final risk).  This is exactly the filter ESA used for its test set, and it
    leaves 10,460 events of which 216 are high risk.

Every feature comes from the usable CDMs only.  The row never sees the CDMs
inside the last two days, and in particular never sees the CDM that carries
the label.  (test_data_private.csv is read to check that the labels we derive
from the raw database agree with ESA's; they do, for all 2,167 test events.)
"""
import os
import json
import numpy as np
import pandas as pd

from config import DATA, PROC, HIGH_RISK, CUTOFF_DAYS, FINAL_WITHIN_DAYS, CATEGORICAL

ROOT = os.path.join(DATA, "Collision Avoidance Challenge - Dataset")
RAW = os.path.join(ROOT, "raw_data", "raw_data_2015-2019.gz")
KEL = os.path.join(ROOT, "kelvins_competition_data")

# Columns summarised over the usable CDMs (the "history" features).  The slope
# is a least-squares slope per day towards closest approach, so a positive
# risk_slope means the risk was rising as the approach got nearer.
HIST_COLS = ["risk", "miss_distance", "mahalanobis_distance", "max_risk_estimate",
             "c_sigma_t", "c_sigma_r", "c_sigma_n", "t_sigma_t", "c_obs_used"]


def fingerprint(df):
    key = ["time_to_tca", "mission_id", "risk", "miss_distance", "relative_speed",
           "t_j2k_sma", "c_j2k_sma", "mahalanobis_distance"]
    return df[key].round(6).astype(str).agg("|".join, axis=1)


def slope(t, y):
    """Least-squares slope of y against time-before-TCA counted forwards (-time_to_tca)."""
    if len(t) < 2:
        return np.nan
    x = -np.asarray(t, float); y = np.asarray(y, float)
    x = x - x.mean()
    d = (x * x).sum()
    return float((x * (y - y.mean())).sum() / d) if d > 0 else np.nan


def main():
    raw = pd.read_csv(RAW)
    assert raw.shape == (199082, 103), raw.shape
    raw = raw.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).reset_index(drop=True)
    g = raw.groupby("event_id", sort=True)

    # ---- label and eligibility, from the whole series ------------------------
    ev = pd.DataFrame({
        "mission_id": g.mission_id.first(),
        "n_cdms_total": g.size(),
        "first_ttca": g.time_to_tca.max(),
        "last_ttca": g.time_to_tca.min(),
        "final_risk": g.risk.last(),                 # last row = smallest time_to_tca
    })
    ev["high_risk"] = (ev.final_risk >= HIGH_RISK).astype(int)
    assert (ev.high_risk.sum(), (ev.final_risk >= -5).sum(), (ev.final_risk >= -4).sum()) == (515, 131, 30)

    # ---- usable CDMs: at least CUTOFF_DAYS before closest approach ------------
    use = raw[raw.time_to_tca >= CUTOFF_DAYS]
    gu = use.groupby("event_id", sort=True)
    latest = gu.last()                                 # smallest time_to_tca >= 2 (sorted descending)
    latest = latest.drop(columns=["mission_id"])
    latest.columns = [c for c in latest.columns]       # keep the CDM names as they are

    hist = pd.DataFrame(index=latest.index)
    hist["n_cdms_used"] = gu.size()
    hist["span_days"] = gu.time_to_tca.max() - gu.time_to_tca.min()
    for c in HIST_COLS:
        hist[f"{c}_first"] = gu[c].first()
        hist[f"{c}_mean"] = gu[c].mean()
        hist[f"{c}_min"] = gu[c].min()
        hist[f"{c}_max"] = gu[c].max()
        hist[f"{c}_std"] = gu[c].std().fillna(0.0)
        hist[f"{c}_change"] = gu[c].last() - gu[c].first()
        hist[f"{c}_slope"] = gu.apply(lambda d, c=c: slope(d.time_to_tca.values, d[c].values))
    hist["risk_max_jump"] = gu.risk.apply(lambda s: float(np.abs(np.diff(s.values)).max()) if len(s) > 1 else 0.0)

    tab = ev.join(latest, how="inner").join(hist, how="inner")
    tab["eligible"] = (tab.last_ttca < FINAL_WITHIN_DAYS) & (tab.n_cdms_used >= 1) & (tab.n_cdms_total >= 2)
    tab = tab[tab.eligible].drop(columns=["eligible"]).copy()
    assert len(tab) == 10460 and tab.high_risk.sum() == 216, (len(tab), tab.high_risk.sum())

    # ---- which events ESA used as its competition test set --------------------
    te = pd.read_csv(os.path.join(KEL, "test_data.csv"))
    tp = pd.read_csv(os.path.join(KEL, "test_data_private.csv")).set_index("event_id")
    rk = fingerprint(raw); raw_ids = pd.Series(raw.event_id.values, index=rk.values)
    te_ids = pd.Series(te.event_id.values, index=fingerprint(te).values)
    m = pd.DataFrame({"te": te_ids}).join(raw_ids.rename("raw"), how="inner")
    te2raw = m.groupby("te").raw.agg(lambda s: s.mode()[0]).astype(int)
    assert te2raw.nunique() == 2167 == len(te2raw)
    agree = np.isclose(tp.true_risk.reindex(te2raw.index).values, ev.final_risk.reindex(te2raw.values).values)
    assert agree.all(), "labels from the raw database disagree with ESA's private test labels"
    tab["esa_test"] = tab.index.isin(te2raw.values).astype(int)
    assert tab.esa_test.sum() == 2167 and tab.loc[tab.esa_test == 1, "high_risk"].sum() == 150

    # ---- the naive forecast, kept as a column for the baseline ---------------
    tab["lrp_high"] = (tab.risk >= HIGH_RISK).astype(int)

    tab.index.name = "event_id"
    out = os.path.join(PROC, "events.csv.gz")
    tab.to_csv(out)

    eda = {
        "cdms_raw": int(len(raw)), "events_raw": int(len(ev)),
        "high_risk_raw": {"-6": 515, "-5": 131, "-4": 30},
        "cdms_with_negative_time_to_tca": int((raw.time_to_tca < 0).sum()),
        "events_with_a_usable_cdm": int((ev.index.isin(latest.index)).sum()),
        "eligible_events": int(len(tab)), "eligible_high_risk": int(tab.high_risk.sum()),
        "eligible_base_rate": round(float(tab.high_risk.mean()), 4),
        "esa_test_events": 2167, "esa_test_high_risk": 150,
        "esa_train_eligible_events": int((tab.esa_test == 0).sum()),
        "esa_train_eligible_high_risk": int(tab.loc[tab.esa_test == 0, "high_risk"].sum()),
        "lrp_confusion_all_eligible": {
            "TP": int(((tab.lrp_high == 1) & (tab.high_risk == 1)).sum()),
            "FP": int(((tab.lrp_high == 1) & (tab.high_risk == 0)).sum()),
            "FN": int(((tab.lrp_high == 0) & (tab.high_risk == 1)).sum()),
            "TN": int(((tab.lrp_high == 0) & (tab.high_risk == 0)).sum())},
        "cdms_used_per_event": tab.n_cdms_used.describe().round(2).to_dict(),
        "latest_usable_time_to_tca": tab.time_to_tca.describe().round(3).to_dict(),
        "missions": tab.groupby("mission_id").high_risk.agg(["size", "sum"]).rename(
            columns={"size": "events", "sum": "high"}).to_dict("index"),
        "object_type": tab.c_object_type.value_counts(dropna=False).to_dict(),
        "missing_fraction_top": tab.isna().mean().sort_values(ascending=False).head(8).round(3).to_dict(),
        "n_columns": int(tab.shape[1]),
    }
    json.dump(eda, open(os.path.join(PROC, "eda.json"), "w"), indent=1, default=str)
    print(json.dumps(eda, indent=1, default=str))
    print("wrote", out, tab.shape)


if __name__ == "__main__":
    main()
