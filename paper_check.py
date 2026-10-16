"""Step 2b — repeat the paper's own machine-learning experiment on our event table.

    python3 paper_check.py                       # all 505 splits in one go (about 40 min)
    python3 paper_check.py --start 0 --stop 50   # a chunk; then  --merge  once all chunks exist

Uriot et al. (2022, section 6.2) report that over 505 stratified random 80/20 splits of
the 10,460 eligible events, the naive forecast (latest risk, decided at -6) has a mean
F2 of 0.59967 (sd 0.04391) and a deliberately weak gradient-boosting regressor trained
on the CHANGE in risk, h = r_final - r_latest, has a mean F2 of 0.60732 (sd 0.04895):
a gain of 1.27 %, significant under a paired t-test (t = 10.2, p = 1.8e-22).

Their recipe: the latest usable CDM only, its 100 numeric columns as they are (no
scaling, no imputation, no class weights), target h passed through a quantile
transformer to a uniform distribution, LightGBM with n_estimators 25, feature_fraction
0.25, learning_rate 0.05 (100 such models per split).  We run exactly that
(LGBMRegressor, when lightgbm is installed), the nearest scikit-learn equivalent
(HistGradientBoostingRegressor with max_iter 25, learning_rate 0.05, max_features 0.25,
which also handles missing values natively), and the stronger settings our models.py
uses for its Δrisk model (150 trees, learning rate 0.1).

The point of the check is whether OUR Δrisk model loses to the naive forecast because
the idea does not work on this data, or because a stronger regressor moves the forecast
too far from the latest risk.  'mean |shift|' is how far, in log10 risk, each model moves
the forecast away from the latest risk on the test events.

Output  results/paper_check.json, results/paper_check_splits.csv and a printed table.
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import fbeta_score, average_precision_score

from config import OUT, SEED, HIGH_RISK
import models

N_SPLITS = 505          # as in the paper
MODELS_PER_SPLIT = 3    # the paper trained 100 per split; 3 is enough for the mean we need


def _lgbm(seed):
    """LGBMRegressor with the paper's three changes to the defaults (section 6.1).
    Returns None when lightgbm is not installed, and the setting is then skipped."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    return LGBMRegressor(n_estimators=25, feature_fraction=0.25, learning_rate=0.05, random_state=seed,
                         n_jobs=1, verbose=-1)


SETTINGS = {
    "lgbm": ("LightGBM exactly as in the paper (25 trees, lr 0.05, 25 % of features per tree)", _lgbm),
    "hgb_paper": ("sklearn HGB with the paper's settings (25 trees, lr 0.05, 25 % of features)",
                  lambda seed: HistGradientBoostingRegressor(max_iter=25, learning_rate=0.05, max_features=0.25,
                                                             random_state=seed)),
    "hgb_ours": ("sklearn HGB with our models.py settings (150 trees, lr 0.1, 15 leaves)",
                 lambda seed: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                                                            min_samples_leaf=20, l2_regularization=1.0,
                                                            random_state=seed)),
}


def f2(y, pred):
    return fbeta_score(y, (pred >= HIGH_RISK).astype(int), beta=2)


def run(start, stop):
    tab = models.load()
    num, _ = models.feature_lists("latest", tab)          # the latest CDM's 100 numeric columns
    X = tab[num].values.astype(float)
    r_latest = tab.risk.values.astype(float)
    h = tab.final_risk.values.astype(float) - r_latest
    y = tab.high_risk.values

    # the same 505 splits whatever chunk is being run
    splits = list(StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=SEED).split(X, y))
    rows, t0 = [], time.time()
    for s in range(start, stop):
        tr, te = splits[s]
        row = {"split": s, "n_test_high": int(y[te].sum()),
               "lrp_f2": f2(y[te], r_latest[te]), "lrp_ap": average_precision_score(y[te], r_latest[te])}
        for key, (name, make) in SETTINGS.items():
            f2s, aps, moved = [], [], []
            for seed in range(MODELS_PER_SPLIT):
                base = make(SEED + seed)
                if base is None:
                    break
                reg = TransformedTargetRegressor(
                    regressor=base,
                    transformer=QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=SEED))
                reg.fit(X[tr], h[tr])
                pred = r_latest[te] + reg.predict(X[te])
                f2s.append(f2(y[te], pred)); aps.append(average_precision_score(y[te], pred))
                moved.append(float(np.mean(np.abs(pred - r_latest[te]))))
            if f2s:
                row[f"{key}_f2"] = float(np.mean(f2s)); row[f"{key}_ap"] = float(np.mean(aps))
                row[f"{key}_shift"] = float(np.mean(moved))
        rows.append(row)
        if (s + 1 - start) % 10 == 0:
            print(f"splits {start}-{s} done, {time.time() - t0:.0f} s", file=sys.stderr)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"paper_check_rows_{start:03d}_{stop:03d}.csv"), index=False)


def merge():
    files = sorted(glob.glob(os.path.join(OUT, "paper_check_rows_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("split").sort_values("split")
    df.to_csv(os.path.join(OUT, "paper_check_splits.csv"), index=False)
    out = {"n_splits": int(len(df)), "models_per_split": MODELS_PER_SPLIT, "test_size": 0.2,
           "lrp_f2_mean": df.lrp_f2.mean(), "lrp_f2_sd": df.lrp_f2.std(ddof=0), "lrp_ap_mean": df.lrp_ap.mean(),
           "paper_reported": {"lrp_f2_mean": 0.59967, "lrp_f2_sd": 0.04391, "lgbm_f2_mean": 0.60732,
                              "lgbm_f2_sd": 0.04895, "gain_pct": 1.2743, "paired_t": 10.23,
                              "p_value": 1.83e-22, "n_splits": 505, "models_per_split": 100}}
    for key, (name, _) in SETTINGS.items():
        if f"{key}_f2" not in df:
            continue
        d = df[f"{key}_f2"] - df.lrp_f2
        t = stats.ttest_rel(df[f"{key}_f2"], df.lrp_f2)
        out[key] = {"settings": name, "f2_mean": df[f"{key}_f2"].mean(), "f2_sd": df[f"{key}_f2"].std(ddof=0),
                    "gain_pct": 100 * (df[f"{key}_f2"].mean() / df.lrp_f2.mean() - 1),
                    "f2_diff_mean": d.mean(), "f2_diff_sd": d.std(ddof=0),
                    "splits_better": int((d > 0).sum()), "splits_worse": int((d < 0).sum()),
                    "paired_t": float(t.statistic), "p_value": float(t.pvalue),
                    "ap_mean": df[f"{key}_ap"].mean(), "mean_abs_shift_of_forecast": df[f"{key}_shift"].mean()}
    json.dump(out, open(os.path.join(OUT, "paper_check.json"), "w"), indent=1, default=float)

    print(f"\n{len(df)} stratified 80/20 splits of the 10,460 eligible events, {MODELS_PER_SPLIT} models per split")
    print(f"naive forecast (LRP)  F2 {out['lrp_f2_mean']:.4f} ± {out['lrp_f2_sd']:.4f}  AP {out['lrp_ap_mean']:.3f}"
          f"   (paper: F2 0.5997 ± 0.0439)")
    for key in SETTINGS:
        if key in out:
            o = out[key]
            print(f"{o['settings']}\n    F2 {o['f2_mean']:.4f} ± {o['f2_sd']:.4f}  gain {o['gain_pct']:+.2f} %  "
                  f"better in {o['splits_better']}/{len(df)} splits  paired t {o['paired_t']:+.1f}  p {o['p_value']:.1e}  "
                  f"AP {o['ap_mean']:.3f}  mean |shift| {o['mean_abs_shift_of_forecast']:.2f}")
    print("(paper: LightGBM F2 0.6073 ± 0.0490, +1.27 %, t = 10.2, p = 1.8e-22)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=N_SPLITS)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        run(a.start, min(a.stop, N_SPLITS))
        if a.start == 0 and a.stop >= N_SPLITS:
            merge()
